#!/usr/bin/env bash
# The whole dispatch A/B in one exclusive single-node job (ab.sh run submits it): build the three variants on
# node-local disk, then for each mode and NVLink setting run one discarded warm-up per build and AB_REPS
# pairs of the runtime build and the matching fixed build, runtime first on odd reps and fixed first on even
# reps (ABBA). Results are copied to Lustre after every run; the local tree is removed at the end.
#SBATCH --exclusive
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00
set -uo pipefail
here=${AB_KIT:?set AB_KIT}
source "$here/ab-env.sh"
export EP_ENV_PRE=$EP_KIT/env-dfw.sh
source "$EP_KIT/env.sh"
export GPUS_PER_NODE=$AB_RANKS
# Node-local work directory: AB_LOCAL, else the first writable candidate.
if [ -z "${AB_LOCAL:-}" ]; then
    for c in $AB_LOCAL_CANDIDATES; do
        [ -d "$c" ] && [ -w "$c" ] && { AB_LOCAL=$c; break; }
    done
fi
export AB_WORK=${AB_LOCAL:?no writable node-local directory}/$USER/dispatch-ab-$SLURM_JOB_ID
mkdir -p "$AB_WORK/results"
trap 'rm -rf "$AB_WORK"' EXIT
echo "[$(date +%T)] node=$(hostname -s) work=$AB_WORK"
df -h "$AB_LOCAL"
# Per-run logs and build info are collected locally and appended to this one Lustre file after every run.
export AB_RESULTS=$AB_WORK/results
results_tar=$AB_DIR/results-$SLURM_JOB_ID.tar
# Stage the source tarball: one sequential read from Lustre.
cp "$AB_TAR" "$AB_WORK/src.tar"
export AB_TAR=$AB_WORK/src.tar

node=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
ctr=(--container-image="$EP_CONTAINER_IMAGE" --container-name="ab-$SLURM_JOB_ID"
     --container-mounts="$EP_CONTAINER_MOUNTS,$AB_WORK:$AB_WORK" --no-container-remap-root)
step=(-N1 -w "$node" --ntasks-per-node=1 --cpus-per-task="${SLURM_CPUS_ON_NODE:-1}" --cpu-bind=none)

srun "${step[@]}" "${ctr[@]}" --export=ALL bash "$here/ab-build.sh" < /dev/null
build_rc=$?
rm -f "$AB_TAR"
tar -cf "$results_tar" -C "$AB_RESULTS" .
(( build_rc == 0 )) || { echo "[$(date +%T)] build failed"; exit 1; }

plan=$AB_WORK/noexp_$AB_RANKS.json
python3 -c "import json, sys; print(json.dumps([list(range(int(sys.argv[1])))]))" "$AB_RANKS" > "$plan"
export EP_RESULTS=$AB_RESULTS

run() {  # run <tag> <variant> <mode> <nvl>
    local tag=$1 v=$2 mode=$3 nvl=$4
    echo "[$(date +%T)] $tag"
    srun "${step[@]}" --kill-on-bad-exit=1 "${ctr[@]}" --export=ALL,EP_ROOT="$AB_WORK/$v" \
        "$EP_KIT/run_node.sh" "$tag" "$mode" "$nvl" "$plan" "$node" < /dev/null
    echo "[$(date +%T)] $tag rc=$?"
    tar -rf "$results_tar" -C "$AB_RESULTS" "./$tag"
}

for mode in $AB_MODES; do
    for nvl in $AB_NVL; do
        run "ab-warm-$mode-$nvl-runtime" runtime "$mode" "$nvl"
        run "ab-warm-$mode-$nvl-fix$mode" "fix$mode" "$mode" "$nvl"
    done
done
for r in $(seq "$AB_REPS"); do
    for mode in $AB_MODES; do
        for nvl in $AB_NVL; do
            if (( r % 2 )); then order="runtime fix$mode"; else order="fix$mode runtime"; fi
            for v in $order; do
                run "ab-$mode-$nvl-$v-r$r" "$v" "$mode" "$nvl"
            done
        done
    done
done
echo "[$(date +%T)] sweep done"
