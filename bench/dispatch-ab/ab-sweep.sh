#!/usr/bin/env bash
# The dispatch A/B inside one single-node allocation (ab.sh run submits it).
# For each mode and NVLink setting, rep r runs the runtime build and the matching fixed build back to back,
# runtime first on odd reps and fixed first on even reps (ABBA), after one discarded warm-up of each.
#SBATCH --exclusive
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00
set -uo pipefail
here=${AB_KIT:?set AB_KIT}
source "$here/ab-env.sh"
export EP_ENV_PRE=$EP_KIT/env-dfw.sh
source "$EP_KIT/env.sh"
export GPUS_PER_NODE=$AB_RANKS
export EP_RESULTS=$AB_DIR/results
mkdir -p "$EP_RESULTS"
python3 "$EP_KIT/mk_plans.py" 8 1 > /dev/null
plan=$EP_KIT/plans/noexp_$AB_RANKS.json
node=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)

run() {  # run <tag> <variant> <mode> <nvl>
    local tag=$1 v=$2 mode=$3 nvl=$4
    [ -e "$EP_RESULTS/$tag/rc.txt" ] && { echo "skip $tag (done)"; return; }
    echo "[$(date +%T)] $tag"
    EP_ROOT=$AB_DIR/$v srun -N1 -w "$node" --ntasks-per-node=1 --cpus-per-task="${SLURM_CPUS_ON_NODE:-1}" \
        --cpu-bind=none --kill-on-bad-exit=1 \
        --container-image="$EP_CONTAINER_IMAGE" --container-name="ab-$SLURM_JOB_ID" \
        --container-mounts="$EP_CONTAINER_MOUNTS" --no-container-remap-root \
        --export=ALL,EP_ROOT="$AB_DIR/$v" \
        "$EP_KIT/run_node.sh" "$tag" "$mode" "$nvl" "$plan" "$node" < /dev/null
    echo "[$(date +%T)] $tag rc=$?"
}

for mode in $AB_MODES; do
    fixed=fix$mode
    for nvl in $AB_NVL; do
        run "ab-warm-$mode-$nvl-runtime" runtime "$mode" "$nvl"
        run "ab-warm-$mode-$nvl-$fixed" "$fixed" "$mode" "$nvl"
    done
done
for r in $(seq "$AB_REPS"); do
    for mode in $AB_MODES; do
        fixed=fix$mode
        for nvl in $AB_NVL; do
            if (( r % 2 )); then order="runtime $fixed"; else order="$fixed runtime"; fi
            for v in $order; do
                run "ab-$mode-$nvl-$v-r$r" "$v" "$mode" "$nvl"
            done
        done
    done
done
echo "[$(date +%T)] sweep done"
