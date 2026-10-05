#!/usr/bin/env bash
# Dispatch A/B on DFW: runtime exec-mode branch vs the mode fixed at compile time. Run from the login node
# with bash (the login shell is csh):
#   bash ab.sh build <nixl-src.tar>    one node, in the image: build runtime, fixdirect and fixproxy
#   bash ab.sh run [sbatch options]    one exclusive node: warm-ups, then AB_REPS ABBA pairs per cell
#   bash ab.sh collect                 the paired-difference table (python3 on the login node is enough)
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
source "$here/ab-env.sh"
source "$EP_KIT/env-dfw.sh"
ctr=(--container-image="$EP_CONTAINER_IMAGE" --container-mounts="$EP_CONTAINER_MOUNTS" --no-container-remap-root)
one=(-A "$DFW_ACCOUNT" -N1 --ntasks=1 --cpus-per-task="${DFW_CPUS:-128}" --gpus-per-node=8 --exclusive)
case "${1:?build|run|collect}" in
    build)
        src=$(realpath "${2:?source tarball}")
        srun "${one[@]}" -p batch_short -t 90 "${ctr[@]}" bash "$here/ab-build.sh" "$src" < /dev/null
        ;;
    run)
        shift
        mkdir -p "$AB_DIR/results"
        sbatch -A "$DFW_ACCOUNT" -p "$DFW_PARTITION" -N1 --gpus-per-node=8 \
            --output="$AB_DIR/results/ab-%j.out" "$@" \
            --export=ALL,AB_KIT="$here" "$here/ab-sweep.sh"
        ;;
    collect)
        python3 "$here/ab-collect.py" "$AB_DIR/results"
        ;;
    *) echo "unknown command: $1" >&2; exit 2 ;;
esac
