#!/usr/bin/env bash
# Dispatch A/B on DFW: runtime exec-mode branch vs the mode fixed at compile time. Run from the login node
# with bash (the login shell is csh):
#   bash ab.sh run [sbatch options]    one exclusive node: build on node-local disk, warm-ups, AB_REPS ABBA
#                                      pairs per cell; results land in $AB_DIR/results-<job>.tar
#   bash ab.sh collect <job>           the paired-difference table, from that tarball unpacked under /tmp
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
source "$here/ab-env.sh"
source "$EP_KIT/env-dfw.sh"
case "${1:?run|collect}" in
    run)
        shift
        sbatch -A "$DFW_ACCOUNT" -p "$DFW_PARTITION" -N1 --gpus-per-node=8 \
            --output="$AB_DIR/ab-%j.out" "$@" \
            --export=ALL,AB_KIT="$here" "$here/ab-sweep.sh"
        ;;
    collect)
        out=$(mktemp -d /tmp/ab-collect-XXXXXX)
        tar -xf "$AB_DIR/results-${2:?job id}.tar" -C "$out"
        python3 "$here/ab-collect.py" "$out"
        cat "$out"/build-*/sass-instructions.txt 2>/dev/null | paste -sd' ' | sed 's/^/sass instructions runtime fixdirect fixproxy: /'
        rm -rf "$out"
        ;;
    *) echo "unknown command: $1" >&2; exit 2 ;;
esac
