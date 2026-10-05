#!/usr/bin/env bash
# Build the three dispatch A/B variants into node-local AB_WORK, inside the DFW image (ab-sweep.sh runs this
# as the job's first step). Each variant gets its own EP_ROOT so the ep-mars kit's run_node.sh can run it
# unchanged:
#   runtime    the tag branch in device_dispatch.cuh, as merged
#   fixdirect  -DNIXL_DEVICE_FIXED_EXEC_MODE=1 (UCX_DIRECT fixed at compile time)
#   fixproxy   -DNIXL_DEVICE_FIXED_EXEC_MODE=2 (PROXY fixed at compile time)
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
source "$here/ab-env.sh"
source "$EP_KIT/env-dfw.sh"
work=${AB_WORK:?node-local work directory}

declare -A flags=([runtime]="" [fixdirect]="-DNIXL_DEVICE_FIXED_EXEC_MODE=1" [fixproxy]="-DNIXL_DEVICE_FIXED_EXEC_MODE=2")

shared=$work/src
mkdir -p "$shared"
echo "=== [$(date +%T)] extracting $AB_TAR into $shared"
tar -xf "$AB_TAR" -C "$shared" --exclude=./subprojects/asio-1.30.2/doc \
    --exclude=./subprojects/taskflow/docs --exclude=./subprojects/taskflow/doxygen

for v in runtime fixdirect fixproxy; do
    root=$work/$v
    # A variant's source/ is symlinks into the shared tree plus its own build/, the layout env.sh and
    # run_node.sh expect.
    mkdir -p "$root/source"
    for e in "$shared"/*; do
        ln -sfn "$e" "$root/source/$(basename "$e")"
    done
    cd "$root"
    sed -e "s#@ROOT@#$root#g" -e "s#@VENV@#$EP_VENV#g" "$EP_KIT/ep-native.ini.in" > ep-native.ini
    sed -e "s#@ROOT@#$root#g" -e "s#@VENV@#$EP_VENV#g" "$EP_KIT/pybind11-config.in" > pybind11-config
    chmod +x pybind11-config
    cd source
    echo "=== [$(date +%T)] $v: cuda_args='${flags[$v]}'"
    PATH=$EP_VENV/bin:$CUDA_HOME/bin:$PATH meson setup build "$shared" --native-file="$root/ep-native.ini" \
        --wrap-mode=nodownload -Ducx_path="$EP_UCX" -Dbuild_examples=true -Dbuild_nixl_ep=true \
        -Dbuild_tests=false -Dnixl_cuda_arch_list="$NIXL_CUDA_ARCH" -Dbuildtype=release \
        -Ddisable_plugins=GPUNETIO -Dcuda_args="${flags[$v]}"
    PATH=$EP_VENV/bin:$CUDA_HOME/bin:$PATH meson compile -C build -j"$(nproc)"
    PYTHONPATH=$root/source/build/examples/device/ep "$EP_VENV/bin/python3" -c "import nixl_ep; print('$v: nixl_ep import ok')"
    # The kernels' register use and static size, the generated-code side of the comparison; small, so
    # they go to Lustre with the results.
    so=$(find build/examples/device/ep -name 'nixl_ep*.so' | head -1)
    mkdir -p "$AB_RESULTS/build-$v"
    "$CUDA_HOME/bin/cuobjdump" -res-usage "$so" > "$AB_RESULTS/build-$v/res-usage.txt" 2>&1 || true
    "$CUDA_HOME/bin/cuobjdump" -sass "$so" > "$root/sass.txt" 2>&1 || true
    grep -cE '^ +/\*[0-9a-f]+\*/' "$root/sass.txt" > "$AB_RESULTS/build-$v/sass-instructions.txt" || true
    echo "=== [$(date +%T)] $v done: tip=$(cat TIP 2>/dev/null) sass_instructions=$(cat "$AB_RESULTS/build-$v/sass-instructions.txt")"
done
