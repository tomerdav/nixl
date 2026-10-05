#!/usr/bin/env bash
# Build the three dispatch A/B variants from one source tarball, inside the DFW image on a compute node
# (ab.sh build runs this through srun). Each variant gets its own EP_ROOT so the ep-mars kit's run_node.sh
# can run it unchanged:
#   runtime    the tag branch in device_dispatch.cuh, as merged
#   fixdirect  -DNIXL_DEVICE_FIXED_EXEC_MODE=1 (UCX_DIRECT fixed at compile time)
#   fixproxy   -DNIXL_DEVICE_FIXED_EXEC_MODE=2 (PROXY fixed at compile time)
#
# usage: ab-build.sh <nixl-src.tar>
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
src_tar=$(realpath "${1:?source tarball}")
source "$here/ab-env.sh"
source "$EP_KIT/env-dfw.sh"

declare -A flags=([runtime]="" [fixdirect]="-DNIXL_DEVICE_FIXED_EXEC_MODE=1" [fixproxy]="-DNIXL_DEVICE_FIXED_EXEC_MODE=2")
for v in runtime fixdirect fixproxy; do
    root=$AB_DIR/$v
    mkdir -p "$root"
    cd "$root"
    [ -d source ] || { mkdir source && tar -xf "$src_tar" -C source; }
    sed -e "s#@ROOT@#$root#g" -e "s#@VENV@#$EP_VENV#g" "$EP_KIT/ep-native.ini.in" > ep-native.ini
    sed -e "s#@ROOT@#$root#g" -e "s#@VENV@#$EP_VENV#g" "$EP_KIT/pybind11-config.in" > pybind11-config
    chmod +x pybind11-config
    cd source
    [ -d build ] && reconf=--reconfigure || reconf=
    echo "=== $v: cuda_args='${flags[$v]}'"
    PATH=$EP_VENV/bin:$CUDA_HOME/bin:$PATH meson setup $reconf build --native-file="$root/ep-native.ini" \
        --wrap-mode=nodownload -Ducx_path="$EP_UCX" -Dbuild_examples=true -Dbuild_nixl_ep=true \
        -Dbuild_tests=false -Dnixl_cuda_arch_list="$NIXL_CUDA_ARCH" -Dbuildtype=release \
        -Ddisable_plugins=GPUNETIO -Dcuda_args="${flags[$v]}"
    PATH=$EP_VENV/bin:$CUDA_HOME/bin:$PATH meson compile -C build -j"$(nproc)"
    PYTHONPATH=$root/source/build/examples/device/ep "$EP_VENV/bin/python3" -c "import nixl_ep; print('$v: nixl_ep import ok')"
    # Record the kernels' register use and static size, the generated-code side of the comparison.
    so=$(find build/examples/device/ep -name 'nixl_ep*.so' | head -1)
    "$CUDA_HOME/bin/cuobjdump" -res-usage "$so" > "$root/res-usage.txt" 2>&1 || true
    "$CUDA_HOME/bin/cuobjdump" -sass "$so" > "$root/sass.txt" 2>&1 || true
    echo "$v: tip=$(cat TIP 2>/dev/null) sass_lines=$(wc -l < "$root/sass.txt")"
done
