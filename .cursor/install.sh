#!/usr/bin/env bash
# Cloud Agent install script for NIXL.
#
# Builds a source install of NIXL for development in a Cloud Agent VM. By default
# it enables build-time CUDA support (CUDA toolkit + UCX built --with-cuda + the
# CUDA-dependent GDS plugin) so agents can develop, compile, and link the CUDA
# code paths. The VM has no NVIDIA GPU, so the CUDA runtime paths (VRAM
# transfers, torch.cuda kernels) compile but cannot execute here; the CPU/DRAM
# data path over UCX is fully runnable.
#
# Set NIXL_ENABLE_CUDA=0 for a CPU-only build (no CUDA toolkit, UCX built
# --without-cuda, no GDS plugin).
#
# Idempotent: CUDA toolkit, UCX, and Abseil are only (re)built when missing;
# NIXL is always rebuilt from the checked-out source.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NPROC="$(nproc)"
UCX_VERSION="v1.23.x"
ABSL_TAG="lts_2025_08_14"
PREFIX="/usr/local"

# Build-time CUDA knobs (all overridable from the environment).
NIXL_ENABLE_CUDA="${NIXL_ENABLE_CUDA:-1}"
CUDA_APT_VERSION="${CUDA_APT_VERSION:-12-9}"   # apt metapackage suffix, e.g. 12-9
# CUDA arch(es) to compile device code for. A single arch keeps builds fast; set
# to your target GPU's SM (e.g. 80=A100, 89=L40S, 90=H100, 100=B200) or a
# comma-separated list. Only relevant for CUDA builds.
NIXL_CUDA_ARCH_LIST="${NIXL_CUDA_ARCH_LIST:-90}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

# NIXL uses GCC; the image's default `c++`/`cc` may point at clang, which cannot
# find libstdc++ here. Pin the compilers explicitly.
export CC=gcc CXX=g++
export PKG_CONFIG_PATH="${PREFIX}/lib/pkgconfig:${PREFIX}/lib64/pkgconfig:/usr/share/pkgconfig:${PKG_CONFIG_PATH:-}"

log() { printf '\n=== %s ===\n' "$*"; }

# ---------------------------------------------------------------------------
# 1. System packages
# ---------------------------------------------------------------------------
log "Installing system packages (apt)"
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends \
    build-essential cmake meson ninja-build pkg-config \
    autoconf automake libtool \
    libnuma-dev numactl libaio-dev libcurl4-openssl-dev zlib1g-dev uuid-dev \
    libasio-dev \
    python3-dev python3-pip python3-venv pybind11-dev \
    git ca-certificates wget

# ---------------------------------------------------------------------------
# 2. CUDA toolkit (build-time; skipped when NIXL_ENABLE_CUDA=0)
# ---------------------------------------------------------------------------
if [ "${NIXL_ENABLE_CUDA}" = "1" ]; then
    if [ -x "${CUDA_HOME}/bin/nvcc" ]; then
        log "CUDA toolkit $("${CUDA_HOME}/bin/nvcc" --version | grep -oP 'release \K[0-9.]+') already installed; skipping"
    elif [ "$(uname -m)" != "x86_64" ]; then
        log "CUDA apt repo is only wired up for x86_64 here; disabling CUDA"
        NIXL_ENABLE_CUDA=0
    else
        log "Installing CUDA toolkit ${CUDA_APT_VERSION}"
        KEYRING_TMP="$(mktemp -d)"
        wget -q -O "${KEYRING_TMP}/cuda-keyring.deb" \
            https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
        sudo dpkg -i "${KEYRING_TMP}/cuda-keyring.deb"
        rm -rf "${KEYRING_TMP}"
        sudo apt-get update -qq
        sudo apt-get install -y --no-install-recommends "cuda-toolkit-${CUDA_APT_VERSION}"
    fi
fi

# Resolve final CUDA availability and put nvcc on PATH for the meson build.
if [ "${NIXL_ENABLE_CUDA}" = "1" ] && [ -x "${CUDA_HOME}/bin/nvcc" ]; then
    HAVE_CUDA=1
    export PATH="${CUDA_HOME}/bin:${PATH}"
    log "Building with CUDA support (targets: ${NIXL_CUDA_ARCH_LIST})"
else
    HAVE_CUDA=0
    log "Building CPU-only (no CUDA)"
fi

# ---------------------------------------------------------------------------
# 3. Python build + runtime dependencies (system interpreter)
# ---------------------------------------------------------------------------
log "Installing Python build/runtime dependencies"
# Install with sudo so console scripts (uv, meson, ninja) land on the system
# PATH (/usr/local/bin) and packages land in a versioned dist-packages that is
# on the system interpreter's sys.path.
PIP="sudo pip3 install --no-cache-dir --break-system-packages"
$PIP meson ninja pybind11 patchelf tomlkit pyyaml uv numpy
if [ "${HAVE_CUDA}" = "1" ]; then
    # Default PyPI Linux wheel is the CUDA build (bundles its own CUDA runtime).
    $PIP torch
else
    $PIP torch --index-url https://download.pytorch.org/whl/cpu
fi

# ---------------------------------------------------------------------------
# 4. UCX (rebuilt when missing, or when CUDA support is needed but absent)
# ---------------------------------------------------------------------------
ucx_ok() {
    pkg-config --modversion ucx 2>/dev/null | grep -q '^1\.23' || return 1
    if [ "${HAVE_CUDA}" = "1" ]; then
        # Require the CUDA UCT module to be present for a CUDA-enabled UCX.
        [ -e "${PREFIX}/lib/ucx/libuct_cuda.so" ]
    fi
}
if ucx_ok; then
    log "UCX $(pkg-config --modversion ucx) already installed with required features; skipping"
else
    if [ "${HAVE_CUDA}" = "1" ]; then
        UCX_CUDA_ARG="--with-cuda=${CUDA_HOME}"
    else
        UCX_CUDA_ARG="--without-cuda"
    fi
    log "Building UCX ${UCX_VERSION} from source (${UCX_CUDA_ARG})"
    UCX_TMP="$(mktemp -d)"
    git clone --depth 1 -b "${UCX_VERSION}" https://github.com/openucx/ucx.git "${UCX_TMP}/ucx"
    (
        cd "${UCX_TMP}/ucx"
        ./autogen.sh
        ./contrib/configure-release-mt \
            --prefix="${PREFIX}" \
            --enable-shared --disable-static \
            --disable-doxygen-doc --enable-optimizations \
            --without-avx --enable-cma --enable-devel-headers \
            --without-gdrcopy "${UCX_CUDA_ARG}"
        make -j"${NPROC}"
        sudo make -j install-strip
    )
    sudo ldconfig
    rm -rf "${UCX_TMP}"
fi

# ---------------------------------------------------------------------------
# 5. Abseil (NIXL needs a newer Abseil than Ubuntu ships; the meson wrap patch
#    host is not reachable under restricted egress, so install it system-wide)
# ---------------------------------------------------------------------------
if pkg-config --exists absl_log_initialize 2>/dev/null; then
    log "Abseil already installed; skipping"
else
    log "Building Abseil ${ABSL_TAG} from source"
    ABSL_TMP="$(mktemp -d)"
    git clone https://github.com/abseil/abseil-cpp.git "${ABSL_TMP}/abseil-cpp"
    (
        cd "${ABSL_TMP}/abseil-cpp"
        git fetch --depth 1 origin "${ABSL_TAG}"
        git checkout "${ABSL_TAG}"
        mkdir -p build && cd build
        cmake .. \
            -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
            -DCMAKE_INSTALL_LIBDIR=lib \
            -DCMAKE_BUILD_TYPE=Release \
            -DBUILD_SHARED_LIBS=ON \
            -DCMAKE_CXX_STANDARD=20 \
            -DABSL_PROPAGATE_CXX_STD=ON \
            -DABSL_ENABLE_INSTALL=ON
        make -j"${NPROC}"
        sudo make install
    )
    sudo ldconfig
    rm -rf "${ABSL_TMP}"
fi

# ---------------------------------------------------------------------------
# 6. NIXL (always rebuilt from the checked-out source)
# ---------------------------------------------------------------------------
log "Building and installing NIXL"
cd "${REPO_ROOT}"
NIXL_PLUGINS="UCX,POSIX"
NIXL_CUDA_ARGS=()
if [ "${HAVE_CUDA}" = "1" ]; then
    # meson auto-detects nvcc (on PATH) and compiles the CUDA/VRAM code paths.
    NIXL_CUDA_ARGS=(-Dnixl_cuda_arch_list="${NIXL_CUDA_ARCH_LIST}")
    # GPUDirect Storage plugin needs cuFile headers, shipped with the toolkit.
    [ -f "${CUDA_HOME}/include/cufile.h" ] && NIXL_PLUGINS="${NIXL_PLUGINS},GDS"
fi
rm -rf build
meson setup build \
    --prefix="${PREFIX}" \
    -Ducx_path="${PREFIX}" \
    -Denable_plugins="${NIXL_PLUGINS}" \
    "${NIXL_CUDA_ARGS[@]}" \
    -Drust=false -Dbuild_docs=false -Dbuild_tests=false -Dbuild_examples=false
ninja -C build
sudo ninja -C build install

# Make the freshly installed NIXL/UCX/Abseil libraries discoverable globally so
# that `import nixl` works without any per-shell environment variables. Include
# the CUDA runtime libdir for CUDA builds (libcudart, libcufile, ...). The
# NVIDIA *driver* (libcuda.so.1) is provided by a GPU host; when absent, UCX
# simply skips its CUDA transport at runtime and the DRAM path still works.
{
    echo "${PREFIX}/lib"
    echo "${PREFIX}/lib/x86_64-linux-gnu"
    [ "${HAVE_CUDA}" = "1" ] && echo "${CUDA_HOME}/lib64"
} | sudo tee /etc/ld.so.conf.d/nixl.conf >/dev/null
sudo ldconfig

# meson installs the extension modules (nixl_cu12) into
# ${PREFIX}/lib/python3/dist-packages, which is not on the system interpreter's
# sys.path. Add a .pth pointing there so `import nixl_cu12` (and thus `nixl`)
# resolves globally.
PYVER="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
echo "${PREFIX}/lib/python3/dist-packages" | \
    sudo tee "${PREFIX}/lib/python${PYVER}/dist-packages/nixl_bindings.pth" >/dev/null

# Install the pure-Python meta package (`nixl`). Its `nixl-cu12` dependency is
# satisfied by the extension modules installed above, so skip dependency
# resolution.
$PIP --no-deps --force-reinstall "$(ls build/src/bindings/python/nixl-meta/nixl-*-py3-none-any.whl)"

log "Verifying installation"
python3 -c "import nixl; from nixl import nixl_agent, nixl_agent_config; \
a = nixl_agent('install-probe', nixl_agent_config(backends=[])); \
print('NIXL plugins:', a.get_plugin_list())"

log "NIXL install complete"
