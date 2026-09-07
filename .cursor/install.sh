#!/usr/bin/env bash
# Cloud Agent install script for NIXL.
#
# Builds a CPU-only, source install of NIXL suitable for development in a
# Cloud Agent VM (no GPU / RDMA hardware): UCX (TCP + shared-memory transports),
# a modern Abseil, and NIXL itself with the vendor-neutral UCX and POSIX
# plugins plus the Python bindings.
#
# Idempotent: system packages, UCX, and Abseil are only (re)built when missing,
# while NIXL is always rebuilt from the checked-out source.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NPROC="$(nproc)"
UCX_VERSION="v1.23.x"
ABSL_TAG="lts_2025_08_14"
PREFIX="/usr/local"

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
# 2. Python build + runtime dependencies (system interpreter)
# ---------------------------------------------------------------------------
log "Installing Python build/runtime dependencies"
# Install with sudo so console scripts (uv, meson, ninja) land on the system
# PATH (/usr/local/bin) and packages land in a versioned dist-packages that is
# on the system interpreter's sys.path.
PIP="sudo pip3 install --no-cache-dir --break-system-packages"
$PIP meson ninja pybind11 patchelf tomlkit pyyaml uv numpy
# CPU-only PyTorch (no CUDA in the Cloud Agent VM). Needed by the examples/tests.
$PIP torch --index-url https://download.pytorch.org/whl/cpu

# ---------------------------------------------------------------------------
# 3. UCX (skip if a matching version is already installed)
# ---------------------------------------------------------------------------
if pkg-config --modversion ucx 2>/dev/null | grep -q '^1\.23'; then
    log "UCX $(pkg-config --modversion ucx) already installed; skipping"
else
    log "Building UCX ${UCX_VERSION} from source"
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
            --without-gdrcopy --without-cuda
        make -j"${NPROC}"
        sudo make -j install-strip
    )
    sudo ldconfig
    rm -rf "${UCX_TMP}"
fi

# ---------------------------------------------------------------------------
# 4. Abseil (NIXL needs a newer Abseil than Ubuntu ships; the meson wrap patch
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
# 5. NIXL (always rebuilt from the checked-out source)
# ---------------------------------------------------------------------------
log "Building and installing NIXL"
cd "${REPO_ROOT}"
rm -rf build
meson setup build \
    --prefix="${PREFIX}" \
    -Ducx_path="${PREFIX}" \
    -Denable_plugins=UCX,POSIX \
    -Drust=false -Dbuild_docs=false -Dbuild_tests=false -Dbuild_examples=false
ninja -C build
sudo ninja -C build install

# Make the freshly installed NIXL/UCX/Abseil libraries discoverable globally so
# that `import nixl` works without any per-shell environment variables.
echo -e "${PREFIX}/lib\n${PREFIX}/lib/x86_64-linux-gnu" | \
    sudo tee /etc/ld.so.conf.d/nixl.conf >/dev/null
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
