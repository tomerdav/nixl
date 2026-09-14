# Cluster validation: CPU-proxy ownership v2

This standalone validation branch contains the complete v2 stack plus the exact
two-process drivers/plans exercised on adv-dev-506. No access to adv-dev-420 or
private file transfer is required.

```bash
git clone --branch codex/cpu-proxy-v2-cluster-validation --single-branch \
  https://github.com/tomerdav/nixl.git nixl-v2
cd nixl-v2
git rev-parse HEAD
```

The runtime base is `aef09810d1b41afda552901b531dedad657eb61f`.
The additional files are validation-only: production EP behavior is unchanged.
Use the bundled drivers named below, not the original `elastic.py`, for these
two-process checks.

The drivers preserve the RC transport allow-list after `disable_ll_nvlink`
overwrites it, and keep a fault-designated rank in the descriptor layout until
disconnect. Both corrections are confined to the validation drivers. The
SIGKILL driver also checks the expected victim/survivor exit codes.

Recorded results on two H100s: expansion, clean contraction, SIGTERM fault cleanup,
and expansion followed by SIGKILL and survivor progress all passed. Cross-node
EP on this cluster still needs to be validated. Tests are correctness checks,
not performance measurements.

## 1. Cluster prerequisites and native build

Obtain the GPU/CPU allocation through your cluster scheduler first. This recipe
targets H100/Hopper (`sm90`), as tested on adv-dev-506. Do not apply arch 90
blindly to a different GPU family. Reserve CPU capacity for the proxy workers
as well as Python/UCX background work. Do not benchmark on login nodes.

Use a Python environment with CUDA-enabled PyTorch, NumPy and pybind11; Meson,
Ninja, a compatible C++ compiler, CUDA toolkit/development files, and RDMA headers
must be available. `pkg-config --modversion pybind11` must succeed. Meson can
fetch its normal fallback dependencies if network access is available; otherwise
prepopulate the dependency sources using your cluster's normal offline process.

Use UCX with the device API needed by this stack. The tested source is OpenUCX
`d65ffe7bf39d8bad835415271a893095b57b42cf` (1.22 development tree), repository
`https://github.com/openucx/ucx.git`. Prefer that revision for reproduction, built
against the target cluster's CUDA and RDMA stack. An arbitrary system UCX 1.20
is not equivalent. Do not copy the 506 binary bundle into a different ABI/CUDA
environment. GDRCopy is optional for the host-mapped control fallback; if enabled,
ensure its headers, userspace library and runtime driver are usable.

Activate your environment before configuring, then set actual cluster paths:

```bash
export UCX_PREFIX=/path/to/private/ucx-install
export UCX_LIBDIR="$UCX_PREFIX/lib"  # use lib64 if that is your install layout
export CUDA_HOME=/path/to/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export PKG_CONFIG_PATH="$UCX_LIBDIR/pkgconfig:${PKG_CONFIG_PATH:-}"
export LD_LIBRARY_PATH="$UCX_LIBDIR:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

python3 -c 'import sys, torch; print(sys.executable, torch.__version__, torch.version.cuda)'
nvcc --version
pkg-config --modversion pybind11

meson setup build \
  -Ducx_path="$UCX_PREFIX" \
  -Dbuild_examples=true -Dbuild_nixl_ep=true -Dbuild_tests=true \
  -Dnixl_cuda_arch_list=90 \
  -Dbuildtype=debugoptimized -Doptimization=3 -Db_ndebug=true -Dwerror=false \
  -Ddisable_plugins=GPUNETIO
meson compile -C build -j8
NIXL_PLUGIN_DIR="$PWD/build/test/gtest/mocks" \
  meson test -C build unit --no-rebuild --print-errorlogs
```

GPUNETIO is excluded because it is unrelated to this UCX proxy test and failed
against 506's installed DOCA API. Use the same exclusion for comparison. The
debugoptimized/O3 build matches the tested EP configuration. Check configure
output: missing Torch, pybind11 or CUDA can cause Meson to skip EP rather than
fail the whole build. Import verification below is mandatory.

## 2. Runtime environment (on every participating node)

Run from the checkout root. Select the **actual RDMA NIC/port for that node**;
`mlx5_0:1` is an example, not a Lyris/DFW assumption. Use `ibdev2netdev`,
`ucx_info -d` and `nvidia-smi topo -m` to inspect available devices/topology.

```bash
export LD_LIBRARY_PATH="$UCX_LIBDIR:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export UCX_MODULE_DIR="$UCX_LIBDIR/ucx"
export NIXL_PLUGIN_DIR="$PWD/build/src/plugins/ucx"
export PYTHONPATH="$PWD/build/examples/device/ep:$PWD/src/bindings/python/nixl-meta"

export NIXL_EP_DEVICE_MODE=proxy
export NIXL_EP_PROXY_CHANNELS=4
export NIXL_EP_PROXY_WORKER_COUNT=2
export UCX_NET_DEVICES=mlx5_0:1  # replace per node
export UCX_TLS=rc,cuda_copy,gdr_copy
export UCX_MAX_RMA_RAILS=1
export NIXL_LOG_LEVEL=WARN
export UCX_LOG_LEVEL=warn

# Test-only settings used in the validated failure runs:
export UCX_RC_TIMEOUT=50ms
export UCX_RC_RETRY_COUNT=3
export UCX_UD_TIMEOUT=2s

python3 -c 'import nixl_ep, sys; print(nixl_ep.buffer.__file__); print([(k, v.__file__) for k, v in sys.modules.items() if "nixl_ep_cpp_torch" in k])'
```

The Python package/native extension must resolve inside this checkout's build,
not a previously installed wheel. Missing optional `gdr_copy` was observed on
506; it must not cause an unreported switch of the GPU data path to TCP/shared
memory. A debug run should show RC zero-copy GPU PUTs. The proper rail variable
is `UCX_MAX_RMA_RAILS`, not `UCX_RMA_MAX_RAILS`.

The bundled drivers restore this transport allow-list after the EP Python
constructor overwrites it. Using only environment exports with the old driver
does not achieve that. Do not replace them with `elastic.py` yet.

These validation drivers select `CUDA_VISIBLE_DEVICES` from the assigned local rank
modulo 2. Use an allocation where local devices 0 and 1 are assigned to this job
(or device 0 on each node). Do not override a scheduler's GPU UUID/index mapping
blindly; adapt that one test-driver assignment to your allocation first.

## 3. Two processes on one two-GPU node

Run tests sequentially; the driver starts control servers on ports 9999/10000.
Use unused ports/no other test instance on this allocation. Define common args:

```bash
EP_TEST_DIR=examples/device/ep/tests/elastic
EP_ARGS=(--num-processes 2 --num-tokens 8 --hidden-dim 2560 \
  --num-experts-per-rank 2 --num-topk 2 --disable-ll-nvlink \
  --timeout-ms 5000 --validate-phase-failures)

for ep_case in expansion contraction fault; do
  timeout --kill-after=10 180 python3 "$EP_TEST_DIR/elastic_two_proc_fault_control.py" \
    --plan "$EP_TEST_DIR/ep-${ep_case}-2proc.json" "${EP_ARGS[@]}" \
    > "ep-${ep_case}.log" 2>&1
  ep_rc=$?
  printf '%s exit=%s\n' "$ep_case" "$ep_rc"
  test "$ep_rc" -eq 0 || break
done

timeout --kill-after=10 180 python3 "$EP_TEST_DIR/elastic_two_proc_sigkill.py" \
  --plan "$EP_TEST_DIR/ep-expand-fault-contract-2proc.json" "${EP_ARGS[@]}" \
  > ep-sigkill.log 2>&1
```

Expected: all parent exit codes 0. Expansion is 1 -> 2 active ranks; contraction
is 2 -> 1; the fault plan kills rank 1 then checks survivor execution. The SIGKILL
test asserts child exits `[0,-9]` in either process order. Receive timeouts naming
rank 1 **after the planned kill** are expected; healthy-phase timeouts are failures.
No hidden/extra failures are permitted by the strict mask check.

The scripts contain a five-second control-path delay after disconnect. It is not
a proxy drain timeout or a benchmark number.

## 4. Two nodes, one process/GPU per node

Use the same source, bundled drivers and ABI-compatible environment on both nodes.
Build locally on both if paths/software differ. Choose a reachable control IP
for node A and open TCP 9999/10000 between the allocated nodes, plus any site-
required NIXL metadata connectivity. The RDMA NIC is independently selected on
each node. Do not infer scheduler/launcher syntax from these direct invocations.

For each test, start node A first **without** `--tcp-server`; it owns the control
servers and must receive global rank 0. Then start node B with A's reachable IP:

```bash
# Node A, after applying the runtime environment:
EP_TEST_DIR=examples/device/ep/tests/elastic
EP_PLAN="$EP_TEST_DIR/ep-expand-fault-contract-2proc.json"
python3 "$EP_TEST_DIR/elastic_two_proc_sigkill.py" --plan "$EP_PLAN" \
  --num-processes 1 --num-tokens 8 --hidden-dim 2560 \
  --num-experts-per-rank 2 --num-topk 2 --disable-ll-nvlink \
  --timeout-ms 5000 --validate-phase-failures > ep-node-a.log 2>&1

# Node B, separate terminal/job step; start after A gets global rank 0:
EP_TEST_DIR=examples/device/ep/tests/elastic
EP_PLAN="$EP_TEST_DIR/ep-expand-fault-contract-2proc.json"
EP_MASTER_IP=NODE_A_CONTROL_IP
python3 "$EP_TEST_DIR/elastic_two_proc_sigkill.py" --plan "$EP_PLAN" \
  --tcp-server "$EP_MASTER_IP" --num-processes 1 \
  --num-tokens 8 --hidden-dim 2560 --num-experts-per-rank 2 --num-topk 2 \
  --disable-ll-nvlink --timeout-ms 5000 --validate-phase-failures > ep-node-b.log 2>&1
```

The two shell commands run concurrently, one per node. In this one-process-per-
invocation mode, node B itself is SIGKILLed: expect shell exit 137 there and 0 on
node A, rather than the single-node parent's aggregated exit check. Configure your
scheduler step so this deliberate victim exit does not automatically kill the
survivor step. Node A must log detection of exactly rank 1 and complete the final
phase. Clean expansion/contraction use the ordinary corrected driver and their
corresponding plans; both nodes should exit 0. Restart servers for each new run.

The cross-node commands follow the driver's existing `--tcp-server` support;
the recorded new EP validation was on two GPUs on **one** node. Cross-node EP on
Lyris/DFW is what these instructions let you validate, not a result already claimed.

## Collect evidence

Keep per-node logs, exit codes, `git rev-parse HEAD`, Python/Torch/CUDA versions,
`ucx_info -v`, `UCX_NET_DEVICES`, GPU allocation and topology. For a failing run,
repeat with `NIXL_LOG_LEVEL=DEBUG UCX_LOG_LEVEL=debug`; retain the selected GPU PUT
protocol and the first failure, not just the final timeout. Do not increase the
receive timeout to mask a healthy-phase transport/progress failure.
