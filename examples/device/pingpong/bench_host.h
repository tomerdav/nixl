#pragma once

#include "nixl.h"
#include "nixl_descriptors.h"
#include "nixl_types.h"
#include <cstdint>
#include <memory>
#include <string>

// ---- Parameter bundle passed to BenchContext::setup --------------------------

struct BenchParams {
    size_t   msg_size     = 8;
    uint64_t num_iters    = 1000;
    uint64_t warmup_iters = 100;
    int      gpu_id       = 0;
    bool     is_sender    = true;
};

// ---- Per-side context ---------------------------------------------------------
// Owns all NIXL and CUDA resources for one side of the ping-pong.
// Resources are released in the destructor (RAII) — no explicit teardown needed.

class BenchContext {
public:
    BenchContext() = default;
    ~BenchContext();

    // Non-copyable; moves suppressed by the user-defined destructor.
    BenchContext(const BenchContext &) = delete;
    BenchContext &operator=(const BenchContext &) = delete;

    // Setup: creates agent + UCX backend, allocates device buffers, registers
    // them with NIXL, exchanges metadata and recv_buf addresses with the peer,
    // and builds memory view handles ready for the GPU kernel.
    //
    //   peer_ip:   hostname/IP of the peer ("127.0.0.1" in single-process mode)
    //   peer_port: port the peer's listen thread is bound to
    //   my_port:   port this side's listen thread will bind to
    nixl_status_t setup(const BenchParams &params,
                        const char *peer_ip, int peer_port, int my_port);

    // ---- State read by the kernel launcher (intentionally public) ------------
    std::unique_ptr<nixlAgent> agent;
    nixlBackendH *ucx_backend = nullptr;
    void        *send_buf     = nullptr; // cudaMalloc'd, buf_size bytes
    void        *recv_buf     = nullptr; // cudaMalloc'd, buf_size bytes
    size_t       buf_size     = 0;       // msg_size + sizeof(uint64_t)
    nixlMemViewH local_mvh    = nullptr; // view of local send_buf
    nixlMemViewH remote_mvh   = nullptr; // view of peer's recv_buf
    bool         is_sender    = true;
    int          gpu_id       = 0;
};
