#pragma once

#include "nixl.h"
#include "nixl_descriptors.h"
#include "nixl_types.h"
#include <cstdint>
#include <string>

// ---- Parameter bundle passed to bench_setup -----------------------------------

struct BenchParams {
    size_t msg_size = 8;
    uint64_t num_iters = 1000;
    uint64_t warmup_iters = 100;
    int gpu_id = 0;
    bool is_sender = true;
};


// ---- Per-side context ---------------------------------------------------------

struct BenchContext {
    nixlAgent *agent = nullptr;
    nixlBackendH *ucx_backend = nullptr;
    void *send_buf = nullptr; // cudaMalloc'd, buf_size bytes
    void *recv_buf = nullptr; // cudaMalloc'd, buf_size bytes
    size_t buf_size = 0; // msg_size + sizeof(uint64_t)
    nixlMemViewH local_mvh = nullptr; // local send view
    nixlMemViewH remote_mvh = nullptr; // remote recv view (peer's recv_buf)
    bool is_sender = true;
    int gpu_id = 0;
};

// ---- API declarations --------------------------------------------------------

// Unified setup for both single-process (loopback) and two-process modes.
// peer_ip:    IP or hostname of the peer ("127.0.0.1" in single-process mode)
// peer_port:  the port the peer's listen thread is bound to
// my_port:    the port this side's listen thread will bind to
nixl_status_t
bench_setup(BenchContext &ctx, const BenchParams &params,
            const char *peer_ip, int peer_port, int my_port);

void
bench_teardown(BenchContext &ctx);
