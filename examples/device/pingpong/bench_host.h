#pragma once

#include "nixl.h"
#include "nixl_descriptors.h"
#include "nixl_types.h"
#include <cstdint>
#include <future>
#include <string>

// ---- Parameter bundle passed to bench_setup -----------------------------------

struct BenchParams {
    size_t msg_size = 8;
    uint64_t num_iters = 1000;
    uint64_t warmup_iters = 100;
    int gpu_id = 0;
    bool is_sender = true;
};

// ---- In-process rendezvous (single-process mode only) -------------------------
// Both sender and receiver threads share one instance of this struct.
// Each side sets its own promise and reads the other's future.

struct BenchRendezvous {
    std::promise<nixl_blob_t> sender_md_promise;
    std::promise<nixl_blob_t> recvr_md_promise;
    std::promise<uintptr_t> sender_addr_promise;
    std::promise<uintptr_t> recvr_addr_promise;
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

// Single-process setup: both sender and receiver call this from their threads.
// rndv must outlive both calls (owned by the spawning thread).
nixl_status_t
bench_setup_single(BenchContext &ctx, const BenchParams &params, BenchRendezvous &rndv);

void
bench_teardown(BenchContext &ctx);
