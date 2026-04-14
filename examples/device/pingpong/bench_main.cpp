// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "bench_host.h"
#include "bench_kernel_iface.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>

// ----------------------------------------------------------------------------
// Minimal usage
// ----------------------------------------------------------------------------
static void
usage(const char *prog) {
    fprintf(stderr,
            "Two-process mode:\n"
            "  %s --role <sender|receiver>\n"
            "     --peer-ip   <ip>    peer's hostname or IP\n"
            "     --peer-port <port>  peer's NIXL listen port\n"
            "     --listen-port <port>  our NIXL listen port\n"
            "    [--msg-size  <bytes>]  (default 8)\n"
            "    [--iters     <n>]      (default 1000)\n"
            "    [--warmup    <n>]      (default 100)\n"
            "    [--gpu       <id>]     (default 0)\n"
            "    [--warp]               use WARP level (default: THREAD)\n"
            "\n"
            "Single-process mode (two threads, one GPU):\n"
            "  %s --single-process\n"
            "    [--msg-size  <bytes>]\n"
            "    [--iters     <n>]\n"
            "    [--warmup    <n>]\n"
            "    [--gpu       <id>]\n"
            "    [--warp]\n",
            prog,
            prog);
    exit(1);
}

// ----------------------------------------------------------------------------
// Single-process benchmark
// ----------------------------------------------------------------------------
static int
single_process_run(size_t msg_size,
                   uint64_t num_iters,
                   uint64_t warmup_iters,
                   int gpu_id,
                   bool use_warp)
{
    fprintf(stderr,
            "[main] single-process mode  msg_size=%zu  iters=%llu  warmup=%llu"
            "  gpu=%d  level=%s\n",
            msg_size, (unsigned long long)num_iters, (unsigned long long)warmup_iters,
            gpu_id, use_warp ? "WARP" : "THREAD");

    // Allocate device counters for elapsed ticks (one per side; only sender's
    // value matters but we allocate both so the receiver launch signature is uniform).
    cudaSetDevice(gpu_id);
    uint64_t *d_elapsed_sender = nullptr, *d_elapsed_recvr = nullptr;
    if (cudaMalloc(&d_elapsed_sender, sizeof(uint64_t)) != cudaSuccess ||
        cudaMalloc(&d_elapsed_recvr,  sizeof(uint64_t)) != cudaSuccess) {
        fprintf(stderr, "[main] cudaMalloc d_elapsed failed\n");
        return 1;
    }
    cudaMemset(d_elapsed_sender, 0, sizeof(uint64_t));
    cudaMemset(d_elapsed_recvr,  0, sizeof(uint64_t));

    BenchRendezvous rndv;
    BenchContext    sender_ctx, recvr_ctx;
    nixl_status_t   sender_st = NIXL_SUCCESS, recvr_st = NIXL_SUCCESS;

    // ---- Sender thread -------------------------------------------------------
    std::thread sender_thr([&]() {
        fprintf(stderr, "[sender] thread started\n");

        BenchParams params;
        params.msg_size     = msg_size;
        params.num_iters    = num_iters;
        params.warmup_iters = warmup_iters;
        params.gpu_id       = gpu_id;
        params.is_sender    = true;

        sender_st = bench_setup_single(sender_ctx, params, rndv);
        if (sender_st != NIXL_SUCCESS) {
            fprintf(stderr, "[sender] setup failed (%d) — exiting thread\n", sender_st);
            return;
        }
        fprintf(stderr, "[sender] setup complete\n");

        gpu_bench_ctx kctx;
        kctx.local_mvh    = sender_ctx.local_mvh;
        kctx.remote_mvh   = sender_ctx.remote_mvh;
        kctx.send_buf     = (uint8_t *)sender_ctx.send_buf;
        kctx.recv_buf     = (uint8_t *)sender_ctx.recv_buf;
        kctx.msg_size     = msg_size;
        kctx.num_iters    = num_iters;
        kctx.warmup_iters = warmup_iters;
        kctx.is_sender    = true;

        cudaStream_t stream;
        cudaStreamCreate(&stream);
        fprintf(stderr, "[sender] launching %s kernel (%llu warmup + %llu timed iters)\n",
                use_warp ? "WARP" : "THREAD",
                (unsigned long long)warmup_iters, (unsigned long long)num_iters);

        if (use_warp)
            launch_pingpong_warp(kctx, d_elapsed_sender, stream);
        else
            launch_pingpong_thread(kctx, d_elapsed_sender, stream);

        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
        fprintf(stderr, "[sender] kernel finished\n");
    });

    // ---- Receiver thread -----------------------------------------------------
    std::thread recvr_thr([&]() {
        fprintf(stderr, "[recvr] thread started\n");

        BenchParams params;
        params.msg_size     = msg_size;
        params.num_iters    = num_iters;
        params.warmup_iters = warmup_iters;
        params.gpu_id       = gpu_id;
        params.is_sender    = false;

        recvr_st = bench_setup_single(recvr_ctx, params, rndv);
        if (recvr_st != NIXL_SUCCESS) {
            fprintf(stderr, "[recvr] setup failed (%d) — exiting thread\n", recvr_st);
            return;
        }
        fprintf(stderr, "[recvr] setup complete\n");

        gpu_bench_ctx kctx;
        kctx.local_mvh    = recvr_ctx.local_mvh;
        kctx.remote_mvh   = recvr_ctx.remote_mvh;
        kctx.send_buf     = (uint8_t *)recvr_ctx.send_buf;
        kctx.recv_buf     = (uint8_t *)recvr_ctx.recv_buf;
        kctx.msg_size     = msg_size;
        kctx.num_iters    = num_iters;
        kctx.warmup_iters = warmup_iters;
        kctx.is_sender    = false;

        cudaStream_t stream;
        cudaStreamCreate(&stream);
        fprintf(stderr, "[recvr] launching %s kernel\n", use_warp ? "WARP" : "THREAD");

        if (use_warp)
            launch_pingpong_warp(kctx, d_elapsed_recvr, stream);
        else
            launch_pingpong_thread(kctx, d_elapsed_recvr, stream);

        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
        fprintf(stderr, "[recvr] kernel finished\n");
    });

    sender_thr.join();
    recvr_thr.join();
    fprintf(stderr, "[main] both threads joined\n");

    // ---- Teardown ------------------------------------------------------------
    fprintf(stderr, "[main] tearing down sender context\n");
    bench_teardown(sender_ctx);
    fprintf(stderr, "[main] tearing down receiver context\n");
    bench_teardown(recvr_ctx);

    if (sender_st != NIXL_SUCCESS || recvr_st != NIXL_SUCCESS) {
        fprintf(stderr, "[main] one or both sides failed — no latency output\n");
        cudaFree(d_elapsed_sender);
        cudaFree(d_elapsed_recvr);
        return 1;
    }

    // ---- Latency computation -------------------------------------------------
    uint64_t h_elapsed = 0;
    cudaMemcpy(&h_elapsed, d_elapsed_sender, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    fprintf(stderr, "[main] elapsed ticks (sender): %llu over %llu timed iters\n",
            (unsigned long long)h_elapsed, (unsigned long long)num_iters);

    // cudaDevAttrClockRate gives the SM clock in kHz
    int clock_khz = 0;
    cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, gpu_id);
    double clock_hz = (double)clock_khz * 1000.0;
    fprintf(stderr, "[main] GPU SM clock: %.3f GHz\n", clock_hz / 1e9);

    double rtt_us     = (double)h_elapsed / (double)num_iters / clock_hz * 1e6;
    double one_way_us = rtt_us / 2.0;

    printf("msg_size=%-6zu  iters=%-6llu  RTT=%.3f us  one-way=%.3f us  [%s]\n",
           msg_size,
           (unsigned long long)num_iters,
           rtt_us,
           one_way_us,
           use_warp ? "WARP" : "THREAD");

    cudaFree(d_elapsed_sender);
    cudaFree(d_elapsed_recvr);
    fprintf(stderr, "[main] done\n");
    return 0;
}

// ----------------------------------------------------------------------------
// main
// ----------------------------------------------------------------------------
int
main(int argc, char *argv[]) {
    // ---- argument parsing --------------------------------------------------
    const char *role_str = nullptr;
    const char *peer_ip = nullptr;
    int peer_port = 0;
    int listen_port = 0;
    size_t msg_size = 8;
    uint64_t num_iters = 1000;
    uint64_t warmup_iters = 100;
    int gpu_id = 0;
    bool use_warp = false;
    bool single_process = false;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--role") && i + 1 < argc)
            role_str = argv[++i];
        else if (!strcmp(argv[i], "--peer-ip") && i + 1 < argc)
            peer_ip = argv[++i];
        else if (!strcmp(argv[i], "--peer-port") && i + 1 < argc)
            peer_port = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--listen-port") && i + 1 < argc)
            listen_port = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--msg-size") && i + 1 < argc)
            msg_size = (size_t)atoll(argv[++i]);
        else if (!strcmp(argv[i], "--iters") && i + 1 < argc)
            num_iters = (uint64_t)atoll(argv[++i]);
        else if (!strcmp(argv[i], "--warmup") && i + 1 < argc)
            warmup_iters = (uint64_t)atoll(argv[++i]);
        else if (!strcmp(argv[i], "--gpu") && i + 1 < argc)
            gpu_id = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--warp"))
            use_warp = true;
        else if (!strcmp(argv[i], "--single-process"))
            single_process = true;
        else
            usage(argv[0]);
    }

    // Default to single-process mode if no two-process args are supplied
    if (!role_str && !peer_ip && peer_port == 0 && listen_port == 0) single_process = true;

    if (single_process) {
        if (role_str || peer_ip || peer_port || listen_port) {
            fprintf(stderr,
                    "--single-process is mutually exclusive with "
                    "--role/--peer-ip/--peer-port/--listen-port\n");
            usage(argv[0]);
        }
        return single_process_run(msg_size, num_iters, warmup_iters, gpu_id, use_warp);
    }

    // Two-process mode (Stage 7 — not yet implemented)
    if (!role_str || !peer_ip || peer_port == 0 || listen_port == 0) usage(argv[0]);

    bool is_sender = (strcmp(role_str, "sender") == 0);
    if (!is_sender && strcmp(role_str, "receiver") != 0) {
        fprintf(stderr, "Unknown role '%s'; expected sender or receiver\n", role_str);
        usage(argv[0]);
    }

    fprintf(stderr, "[main] two-process mode not yet implemented (Stage 7)\n");
    (void)is_sender;
    return 1;
}
