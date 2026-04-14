#include "bench_host.h"

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <string>
#include <unordered_map>

nixl_status_t
bench_setup_single(BenchContext &ctx, const BenchParams &params, BenchRendezvous &rndv)
{
    ctx.is_sender = params.is_sender;
    ctx.gpu_id    = params.gpu_id;
    ctx.buf_size  = params.msg_size + sizeof(uint64_t);

    // 1. Bind CUDA device
    cudaSetDevice(params.gpu_id);

    // 2. Create NIXL agent with progress thread (avoids manual progress() calls)
    std::string my_name   = params.is_sender ? "sender"   : "receiver";

    nixlAgentConfig cfg(/*useProgThread=*/true);
    ctx.agent = new nixlAgent(my_name, cfg);

    // 3. Create UCX backend
    nixl_b_params_t bparams;
    nixl_status_t st = ctx.agent->createBackend("UCX", bparams, ctx.ucx_backend);
    if (st != NIXL_SUCCESS) {
        fprintf(stderr, "[%s] createBackend failed: %d\n", my_name.c_str(), st);
        return st;
    }

    // 4. Allocate and zero device buffers
    if (cudaMalloc(&ctx.send_buf, ctx.buf_size) != cudaSuccess ||
        cudaMalloc(&ctx.recv_buf, ctx.buf_size) != cudaSuccess) {
        fprintf(stderr, "[%s] cudaMalloc failed\n", my_name.c_str());
        return NIXL_ERR_NOT_FOUND;
    }
    cudaMemset(ctx.send_buf, 0, ctx.buf_size);
    cudaMemset(ctx.recv_buf, 0, ctx.buf_size);

    // 5. Register send and recv buffers with NIXL
    nixl_reg_dlist_t send_dlist(VRAM_SEG), recv_dlist(VRAM_SEG);
    send_dlist.addDesc(nixlBlobDesc((uintptr_t)ctx.send_buf, ctx.buf_size, params.gpu_id, ""));
    recv_dlist.addDesc(nixlBlobDesc((uintptr_t)ctx.recv_buf, ctx.buf_size, params.gpu_id, ""));

    st = ctx.agent->registerMem(send_dlist);
    if (st != NIXL_SUCCESS) {
        fprintf(stderr, "[%s] registerMem(send) failed: %d\n", my_name.c_str(), st);
        return st;
    }
    st = ctx.agent->registerMem(recv_dlist);
    if (st != NIXL_SUCCESS) {
        fprintf(stderr, "[%s] registerMem(recv) failed: %d\n", my_name.c_str(), st);
        return st;
    }

    // 6. Metadata exchange via in-process promises/futures
    //    Each side publishes its own MD blob then waits for the peer's.
    nixl_blob_t my_md;
    st = ctx.agent->getLocalMD(my_md);
    if (st != NIXL_SUCCESS) {
        fprintf(stderr, "[%s] getLocalMD failed: %d\n", my_name.c_str(), st);
        return st;
    }

    nixl_blob_t peer_md;
    if (params.is_sender) {
        rndv.sender_md_promise.set_value(my_md);
        peer_md = rndv.recvr_md_promise.get_future().get();
    } else {
        rndv.recvr_md_promise.set_value(my_md);
        peer_md = rndv.sender_md_promise.get_future().get();
    }

    std::string peer_name;
    st = ctx.agent->loadRemoteMD(peer_md, peer_name);
    if (st != NIXL_SUCCESS) {
        fprintf(stderr, "[%s] loadRemoteMD failed: %d\n", my_name.c_str(), st);
        return st;
    }

    // 7. Exchange recv_buf addresses so each side knows where to PUT
    uintptr_t my_recv_addr   = (uintptr_t)ctx.recv_buf;
    uintptr_t peer_recv_addr = 0;

    if (params.is_sender) {
        rndv.sender_addr_promise.set_value(my_recv_addr);
        peer_recv_addr = rndv.recvr_addr_promise.get_future().get();
    } else {
        rndv.recvr_addr_promise.set_value(my_recv_addr);
        peer_recv_addr = rndv.sender_addr_promise.get_future().get();
    }

    // 8. Build memory view handles
    //    Local: our send_buf (we PUT from here)
    nixl_local_dlist_t local_send_dlist(VRAM_SEG);
    local_send_dlist.addDesc(nixlBasicDesc((uintptr_t)ctx.send_buf, ctx.buf_size, params.gpu_id));

    //    Remote: peer's recv_buf (we PUT into there)
    nixl_remote_dlist_t remote_dlist(VRAM_SEG);
    remote_dlist.addDesc(nixlRemoteDesc(peer_recv_addr, ctx.buf_size, params.gpu_id, peer_name));

    // prepMemView may need retries even with useProgThread=true
    while ((st = ctx.agent->prepMemView(local_send_dlist, ctx.local_mvh)) != NIXL_SUCCESS) {}
    while ((st = ctx.agent->prepMemView(remote_dlist,     ctx.remote_mvh)) != NIXL_SUCCESS) {}

    return NIXL_SUCCESS;
}

void
bench_teardown(BenchContext &ctx)
{
    if (!ctx.agent) return;

    if (ctx.local_mvh)  ctx.agent->releaseMemView(ctx.local_mvh);
    if (ctx.remote_mvh) ctx.agent->releaseMemView(ctx.remote_mvh);

    // Reconstruct the registration dlists to deregister
    if (ctx.send_buf && ctx.buf_size > 0) {
        nixl_reg_dlist_t send_dlist(VRAM_SEG);
        send_dlist.addDesc(nixlBlobDesc((uintptr_t)ctx.send_buf, ctx.buf_size, ctx.gpu_id, ""));
        ctx.agent->deregisterMem(send_dlist);
    }
    if (ctx.recv_buf && ctx.buf_size > 0) {
        nixl_reg_dlist_t recv_dlist(VRAM_SEG);
        recv_dlist.addDesc(nixlBlobDesc((uintptr_t)ctx.recv_buf, ctx.buf_size, ctx.gpu_id, ""));
        ctx.agent->deregisterMem(recv_dlist);
    }

    if (ctx.send_buf) { cudaFree(ctx.send_buf); ctx.send_buf = nullptr; }
    if (ctx.recv_buf) { cudaFree(ctx.recv_buf); ctx.recv_buf = nullptr; }

    delete ctx.agent;
    ctx.agent     = nullptr;
    ctx.local_mvh = nullptr;
    ctx.remote_mvh = nullptr;
}
