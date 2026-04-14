#include "bench_host.h"

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <chrono>

nixl_status_t
bench_setup(BenchContext &ctx, const BenchParams &params,
            const char *peer_ip, int peer_port, int my_port)
{
    ctx.is_sender = params.is_sender;
    ctx.gpu_id    = params.gpu_id;
    ctx.buf_size  = params.msg_size + sizeof(uint64_t);

    const std::string my_name   = params.is_sender ? "sender"   : "receiver";
    const std::string peer_name = params.is_sender ? "receiver" : "sender";

    // 1. Bind CUDA device
    cudaSetDevice(params.gpu_id);

    // 2. Create NIXL agent with both a progress thread (drives UCX completions)
    //    and a listen thread (accepts incoming TCP metadata connections from peer).
    nixlAgentConfig cfg(/*useProgThread=*/true, /*useListenThread=*/true, my_port);
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

    // 5. Register both buffers.
    //    recv_buf must be registered so its UCX rkey is serialised into the
    //    metadata blob; the peer needs that rkey to PUT into our recv_buf.
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

    // 6. Metadata exchange via TCP.
    //
    //    The metadata blob contains UCX connection info and rkeys for all
    //    registered buffers. fetchRemoteMD connects to the peer's listen port
    //    and downloads the peer's blob; sendLocalMD connects and uploads ours.
    //
    //    Sender drives both calls; receiver's listen thread handles them passively
    //    — no explicit metadata API calls on the receiver side.
    nixl_opt_args_t md_args;
    md_args.ipAddr = peer_ip;
    md_args.port   = peer_port;   // peer's listen port

    if (params.is_sender) {
        fprintf(stderr, "[%s] fetching remote MD from %s:%d ...\n",
                my_name.c_str(), peer_ip, peer_port);
        // Retry: receiver's listen thread may not be ready yet.
        while ((st = ctx.agent->fetchRemoteMD(peer_name, &md_args)) != NIXL_SUCCESS) {
            fprintf(stderr, "[%s] fetchRemoteMD not ready (%d), retrying...\n",
                    my_name.c_str(), st);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        fprintf(stderr, "[%s] fetchRemoteMD done, pushing local MD...\n", my_name.c_str());
        st = ctx.agent->sendLocalMD(&md_args);
        if (st != NIXL_SUCCESS) {
            fprintf(stderr, "[%s] sendLocalMD failed: %d\n", my_name.c_str(), st);
            return st;
        }
        fprintf(stderr, "[%s] metadata exchange complete\n", my_name.c_str());
    } else {
        fprintf(stderr, "[%s] listen thread will handle incoming metadata\n",
                my_name.c_str());
    }

    // 7. Address exchange via NIXL notifications.
    //
    //    We need each side's recv_buf device address to build nixlRemoteDesc.
    //    genNotif sends a small blob to the peer; getNotifs drains the local
    //    inbox.  genNotif returns non-SUCCESS until the peer's metadata is
    //    loaded locally, so spinning on it is the correct wait for the receiver
    //    (whose listen thread loads the sender's MD asynchronously).
    nixl_blob_t addr_blob(sizeof(uintptr_t), '\0');
    uintptr_t my_recv_addr = (uintptr_t)ctx.recv_buf;
    memcpy(&addr_blob[0], &my_recv_addr, sizeof(uintptr_t));

    fprintf(stderr, "[%s] sending recv_buf addr to peer...\n", my_name.c_str());
    while ((st = ctx.agent->genNotif(peer_name, addr_blob)) != NIXL_SUCCESS)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

    fprintf(stderr, "[%s] waiting for peer recv_buf addr...\n", my_name.c_str());
    nixl_notifs_t notifs;
    while (notifs[peer_name].empty()) {
        ctx.agent->getNotifs(notifs);
        if (notifs[peer_name].empty())
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    uintptr_t peer_recv_addr = 0;
    memcpy(&peer_recv_addr, notifs[peer_name][0].data(), sizeof(uintptr_t));
    fprintf(stderr, "[%s] peer recv_buf addr: 0x%lx\n", my_name.c_str(), peer_recv_addr);

    // 8. Build memory view handles.
    //    local_mvh  — covers our send_buf; the kernel POSTs PUTs from here.
    //    remote_mvh — covers peer's recv_buf; the kernel PUTs into there.
    nixl_local_dlist_t local_send_dlist(VRAM_SEG);
    local_send_dlist.addDesc(nixlBasicDesc((uintptr_t)ctx.send_buf, ctx.buf_size, params.gpu_id));

    nixl_remote_dlist_t remote_dlist(VRAM_SEG);
    remote_dlist.addDesc(nixlRemoteDesc(peer_recv_addr, ctx.buf_size, params.gpu_id, peer_name));

    while ((st = ctx.agent->prepMemView(local_send_dlist, ctx.local_mvh)) != NIXL_SUCCESS) {}
    while ((st = ctx.agent->prepMemView(remote_dlist,     ctx.remote_mvh)) != NIXL_SUCCESS) {}
    fprintf(stderr, "[%s] memory views ready — setup complete\n", my_name.c_str());

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
    ctx.agent      = nullptr;
    ctx.local_mvh  = nullptr;
    ctx.remote_mvh = nullptr;
}
