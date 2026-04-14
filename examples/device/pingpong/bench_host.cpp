#include "bench_host.h"

#include <cuda_runtime.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>

// ---- BenchContext::setup -----------------------------------------------------

nixl_status_t
BenchContext::setup(const BenchParams &params,
                    const char *peer_ip, int peer_port, int my_port)
{
    is_sender = params.is_sender;
    gpu_id    = params.gpu_id;
    buf_size  = params.msg_size + sizeof(uint64_t);

    const std::string my_name   = params.is_sender ? "sender"   : "receiver";
    const std::string peer_name = params.is_sender ? "receiver" : "sender";

    // 1. Bind CUDA device
    cudaSetDevice(params.gpu_id);

    // 2. Create NIXL agent with both a progress thread (drives UCX completions)
    //    and a listen thread (accepts incoming TCP metadata connections from peer).
    //    The constructor throws std::runtime_error if the listen port is in use.
    nixlAgentConfig cfg(/*useProgThread=*/true, /*useListenThread=*/true, my_port);
    try {
        agent = std::make_unique<nixlAgent>(my_name, cfg);
    } catch (const std::exception &e) {
        fprintf(stderr, "[%s] nixlAgent construction failed (port %d in use?): %s\n",
                my_name.c_str(), my_port, e.what());
        return NIXL_ERR_NOT_FOUND;
    }

    // 3. Create UCX backend
    nixl_b_params_t bparams;
    nixl_status_t st = agent->createBackend("UCX", bparams, ucx_backend);
    if (st != NIXL_SUCCESS) {
        fprintf(stderr, "[%s] createBackend failed: %d\n", my_name.c_str(), st);
        return st;
    }

    // 4. Allocate and zero device buffers
    if (cudaMalloc(&send_buf, buf_size) != cudaSuccess ||
        cudaMalloc(&recv_buf, buf_size) != cudaSuccess) {
        fprintf(stderr, "[%s] cudaMalloc failed\n", my_name.c_str());
        return NIXL_ERR_NOT_FOUND;
    }
    cudaMemset(send_buf, 0, buf_size);
    cudaMemset(recv_buf, 0, buf_size);

    // 5. Register both buffers.
    //    recv_buf must be registered so its UCX rkey is serialised into the
    //    metadata blob; the peer needs that rkey to PUT into our recv_buf.
    nixl_reg_dlist_t send_dlist(VRAM_SEG), recv_dlist(VRAM_SEG);
    send_dlist.addDesc(nixlBlobDesc((uintptr_t)send_buf, buf_size, params.gpu_id, ""));
    recv_dlist.addDesc(nixlBlobDesc((uintptr_t)recv_buf, buf_size, params.gpu_id, ""));

    st = agent->registerMem(send_dlist);
    if (st != NIXL_SUCCESS) {
        fprintf(stderr, "[%s] registerMem(send) failed: %d\n", my_name.c_str(), st);
        return st;
    }
    st = agent->registerMem(recv_dlist);
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
    //    Sender drives both calls; receiver's listen thread handles them
    //    passively — no explicit metadata API calls on the receiver side.
    nixl_opt_args_t md_args;
    md_args.ipAddr = peer_ip;
    md_args.port   = peer_port;   // peer's listen port

    if (params.is_sender) {
        fprintf(stderr, "[%s] fetching remote MD from %s:%d ...\n",
                my_name.c_str(), peer_ip, peer_port);
        // Retry: receiver's listen thread may not be ready yet.
        while ((st = agent->fetchRemoteMD(peer_name, &md_args)) != NIXL_SUCCESS) {
            fprintf(stderr, "[%s] fetchRemoteMD not ready (%d), retrying...\n",
                    my_name.c_str(), st);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        fprintf(stderr, "[%s] fetchRemoteMD done, pushing local MD...\n", my_name.c_str());
        st = agent->sendLocalMD(&md_args);
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
    auto my_recv_addr = reinterpret_cast<uintptr_t>(recv_buf);
    memcpy(addr_blob.data(), &my_recv_addr, sizeof(uintptr_t));

    // Wait silently until the listen thread has loaded the peer's metadata.
    // checkRemoteMD returns NIXL_ERR_NOT_FOUND (without logging) until
    // remoteBackends_ is populated; genNotif would log ERROR on every retry.
    nixl_xfer_dlist_t empty_dlist(VRAM_SEG);
    fprintf(stderr, "[%s] waiting for peer metadata to be ready...\n", my_name.c_str());
    while (agent->checkRemoteMD(peer_name, empty_dlist) != NIXL_SUCCESS)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

    nixl_opt_args_t notif_args;
    notif_args.backends.push_back(ucx_backend);

    fprintf(stderr, "[%s] sending recv_buf addr to peer...\n", my_name.c_str());
    while ((st = agent->genNotif(peer_name, addr_blob, &notif_args)) != NIXL_SUCCESS)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

    fprintf(stderr, "[%s] waiting for peer recv_buf addr...\n", my_name.c_str());
    nixl_notifs_t notifs;
    while (notifs[peer_name].empty()) {
        agent->getNotifs(notifs);
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
    local_send_dlist.addDesc(nixlBasicDesc((uintptr_t)send_buf, buf_size, params.gpu_id));

    nixl_remote_dlist_t remote_dlist(VRAM_SEG);
    remote_dlist.addDesc(nixlRemoteDesc(peer_recv_addr, buf_size, params.gpu_id, peer_name));

    while ((st = agent->prepMemView(local_send_dlist, local_mvh)) != NIXL_SUCCESS) {}
    while ((st = agent->prepMemView(remote_dlist,     remote_mvh)) != NIXL_SUCCESS) {}
    fprintf(stderr, "[%s] memory views ready — setup complete\n", my_name.c_str());

    return NIXL_SUCCESS;
}

// ---- BenchContext::~BenchContext ---------------------------------------------

BenchContext::~BenchContext()
{
    if (!agent) return;

    if (local_mvh)  agent->releaseMemView(local_mvh);
    if (remote_mvh) agent->releaseMemView(remote_mvh);

    if (send_buf && buf_size > 0) {
        nixl_reg_dlist_t send_dlist(VRAM_SEG);
        send_dlist.addDesc(nixlBlobDesc((uintptr_t)send_buf, buf_size, gpu_id, ""));
        agent->deregisterMem(send_dlist);
        cudaFree(send_buf);
    }
    if (recv_buf && buf_size > 0) {
        nixl_reg_dlist_t recv_dlist(VRAM_SEG);
        recv_dlist.addDesc(nixlBlobDesc((uintptr_t)recv_buf, buf_size, gpu_id, ""));
        agent->deregisterMem(recv_dlist);
        cudaFree(recv_buf);
    }

    // agent unique_ptr destroyed after this body — agent outlives the cleanup above.
}
