#include "nixl.h"
#include "nixl_params.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

// ----------------------------------------------------------------------------
// Minimal usage
// ----------------------------------------------------------------------------
static void
usage(const char *prog) {
    fprintf(stderr,
            "Usage: %s --role <sender|receiver>\n"
            "          --peer-ip   <ip>    peer's hostname or IP\n"
            "          --peer-port <port>  peer's NIXL listen port\n"
            "          --listen-port <port>  our NIXL listen port\n"
            "         [--msg-size  <bytes>]  (default 8)\n"
            "         [--iters     <n>]      (default 1000)\n"
            "         [--warmup    <n>]      (default 100)\n"
            "         [--gpu       <id>]     (default 0)\n"
            "         [--warp]               use WARP level (default: THREAD)\n",
            prog);
    exit(1);
}

int
main(int argc, char *argv[]) {
    // ---- argument parsing --------------------------------------------------
    const char *role_str    = nullptr;
    const char *peer_ip     = nullptr;
    int         peer_port   = 0;
    int         listen_port = 0;
    size_t      msg_size    = 8;
    uint64_t    num_iters   = 1000;
    uint64_t    warmup_iters = 100;
    int         gpu_id      = 0;
    bool        use_warp    = false;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--role")         && i+1 < argc) role_str    = argv[++i];
        else if (!strcmp(argv[i], "--peer-ip")      && i+1 < argc) peer_ip     = argv[++i];
        else if (!strcmp(argv[i], "--peer-port")    && i+1 < argc) peer_port   = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--listen-port")  && i+1 < argc) listen_port = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--msg-size")     && i+1 < argc) msg_size    = (size_t)atoll(argv[++i]);
        else if (!strcmp(argv[i], "--iters")        && i+1 < argc) num_iters   = (uint64_t)atoll(argv[++i]);
        else if (!strcmp(argv[i], "--warmup")       && i+1 < argc) warmup_iters = (uint64_t)atoll(argv[++i]);
        else if (!strcmp(argv[i], "--gpu")          && i+1 < argc) gpu_id      = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--warp"))                        use_warp    = true;
        else usage(argv[0]);
    }

    if (!role_str || !peer_ip || peer_port == 0 || listen_port == 0)
        usage(argv[0]);

    bool is_sender = (strcmp(role_str, "sender") == 0);
    if (!is_sender && strcmp(role_str, "receiver") != 0) {
        fprintf(stderr, "Unknown role '%s'; expected sender or receiver\n", role_str);
        usage(argv[0]);
    }

    // TODO: pass to bench_setup()
    (void)msg_size;
    (void)num_iters;
    (void)warmup_iters;
    (void)gpu_id;
    (void)use_warp;

    // ---- NIXL agent --------------------------------------------------------
    std::string agent_name = is_sender ? "sender" : "receiver";
    std::string peer_name  = is_sender ? "receiver" : "sender";

    nixlAgentConfig cfg;
    cfg.useListenThread = true;
    cfg.listenPort      = listen_port;

    nixlAgent agent(agent_name, cfg);

    nixl_b_params_t bparams;
    nixlBackendH *ucx_backend;
    agent.createBackend("UCX", bparams, ucx_backend);

    // TODO: bench_setup(), kernel launch, latency output, teardown

    return 0;
}
