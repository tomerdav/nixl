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
        // --role/--peer-ip/--peer-port/--listen-port are not used in single-process mode
        if (role_str || peer_ip || peer_port || listen_port) {
            fprintf(stderr,
                    "--single-process is mutually exclusive with "
                    "--role/--peer-ip/--peer-port/--listen-port\n");
            usage(argv[0]);
        }
    } else {
        if (!role_str || !peer_ip || peer_port == 0 || listen_port == 0) usage(argv[0]);

        bool is_sender = (strcmp(role_str, "sender") == 0);
        if (!is_sender && strcmp(role_str, "receiver") != 0) {
            fprintf(stderr, "Unknown role '%s'; expected sender or receiver\n", role_str);
            usage(argv[0]);
        }
    }

    // TODO: pass all params to bench_setup() / single_process_run()
    (void)msg_size;
    (void)num_iters;
    (void)warmup_iters;
    (void)gpu_id;
    (void)use_warp;
    (void)role_str;
    (void)peer_ip;
    (void)peer_port;
    (void)listen_port;
    (void)single_process;

    // TODO (two-process): bench_setup(), kernel launch, latency output, teardown
    // TODO (single_process):    spawn sender/receiver threads, exchange MD via
    //                     getLocalMD/loadRemoteMD, run kernels concurrently,
    //                     collect latency from sender thread

    return 0;
}
