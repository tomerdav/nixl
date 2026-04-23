#include "bench_kernel.cuh"
#include "nixl_device.cuh"
#include "nixl_types.h"
#include <cstdint>

__device__ static void
wait_sequence_number(volatile uint64_t *counter, uint64_t expected_value) {
    while (*counter < expected_value) {
        __nanosleep(50); // Sleep for 50 nanoseconds to reduce contention, taken from UCX
    }
}

template<nixl_gpu_level_t level>
__device__ static nixl_status_t
do_put_async(nixlMemViewH local_mvh,
             nixlMemViewH remote_mvh,
             size_t total_size,
             nixlGpuXferStatusH &xfer_status) {
    nixlMemViewElem src{local_mvh, 0, 0};
    nixlMemViewElem dst{remote_mvh, 0, 0};
    nixl_status_t status = nixlPut<level>(src, dst, total_size, 0, 0, &xfer_status);
    if (status != NIXL_IN_PROG) {
        if (threadIdx.x == 0) {
            printf("nixlPut failed with status %d\n", status);
        }
        return status;
    }
    return status;
}


template<nixl_gpu_level_t level>
__device__ static nixl_status_t
do_put_sync(nixlMemViewH local_mvh,
            nixlMemViewH remote_mvh,
            size_t total_size,
            nixlGpuXferStatusH &xfer_status) {
    nixlMemViewElem src{local_mvh, 0, 0};
    nixlMemViewElem dst{remote_mvh, 0, 0};

    // Initiate the transfer
    do_put_async<level>(local_mvh, remote_mvh, total_size, xfer_status);
    // Wait for the transfer to complete
    do {
        status = nixlGpuGetXferStatus<level>(xfer_status);
    } while (status == NIXL_IN_PROG);
    return status;
}


template<nixl_gpu_level_t level>
__global__ void
nixl_pingpong_latency_kernel(gpu_bench_ctx ctx, uint64_t *elapsed_device) {
    constexpr bool is_warp = (level == nixl_gpu_level_t::WARP);

    if constexpr (!is_warp) {
        if (threadIdx.x != 0) return; // Only one thread does the work for non-warp level
    }

    const int lane_id = threadIdx.x % 32; // Get the lane ID for warp-level operations

    // Setup counter pointers
    volatile uint64_t *send_counter =
        reinterpret_cast<volatile uint64_t *>(ctx.send_buf + ctx.msg_size);
    volatile uint64_t *recv_counter =
        reinterpret_cast<volatile uint64_t *>(ctx.recv_buf + ctx.msg_size);

    const size_t total_size = ctx.msg_size + sizeof(uint64_t); // Message size + counter
    nixlGpuXferStatusH xfer_status;

    // warmup
    const uint64_t total_iters = ctx.num_iters + ctx.warmup_iters;
    uint64_t start_time = 0;

    for (uint64_t i = 0; i < total_iters; i++) {
        if (ctx.is_sender && lane_id == 0 && i == ctx.warmup_iters) {
            start_time = clock64(); // Start timing after warmup
        }

        // ping pong body
        if (ctx.is_sender) {
            if (lane_id == 0) {
                *send_counter = i + 1; // Increment send counter to signal the receiver
            }
            if constexpr (is_warp) {
                __syncwarp(); // Ensure all threads see the updated counter
            }
            do_put_async<level>(ctx.local_mvh, ctx.remote_mvh, total_size, xfer_status);

            if (lane_id == 0) {
                wait_sequence_number(recv_counter,
                                     i + 1); // Wait for the receiver to process the message
            }
            if constexpr (is_warp) {
                __syncwarp(); // Ensure all threads are synchronized before the next iteration
            }
        } else {
            if (lane_id == 0) {
                wait_sequence_number(recv_counter, i + 1); // Wait for the sender to signal
            }
            if constexpr (is_warp) {
                __syncwarp(); // Ensure all threads see the updated counter
            }

            if (lane_id == 0) {
                *send_counter = i + 1; // Increment send counter to signal the sender
            }

            do_put_async<level>(ctx.local_mvh, ctx.remote_mvh, total_size, xfer_status);
        }
    }

    if (ctx.is_sender && lane_id == 0) {
        uint64_t end_time = clock64();
        *elapsed_device = end_time - start_time;
    }
}

// Explicit template instantiations for the desired levels
template __global__ void
nixl_pingpong_latency_kernel<nixl_gpu_level_t::THREAD>(gpu_bench_ctx ctx, uint64_t *elapsed_device);
template __global__ void
nixl_pingpong_latency_kernel<nixl_gpu_level_t::WARP>(gpu_bench_ctx ctx, uint64_t *elapsed_device);

void
launch_pingpong_thread(gpu_bench_ctx ctx, uint64_t *d_elapsed, cudaStream_t stream) {
    nixl_pingpong_latency_kernel<nixl_gpu_level_t::THREAD><<<1, 1, 0, stream>>>(ctx, d_elapsed);
}

void
launch_pingpong_warp(gpu_bench_ctx ctx, uint64_t *d_elapsed, cudaStream_t stream) {
    nixl_pingpong_latency_kernel<nixl_gpu_level_t::WARP><<<1, 32, 0, stream>>>(ctx, d_elapsed);
}
