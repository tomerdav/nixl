#pragma once

// Device-only header: pulls in nixl_device.cuh (requires nvcc).
// Include bench_kernel_iface.h instead when compiling host (.cpp) code.

#include "bench_kernel_iface.h"
#include "nixl_device.cuh"
