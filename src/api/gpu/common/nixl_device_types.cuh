/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef NIXL_SRC_API_GPU_COMMON_NIXL_DEVICE_TYPES_CUH
#define NIXL_SRC_API_GPU_COMMON_NIXL_DEVICE_TYPES_CUH

#include <cstddef>
#include <cstdint>

#include <nixl_types.h>

struct nixlGpuXferStatusH {
    alignas(16) unsigned char storage[64] = {};
};

enum class nixl_gpu_level_t : uint64_t { THREAD = 0, WARP = 1, BLOCK = 2, GRID = 3 };

namespace nixl_gpu_flags {
constexpr uint64_t defer = 1;
} // namespace nixl_gpu_flags

struct nixlMemViewElem {
    nixlMemViewH mvh;
    size_t index; /**< Index in the memory view */
    size_t offset; /**< Offset within the buffer */
};

/** Optional per-put diagnostics filled by the GPU put path (proxy backend). */
struct nixlGpuPutStats {
    /** Set to 1 when enqueue waited on a host-confirmed full work ring. */
    uint32_t ring_backpressure = 0;
    /** GPU cycles spent atomically claiming a proxy-ring ticket. */
    uint64_t ticket_claim_cycles = 0;
    /** GPU cycles spent writing the proxy-ring submission record. */
    uint64_t submission_write_cycles = 0;
    /** GPU cycles spent publishing the record to the proxy worker. */
    uint64_t publication_cycles = 0;
};

#endif // NIXL_SRC_API_GPU_COMMON_NIXL_DEVICE_TYPES_CUH
