/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef NIXL_SRC_UTILS_DEVICE_DEVICE_ALLOCATOR_PLUGIN_H
#define NIXL_SRC_UTILS_DEVICE_DEVICE_ALLOCATOR_PLUGIN_H

#include <cstdint>

#include "device_allocator.h"

enum class nixlDeviceRuntime : uint32_t {
    CUDA = 1,
    HIP = 2,
};

constexpr uint32_t NIXL_DEVICE_ALLOCATOR_PLUGIN_API_VERSION = 1;

struct nixlDeviceAllocatorPluginV1 {
    uint32_t apiVersion;
    nixlDeviceRuntime runtime;
    const char *name;
    const char *version;
    nixlDeviceAllocator *(*getAllocator)() noexcept;
};

using nixlDeviceAllocatorPluginInit = nixlDeviceAllocatorPluginV1 *(*)() noexcept;

extern "C" NIXL_DEVICE_ALLOCATOR_EXPORT nixlDeviceAllocatorPluginV1 *
nixl_device_allocator_plugin_init() noexcept;

#endif // NIXL_SRC_UTILS_DEVICE_DEVICE_ALLOCATOR_PLUGIN_H
