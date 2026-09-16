/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "device/device_allocator_plugin.h"

namespace {
nixlDeviceAllocator *
getAllocator() noexcept {
#ifdef MOCK_DEVICE_ALLOCATOR_PROBE_SUCCESS
    return &nixlGetUnsupportedDeviceAllocator();
#else
    return nullptr;
#endif
}
} // namespace

extern "C" NIXL_DEVICE_ALLOCATOR_EXPORT nixlDeviceAllocatorPluginV1 *
nixl_device_allocator_plugin_init() noexcept {
    static nixlDeviceAllocatorPluginV1 plugin = {MOCK_DEVICE_ALLOCATOR_API_VERSION,
                                                 MOCK_DEVICE_ALLOCATOR_RUNTIME,
                                                 "mock",
                                                 "1.0.0",
                                                 getAllocator};
    return &plugin;
}
