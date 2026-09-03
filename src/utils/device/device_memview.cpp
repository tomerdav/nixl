/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "device/device_memview.h"

#include <gpu/device_types.cuh>

#include "device/device_allocator.h"

static_assert(sizeof(nixl_device_exec_mode_t) == 1);

nixl_status_t
nixlDeviceMemViewAllocate(nixl_device_exec_mode_t execution_mode,
                          nixlMemViewH backend_memview,
                          nixlMemViewH &wrapper_out) noexcept {
    wrapper_out = nullptr;
    if (backend_memview == nullptr) {
        return NIXL_ERR_INVALID_PARAM;
    }

    nixlDeviceAllocator &allocator = nixlGetDeviceAllocator();
    nixlDeviceMem wrapper_mem;
    auto status = allocator.allocDeviceMem(sizeof(nixlDeviceMemViewWrapper), wrapper_mem);
    if (status != NIXL_SUCCESS) {
        return status;
    }

    const nixlDeviceMemViewWrapper host_wrapper{
        execution_mode,
        backend_memview,
    };
    status = allocator.copyHostToDevice(wrapper_mem.get(), &host_wrapper, sizeof(host_wrapper));
    if (status != NIXL_SUCCESS) {
        return status;
    }

    /* Ownership crosses the public handle boundary; reclaimed in
     * nixlDeviceMemViewFree. */
    wrapper_out = wrapper_mem.release();
    return NIXL_SUCCESS;
}

nixl_status_t
nixlDeviceMemViewGetBackend(nixlMemViewH wrapper, nixlMemViewH &backend_out) noexcept {
    backend_out = nullptr;
    if (wrapper == nullptr) {
        return NIXL_ERR_INVALID_PARAM;
    }

    nixlDeviceMemViewWrapper host_wrapper{};
    const auto status =
        nixlGetDeviceAllocator().copyDeviceToHost(&host_wrapper, wrapper, sizeof(host_wrapper));
    if (status != NIXL_SUCCESS) {
        return status;
    }

    backend_out = host_wrapper.backend_memview;
    return NIXL_SUCCESS;
}

void
nixlDeviceMemViewFree(nixlMemViewH wrapper) noexcept {
    nixlGetDeviceAllocator().freeDeviceMem(wrapper);
}
