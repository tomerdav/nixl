/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef NIXL_SRC_UTILS_DEVICE_DEVICE_MEMVIEW_H
#define NIXL_SRC_UTILS_DEVICE_DEVICE_MEMVIEW_H

#include <nixl_types.h>

#include <gpu/device_types.cuh>

/**
 * Wrap a backend memview into the device-dispatch handle. The backend that
 * created the memview decides the execution mode; this layer is
 * backend-agnostic.
 */
[[nodiscard]] nixl_status_t
nixlDeviceMemViewAllocate(nixl_device_exec_mode_t execution_mode,
                          nixlMemViewH backend_memview,
                          nixlMemViewH &wrapper_out) noexcept;

[[nodiscard]] nixl_status_t
nixlDeviceMemViewGetBackend(nixlMemViewH wrapper, nixlMemViewH &backend_out) noexcept;

void
nixlDeviceMemViewFree(nixlMemViewH wrapper) noexcept;

#endif // NIXL_SRC_UTILS_DEVICE_DEVICE_MEMVIEW_H
