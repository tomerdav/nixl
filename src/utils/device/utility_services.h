/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef NIXL_SRC_UTILS_DEVICE_UTILITY_SERVICES_H
#define NIXL_SRC_UTILS_DEVICE_UTILITY_SERVICES_H

#include "device_allocator.h"

class nixlUtilityServices {
public:
    virtual ~nixlUtilityServices() = default;

    [[nodiscard]] virtual nixlDeviceAllocator &
    deviceAllocator() noexcept = 0;
};

#endif // NIXL_SRC_UTILS_DEVICE_UTILITY_SERVICES_H
