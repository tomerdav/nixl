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
#include "device/device_allocator.h"

namespace {
class nixlUnsupportedDeviceAllocator final : public nixlDeviceAllocator {
public:
    nixl_status_t
    doAllocDeviceMem(void *&, size_t) noexcept override {
        return NIXL_ERR_NOT_SUPPORTED;
    }

    void
    doFreeDeviceMem(void *) noexcept override {}

    nixl_status_t
    doAllocMappedHostMem(void *&, void *&, size_t) noexcept override {
        return NIXL_ERR_NOT_SUPPORTED;
    }

    void
    doFreeMappedHostMem(void *) noexcept override {}

    nixl_status_t
    copyHostToDevice(void *, const void *, size_t) noexcept override {
        return NIXL_ERR_NOT_SUPPORTED;
    }

    nixl_status_t
    copyDeviceToHost(void *, const void *, size_t) noexcept override {
        return NIXL_ERR_NOT_SUPPORTED;
    }

    nixl_status_t
    memsetDeviceMem(void *, int, size_t) noexcept override {
        return NIXL_ERR_NOT_SUPPORTED;
    }

    nixl_status_t
    synchronize() noexcept override {
        return NIXL_ERR_NOT_SUPPORTED;
    }

    nixl_status_t
    getActiveDevice(int &) noexcept override {
        return NIXL_ERR_NOT_SUPPORTED;
    }

    nixl_status_t
    setActiveDevice(int) noexcept override {
        return NIXL_ERR_NOT_SUPPORTED;
    }
};
} // namespace

nixlDeviceAllocator &
nixlGetUnsupportedDeviceAllocator() noexcept {
    static nixlUnsupportedDeviceAllocator unsupported;
    return unsupported;
}
