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

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <thread>
#include <vector>

#include "device/device_allocator.h"
#include "gpu_utils.h"
#include "plugin_manager.h"

class nixlPluginManagerTestPeer {
public:
    static std::shared_ptr<const nixlDeviceAllocatorPluginHandle>
    loadDeviceAllocatorPlugin(const std::string &path, nixlDeviceRuntime runtime) {
        return nixlPluginManager::loadDeviceAllocatorPluginFromPath(path, runtime);
    }
};

namespace gtest {
namespace device_allocator {

    constexpr size_t kSize = 4096;

    TEST(deviceMemHost, EmptyHandleOperationsAreSafe) {
        nixlDeviceMem mem;
        EXPECT_FALSE(static_cast<bool>(mem));
        EXPECT_EQ(mem.devicePointer(), nullptr);
        EXPECT_EQ(mem.size(), 0u);
        mem.reset();
        EXPECT_EQ(mem.release(), nullptr);

        nixlDeviceMem moved(std::move(mem));
        EXPECT_FALSE(static_cast<bool>(moved));
    }

    TEST(mappedHostMemHost, EmptyHandleOperationsAreSafe) {
        nixlMappedHostMem mapped;
        EXPECT_FALSE(static_cast<bool>(mapped));
        EXPECT_EQ(mapped.hostPtr(), nullptr);
        EXPECT_EQ(mapped.devicePointer(), nullptr);
        EXPECT_EQ(mapped.size(), 0u);
        mapped.reset();

        nixlMappedHostMem moved(std::move(mapped));
        EXPECT_FALSE(static_cast<bool>(moved));
    }

    TEST(deviceAllocatorHost, AccessorIsStableAndFreeNullIsSafe) {
        auto &services = nixlPluginManager::getInstance();
        nixlDeviceAllocator &allocator = services.deviceAllocator();
        EXPECT_EQ(&allocator, &services.deviceAllocator());
        allocator.freeDeviceMem(nullptr);
    }

    TEST(deviceAllocatorHost, SelectsProviderFromHardwareInventory) {
        EXPECT_EQ(nixlPluginManager::deviceAllocatorProbeOrder(1, 0),
                  std::vector<nixlDeviceRuntime>{nixlDeviceRuntime::CUDA});
        EXPECT_EQ(nixlPluginManager::deviceAllocatorProbeOrder(0, 1),
                  std::vector<nixlDeviceRuntime>{nixlDeviceRuntime::HIP});
        EXPECT_EQ(nixlPluginManager::deviceAllocatorProbeOrder(1, 1),
                  std::vector<nixlDeviceRuntime>{nixlDeviceRuntime::CUDA});
        EXPECT_EQ(
            nixlPluginManager::deviceAllocatorProbeOrder(0, 0),
            (std::vector<nixlDeviceRuntime>{nixlDeviceRuntime::CUDA, nixlDeviceRuntime::HIP}));
    }

    TEST(deviceAllocatorHost, AccessorIsThreadSafe) {
        constexpr size_t kThreads = 8;
        std::array<nixlDeviceAllocator *, kThreads> allocators{};
        std::array<std::thread, kThreads> threads;
        for (size_t i = 0; i < kThreads; ++i) {
            threads[i] = std::thread([i, &allocators]() {
                allocators[i] = &nixlPluginManager::getInstance().deviceAllocator();
            });
        }
        for (auto &thread : threads) {
            thread.join();
        }
        EXPECT_TRUE(std::all_of(allocators.begin(), allocators.end(), [&](const auto *allocator) {
            return allocator == allocators.front();
        }));
    }

    TEST(deviceAllocatorHost, UnsupportedAllocatorPreservesNotSupported) {
        auto &allocator = nixlGetUnsupportedDeviceAllocator();
        nixlDeviceMem device_mem;
        nixlMappedHostMem mapped_mem;
        EXPECT_EQ(allocator.allocDeviceMem(kSize, device_mem), NIXL_ERR_NOT_SUPPORTED);
        EXPECT_EQ(allocator.allocMappedHostMem(kSize, mapped_mem), NIXL_ERR_NOT_SUPPORTED);
        EXPECT_FALSE(device_mem);
        EXPECT_FALSE(mapped_mem);
    }

    TEST(deviceAllocatorHost, ValidatesProviderBeforeUsingIt) {
        EXPECT_NE(nixlPluginManagerTestPeer::loadDeviceAllocatorPlugin(MOCK_DEVICE_ALLOCATOR_VALID,
                                                                       nixlDeviceRuntime::CUDA),
                  nullptr);
        EXPECT_EQ(nixlPluginManagerTestPeer::loadDeviceAllocatorPlugin(
                      "/missing/device/allocator/plugin.so", nixlDeviceRuntime::CUDA),
                  nullptr);
        EXPECT_EQ(nixlPluginManagerTestPeer::loadDeviceAllocatorPlugin(
                      MOCK_DEVICE_ALLOCATOR_WRONG_ABI, nixlDeviceRuntime::CUDA),
                  nullptr);
        EXPECT_EQ(nixlPluginManagerTestPeer::loadDeviceAllocatorPlugin(
                      MOCK_DEVICE_ALLOCATOR_WRONG_RUNTIME, nixlDeviceRuntime::CUDA),
                  nullptr);
        EXPECT_EQ(nixlPluginManagerTestPeer::loadDeviceAllocatorPlugin(
                      MOCK_DEVICE_ALLOCATOR_FAILED_PROBE, nixlDeviceRuntime::CUDA),
                  nullptr);
    }

    TEST(deviceAllocatorHost, ZeroSizeAllocationsLeaveOutputsEmpty) {
        nixlDeviceAllocator &allocator = nixlPluginManager::getInstance().deviceAllocator();
        nixlDeviceMem device_mem;
        EXPECT_NE(allocator.allocDeviceMem(0, device_mem), NIXL_SUCCESS);
        EXPECT_FALSE(device_mem);

        nixlMappedHostMem mapped_mem;
        EXPECT_NE(allocator.allocMappedHostMem(0, mapped_mem), NIXL_SUCCESS);
        EXPECT_FALSE(mapped_mem);
    }

    class deviceAllocatorTest : public testing::Test {
    protected:
        nixlDeviceAllocator *allocator_ = nullptr;

        void
        SetUp() override {
            int count = 0;
            gpuGetDeviceCount(&count, "Probing GPU devices");
            if (count < 1) {
                GTEST_SKIP() << "No GPU is available.";
            }
            gpuSetDevice(0, "Selecting GPU 0");
            allocator_ = &nixlPluginManager::getInstance().deviceAllocator();
            int device = 0;
            const nixl_status_t status = allocator_->getActiveDevice(device);
            if (status == NIXL_ERR_NOT_SUPPORTED) {
                GTEST_SKIP() << "No device allocator implementation is available.";
            }
            ASSERT_EQ(status, NIXL_SUCCESS);
        }
    };

    TEST_F(deviceAllocatorTest, AllocCopyFree) {
        nixlDeviceAllocator &allocator = *allocator_;

        nixlDeviceMem mem;
        ASSERT_EQ(allocator.allocDeviceMem(kSize, mem), NIXL_SUCCESS);
        void *const device_ptr = mem.devicePointer();
        EXPECT_NE(allocator.allocDeviceMem(0, mem), NIXL_SUCCESS);
        EXPECT_EQ(mem.devicePointer(), device_ptr);
        EXPECT_EQ(mem.size(), kSize);

        std::vector<unsigned char> src(kSize, 0xA5);
        std::vector<unsigned char> dst(kSize, 0);
        ASSERT_EQ(allocator.copyHostToDevice(mem.devicePointer(), src.data(), kSize), NIXL_SUCCESS);
        ASSERT_EQ(allocator.copyDeviceToHost(dst.data(), mem.devicePointer(), kSize), NIXL_SUCCESS);
        EXPECT_EQ(dst, src);

        ASSERT_EQ(allocator.memsetDeviceMem(mem.devicePointer(), 0, kSize), NIXL_SUCCESS);
        ASSERT_EQ(allocator.copyDeviceToHost(dst.data(), mem.devicePointer(), kSize), NIXL_SUCCESS);
        EXPECT_EQ(dst, std::vector<unsigned char>(kSize, 0));

        allocator.freeDeviceMem(mem.release());

        nixlMappedHostMem mapped;
        ASSERT_EQ(allocator.allocMappedHostMem(kSize, mapped), NIXL_SUCCESS);
        void *const host_ptr = mapped.hostPtr();
        void *const mapped_device_ptr = mapped.devicePointer();
        EXPECT_NE(allocator.allocMappedHostMem(0, mapped), NIXL_SUCCESS);
        EXPECT_EQ(mapped.hostPtr(), host_ptr);
        EXPECT_EQ(mapped.devicePointer(), mapped_device_ptr);
        EXPECT_EQ(mapped.size(), kSize);
        std::fill_n(mapped.asHost<unsigned char>(), kSize, 0x5A);
        ASSERT_EQ(allocator.copyDeviceToHost(dst.data(), mapped.devicePointer(), kSize),
                  NIXL_SUCCESS);
        EXPECT_EQ(dst, std::vector<unsigned char>(kSize, 0x5A));
    }

} // namespace device_allocator
} // namespace gtest
