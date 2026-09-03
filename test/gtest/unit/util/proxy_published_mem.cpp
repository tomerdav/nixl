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

#include <cstdint>
#include <limits>
#include <memory>

#include "device/proxy/proxy_published_mem.h"
#include "mocks/proxy_mocks.h"

namespace gtest {
namespace proxy_published_mem {

    using proxy_mocks::MockDeviceOps;

    /** Mock device memory cannot be GDRCopy-pinned, so these tests exercise the fallback. */
    TEST(ProxyPublishedMemTest, FallbackPublishesWordsTheDeviceAliasReads) {
        MockDeviceOps ops;
        std::unique_ptr<nixl::hostPublishedDeviceMem> mem;
        ASSERT_EQ(nixl::hostPublishedDeviceMem::create(ops, 3 * sizeof(uint64_t), mem),
                  NIXL_SUCCESS);
        ASSERT_NE(mem, nullptr);
        EXPECT_EQ(mem->size(), 3 * sizeof(uint64_t));
        EXPECT_EQ(mem->backingKind(), nixl::hostPublishedDeviceMem::backing::mapped_host);

        const uint64_t *words = ops.hostAlias(static_cast<const uint64_t *>(mem->devicePointer()));
        ASSERT_NE(words, nullptr);
        EXPECT_NE(static_cast<const void *>(words), mem->devicePointer());
        for (size_t i = 0; i < 3; ++i) {
            EXPECT_EQ(words[i], 0u) << i;
        }

        EXPECT_EQ(mem->publish(0, 7), NIXL_SUCCESS);
        EXPECT_EQ(mem->publish(2 * sizeof(uint64_t), 0x123456789abcdef0), NIXL_SUCCESS);
        EXPECT_EQ(words[0], 7u);
        EXPECT_EQ(words[1], 0u);
        EXPECT_EQ(words[2], 0x123456789abcdef0u);

        mem.reset();
        EXPECT_EQ(ops.liveAllocations(), 0u);
    }

    TEST(ProxyPublishedMemTest, ZeroAndOversizeAreRejected) {
        MockDeviceOps ops;
        std::unique_ptr<nixl::hostPublishedDeviceMem> mem;
        EXPECT_EQ(nixl::hostPublishedDeviceMem::create(ops, 0, mem), NIXL_ERR_INVALID_PARAM);
        EXPECT_EQ(mem, nullptr);
        EXPECT_EQ(
            nixl::hostPublishedDeviceMem::create(ops, std::numeric_limits<size_t>::max(), mem),
            NIXL_ERR_INVALID_PARAM);
        EXPECT_EQ(mem, nullptr);
        EXPECT_EQ(ops.liveAllocations(), 0u);
    }

    TEST(ProxyPublishedMemTest, PublishOutsideTheRangeIsRejected) {
        MockDeviceOps ops;
        std::unique_ptr<nixl::hostPublishedDeviceMem> mem;
        ASSERT_EQ(nixl::hostPublishedDeviceMem::create(ops, 2 * sizeof(uint64_t), mem),
                  NIXL_SUCCESS);
        const uint64_t *words = ops.hostAlias(static_cast<const uint64_t *>(mem->devicePointer()));
        ASSERT_NE(words, nullptr);

        EXPECT_EQ(mem->publish(2 * sizeof(uint64_t), 1), NIXL_ERR_INVALID_PARAM);
        EXPECT_EQ(mem->publish(sizeof(uint64_t) + 4, 1), NIXL_ERR_INVALID_PARAM);
        EXPECT_EQ(mem->publish(4, 1), NIXL_ERR_INVALID_PARAM);
        EXPECT_EQ(mem->publish(std::numeric_limits<size_t>::max() - 7, 1), NIXL_ERR_INVALID_PARAM);
        EXPECT_EQ(words[0], 0u);
        EXPECT_EQ(words[1], 0u);

        EXPECT_EQ(mem->publish(sizeof(uint64_t), 1), NIXL_SUCCESS);
        EXPECT_EQ(words[1], 1u);
    }

} // namespace proxy_published_mem
} // namespace gtest
