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
#include <string>
#include <vector>

#include "device/proxy/proxy_registry.h"
#include "mocks/proxy_mocks.h"

namespace gtest {
namespace proxy_memview_registry {

    using proxy_mocks::DummyBackendMD;
    using proxy_mocks::makeLocalDlist;
    using proxy_mocks::makeRemoteDesc;
    using proxy_mocks::MockDeviceAllocator;

    class ProxyMemViewRegistryTest : public testing::Test {
    protected:
        /** Device memory is host memory under the mock: the view is readable in place. */
        static const nixlProxyDeviceMemView &
        view(nixlMemViewH handle) {
            return *static_cast<const nixlProxyDeviceMemView *>(handle);
        }

        static uint32_t
        idOf(nixlMemViewH handle) {
            return handle == nullptr ? 0 : view(handle).proxy_memview_id;
        }

        nixlMemViewH
        prepLocal(uintptr_t addr, size_t len = 64, uint64_t dev_id = 0) {
            nixlMemViewH handle = nullptr;
            EXPECT_EQ(registry_.prepLocal(makeLocalDlist(addr, len, dev_id, &local_md_), handle),
                      NIXL_SUCCESS);
            return handle;
        }

        nixlMemViewH
        prepRemote(const nixl_remote_meta_dlist_t &dlist,
                   const std::vector<void *> &direct_ptrs = {}) {
            nixlMemViewH handle = nullptr;
            EXPECT_EQ(registry_.prepRemote(dlist, direct_ptrs, handle), NIXL_SUCCESS);
            return handle;
        }

        nixlMemViewH
        prepRemote(const std::string &agent = "peer",
                   uintptr_t addr = 0x2000,
                   size_t len = 64,
                   uint64_t dev_id = 0) {
            nixl_remote_meta_dlist_t dlist(VRAM_SEG);
            dlist.addDesc(makeRemoteDesc(agent, addr, len, dev_id, &remote_md_));
            return prepRemote(dlist);
        }

        static nixlProxySubmission
        put(uint32_t src,
            uint32_t dst,
            uint64_t size,
            uint64_t src_offset = 0,
            uint64_t dst_offset = 0) {
            nixlProxySubmission record{};
            record.opcode = nixl_proxy_opcode_t::PUT;
            record.src_proxy_memview_id = src;
            record.src_offset = src_offset;
            record.dst_proxy_memview_id = dst;
            record.dst_offset = dst_offset;
            record.size = size;
            return record;
        }

        static nixlProxySubmission
        atomicAdd(uint32_t dst, uint64_t dst_offset = 0, uint64_t value = 42) {
            nixlProxySubmission record{};
            record.opcode = nixl_proxy_opcode_t::ATOMIC_ADD;
            record.dst_proxy_memview_id = dst;
            record.dst_offset = dst_offset;
            record.size = sizeof(uint64_t);
            record.value = value;
            return record;
        }

        nixl_status_t
        prepare(const nixlProxySubmission &record, nixlBackendProxySubmission &prepared) {
            return registry_.prepareSubmission(record, prepared);
        }

        MockDeviceAllocator allocator_;
        nixlProxyDeviceContextData context_{};
        nixlProxyMemViewRegistry registry_{allocator_, &context_};
        DummyBackendMD local_md_;
        DummyBackendMD remote_md_;
    };

    // The data path: every shape of record a worker can dequeue, resolved
    // against the descriptors it names or rejected.
    TEST_F(ProxyMemViewRegistryTest, PrepareSubmissionResolvesAndValidates) {
        const uint32_t src = idOf(prepLocal(0x1000, 64, /*dev_id=*/7));
        const uint32_t dst = idOf(prepRemote("remote-agent", 0x2000, 64, /*dev_id=*/11));
        const uint32_t unnamed = idOf(prepRemote(""));
        const uint32_t null_agent = idOf(prepRemote(nixl_null_agent));
        const uint32_t empty = idOf(prepRemote(nixl_remote_meta_dlist_t(VRAM_SEG)));
        constexpr uint64_t kLarge = (uint64_t{1} << 32) + 64;
        const uint32_t big_src = idOf(prepLocal(0x5000, kLarge + 64));
        const uint32_t big_dst = idOf(prepRemote("peer", 0x6000, kLarge + 64));

        // A put resolves both sides through the record's offsets and carries
        // the record's own fields along.
        nixlBackendProxySubmission prepared;
        nixlProxySubmission record = put(src, dst, 16, 5, 9);
        record.op_idx = 7;
        record.channel_id = 3;
        ASSERT_EQ(prepare(record, prepared), NIXL_SUCCESS);
        EXPECT_EQ(prepared.op_idx, 7u);
        EXPECT_EQ(prepared.channel_id, 3u);
        EXPECT_EQ(prepared.opcode, nixl_proxy_opcode_t::PUT);
        EXPECT_EQ(prepared.size, 16u);
        EXPECT_EQ(prepared.local.mem_type, DRAM_SEG);
        EXPECT_EQ(prepared.local.desc.addr, 0x1005u);
        EXPECT_EQ(prepared.local.desc.len, 16u);
        EXPECT_EQ(prepared.local.desc.devId, 7u);
        EXPECT_EQ(prepared.local.desc.metadataP, &local_md_);
        EXPECT_EQ(prepared.remote.mem_type, VRAM_SEG);
        EXPECT_EQ(prepared.remote.desc.addr, 0x2009u);
        EXPECT_EQ(prepared.remote.desc.len, 16u);
        EXPECT_EQ(prepared.remote.desc.devId, 11u);
        EXPECT_EQ(prepared.remote.desc.metadataP, &remote_md_);
        EXPECT_EQ(prepared.remote_agent, "remote-agent");

        // An atomic add is sized by the counter, whatever the record says, and
        // carries its value.
        record = atomicAdd(dst, 9, 42);
        record.size = 3;
        ASSERT_EQ(prepare(record, prepared), NIXL_SUCCESS);
        EXPECT_EQ(prepared.opcode, nixl_proxy_opcode_t::ATOMIC_ADD);
        EXPECT_EQ(prepared.size, sizeof(uint64_t));
        EXPECT_EQ(prepared.remote.desc.addr, 0x2009u);
        EXPECT_EQ(prepared.remote.desc.len, sizeof(uint64_t));
        EXPECT_EQ(prepared.value, 42u);

        // Offsets and sizes are 64-bit end to end.
        ASSERT_EQ(prepare(put(big_src, big_dst, 32, kLarge, kLarge), prepared), NIXL_SUCCESS);
        EXPECT_EQ(prepared.local.desc.addr, uintptr_t{0x5000} + kLarge);
        EXPECT_EQ(prepared.remote.desc.addr, uintptr_t{0x6000} + kLarge);
        ASSERT_EQ(prepare(put(big_src, big_dst, kLarge), prepared), NIXL_SUCCESS);
        EXPECT_EQ(prepared.size, kLarge);
        EXPECT_EQ(prepared.local.desc.len, kLarge);

        // A range may end exactly at the descriptor boundary.
        ASSERT_EQ(prepare(put(src, dst, 16, 48, 48), prepared), NIXL_SUCCESS);
        EXPECT_EQ(prepared.local.desc.addr, 0x1030u);
        EXPECT_EQ(prepared.remote.desc.addr, 0x2030u);

        // Everything else is rejected, and a rejected record leaves the output
        // untouched.
        nixlProxySubmission unsupported = atomicAdd(dst);
        unsupported.opcode = static_cast<nixl_proxy_opcode_t>(99);

        struct Row {
            const char *name;
            nixlProxySubmission record;
            nixl_status_t expected;
        };

        const std::vector<Row> rows = {
            {"source past the end", put(src, dst, 8, 60, 0), NIXL_ERR_INVALID_PARAM},
            {"destination past the end", put(src, dst, 8, 0, 60), NIXL_ERR_INVALID_PARAM},
            {"offset far past the end",
             put(src, dst, 1, 0, std::numeric_limits<uint32_t>::max()),
             NIXL_ERR_INVALID_PARAM},
            {"counter past the end", atomicAdd(dst, 60), NIXL_ERR_INVALID_PARAM},
            {"roles swapped", put(dst, src, 16), NIXL_ERR_INVALID_PARAM},
            {"empty remote agent", atomicAdd(unnamed), NIXL_ERR_INVALID_PARAM},
            {"null remote agent", atomicAdd(null_agent), NIXL_ERR_INVALID_PARAM},
            {"empty descriptor list", atomicAdd(empty), NIXL_ERR_INVALID_PARAM},
            {"unknown ids", put(97, 99, 16), NIXL_ERR_NOT_FOUND},
            {"unsupported opcode", unsupported, NIXL_ERR_NOT_SUPPORTED},
        };
        for (const auto &row : rows) {
            prepared.op_idx = 123;
            EXPECT_EQ(prepare(row.record, prepared), row.expected) << row.name;
            EXPECT_EQ(prepared.op_idx, 123u) << row.name;
        }
    }

    // The control path: handles, ids, and retirement.
    TEST_F(ProxyMemViewRegistryTest, HandlesIdsAndRetirement) {
        // Ids are assigned in order, and every handle is its own device view
        // stamped with the runtime's context.
        nixlMemViewH first = prepLocal(0x1000);
        nixlMemViewH second = prepRemote("peer", 0x2000);
        nixlMemViewH third = prepLocal(0x3000);
        EXPECT_NE(first, second);
        EXPECT_NE(second, third);
        EXPECT_NE(first, third);
        EXPECT_EQ(idOf(first), 1u);
        EXPECT_EQ(idOf(second), 2u);
        EXPECT_EQ(idOf(third), 3u);
        EXPECT_EQ(view(first).direct_ptr_count, 0u);
        EXPECT_EQ(view(first).context, &context_);

        // A rejected prep consumes no id and leaves no allocation behind.
        const size_t live = allocator_.liveAllocations();
        nixl_remote_meta_dlist_t dram(DRAM_SEG);
        dram.addDesc(makeRemoteDesc("peer", 0x2000, 64, 0, &remote_md_));
        nixlMemViewH rejected = nullptr;
        EXPECT_EQ(registry_.prepRemote(dram, {}, rejected), NIXL_ERR_INVALID_PARAM);
        EXPECT_EQ(rejected, nullptr);
        EXPECT_EQ(allocator_.liveAllocations(), live);

        // Direct pointers are copied into the trailing run of the device view.
        nixl_remote_meta_dlist_t two_peers(VRAM_SEG);
        two_peers.addDesc(makeRemoteDesc("peer0", 0x4000, 64, 0, &remote_md_));
        two_peers.addDesc(makeRemoteDesc("peer1", 0x5000, 64, 1, &remote_md_));
        const std::vector<void *> direct_ptrs{reinterpret_cast<void *>(uintptr_t{0xfeed0000}),
                                              nullptr};
        nixlMemViewH fourth = prepRemote(two_peers, direct_ptrs);
        EXPECT_EQ(idOf(fourth), 4u);
        ASSERT_EQ(view(fourth).direct_ptr_count, 2u);
        void *const *stored = nixlProxyDeviceMemViewDirectPtrs(&view(fourth));
        EXPECT_EQ(std::vector<void *>(stored, stored + 2), direct_ptrs);

        // resolve() knows live handles and nothing else.
        nixlMemViewH backend = nullptr;
        EXPECT_TRUE(registry_.resolve(first, backend));
        EXPECT_FALSE(registry_.resolve(reinterpret_cast<nixlMemViewH>(uintptr_t{0x99}), backend));

        // Retiring frees the device view, stops dispatch to that id, leaves the
        // other entries usable, and never hands the id out again.
        ASSERT_EQ(registry_.unregister(second), NIXL_SUCCESS);
        EXPECT_TRUE(allocator_.wasFreed(second));
        EXPECT_EQ(registry_.unregister(second), NIXL_ERR_INVALID_PARAM);
        EXPECT_FALSE(registry_.resolve(second, backend));
        nixlBackendProxySubmission prepared;
        EXPECT_EQ(prepare(put(1, 2, 8), prepared), NIXL_ERR_NOT_FOUND);
        ASSERT_EQ(prepare(put(1, 4, 8), prepared), NIXL_SUCCESS);
        EXPECT_EQ(prepared.remote.desc.addr, 0x4000u);
        EXPECT_EQ(idOf(prepRemote()), 5u);
    }

} // namespace proxy_memview_registry
} // namespace gtest
