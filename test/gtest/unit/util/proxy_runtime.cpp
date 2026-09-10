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

// The proxy runtime driven end to end on the host. The mock allocator hands
// out host memory, the mock backend stands in for the transport, and the test
// plays the GPU: it writes ring records with the publication order
// proxy_protocol.h specifies and polls the completion slot. No CUDA involved.

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "device/proxy/proxy_config.h"
#include "device/proxy/proxy_protocol.h"
#include "device/proxy/proxy_runtime.h"
#include "mocks/proxy_mocks.h"

namespace gtest {
namespace proxy_runtime {

    using proxy_mocks::DummyBackendMD;
    using proxy_mocks::makeLocalDlist;
    using proxy_mocks::makeRemoteDesc;
    using proxy_mocks::MockBackend;
    using proxy_mocks::MockDeviceAllocator;

    constexpr uint32_t kRingDepth = 4;

    class ProxyRuntimeTest : public testing::Test {
    protected:
        void
        TearDown() override {
            backend_.completeEverything();
            runtime_.reset();
            // Every device and mapped-host allocation the runtime made is gone.
            EXPECT_EQ(allocator_.liveAllocations(), 0u);
        }

        nixlProxyConfig
        makeConfig(uint32_t channel_count, uint32_t max_peers, uint32_t thread_count) const {
            nixlProxyConfig config;
            config.enabled = true;
            config.channel_count = channel_count;
            config.thread_count = thread_count == 0 ? channel_count : thread_count;
            config.max_peers = max_peers;
            config.ring_depth = kRingDepth;
            return config;
        }

        nixl_status_t
        createRuntime(uint32_t channel_count = 1,
                      uint32_t max_peers = 1,
                      uint32_t thread_count = 0) {
            max_peers_ = max_peers;
            return nixlProxyRuntime::create(backend_.ops(),
                                            makeConfig(channel_count, max_peers, thread_count),
                                            runtime_,
                                            allocator_);
        }

        /**
         * The GPU's view of one ring, translated to host pointers. The ring
         * descriptor and its producer words are device memory, which the mock
         * keeps directly readable. The records, the completion slot, and the
         * consumer index are mapped host memory that the runtime publishes to
         * the GPU through their device alias, which only the mock can
         * translate back - so a host pointer published in their place shows
         * up here as a failed translation.
         */
        struct ChannelAccess {
            const nixlProxyWorkRing *ring = nullptr;
            nixlProxySubmission *records = nullptr;
            const nixlProxyCompletionSlot *completion = nullptr;
            const uint64_t *consumer_idx = nullptr;
        };

        ChannelAccess
        channel(uint32_t channel_id = 0, uint32_t peer = 0) const {
            const size_t slot = static_cast<size_t>(channel_id) * max_peers_ + peer;
            const nixlProxyChannelView &view = runtime_->deviceChannelViews()[slot];
            ChannelAccess access;
            access.ring = view.work_ring;
            access.records = allocator_.hostAlias(access.ring->records);
            access.completion = allocator_.hostAlias(view.completion_slot);
            access.consumer_idx = allocator_.hostAlias(access.ring->consumer_idx);
            EXPECT_NE(access.records, nullptr);
            EXPECT_NE(access.completion, nullptr);
            EXPECT_NE(access.consumer_idx, nullptr);
            return access;
        }

        /**
         * Enqueue a record the way the GPU must: fill the body first, then
         * release-store op_idx, which lives at offset 0 and publishes readiness.
         */
        static void
        publish(const ChannelAccess &access,
                uint64_t producer_idx,
                const nixlProxySubmission &record,
                uint64_t op_idx) {
            *access.ring->producer_idx = producer_idx + 1;
            const uint32_t slot = static_cast<uint32_t>(producer_idx % kRingDepth);
            std::memcpy(reinterpret_cast<char *>(&access.records[slot]) + sizeof(uint64_t),
                        reinterpret_cast<const char *>(&record) + sizeof(uint64_t),
                        sizeof(record) - sizeof(uint64_t));
            __atomic_store_n(&access.records[slot].op_idx, op_idx, __ATOMIC_RELEASE);
        }

        static uint64_t
        consumerIdx(const ChannelAccess &access) {
            return __atomic_load_n(access.consumer_idx, __ATOMIC_ACQUIRE);
        }

        static uint64_t
        completedIdx(const ChannelAccess &access) {
            return __atomic_load_n(&access.completion->completed_idx, __ATOMIC_ACQUIRE);
        }

        template<typename Predicate>
        static bool
        waitFor(Predicate predicate) {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
            while (std::chrono::steady_clock::now() < deadline) {
                if (predicate()) {
                    return true;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            return predicate();
        }

        /** A local source view and a remote view with one 64-byte descriptor per agent. */
        void
        prepMemViews(nixlMemViewH &src,
                     nixlMemViewH &dst,
                     const std::vector<std::string> &agents = {"peer"}) {
            ASSERT_EQ(runtime_->prepMemView(makeLocalDlist(0x1000, 64, 0, &local_md_), &src),
                      NIXL_SUCCESS);
            nixl_remote_meta_dlist_t remote(VRAM_SEG);
            for (const auto &agent : agents) {
                remote.addDesc(makeRemoteDesc(agent, 0x2000, 64, 0, &remote_md_));
            }
            ASSERT_EQ(runtime_->prepMemView(remote, &dst), NIXL_SUCCESS);
        }

        static uint64_t
        memViewId(nixlMemViewH view) {
            return static_cast<const nixlProxyDeviceMemView *>(view)->host_view;
        }

        static nixlProxySubmission
        makePut(nixlMemViewH src,
                nixlMemViewH dst,
                uint32_t channel_id = 0,
                uint32_t dst_index = 0) {
            nixlProxySubmission record{};
            record.opcode = nixl_proxy_opcode_t::PUT;
            record.channel_id = static_cast<uint16_t>(channel_id);
            record.src_view = memViewId(src);
            record.dst_view = memViewId(dst);
            record.dst_index = dst_index;
            record.operand = 4;
            record.dst_offset = 8;
            record.size = 32;
            return record;
        }

        static nixlProxySubmission
        makeAtomicAdd(nixlMemViewH dst, uint64_t value = 42) {
            nixlProxySubmission record{};
            record.opcode = nixl_proxy_opcode_t::ATOMIC_ADD;
            record.dst_view = memViewId(dst);
            record.dst_offset = 8;
            record.size = sizeof(uint64_t);
            record.operand = value;
            return record;
        }

        MockDeviceAllocator allocator_;
        MockBackend backend_;
        DummyBackendMD local_md_;
        DummyBackendMD remote_md_;
        uint32_t max_peers_ = 1;
        // Declared last: the runtime's callbacks and memory outlive nothing here,
        // so it must die before the mocks it points at.
        std::unique_ptr<nixlProxyRuntime> runtime_;
    };

    TEST_F(ProxyRuntimeTest, CreateRejectsBadInputsWithoutLeaking) {
        struct Row {
            const char *name;
            std::function<void(nixlProxyConfig &, nixlProxyBackendOps &)> mutate;
            nixl_status_t expected;
        };

        const std::vector<Row> rows = {
            {"incomplete callbacks",
             [](nixlProxyConfig &, nixlProxyBackendOps &ops) { ops.submit = nullptr; },
             NIXL_ERR_INVALID_PARAM},
            {"missing quiesce",
             [](nixlProxyConfig &, nixlProxyBackendOps &ops) { ops.quiesce = nullptr; },
             NIXL_ERR_INVALID_PARAM},
            {"zero peers",
             [](nixlProxyConfig &config, nixlProxyBackendOps &) { config.max_peers = 0; },
             NIXL_ERR_INVALID_PARAM},
            {"zero channels",
             [](nixlProxyConfig &config, nixlProxyBackendOps &) { config.channel_count = 0; },
             NIXL_ERR_INVALID_PARAM},
            {"zero threads",
             [](nixlProxyConfig &config, nixlProxyBackendOps &) { config.thread_count = 0; },
             NIXL_ERR_INVALID_PARAM},
            {"zero ring depth",
             [](nixlProxyConfig &config, nixlProxyBackendOps &) { config.ring_depth = 0; },
             NIXL_ERR_INVALID_PARAM},
            {"backend init failure",
             [this](nixlProxyConfig &, nixlProxyBackendOps &) {
                 backend_.failInit(NIXL_ERR_BACKEND);
             },
             NIXL_ERR_BACKEND},
        };
        for (const auto &row : rows) {
            nixlProxyConfig config = makeConfig(1, 1, 1);
            nixlProxyBackendOps ops = backend_.ops();
            row.mutate(config, ops);
            EXPECT_EQ(nixlProxyRuntime::create(std::move(ops), config, runtime_, allocator_),
                      row.expected)
                << row.name;
            EXPECT_EQ(runtime_, nullptr) << row.name;
            EXPECT_EQ(allocator_.liveAllocations(), 0u) << row.name;
        }
        backend_.failInit(NIXL_SUCCESS);

        // Accepted: the backend is told how many threads will actually run,
        // and a null output is refused before anything is registered.
        ASSERT_EQ(createRuntime(/*channel_count=*/3, /*max_peers=*/2, /*thread_count=*/8),
                  NIXL_SUCCESS);
        EXPECT_EQ(backend_.initThreadCount(), 3u);
        EXPECT_EQ(runtime_->prepMemView(makeLocalDlist(0x1000, 64, 0, &local_md_), nullptr),
                  NIXL_ERR_INVALID_PARAM);
    }

    TEST_F(ProxyRuntimeTest, PartialAllocationRollsBack) {
        for (int allocation = 0; allocation < 20; ++allocation) {
            allocator_.fail_after = allocation;
            const auto status = createRuntime(2, 2);
            if (status == NIXL_SUCCESS) {
                runtime_.reset();
            } else {
                EXPECT_EQ(runtime_, nullptr);
            }
            EXPECT_EQ(allocator_.liveAllocations(), 0u) << allocation;
        }
        allocator_.fail_after = -1;
    }

    TEST_F(ProxyRuntimeTest, QuiescenceFailureNeverFreesViews) {
        EXPECT_DEATH(
            {
                auto ops = backend_.ops();
                ops.quiesce = [](uint32_t, uint32_t) { return NIXL_ERR_BACKEND; };
                ASSERT_EQ(nixlProxyRuntime::create(ops, makeConfig(1, 1, 1), runtime_, allocator_),
                          NIXL_SUCCESS);
                ASSERT_EQ(runtime_->startWorkers(), NIXL_SUCCESS);
                static_cast<void>(runtime_->shutdown());
            },
            "Failed to quiesce proxy backend");
    }

    TEST_F(ProxyRuntimeTest, DeviceContextAndRingsStartInitialized) {
        ASSERT_EQ(createRuntime(/*channel_count=*/3, /*max_peers=*/2), NIXL_SUCCESS);

        const nixlProxyDeviceContextData *context = runtime_->deviceContext();
        ASSERT_NE(context, nullptr);
        EXPECT_EQ(context->max_peers, 2u);
        EXPECT_EQ(context->num_channels, 3u);
        EXPECT_EQ(context->protocol_version, kProxyProtocolVersion);
        ASSERT_NE(context->channels, nullptr);
        ASSERT_NE(context->shutdown_word, nullptr);

        // Every ring starts empty, with its own records, and the device copy of
        // the view table matches what the host handed out.
        std::set<const void *> record_arrays;
        for (uint32_t channel_id = 0; channel_id < 3; ++channel_id) {
            for (uint32_t peer = 0; peer < 2; ++peer) {
                const size_t slot = channel_id * 2 + peer;
                const ChannelAccess access = channel(channel_id, peer);
                EXPECT_EQ(context->channels[slot].work_ring, access.ring);
                EXPECT_EQ(context->channels[slot].completion_slot,
                          runtime_->deviceChannelViews()[slot].completion_slot);
                EXPECT_EQ(access.ring->depth, kRingDepth);
                EXPECT_EQ(*access.ring->producer_idx, 0u);
                EXPECT_EQ(*access.ring->consumer_idx_cache, 0u);
                EXPECT_EQ(consumerIdx(access), 0u);
                EXPECT_EQ(completedIdx(access), 0u);
                EXPECT_EQ(access.completion->completion_status, NIXL_IN_PROG);
                record_arrays.insert(access.records);
            }
        }
        EXPECT_EQ(record_arrays.size(), 6u);

        // Starting publishes RUNNING to the word the kernels poll.
        ASSERT_EQ(runtime_->startWorkers(), NIXL_SUCCESS);
        const uint64_t *shutdown_word = allocator_.hostAlias(context->shutdown_word);
        ASSERT_NE(shutdown_word, nullptr);
        EXPECT_EQ(__atomic_load_n(shutdown_word, __ATOMIC_ACQUIRE),
                  static_cast<uint64_t>(nixl_proxy_control_state_t::RUNNING));

        // A prepared view is stamped with the context so device code can find the rings.
        nixlMemViewH src = nullptr;
        ASSERT_EQ(runtime_->prepMemView(makeLocalDlist(0x1000, 64, 0, &local_md_), &src),
                  NIXL_SUCCESS);
        EXPECT_EQ(static_cast<const nixlProxyDeviceMemView *>(src)->context, context);

        EXPECT_EQ(runtime_->shutdown(), NIXL_SUCCESS);
        EXPECT_EQ(runtime_->deviceContext(), nullptr);
    }

    TEST_F(ProxyRuntimeTest, StartAndShutdownLifecycle) {
        ASSERT_EQ(createRuntime(), NIXL_SUCCESS);
        ASSERT_EQ(runtime_->startWorkers(), NIXL_SUCCESS);
        // A second start is refused rather than doubling the threads.
        EXPECT_EQ(runtime_->startWorkers(), NIXL_ERR_INVALID_PARAM);

        EXPECT_EQ(runtime_->shutdown(), NIXL_SUCCESS);
        EXPECT_EQ(runtime_->deviceContext(), nullptr);
        EXPECT_EQ(backend_.shutdownCalls(), 1u);
        // Shutting down again is a no-op that does not reach the backend twice.
        EXPECT_EQ(runtime_->shutdown(), NIXL_SUCCESS);
        EXPECT_EQ(backend_.shutdownCalls(), 1u);

        // Nothing is left behind for the next runtime, which may also be shut
        // down without ever having started.
        runtime_.reset();
        EXPECT_EQ(allocator_.liveAllocations(), 0u);
        ASSERT_EQ(createRuntime(), NIXL_SUCCESS);
        EXPECT_EQ(runtime_->shutdown(), NIXL_SUCCESS);
        EXPECT_EQ(backend_.shutdownCalls(), 2u);
    }

    TEST_F(ProxyRuntimeTest, RecordPathSubmitsAndCompletesInOrder) {
        ASSERT_EQ(createRuntime(), NIXL_SUCCESS);
        nixlMemViewH src = nullptr, dst = nullptr;
        prepMemViews(src, dst, {"remote-agent"});
        ASSERT_EQ(runtime_->startWorkers(), NIXL_SUCCESS);

        const ChannelAccess access = channel();
        publish(access, 0, makePut(src, dst), 7);
        publish(access, 1, makeAtomicAdd(dst, 42), 8);
        ASSERT_TRUE(waitFor([&]() { return backend_.submissionCount() == 2; }));

        // The worker resolved both records through the registry into
        // transport-ready descriptors.
        const auto submissions = backend_.submissions();
        const nixlBackendProxySubmission &put = submissions[0];
        EXPECT_EQ(put.op_idx, 7u);
        EXPECT_EQ(put.opcode, nixl_proxy_opcode_t::PUT);
        EXPECT_EQ(put.channel_id, 0u);
        EXPECT_EQ(put.peer_index, 0u);
        EXPECT_EQ(put.size, 32u);
        EXPECT_EQ(put.local.mem_type, DRAM_SEG);
        EXPECT_EQ(put.local.desc.addr, 0x1004u);
        EXPECT_EQ(put.local.desc.len, 32u);
        EXPECT_EQ(put.local.desc.metadataP, &local_md_);
        EXPECT_EQ(put.remote.mem_type, VRAM_SEG);
        EXPECT_EQ(put.remote.desc.addr, 0x2008u);
        EXPECT_EQ(put.remote.desc.len, 32u);
        EXPECT_EQ(put.remote.desc.metadataP, &remote_md_);
        const nixlBackendProxySubmission &atomic = submissions[1];
        EXPECT_EQ(atomic.op_idx, 8u);
        EXPECT_EQ(atomic.opcode, nixl_proxy_opcode_t::ATOMIC_ADD);
        EXPECT_EQ(atomic.size, sizeof(uint64_t));
        EXPECT_EQ(atomic.value, 42u);
        EXPECT_EQ(atomic.remote.desc.addr, 0x2008u);
        EXPECT_EQ(atomic.remote.desc.len, sizeof(uint64_t));
        EXPECT_EQ(atomic.remote.desc.metadataP, &remote_md_);

        // Nothing is published back until the backend completes, and completing
        // the second request first does not let it overtake the first.
        EXPECT_EQ(completedIdx(access), 0u);
        EXPECT_EQ(consumerIdx(access), 0u);
        backend_.complete(backend_.token(1));
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        EXPECT_EQ(completedIdx(access), 0u);
        EXPECT_EQ(consumerIdx(access), 0u);

        backend_.complete(backend_.token(0));
        ASSERT_TRUE(waitFor([&]() { return consumerIdx(access) == 2u; }));
        EXPECT_EQ(completedIdx(access), 8u);
        EXPECT_EQ(access.completion->completion_status, NIXL_SUCCESS);
    }

    TEST_F(ProxyRuntimeTest, FullRingBackpressuresUntilTheConsumerIndexAdvances) {
        ASSERT_EQ(createRuntime(), NIXL_SUCCESS);
        nixlMemViewH src = nullptr, dst = nullptr;
        prepMemViews(src, dst);
        ASSERT_EQ(runtime_->startWorkers(), NIXL_SUCCESS);

        const ChannelAccess access = channel();
        const nixlProxySubmission record = makePut(src, dst);
        for (uint64_t i = 0; i < kRingDepth; ++i) {
            publish(access, i, record, i + 1);
        }
        ASSERT_TRUE(waitFor([&]() { return backend_.submissionCount() == kRingDepth; }));
        EXPECT_EQ(consumerIdx(access), 0u);

        // The ring is full: a record written into the wrapped slot must wait.
        publish(access, kRingDepth, record, kRingDepth + 1);
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        EXPECT_EQ(backend_.submissionCount(), size_t{kRingDepth});

        for (uint32_t i = 0; i < kRingDepth; ++i) {
            backend_.complete(backend_.token(i));
        }
        ASSERT_TRUE(waitFor([&]() { return consumerIdx(access) == kRingDepth; }));
        EXPECT_EQ(completedIdx(access), uint64_t{kRingDepth});

        // With space freed the wrapped record is picked up.
        ASSERT_TRUE(waitFor([&]() { return backend_.submissionCount() == kRingDepth + 1; }));
        EXPECT_EQ(backend_.submissions().back().op_idx, uint64_t{kRingDepth} + 1);
        backend_.complete(backend_.token(kRingDepth));
        ASSERT_TRUE(waitFor([&]() { return consumerIdx(access) == uint64_t{kRingDepth} + 1; }));
    }

    TEST_F(ProxyRuntimeTest, FirstErrorLatchesWhileLaterWorkIsRetired) {
        ASSERT_EQ(createRuntime(), NIXL_SUCCESS);
        nixlMemViewH src = nullptr, dst = nullptr;
        prepMemViews(src, dst);
        ASSERT_EQ(runtime_->startWorkers(), NIXL_SUCCESS);
        const ChannelAccess access = channel();
        const nixlProxySubmission good = makePut(src, dst);

        // A record the registry cannot resolve fails at preparation: it is
        // retired without reaching the backend, and its error latches the slot.
        nixlProxySubmission unknown = good;
        unknown.dst_view = 0;
        publish(access, 0, unknown, 1);
        ASSERT_TRUE(waitFor([&]() { return consumerIdx(access) == 1u; }));
        EXPECT_EQ(completedIdx(access), 1u);
        const nixl_status_t first_error = access.completion->completion_status;
        EXPECT_LT(first_error, 0);
        EXPECT_EQ(backend_.submissionCount(), 0u);

        // A completion error, a submit error, and a success all retire their
        // records afterwards, but the latched status never moves again.
        publish(access, 1, good, 2);
        ASSERT_TRUE(waitFor([&]() { return backend_.submissionCount() == 1; }));
        backend_.complete(backend_.token(0), NIXL_ERR_BACKEND);
        ASSERT_TRUE(waitFor([&]() { return consumerIdx(access) == 2u; }));

        backend_.failNextSubmit(NIXL_ERR_BACKEND);
        publish(access, 2, good, 3);
        ASSERT_TRUE(waitFor([&]() { return consumerIdx(access) == 3u; }));

        publish(access, 3, good, 4);
        ASSERT_TRUE(waitFor([&]() { return backend_.submissionCount() == 3; }));
        backend_.complete(backend_.token(2));
        ASSERT_TRUE(waitFor([&]() { return consumerIdx(access) == 4u; }));

        EXPECT_EQ(completedIdx(access), 1u);
        EXPECT_EQ(access.completion->completion_status, first_error);
    }

    TEST_F(ProxyRuntimeTest, ReplacementViewWorksAfterDrainAndRelease) {
        ASSERT_EQ(createRuntime(), NIXL_SUCCESS);
        nixlMemViewH src = nullptr, dst = nullptr;
        prepMemViews(src, dst);
        ASSERT_EQ(runtime_->startWorkers(), NIXL_SUCCESS);
        const ChannelAccess access = channel();
        const nixlProxySubmission record = makePut(src, dst);
        publish(access, 0, record, 1);
        ASSERT_TRUE(waitFor([&]() { return backend_.submissionCount() == 1; }));
        backend_.complete(backend_.token(0));
        ASSERT_TRUE(waitFor([&]() { return consumerIdx(access) == 1u; }));

        // Retiring drains the rings first, then frees the device view at once -
        // only the GPU reads it, and the caller has quiesced the GPU, the same
        // contract as the direct path's ucp_device_mem_list_release - and
        // forgets the handle.
        ASSERT_EQ(runtime_->unregisterProxyMemView(dst), NIXL_SUCCESS);
        EXPECT_TRUE(allocator_.wasFreed(dst));
        EXPECT_EQ(runtime_->unregisterProxyMemView(dst), NIXL_ERR_INVALID_PARAM);

        // The drain rearmed the ring, so the next generation starts over.
        EXPECT_EQ(consumerIdx(access), 0u);
        EXPECT_EQ(completedIdx(access), 0u);
        EXPECT_EQ(access.completion->completion_status, NIXL_IN_PROG);

        nixl_remote_meta_dlist_t remote(VRAM_SEG);
        remote.addDesc(makeRemoteDesc("peer", 0x9000, 64, 0, &remote_md_));
        nixlMemViewH replacement = nullptr;
        ASSERT_EQ(runtime_->prepMemView(remote, &replacement), NIXL_SUCCESS);
        publish(access, 0, makePut(src, replacement), 2);
        ASSERT_TRUE(waitFor([&]() { return backend_.submissionCount() == 2; }));
        backend_.complete(backend_.token(1));
        ASSERT_TRUE(waitFor([&]() { return consumerIdx(access) == 1u; }));
        EXPECT_EQ(backend_.submissionCount(), 2u);
        EXPECT_EQ(completedIdx(access), 2u);
        EXPECT_EQ(access.completion->completion_status, NIXL_SUCCESS);
    }

    TEST_F(ProxyRuntimeTest, RemoteDirectPointersFollowTheResolver) {
        ASSERT_EQ(createRuntime(/*channel_count=*/1, /*max_peers=*/2), NIXL_SUCCESS);
        nixl_remote_meta_dlist_t remote(VRAM_SEG);
        remote.addDesc(makeRemoteDesc("peer0", 0x2000, 64, 0, &remote_md_));
        remote.addDesc(makeRemoteDesc("peer1", 0x3000, 64, 1, &remote_md_));

        // Whatever the backend resolves lands in the device view, one slot per descriptor.
        const std::vector<void *> direct_ptrs{reinterpret_cast<void *>(uintptr_t{0xabc00000}),
                                              nullptr};
        backend_.setDirectPointers(direct_ptrs);
        nixlMemViewH dst = nullptr;
        ASSERT_EQ(runtime_->prepMemView(remote, &dst), NIXL_SUCCESS);
        EXPECT_EQ(backend_.resolveCalls(), 1u);
        EXPECT_EQ(backend_.lastResolvedDescCount(), 2u);
        const auto *view = static_cast<const nixlProxyDeviceMemView *>(dst);
        ASSERT_EQ(view->direct_ptr_count, 2u);
        void *const *stored = nixlProxyDeviceMemViewDirectPtrs(view);
        EXPECT_EQ(std::vector<void *>(stored, stored + 2), direct_ptrs);

        // A resolver error fails the prep before anything is registered.
        backend_.setResolverStatus(NIXL_ERR_INVALID_PARAM);
        const size_t live = allocator_.liveAllocations();
        nixlMemViewH failed = nullptr;
        EXPECT_EQ(runtime_->prepMemView(remote, &failed), NIXL_ERR_INVALID_PARAM);
        EXPECT_EQ(failed, nullptr);
        EXPECT_EQ(allocator_.liveAllocations(), live);

        // A backend without a resolver is not asked; the view simply carries none.
        runtime_.reset();
        backend_.setResolverStatus(NIXL_SUCCESS);
        backend_.disableResolver();
        ASSERT_EQ(createRuntime(/*channel_count=*/1, /*max_peers=*/2), NIXL_SUCCESS);
        ASSERT_EQ(runtime_->prepMemView(remote, &dst), NIXL_SUCCESS);
        EXPECT_EQ(backend_.resolveCalls(), 2u);
        EXPECT_EQ(static_cast<const nixlProxyDeviceMemView *>(dst)->direct_ptr_count, 0u);
    }

    TEST_F(ProxyRuntimeTest, ChannelsAndPeersKeepToTheirOwnRings) {
        // Three channels over two workers: worker 0 owns channels 0 and 2,
        // worker 1 owns channel 1, and each owns every peer of its channels.
        ASSERT_EQ(createRuntime(/*channel_count=*/3, /*max_peers=*/2, /*thread_count=*/2),
                  NIXL_SUCCESS);
        nixlMemViewH src = nullptr, dst = nullptr;
        ASSERT_EQ(runtime_->prepMemView(makeLocalDlist(0x1000, 64, 0, &local_md_), &src),
                  NIXL_SUCCESS);
        nixl_remote_meta_dlist_t remote(VRAM_SEG);
        remote.addDesc(makeRemoteDesc("peer0", 0x2000, 64, 0, &remote_md_));
        remote.addDesc(makeRemoteDesc("peer1", 0x3000, 64, 0, &remote_md_));
        ASSERT_EQ(runtime_->prepMemView(remote, &dst), NIXL_SUCCESS);
        ASSERT_EQ(runtime_->startWorkers(), NIXL_SUCCESS);

        const ChannelAccess striped = channel(2, 1);
        const ChannelAccess middle = channel(1, 0);
        const ChannelAccess idle = channel(0, 0);
        publish(striped, 0, makePut(src, dst, /*channel_id=*/2, /*dst_index=*/1), 5);
        publish(middle, 0, makePut(src, dst, /*channel_id=*/1, /*dst_index=*/0), 6);
        ASSERT_TRUE(waitFor([&]() { return backend_.submissionCount() == 2; }));

        // Each submission names the ring it came from and the agent that ring serves.
        for (const auto &submission : backend_.submissions()) {
            if (submission.op_idx == 5) {
                EXPECT_EQ(submission.channel_id, 2u);
                EXPECT_EQ(submission.peer_index, 1u);
                EXPECT_EQ(submission.remote.desc.addr, 0x3008u);
            } else {
                EXPECT_EQ(submission.op_idx, 6u);
                EXPECT_EQ(submission.channel_id, 1u);
                EXPECT_EQ(submission.peer_index, 0u);
                EXPECT_EQ(submission.remote.desc.addr, 0x2008u);
            }
        }

        // Completions land in their own ring and nowhere else.
        backend_.complete(backend_.token(0));
        backend_.complete(backend_.token(1));
        ASSERT_TRUE(waitFor([&]() { return consumerIdx(striped) == 1u && consumerIdx(middle) == 1u; }));
        EXPECT_EQ(completedIdx(striped), 5u);
        EXPECT_EQ(completedIdx(middle), 6u);
        EXPECT_EQ(consumerIdx(striped), 1u);
        EXPECT_EQ(consumerIdx(middle), 1u);
        EXPECT_EQ(consumerIdx(idle), 0u);
        EXPECT_EQ(completedIdx(idle), 0u);
    }

    // Queued records must finish before their view is retired.
    // repo docs/issues/005 G1: a record already in the ring when its memview is
    // retired used to fail prepareSubmission and vanish, taking transfers to
    // healthy peers with it. Retiring now drains first.
    TEST_F(ProxyRuntimeTest, DrainSubmitsQueuedRecordsBeforeRetire) {
        backend_.completeEverything();
        ASSERT_EQ(createRuntime(), NIXL_SUCCESS);
        nixlMemViewH src = nullptr, dst = nullptr;
        prepMemViews(src, dst);
        ASSERT_EQ(runtime_->startWorkers(), NIXL_SUCCESS);

        const ChannelAccess access = channel();
        const nixlProxySubmission record = makePut(src, dst);
        for (uint64_t i = 0; i < kRingDepth; ++i) {
            publish(access, i, record, i + 1);
        }

        // No waiting: retire straight away, racing the worker.
        ASSERT_EQ(runtime_->unregisterProxyMemView(dst), NIXL_SUCCESS);

        EXPECT_EQ(backend_.submissionCount(), size_t{kRingDepth});
        for (const auto &submission : backend_.submissions()) {
            EXPECT_EQ(submission.remote.desc.addr, 0x2008u);
        }
        // Drained to empty and rearmed, so the next generation starts clean.
        EXPECT_EQ(consumerIdx(access), 0u);
        EXPECT_EQ(access.completion->completion_status, NIXL_IN_PROG);
    }

    // repo docs/issues/005 G2: re-adding a rank used to inherit whatever state
    // the wedged ring was left in.
    TEST_F(ProxyRuntimeTest, RingsAreUsableAfterDrain) {
        ASSERT_EQ(createRuntime(), NIXL_SUCCESS);
        nixlMemViewH src = nullptr, dst = nullptr;
        prepMemViews(src, dst);
        ASSERT_EQ(runtime_->startWorkers(), NIXL_SUCCESS);

        const ChannelAccess access = channel();
        publish(access, 0, makePut(src, dst), 1);
        ASSERT_TRUE(waitFor([&]() { return backend_.submissionCount() == 1; }));

        backend_.complete(backend_.token(0), NIXL_ERR_BACKEND);
        // A terminal transfer error still permits safe retirement.
        ASSERT_EQ(runtime_->unregisterProxyMemView(dst), NIXL_SUCCESS);
        ASSERT_EQ(runtime_->unregisterProxyMemView(src), NIXL_SUCCESS);

        // Same ring, new generation of memviews - it has to work.
        nixlMemViewH new_src = nullptr, new_dst = nullptr;
        prepMemViews(new_src, new_dst);
        backend_.completeEverything();
        publish(access, 0, makePut(new_src, new_dst), 7);

        ASSERT_TRUE(waitFor([&]() { return completedIdx(access) == 7u; }));
        EXPECT_EQ(access.completion->completion_status, NIXL_SUCCESS);
        EXPECT_EQ(backend_.submissionCount(), 2u);
    }

    TEST_F(ProxyRuntimeTest, DrainOnEmptyRingsCancelsNothing) {
        ASSERT_EQ(createRuntime(), NIXL_SUCCESS);
        nixlMemViewH src = nullptr, dst = nullptr;
        prepMemViews(src, dst);
        ASSERT_EQ(runtime_->startWorkers(), NIXL_SUCCESS);

        ASSERT_EQ(runtime_->unregisterProxyMemView(dst), NIXL_SUCCESS);
        ASSERT_EQ(runtime_->unregisterProxyMemView(src), NIXL_SUCCESS);

        EXPECT_EQ(backend_.submissionCount(), 0u);
    }

    TEST_F(ProxyRuntimeTest, ShutdownAfterCompletionReleasesNothingAndFreesEverything) {
        ASSERT_EQ(createRuntime(), NIXL_SUCCESS);
        nixlMemViewH src = nullptr, dst = nullptr;
        prepMemViews(src, dst);
        ASSERT_EQ(runtime_->startWorkers(), NIXL_SUCCESS);
        const ChannelAccess access = channel();
        const nixlProxySubmission record = makePut(src, dst);
        publish(access, 0, record, 1);
        publish(access, 1, record, 2);
        ASSERT_TRUE(waitFor([&]() { return backend_.submissionCount() == 2; }));
        backend_.complete(backend_.token(0));
        backend_.complete(backend_.token(1));
        ASSERT_TRUE(waitFor([&]() { return consumerIdx(access) == 2u; }));

        // Every request reached a terminal status through check_completion, so
        // shutdown has nothing to hand back; the views left registered go with
        // the registry instead of leaking into the next generation.
        EXPECT_FALSE(allocator_.wasFreed(src));
        EXPECT_FALSE(allocator_.wasFreed(dst));
        EXPECT_EQ(runtime_->shutdown(), NIXL_SUCCESS);
        EXPECT_EQ(backend_.shutdownCalls(), 1u);
        EXPECT_TRUE(allocator_.wasFreed(src));
        EXPECT_TRUE(allocator_.wasFreed(dst));
    }

    TEST_F(ProxyRuntimeTest, ReleaseWaitsForTerminalErrorsBeyondOldDeadline) {
        ASSERT_EQ(createRuntime(1, 2), NIXL_SUCCESS);
        nixlMemViewH src = nullptr, dst = nullptr;
        prepMemViews(src, dst, {"slow", "healthy"});
        ASSERT_EQ(runtime_->startWorkers(), NIXL_SUCCESS);
        publish(channel(0, 0), 0, makePut(src, dst, 0, 0), 1);
        publish(channel(0, 1), 0, makePut(src, dst, 0, 1), 1);
        ASSERT_TRUE(waitFor([&] { return backend_.submissionCount() == 2; }));
        backend_.completePeer(1);
        std::atomic<bool> done{false};
        std::thread release([&] {
            EXPECT_EQ(runtime_->unregisterProxyMemView(dst), NIXL_SUCCESS);
            done.store(true, std::memory_order_release);
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
        EXPECT_FALSE(done.load(std::memory_order_acquire));
        EXPECT_FALSE(allocator_.wasFreed(dst));
        EXPECT_EQ(backend_.quiesceCalls(), 0u);
        EXPECT_TRUE(waitFor([&] { return completedIdx(channel(0, 1)) == 1; }));
        const auto submissions = backend_.submissions();
        for (size_t i = 0; i < submissions.size(); ++i) {
            if (submissions[i].peer_index == 0) {
                backend_.complete(backend_.token(i), NIXL_ERR_REMOTE_DISCONNECT);
            }
        }
        release.join();
        EXPECT_TRUE(done.load());
        EXPECT_TRUE(allocator_.wasFreed(dst));
        EXPECT_EQ(backend_.quiesceCalls(), 2u);
    }

    TEST_F(ProxyRuntimeTest, QuiescencePrecedesResetAndFree) {
        auto ops = backend_.ops();
        nixlMemViewH src = nullptr, dst = nullptr;
        const auto caller = std::this_thread::get_id();
        unsigned quiesced = 0;
        ops.quiesce = [&](uint32_t, uint32_t) {
            EXPECT_NE(std::this_thread::get_id(), caller);
            if (quiesced++ == 0) {
                EXPECT_EQ(consumerIdx(channel()), 1u);
                EXPECT_EQ(completedIdx(channel()), 7u);
                EXPECT_FALSE(allocator_.wasFreed(dst));
            }
            return NIXL_SUCCESS;
        };
        ASSERT_EQ(nixlProxyRuntime::create(ops, makeConfig(1, 1, 1), runtime_, allocator_),
                  NIXL_SUCCESS);
        prepMemViews(src, dst);
        backend_.completeEverything();
        ASSERT_EQ(runtime_->startWorkers(), NIXL_SUCCESS);
        publish(channel(), 0, makePut(src, dst), 7);
        EXPECT_EQ(runtime_->unregisterProxyMemView(dst), NIXL_SUCCESS);
        EXPECT_TRUE(allocator_.wasFreed(dst));
        EXPECT_EQ(consumerIdx(channel()), 0u);
        EXPECT_EQ(runtime_->shutdown(), NIXL_SUCCESS);
        EXPECT_EQ(quiesced, 2u);
    }

    TEST_F(ProxyRuntimeTest, ShutdownWaitsForTerminalWork) {
        ASSERT_EQ(createRuntime(), NIXL_SUCCESS);
        nixlMemViewH src = nullptr, dst = nullptr;
        prepMemViews(src, dst);
        ASSERT_EQ(runtime_->startWorkers(), NIXL_SUCCESS);
        const auto *shutdown = allocator_.hostAlias(runtime_->deviceContext()->shutdown_word);
        publish(channel(), 0, makePut(src, dst), 1);
        ASSERT_TRUE(waitFor([&] { return backend_.submissionCount() == 1; }));
        std::atomic<bool> done{false};
        std::thread stop([&] {
            EXPECT_EQ(runtime_->shutdown(), NIXL_SUCCESS);
            done.store(true, std::memory_order_release);
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
        EXPECT_FALSE(done.load());
        EXPECT_EQ(__atomic_load_n(shutdown, __ATOMIC_ACQUIRE),
                  static_cast<uint64_t>(nixl_proxy_control_state_t::SHUTDOWN));
        EXPECT_FALSE(allocator_.wasFreed(dst));
        backend_.complete(backend_.token(0), NIXL_ERR_REMOTE_DISCONNECT);
        stop.join();
        EXPECT_TRUE(allocator_.wasFreed(dst));
    }

    TEST_F(ProxyRuntimeTest, UnpublishedTicketsNeverRearmSilently) {
        EXPECT_DEATH(
            {
                ASSERT_EQ(createRuntime(), NIXL_SUCCESS);
                nixlMemViewH src = nullptr;
                nixlMemViewH dst = nullptr;
                prepMemViews(src, dst);
                backend_.completeEverything();
                publish(channel(), 0, makePut(src, dst), 1);
                publish(channel(), 2, makePut(src, dst), 3);
                ASSERT_EQ(runtime_->startWorkers(), NIXL_SUCCESS);
                static_cast<void>(runtime_->unregisterProxyMemView(dst));
            },
            "unpublished producer tickets");
    }

    TEST_F(ProxyRuntimeTest, UnstartedRuntimeCannotDiscardQueuedWork) {
        EXPECT_DEATH(
            {
                ASSERT_EQ(createRuntime(), NIXL_SUCCESS);
                nixlMemViewH src = nullptr;
                nixlMemViewH dst = nullptr;
                prepMemViews(src, dst);
                publish(channel(), 0, makePut(src, dst), 1);
                static_cast<void>(runtime_->shutdown());
            },
            "unfinished or unpublished");
    }

    TEST_F(ProxyRuntimeTest, RetirementMakesQueuedPutAndAtomicVisibleAcrossRings) {
        uint64_t source = 0x123456789abcdef0;
        uint64_t destinations[2][4]{};
        std::vector<nixlBackendProxySubmission> pending[2];
        auto ops = backend_.ops();
        ops.submit = [&](const nixlBackendProxySubmission &op, nixlBackendProxyRequest &) {
            pending[op.channel_id].push_back(op);
            return NIXL_SUCCESS;
        };
        ops.quiesce = [&](uint32_t channel, uint32_t) {
            // Local completion need not imply remote visibility.
            for (const auto &op : pending[channel]) {
                auto *dst = reinterpret_cast<uint64_t *>(op.remote.desc.addr);
                if (op.opcode == nixl_proxy_opcode_t::PUT) {
                    std::memcpy(dst, reinterpret_cast<const void *>(op.local.desc.addr), op.size);
                } else {
                    *dst += op.value;
                }
            }
            pending[channel].clear();
            return NIXL_SUCCESS;
        };
        max_peers_ = 2;
        ASSERT_EQ(nixlProxyRuntime::create(ops, makeConfig(2, 2, 2), runtime_, allocator_),
                  NIXL_SUCCESS);
        nixlMemViewH src = nullptr;
        ASSERT_EQ(
            runtime_->prepMemView(
                makeLocalDlist(reinterpret_cast<uintptr_t>(&source), sizeof(source), 0, &local_md_),
                &src),
            NIXL_SUCCESS);
        for (int replacement = 0; replacement < 3; ++replacement) {
            nixl_remote_meta_dlist_t remote(VRAM_SEG);
            for (unsigned peer = 0; peer < 2; ++peer) {
                remote.addDesc(makeRemoteDesc(std::to_string(peer),
                                              reinterpret_cast<uintptr_t>(destinations[peer]),
                                              sizeof(destinations[peer]),
                                              0,
                                              &remote_md_));
            }
            nixlMemViewH dst = nullptr;
            ASSERT_EQ(runtime_->prepMemView(remote, &dst), NIXL_SUCCESS);
            for (unsigned channel_id = 0; channel_id < 2; ++channel_id) {
                for (unsigned peer = 0; peer < 2; ++peer) {
                    auto put = makePut(src, dst, channel_id, peer);
                    put.operand = 0;
                    put.dst_offset = channel_id * 2 * sizeof(uint64_t);
                    put.size = sizeof(uint64_t);
                    publish(channel(channel_id, peer), 0, put, 1);
                    auto atomic = put;
                    atomic.opcode = nixl_proxy_opcode_t::ATOMIC_ADD;
                    atomic.dst_offset += sizeof(uint64_t);
                    atomic.operand = 3;
                    publish(channel(channel_id, peer), 1, atomic, 2);
                }
            }
            if (replacement == 0) {
                ASSERT_EQ(runtime_->startWorkers(), NIXL_SUCCESS);
            }
            ASSERT_EQ(runtime_->unregisterProxyMemView(dst), NIXL_SUCCESS);
            for (const auto &peer : destinations) {
                for (unsigned channel_id = 0; channel_id < 2; ++channel_id) {
                    EXPECT_EQ(peer[channel_id * 2], source);
                    EXPECT_EQ(peer[channel_id * 2 + 1], 3u * (replacement + 1));
                }
            }
        }
        EXPECT_EQ(runtime_->unregisterProxyMemView(src), NIXL_SUCCESS);
        EXPECT_EQ(runtime_->shutdown(), NIXL_SUCCESS);
    }

} // namespace proxy_runtime
} // namespace gtest
