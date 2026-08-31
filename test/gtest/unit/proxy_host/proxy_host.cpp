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

// The proxy runtime driven end to end on the host: a mock nixlDeviceAllocator
// hands out ordinary host memory (device pointer == host pointer) and mock
// nixlProxyBackendOps stand in for the transport, so the test itself plays the
// GPU - writing ring records with the publication order proxy_protocol.h
// specifies. No CUDA and no UCX are involved.

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "device/proxy/proxy_protocol.h"

#include "device/device_allocator.h"
#include "device/proxy/proxy_backend_ops.h"
#include "device/proxy/proxy_config.h"
#include "device/proxy/proxy_runtime.h"

namespace gtest {
namespace proxy_host {

    constexpr uint32_t kRingDepth = 4;

    /**
     * Host-memory stand-in for the GPU allocator: "device" memory is malloc'd, so
     * every device pointer the runtime hands out is directly dereferenceable by
     * the test. Also records which pointers were freed, which is how the
     * lifetime tests observe unregister() and shutdown().
     */
    class MockDeviceAllocator : public nixlDeviceAllocator {
    public:
        nixl_status_t
        copyHostToDevice(void *dst, const void *src, size_t size) noexcept override {
            std::memcpy(dst, src, size);
            return NIXL_SUCCESS;
        }

        nixl_status_t
        copyDeviceToHost(void *dst, const void *src, size_t size) noexcept override {
            std::memcpy(dst, src, size);
            return NIXL_SUCCESS;
        }

        nixl_status_t
        memsetDeviceMem(void *ptr, int value, size_t size) noexcept override {
            std::memset(ptr, value, size);
            return NIXL_SUCCESS;
        }

        nixl_status_t
        synchronize() noexcept override {
            return NIXL_SUCCESS;
        }

        nixl_status_t
        getActiveDevice(int &device_id) noexcept override {
            device_id = 0;
            return NIXL_SUCCESS;
        }

        nixl_status_t
        setActiveDevice(int) noexcept override {
            return NIXL_SUCCESS;
        }

        size_t
        liveAllocations() const {
            const std::lock_guard<std::mutex> lock(mutex_);
            return live_.size();
        }

        bool
        wasFreed(const void *ptr) const {
            const std::lock_guard<std::mutex> lock(mutex_);
            return freed_.count(ptr) != 0;
        }

    protected:
        nixl_status_t
        doAllocDeviceMem(void **ptr, size_t size) noexcept override {
            // Aligned so the control buffer's GPU-page rounding stays in bounds.
            void *allocation = std::aligned_alloc(64, (size + 63) & ~size_t{63});
            if (allocation == nullptr) {
                return NIXL_ERR_BACKEND;
            }
            std::memset(allocation, 0, size);
            track(allocation);
            *ptr = allocation;
            return NIXL_SUCCESS;
        }

        void
        doFreeDeviceMem(void *ptr) noexcept override {
            untrack(ptr);
            std::free(ptr);
        }

        nixl_status_t
        doAllocMappedHostMem(void **host_ptr, void **dev_ptr, size_t size) noexcept override {
            void *allocation = std::aligned_alloc(64, (size + 63) & ~size_t{63});
            if (allocation == nullptr) {
                return NIXL_ERR_BACKEND;
            }
            std::memset(allocation, 0, size);
            track(allocation);
            // The whole point of mapped host memory: one buffer, two aliases.
            *host_ptr = allocation;
            *dev_ptr = allocation;
            return NIXL_SUCCESS;
        }

        void
        doFreeMappedHostMem(void *host_ptr) noexcept override {
            untrack(host_ptr);
            std::free(host_ptr);
        }

    private:
        void
        track(void *ptr) {
            const std::lock_guard<std::mutex> lock(mutex_);
            live_.insert(ptr);
            freed_.erase(ptr);
        }

        void
        untrack(void *ptr) {
            if (ptr == nullptr) {
                return;
            }
            const std::lock_guard<std::mutex> lock(mutex_);
            live_.erase(ptr);
            freed_.insert(ptr);
        }

        mutable std::mutex mutex_;
        std::set<const void *> live_;
        std::set<const void *> freed_;
    };

    /** Mock transport: records submissions and completes them on demand. */
    class MockBackend {
    public:
        nixlProxyBackendOps
        ops() {
            nixlProxyBackendOps ops;
            ops.init = [this](const nixlProxyConfig &config) {
                const std::lock_guard<std::mutex> lock(mutex_);
                init_calls_++;
                init_thread_count_ = config.effectiveThreadCount();
                return NIXL_SUCCESS;
            };
            ops.submit = [this](const nixlBackendProxySubmission &submission,
                                nixlBackendProxyRequest &request) {
                const std::lock_guard<std::mutex> lock(mutex_);
                request = nixlBackendProxyRequest{++next_token_, submission.channel_id};
                submissions_.push_back(submission);
                tokens_.push_back(request.token);
                token_peer_[request.token] = submission.peer_index;
                return NIXL_IN_PROG;
            };
            ops.check_completion = [this](const nixlBackendProxyRequest &request) {
                const std::lock_guard<std::mutex> lock(mutex_);
                if (complete_on_check_) {
                    return NIXL_SUCCESS;
                }
                const auto peer = token_peer_.find(request.token);
                if (peer != token_peer_.end() && healthy_peers_.count(peer->second) != 0) {
                    return NIXL_SUCCESS;
                }
                const auto it = completed_.find(request.token);
                return it == completed_.end() ? NIXL_IN_PROG : it->second;
            };
            ops.release_request = [this](const nixlBackendProxyRequest &request) {
                const std::lock_guard<std::mutex> lock(mutex_);
                released_.push_back(request.token);
            };
            ops.progress = [this](uint32_t, uint32_t) {
                progress_calls_.fetch_add(1, std::memory_order_relaxed);
                return NIXL_SUCCESS;
            };
            ops.shutdown = [this]() {
                const std::lock_guard<std::mutex> lock(mutex_);
                shutdown_calls_++;
                return NIXL_SUCCESS;
            };
            return ops;
        }

        void
        complete(uint64_t token, nixl_status_t status = NIXL_SUCCESS) {
            const std::lock_guard<std::mutex> lock(mutex_);
            completed_[token] = status;
        }

        /** Every request finishes as soon as it is checked. */
        void
        completeEverything() {
            const std::lock_guard<std::mutex> lock(mutex_);
            complete_on_check_ = true;
        }

        /** Requests bound for this peer finish; every other peer stays wedged. */
        void
        completePeer(uint32_t peer_index) {
            const std::lock_guard<std::mutex> lock(mutex_);
            healthy_peers_.insert(peer_index);
        }

        size_t
        submissionCountForPeer(uint32_t peer_index) const {
            const std::lock_guard<std::mutex> lock(mutex_);
            size_t count = 0;
            for (const auto &submission : submissions_) {
                count += submission.peer_index == peer_index ? 1 : 0;
            }
            return count;
        }

        size_t
        submissionCount() const {
            const std::lock_guard<std::mutex> lock(mutex_);
            return submissions_.size();
        }

        std::vector<nixlBackendProxySubmission>
        submissions() const {
            const std::lock_guard<std::mutex> lock(mutex_);
            return submissions_;
        }

        /** Backend token of the n-th submission, in submission order. */
        uint64_t
        token(size_t index) const {
            const std::lock_guard<std::mutex> lock(mutex_);
            return index < tokens_.size() ? tokens_[index] : 0;
        }

        std::vector<uint64_t>
        released() const {
            const std::lock_guard<std::mutex> lock(mutex_);
            return released_;
        }

        uint32_t
        initThreadCount() const {
            const std::lock_guard<std::mutex> lock(mutex_);
            return init_thread_count_;
        }

        size_t
        shutdownCalls() const {
            const std::lock_guard<std::mutex> lock(mutex_);
            return shutdown_calls_;
        }

    private:
        mutable std::mutex mutex_;
        std::vector<nixlBackendProxySubmission> submissions_;
        std::vector<uint64_t> tokens_;
        std::vector<uint64_t> released_;
        std::map<uint64_t, nixl_status_t> completed_;
        std::atomic<uint64_t> progress_calls_{0};
        std::map<uint64_t, uint32_t> token_peer_;
        std::set<uint32_t> healthy_peers_;
        bool complete_on_check_ = false;
        uint64_t next_token_ = 0;
        uint32_t init_thread_count_ = 0;
        size_t init_calls_ = 0;
        size_t shutdown_calls_ = 0;
    };

    class DummyBackendMD : public nixlBackendMD {
    public:
        DummyBackendMD() : nixlBackendMD(false) {}
    };

    class ProxyHostTest : public testing::Test {
    protected:
        void
        TearDown() override {
            runtime_.reset();
            // Every device and mapped-host allocation the runtime made is gone.
            EXPECT_EQ(allocator_.liveAllocations(), 0u);
        }

        nixl_status_t
        createRuntime(uint32_t channel_count = 1, uint32_t max_peers = 1) {
            nixlProxyConfig config;
            config.enabled = true;
            config.channel_count = channel_count;
            config.thread_count = channel_count;
            config.max_peers = max_peers;
            config.ring_depth = kRingDepth;
            return nixlProxyRuntime::create(backend_.ops(), config, runtime_, allocator_);
        }

        /** The GPU's view of a channel; every pointer here is host-dereferenceable. */
        struct ChannelAccess {
            nixlProxyWorkRing *ring = nullptr;
            nixlProxySubmission *records = nullptr;
            nixlProxyCompletionSlot *completion = nullptr;
            const uint64_t *consumer_idx = nullptr;
        };

        ChannelAccess
        channel(size_t slot = 0) const {
            const nixlProxyChannelView &view = runtime_->deviceChannelViews()[slot];
            ChannelAccess access;
            access.ring = view.work_ring;
            access.records = access.ring->records;
            access.completion = view.completion_slot;
            access.consumer_idx = access.ring->consumer_idx;
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
            const uint32_t slot = static_cast<uint32_t>(producer_idx % kRingDepth);
            nixlProxySubmission staged = record;
            staged.op_idx = 0;
            access.records[slot] = staged;
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

        /** A local and a remote memview over 64-byte descriptors. */
        void
        prepMemViews(nixlMemViewH &src, nixlMemViewH &dst, const std::string &agent = "peer") {
            nixl_meta_dlist_t local_dlist(DRAM_SEG);
            local_dlist.addDesc(nixlMetaDesc(0x1000, 64, 0, &local_md_));
            ASSERT_EQ(runtime_->prepMemView(local_dlist, &src), NIXL_SUCCESS);

            nixl_remote_meta_dlist_t remote_dlist(VRAM_SEG);
            nixlRemoteMetaDesc remote_desc(agent);
            remote_desc.addr = 0x2000;
            remote_desc.len = 64;
            remote_desc.devId = 0;
            remote_desc.metadataP = &remote_md_;
            remote_dlist.addDesc(remote_desc);
            ASSERT_EQ(runtime_->prepMemView(remote_dlist, &dst), NIXL_SUCCESS);
        }

        static uint32_t
        memViewId(nixlMemViewH proxy_memview) {
            return static_cast<const nixlProxyDeviceMemView *>(proxy_memview)->proxy_memview_id;
        }

        static nixlProxySubmission
        makePut(nixlMemViewH src, nixlMemViewH dst, uint64_t size = 32) {
            nixlProxySubmission record{};
            record.opcode = nixl_proxy_opcode_t::PUT;
            record.channel_id = 0;
            record.src_proxy_memview_id = memViewId(src);
            record.dst_proxy_memview_id = memViewId(dst);
            record.src_offset = 4;
            record.dst_offset = 8;
            record.size = size;
            return record;
        }

        MockDeviceAllocator allocator_;
        MockBackend backend_;
        DummyBackendMD local_md_;
        DummyBackendMD remote_md_;
        // Declared last: the runtime's callbacks and memory outlive nothing here,
        // so it must die before the mocks it points at.
        std::unique_ptr<nixlProxyRuntime> runtime_;
    };

    TEST_F(ProxyHostTest, CreateAndShutdownWithoutWork) {
        ASSERT_EQ(createRuntime(), NIXL_SUCCESS);
        EXPECT_EQ(backend_.initThreadCount(), 1u);
        ASSERT_EQ(runtime_->startWorkers(), NIXL_SUCCESS);
        EXPECT_EQ(runtime_->shutdown(), NIXL_SUCCESS);
        EXPECT_EQ(backend_.shutdownCalls(), 1u);
    }

    TEST_F(ProxyHostTest, SingleSubmissionReachesBackendAndCompletes) {
        ASSERT_EQ(createRuntime(), NIXL_SUCCESS);
        nixlMemViewH src = nullptr, dst = nullptr;
        prepMemViews(src, dst, "remote-agent");
        ASSERT_EQ(runtime_->startWorkers(), NIXL_SUCCESS);

        const ChannelAccess access = channel();
        publish(access, 0, makePut(src, dst), 7);

        ASSERT_TRUE(waitFor([&]() { return backend_.submissionCount() == 1; }));
        const auto prepared = backend_.submissions().front();
        EXPECT_EQ(prepared.op_idx, 7u);
        EXPECT_EQ(prepared.opcode, nixl_proxy_opcode_t::PUT);
        EXPECT_EQ(prepared.channel_id, 0u);
        EXPECT_EQ(prepared.peer_index, 0u);
        EXPECT_EQ(prepared.size, 32u);
        EXPECT_EQ(prepared.local.mem_type, DRAM_SEG);
        EXPECT_EQ(prepared.local.desc.addr, 0x1004u);
        EXPECT_EQ(prepared.local.desc.len, 32u);
        EXPECT_EQ(prepared.local.desc.metadataP, &local_md_);
        EXPECT_EQ(prepared.remote.mem_type, VRAM_SEG);
        EXPECT_EQ(prepared.remote.desc.addr, 0x2008u);
        EXPECT_EQ(prepared.remote.desc.len, 32u);
        EXPECT_EQ(prepared.remote.desc.metadataP, &remote_md_);
        EXPECT_EQ(prepared.remote_agent, "remote-agent");

        // Nothing is published back until the backend says the transfer is done.
        EXPECT_EQ(completedIdx(access), 0u);
        EXPECT_EQ(consumerIdx(access), 0u);

        backend_.complete(backend_.token(0));
        ASSERT_TRUE(waitFor([&]() { return completedIdx(access) == 7u; }));
        EXPECT_EQ(access.completion->next_status, NIXL_SUCCESS);
        EXPECT_TRUE(waitFor([&]() { return consumerIdx(access) == 1u; }));

        EXPECT_EQ(runtime_->shutdown(), NIXL_SUCCESS);
    }

    TEST_F(ProxyHostTest, CompletionsPublishInSubmissionOrder) {
        ASSERT_EQ(createRuntime(), NIXL_SUCCESS);
        nixlMemViewH src = nullptr, dst = nullptr;
        prepMemViews(src, dst);
        ASSERT_EQ(runtime_->startWorkers(), NIXL_SUCCESS);

        const ChannelAccess access = channel();
        const nixlProxySubmission record = makePut(src, dst);
        publish(access, 0, record, 11);
        ASSERT_TRUE(waitFor([&]() { return backend_.submissionCount() == 1; }));
        publish(access, 1, record, 12);
        ASSERT_TRUE(waitFor([&]() { return backend_.submissionCount() == 2; }));

        // Completing the second first must not let it overtake the first.
        backend_.complete(backend_.token(1));
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        EXPECT_EQ(completedIdx(access), 0u);
        EXPECT_EQ(consumerIdx(access), 0u);

        backend_.complete(backend_.token(0));
        ASSERT_TRUE(waitFor([&]() { return consumerIdx(access) == 2u; }));
        EXPECT_EQ(completedIdx(access), 12u);

        EXPECT_EQ(runtime_->shutdown(), NIXL_SUCCESS);
    }

    TEST_F(ProxyHostTest, FullRingBackpressuresUntilTheConsumerIndexAdvances) {
        ASSERT_EQ(createRuntime(), NIXL_SUCCESS);
        nixlMemViewH src = nullptr, dst = nullptr;
        prepMemViews(src, dst);
        ASSERT_EQ(runtime_->startWorkers(), NIXL_SUCCESS);

        const ChannelAccess access = channel();
        EXPECT_EQ(access.ring->depth, kRingDepth);

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

        EXPECT_EQ(runtime_->shutdown(), NIXL_SUCCESS);
    }

    TEST_F(ProxyHostTest, ShutdownReleasesInFlightRequests) {
        ASSERT_EQ(createRuntime(), NIXL_SUCCESS);
        nixlMemViewH src = nullptr, dst = nullptr;
        prepMemViews(src, dst);
        ASSERT_EQ(runtime_->startWorkers(), NIXL_SUCCESS);

        const ChannelAccess access = channel();
        const nixlProxySubmission record = makePut(src, dst);
        publish(access, 0, record, 1);
        ASSERT_TRUE(waitFor([&]() { return backend_.submissionCount() == 1; }));
        publish(access, 1, record, 2);
        ASSERT_TRUE(waitFor([&]() { return backend_.submissionCount() == 2; }));

        // Neither request ever completes; shutdown must not wait for them.
        EXPECT_EQ(runtime_->shutdown(), NIXL_SUCCESS);

        EXPECT_EQ(backend_.released(),
                  (std::vector<uint64_t>{backend_.token(0), backend_.token(1)}));
    }

    TEST_F(ProxyHostTest, UnregisteredMemViewIsFreedAndMissesEverySubmission) {
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

        ASSERT_EQ(runtime_->unregisterProxyMemView(dst), NIXL_SUCCESS);
        nixlMemViewH resolved = nullptr;
        EXPECT_FALSE(runtime_->resolveProxyMemView(dst, resolved));
        // The device allocation goes immediately, like the direct path's
        // ucp_device_mem_list_release; only the GPU reads it, and the caller
        // has quiesced the GPU by the time it unregisters.
        EXPECT_TRUE(allocator_.wasFreed(dst));

        // Retiring drained and rearmed the ring, so the producer starts over.
        EXPECT_EQ(consumerIdx(access), 0u);
        EXPECT_EQ(completedIdx(access), 0u);

        // The worker survives it: the host-side entry is still alive, and a
        // record naming the retired id is rejected rather than dispatched.
        publish(access, 0, record, 2);
        ASSERT_TRUE(waitFor([&]() { return consumerIdx(access) == 1u; }));
        EXPECT_EQ(backend_.submissionCount(), 1u);
        EXPECT_LT(access.completion->next_status, 0);

        EXPECT_EQ(runtime_->shutdown(), NIXL_SUCCESS);
    }

    // Memviews left registered at shutdown are freed with the registry, not
    // leaked into the next generation.
    TEST_F(ProxyHostTest, LiveMemViewsAreFreedAtShutdown) {
        ASSERT_EQ(createRuntime(), NIXL_SUCCESS);
        nixlMemViewH src = nullptr, dst = nullptr;
        prepMemViews(src, dst);
        ASSERT_EQ(runtime_->startWorkers(), NIXL_SUCCESS);

        EXPECT_FALSE(allocator_.wasFreed(src));
        EXPECT_FALSE(allocator_.wasFreed(dst));

        EXPECT_EQ(runtime_->shutdown(), NIXL_SUCCESS);
        EXPECT_TRUE(allocator_.wasFreed(src));
        EXPECT_TRUE(allocator_.wasFreed(dst));
    }

    TEST_F(ProxyHostTest, MultipleChannelsAndPeersEachGetTheirOwnRing) {
        ASSERT_EQ(createRuntime(/*channel_count=*/2, /*max_peers=*/2), NIXL_SUCCESS);
        nixlMemViewH src = nullptr, dst = nullptr;

        nixl_meta_dlist_t local_dlist(DRAM_SEG);
        local_dlist.addDesc(nixlMetaDesc(0x1000, 64, 0, &local_md_));
        ASSERT_EQ(runtime_->prepMemView(local_dlist, &src), NIXL_SUCCESS);

        nixl_remote_meta_dlist_t remote_dlist(VRAM_SEG);
        for (const char *agent : {"peer0", "peer1"}) {
            nixlRemoteMetaDesc remote_desc(agent);
            remote_desc.addr = 0x2000;
            remote_desc.len = 64;
            remote_desc.devId = 0;
            remote_desc.metadataP = &remote_md_;
            remote_dlist.addDesc(remote_desc);
        }
        ASSERT_EQ(runtime_->prepMemView(remote_dlist, &dst), NIXL_SUCCESS);
        ASSERT_EQ(runtime_->startWorkers(), NIXL_SUCCESS);

        // Channel 1, peer 1: slot = channel * max_peers + peer.
        const ChannelAccess access = channel(1 * 2 + 1);
        nixlProxySubmission record = makePut(src, dst);
        record.channel_id = 1;
        record.dst_index = 1;
        publish(access, 0, record, 5);

        ASSERT_TRUE(waitFor([&]() { return backend_.submissionCount() == 1; }));
        const auto prepared = backend_.submissions().front();
        EXPECT_EQ(prepared.channel_id, 1u);
        EXPECT_EQ(prepared.peer_index, 1u);
        EXPECT_EQ(prepared.remote_agent, "peer1");

        backend_.complete(backend_.token(0));
        ASSERT_TRUE(waitFor([&]() { return completedIdx(access) == 5u; }));

        EXPECT_EQ(runtime_->shutdown(), NIXL_SUCCESS);
    }

    // repo docs/issues/005 G1: a record already in the ring when its memview is
    // retired used to fail prepareSubmission and vanish, taking transfers to
    // healthy peers with it. Retiring now drains first.
    TEST_F(ProxyHostTest, DrainSubmitsQueuedRecordsBeforeRetire) {
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
        EXPECT_EQ(access.completion->next_status, NIXL_IN_PROG);

        EXPECT_EQ(runtime_->shutdown(), NIXL_SUCCESS);
    }

    // A peer that never completes must not hold a membership change hostage.
    TEST_F(ProxyHostTest, DrainCancelsRequestsThatNeverComplete) {
        ASSERT_EQ(createRuntime(), NIXL_SUCCESS);
        nixlMemViewH src = nullptr, dst = nullptr;
        prepMemViews(src, dst);
        ASSERT_EQ(runtime_->startWorkers(), NIXL_SUCCESS);

        const ChannelAccess access = channel();
        const nixlProxySubmission record = makePut(src, dst);
        publish(access, 0, record, 1);
        publish(access, 1, record, 2);
        ASSERT_TRUE(waitFor([&]() { return backend_.submissionCount() == 2; }));

        // Nothing is ever completed, so the drain has to give up on its own.
        ASSERT_EQ(runtime_->unregisterProxyMemView(dst), NIXL_SUCCESS);

        EXPECT_EQ(backend_.released(),
                  (std::vector<uint64_t>{backend_.token(0), backend_.token(1)}));
        EXPECT_EQ(consumerIdx(access), 0u);

        EXPECT_EQ(runtime_->shutdown(), NIXL_SUCCESS);
    }

    // repo docs/issues/005 G2: re-adding a rank used to inherit whatever state
    // the wedged ring was left in.
    TEST_F(ProxyHostTest, RingsAreUsableAfterDrain) {
        ASSERT_EQ(createRuntime(), NIXL_SUCCESS);
        nixlMemViewH src = nullptr, dst = nullptr;
        prepMemViews(src, dst);
        ASSERT_EQ(runtime_->startWorkers(), NIXL_SUCCESS);

        const ChannelAccess access = channel();
        publish(access, 0, makePut(src, dst), 1);
        ASSERT_TRUE(waitFor([&]() { return backend_.submissionCount() == 1; }));

        // Wedge it: the request never completes, so this retire cancels.
        ASSERT_EQ(runtime_->unregisterProxyMemView(dst), NIXL_SUCCESS);
        ASSERT_EQ(runtime_->unregisterProxyMemView(src), NIXL_SUCCESS);

        // Same ring, new generation of memviews - it has to work.
        nixlMemViewH new_src = nullptr, new_dst = nullptr;
        prepMemViews(new_src, new_dst);
        backend_.completeEverything();
        publish(access, 0, makePut(new_src, new_dst), 7);

        ASSERT_TRUE(waitFor([&]() { return completedIdx(access) == 7u; }));
        EXPECT_EQ(access.completion->next_status, NIXL_SUCCESS);
        EXPECT_EQ(backend_.submissionCount(), 2u);

        EXPECT_EQ(runtime_->shutdown(), NIXL_SUCCESS);
    }

    TEST_F(ProxyHostTest, DrainOnEmptyRingsCancelsNothing) {
        ASSERT_EQ(createRuntime(), NIXL_SUCCESS);
        nixlMemViewH src = nullptr, dst = nullptr;
        prepMemViews(src, dst);
        ASSERT_EQ(runtime_->startWorkers(), NIXL_SUCCESS);

        ASSERT_EQ(runtime_->unregisterProxyMemView(dst), NIXL_SUCCESS);
        ASSERT_EQ(runtime_->unregisterProxyMemView(src), NIXL_SUCCESS);

        EXPECT_TRUE(backend_.released().empty());
        EXPECT_EQ(backend_.submissionCount(), 0u);

        EXPECT_EQ(runtime_->shutdown(), NIXL_SUCCESS);
    }

    // One wedged peer must not take a healthy peer's queued records with it -
    // that is the whole point of draining rather than retiring underneath them.
    TEST_F(ProxyHostTest, WedgedPeerDoesNotDropAHealthyPeersRecords) {
        ASSERT_EQ(createRuntime(/*channel_count=*/1, /*max_peers=*/2), NIXL_SUCCESS);

        nixlMemViewH src = nullptr, dst = nullptr;
        nixl_meta_dlist_t local_dlist(DRAM_SEG);
        local_dlist.addDesc(nixlMetaDesc(0x1000, 64, 0, &local_md_));
        ASSERT_EQ(runtime_->prepMemView(local_dlist, &src), NIXL_SUCCESS);

        nixl_remote_meta_dlist_t remote_dlist(VRAM_SEG);
        for (const char *agent : {"wedged", "healthy"}) {
            nixlRemoteMetaDesc remote_desc(agent);
            remote_desc.addr = 0x2000;
            remote_desc.len = 64;
            remote_desc.devId = 0;
            remote_desc.metadataP = &remote_md_;
            remote_dlist.addDesc(remote_desc);
        }
        ASSERT_EQ(runtime_->prepMemView(remote_dlist, &dst), NIXL_SUCCESS);
        ASSERT_EQ(runtime_->startWorkers(), NIXL_SUCCESS);

        // Peer 0 is drained first and never completes, so it burns the deadline.
        const ChannelAccess wedged = channel(0);
        nixlProxySubmission wedged_record = makePut(src, dst);
        wedged_record.dst_index = 0;
        publish(wedged, 0, wedged_record, 1);

        // Fill the healthy peer's ring so the worker cannot pick anything else
        // up until something completes - that is what makes this deterministic.
        const ChannelAccess healthy = channel(1);
        nixlProxySubmission healthy_record = makePut(src, dst);
        healthy_record.dst_index = 1;
        for (uint64_t i = 0; i < kRingDepth; ++i) {
            publish(healthy, i, healthy_record, 100 + i);
        }
        ASSERT_TRUE(waitFor([&]() {
            return backend_.submissionCountForPeer(1) == kRingDepth &&
                backend_.submissionCountForPeer(0) == 1;
        }));

        // One more behind the full ring, which only a drain can get to.
        publish(healthy, kRingDepth, healthy_record, 200);
        backend_.completePeer(1);

        ASSERT_EQ(runtime_->unregisterProxyMemView(dst), NIXL_SUCCESS);

        // Every healthy record made it out; only the wedged peer lost anything.
        EXPECT_EQ(backend_.submissionCountForPeer(1), size_t{kRingDepth} + 1);
        EXPECT_EQ(backend_.released(), (std::vector<uint64_t>{backend_.token(0)}));

        EXPECT_EQ(runtime_->shutdown(), NIXL_SUCCESS);
    }

} // namespace proxy_host
} // namespace gtest
