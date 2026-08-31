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
#ifndef NIXL_TEST_GTEST_MOCKS_PROXY_MOCKS_H
#define NIXL_TEST_GTEST_MOCKS_PROXY_MOCKS_H

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "backend_aux.h"
#include "device/device_allocator.h"
#include "device/proxy/proxy_backend_ops.h"

namespace gtest {
namespace proxy_mocks {

    /**
     * Host-memory stand-in for the GPU allocator, so the device proxy can be
     * tested without a GPU.
     *
     * Device memory is ordinary host memory: a test reads what the code under
     * test wrote to the device by dereferencing the device pointer. Mapped host
     * memory has two aliases, as it does on real hardware. The host alias is the
     * allocation; the device alias is a tagged pointer that cannot be
     * dereferenced. A test playing the GPU translates the device alias it finds
     * in a device-visible structure back with hostAlias(). Code that mixes the
     * two up fails that translation or faults, instead of passing because both
     * aliases happened to be the same address.
     *
     * Also records what was freed, which is how lifetime tests observe retirement
     * and shutdown.
     */
    class MockDeviceAllocator : public nixlDeviceAllocator {
    public:
        /** Fail once after N allocation/H2D calls; -1 disables injection. */
        int fail_after = -1;

        nixl_status_t
        copyHostToDevice(void *dst, const void *src, size_t size) noexcept override {
            if (fail_after >= 0 && fail_after-- == 0) {
                return NIXL_ERR_BACKEND;
            }
            std::memcpy(resolve(dst), src, size);
            return NIXL_SUCCESS;
        }

        nixl_status_t
        copyDeviceToHost(void *dst, const void *src, size_t size) noexcept override {
            std::memcpy(dst, resolve(src), size);
            return NIXL_SUCCESS;
        }

        nixl_status_t
        memsetDeviceMem(void *ptr, int value, size_t size) noexcept override {
            std::memset(resolve(ptr), value, size);
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

        /**
         * The host alias behind a mapped allocation's device alias, or null for a
         * pointer that is not one: a host pointer published where the GPU expects
         * a device alias, or an alias into memory that was already freed.
         */
        template<class T>
        T *
        hostAlias(const T *device_alias) const {
            const uintptr_t raw = reinterpret_cast<uintptr_t>(device_alias);
            if ((raw & kDeviceAliasTag) == 0) {
                return nullptr;
            }
            const uintptr_t host = raw & ~kDeviceAliasTag;

            const std::lock_guard<std::mutex> lock(mutex_);
            auto it = mapped_.upper_bound(host);
            if (it == mapped_.begin()) {
                return nullptr;
            }
            --it;
            if (host >= it->first + it->second) {
                return nullptr;
            }
            return reinterpret_cast<T *>(host);
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
            if (fail_after >= 0 && fail_after-- == 0) {
                return NIXL_ERR_BACKEND;
            }
            void *allocation = allocate(size);
            if (allocation == nullptr) {
                return NIXL_ERR_BACKEND;
            }
            *ptr = allocation;
            return NIXL_SUCCESS;
        }

        void
        doFreeDeviceMem(void *ptr) noexcept override {
            release(ptr);
        }

        nixl_status_t
        doAllocMappedHostMem(void **host_ptr, void **dev_ptr, size_t size) noexcept override {
            void *allocation = allocate(size);
            if (allocation == nullptr) {
                return NIXL_ERR_BACKEND;
            }
            {
                const std::lock_guard<std::mutex> lock(mutex_);
                mapped_[reinterpret_cast<uintptr_t>(allocation)] = size;
            }
            *host_ptr = allocation;
            *dev_ptr =
                reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(allocation) | kDeviceAliasTag);
            return NIXL_SUCCESS;
        }

        void
        doFreeMappedHostMem(void *host_ptr) noexcept override {
            {
                const std::lock_guard<std::mutex> lock(mutex_);
                mapped_.erase(reinterpret_cast<uintptr_t>(host_ptr));
            }
            release(host_ptr);
        }

    private:
        /** Never set in a user-space address, and non-canonical, so a dereference faults. */
        static constexpr uintptr_t kDeviceAliasTag = uintptr_t{1} << 62;

        /** Device-memory pointers are used as they are; a device alias is untagged. */
        static void *
        resolve(const void *ptr) noexcept {
            return reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(ptr) & ~kDeviceAliasTag);
        }

        void *
        allocate(size_t size) noexcept {
            // Cache-line aligned so the control buffer's GPU-page rounding stays
            // in bounds, and zeroed like freshly allocated device memory.
            const size_t rounded = std::max<size_t>(64, (size + 63) & ~size_t{63});
            void *allocation = std::aligned_alloc(64, rounded);
            if (allocation == nullptr) {
                return nullptr;
            }
            std::memset(allocation, 0, rounded);
            const std::lock_guard<std::mutex> lock(mutex_);
            live_.insert(allocation);
            freed_.erase(allocation);
            return allocation;
        }

        void
        release(void *ptr) noexcept {
            if (ptr == nullptr) {
                return;
            }
            {
                const std::lock_guard<std::mutex> lock(mutex_);
                live_.erase(ptr);
                freed_.insert(ptr);
            }
            std::free(ptr);
        }

        mutable std::mutex mutex_;
        std::set<const void *> live_;
        std::set<const void *> freed_;
        /** Host base to size of every live mapped allocation. */
        std::map<uintptr_t, size_t> mapped_;
    };

    /** The registry stores backend metadata by pointer and never looks inside. */
    class DummyBackendMD : public nixlBackendMD {
    public:
        DummyBackendMD() : nixlBackendMD(false) {}
    };

    inline nixl_meta_dlist_t
    makeLocalDlist(uintptr_t addr, size_t len, uint64_t dev_id, nixlBackendMD *md) {
        nixl_meta_dlist_t dlist(DRAM_SEG);
        dlist.addDesc(nixlMetaDesc(addr, len, dev_id, md));
        return dlist;
    }

    inline nixlRemoteMetaDesc
    makeRemoteDesc(const std::string &agent,
                   uintptr_t addr,
                   size_t len,
                   uint64_t dev_id,
                   nixlBackendMD *md) {
        nixlRemoteMetaDesc desc(agent);
        desc.addr = addr;
        desc.len = len;
        desc.devId = dev_id;
        desc.metadataP = md;
        return desc;
    }

    /**
     * Mock transport behind nixlProxyBackendOps. Records every submission and
     * completes a request only when the test says so, which puts the moment a
     * worker observes a terminal status under the test's control.
     */
    class MockBackend {
    public:
        nixlProxyBackendOps
        ops() {
            nixlProxyBackendOps ops;
            ops.init = [this](const nixlProxyConfig &config) {
                const std::lock_guard<std::mutex> lock(mutex_);
                init_thread_count_ = config.effectiveThreadCount();
                return init_status_;
            };
            ops.submit = [this](const nixlBackendProxySubmission &submission,
                                nixlBackendProxyRequest &request) {
                const std::lock_guard<std::mutex> lock(mutex_);
                submissions_.push_back(submission);
                const uint64_t token = ++next_token_;
                tokens_.push_back(token);
                token_peer_[token] = submission.peer_index;
                nixl_status_t status = NIXL_IN_PROG;
                if (!submit_statuses_.empty()) {
                    status = submit_statuses_.front();
                    submit_statuses_.erase(submit_statuses_.begin());
                }
                request = status == NIXL_IN_PROG ?
                    nixlBackendProxyRequest{token, submission.channel_id} :
                    nixlBackendProxyRequest{};
                return status;
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
            ops.progress = [](uint32_t, uint32_t) { return NIXL_SUCCESS; };
            ops.shutdown = [this]() {
                const std::lock_guard<std::mutex> lock(mutex_);
                ++shutdown_calls_;
                return NIXL_SUCCESS;
            };
            if (resolver_enabled_) {
                ops.resolve_direct_ptrs = [this](const nixl_remote_meta_dlist_t &dlist,
                                                 std::vector<void *> &direct_ptrs) {
                    const std::lock_guard<std::mutex> lock(mutex_);
                    ++resolve_calls_;
                    last_resolved_desc_count_ = static_cast<size_t>(dlist.descCount());
                    if (resolver_status_ == NIXL_SUCCESS) {
                        direct_ptrs = direct_ptrs_;
                    }
                    return resolver_status_;
                };
            }
            return ops;
        }

        /** What init() returns; the runtime must fail to build on an error. */
        void
        failInit(nixl_status_t status) {
            const std::lock_guard<std::mutex> lock(mutex_);
            init_status_ = status;
        }

        /** Finish one request; the worker sees the status on its next check. */
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

        /** Requests bound for this peer finish as soon as they are checked. */
        void
        completePeer(uint32_t peer_index) {
            const std::lock_guard<std::mutex> lock(mutex_);
            healthy_peers_.insert(peer_index);
        }

        /** The next submit() returns this status instead of accepting the request. */
        void
        failNextSubmit(nixl_status_t status) {
            const std::lock_guard<std::mutex> lock(mutex_);
            submit_statuses_.push_back(status);
        }

        void
        setDirectPointers(std::vector<void *> direct_ptrs) {
            const std::lock_guard<std::mutex> lock(mutex_);
            direct_ptrs_ = std::move(direct_ptrs);
        }

        void
        setResolverStatus(nixl_status_t status) {
            const std::lock_guard<std::mutex> lock(mutex_);
            resolver_status_ = status;
        }

        /** Leave resolve_direct_ptrs unset in the next ops(). */
        void
        disableResolver() {
            resolver_enabled_ = false;
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

        size_t
        resolveCalls() const {
            const std::lock_guard<std::mutex> lock(mutex_);
            return resolve_calls_;
        }

        size_t
        lastResolvedDescCount() const {
            const std::lock_guard<std::mutex> lock(mutex_);
            return last_resolved_desc_count_;
        }

    private:
        mutable std::mutex mutex_;
        std::vector<nixlBackendProxySubmission> submissions_;
        std::vector<uint64_t> tokens_;
        std::vector<uint64_t> released_;
        std::vector<nixl_status_t> submit_statuses_;
        std::map<uint64_t, nixl_status_t> completed_;
        std::map<uint64_t, uint32_t> token_peer_;
        std::set<uint32_t> healthy_peers_;
        std::vector<void *> direct_ptrs_;
        nixl_status_t init_status_ = NIXL_SUCCESS;
        nixl_status_t resolver_status_ = NIXL_SUCCESS;
        bool complete_on_check_ = false;
        bool resolver_enabled_ = true;
        uint64_t next_token_ = 0;
        uint32_t init_thread_count_ = 0;
        size_t shutdown_calls_ = 0;
        size_t resolve_calls_ = 0;
        size_t last_resolved_desc_count_ = 0;
    };

} // namespace proxy_mocks
} // namespace gtest

#endif // NIXL_TEST_GTEST_MOCKS_PROXY_MOCKS_H
