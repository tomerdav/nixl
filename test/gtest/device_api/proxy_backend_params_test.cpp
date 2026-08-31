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

#include "nixl.h"

namespace {

/**
 * Engine-owned proxy lifecycle: the runtime is created and torn down purely
 * via UCX backend params, with no proxy configuration on the agent.
 */
TEST(ProxyBackendParamsTest, CreateAndTeardown) {
    nixlAgentConfig cfg(false);
    nixlAgent agent("proxy_params_create", std::move(cfg));

    nixl_b_params_t params{{"device_proxy", "true"},
                           {"proxy_channel_count", "2"},
                           {"proxy_max_peers", "4"}};
    nixlBackendH *backend = nullptr;
    ASSERT_EQ(agent.createBackend("UCX", params, backend), NIXL_SUCCESS);
    ASSERT_NE(backend, nullptr);
    // Teardown (agent destructor) must join the proxy threads cleanly.
}

// Real callers seed their parameters from getPluginParams() rather than
// building the map by hand, and that seeds the plugin's own default of
// num_workers=1. In proxy mode the engine derives the worker count from
// channels x peers, so an inherited 1 is an explicit value that disagrees -
// and the caller has to erase it, not merely refrain from setting it.
TEST(ProxyBackendParamsTest, PluginDefaultNumWorkersMustBeErasedForProxy) {
    nixlAgentConfig cfg(false);
    nixlAgent agent("proxy_params_plugin_defaults", std::move(cfg));

    nixl_mem_list_t mems;
    nixl_b_params_t params;
    ASSERT_EQ(agent.getPluginParams("UCX", mems, params), NIXL_SUCCESS);
    ASSERT_EQ(params.count("num_workers"), 1u) << "plugin no longer seeds num_workers; "
                                                  "the erase below may be unnecessary";

    params["device_proxy"] = "true";
    params["proxy_channel_count"] = "2";
    params["proxy_max_peers"] = "4";

    // Left in place, the inherited default conflicts with 2 x 4 = 8.
    nixlBackendH *rejected = nullptr;
    EXPECT_NE(agent.createBackend("UCX", params, rejected), NIXL_SUCCESS);

    params.erase("num_workers");
    nixlBackendH *backend = nullptr;
    EXPECT_EQ(agent.createBackend("UCX", params, backend), NIXL_SUCCESS);
}

TEST(ProxyBackendParamsTest, MatchingNumWorkersAccepted) {
    nixlAgentConfig cfg(false);
    nixlAgent agent("proxy_params_match", std::move(cfg));

    nixl_b_params_t params{{"device_proxy", "true"},
                           {"proxy_channel_count", "2"},
                           {"proxy_max_peers", "4"},
                           {"num_workers", "8"}};
    nixlBackendH *backend = nullptr;
    EXPECT_EQ(agent.createBackend("UCX", params, backend), NIXL_SUCCESS);
}

TEST(ProxyBackendParamsTest, NumWorkersMismatchRejected) {
    nixlAgentConfig cfg(false);
    nixlAgent agent("proxy_params_mismatch", std::move(cfg));

    nixl_b_params_t params{{"device_proxy", "true"},
                           {"proxy_channel_count", "2"},
                           {"proxy_max_peers", "4"},
                           {"num_workers", "3"}};
    nixlBackendH *backend = nullptr;
    EXPECT_NE(agent.createBackend("UCX", params, backend), NIXL_SUCCESS);
}

/**
 * Memviews served by the engine-owned proxy: prep must return a PROXY-tagged
 * wrapped handle and release must unwind the registry entry - all with zero
 * proxy configuration on the agent.
 */
TEST(ProxyBackendParamsTest, MemViewRoundTripViaParams) {
    nixlAgentConfig cfg(false);
    nixlAgent agent("proxy_params_memview", std::move(cfg));

    nixl_b_params_t params{{"device_proxy", "true"},
                           {"proxy_channel_count", "2"},
                           {"proxy_max_peers", "4"}};
    nixlBackendH *backend = nullptr;
    ASSERT_EQ(agent.createBackend("UCX", params, backend), NIXL_SUCCESS);

    std::vector<uint8_t> buffer(4096);
    const auto addr = reinterpret_cast<uintptr_t>(buffer.data());

    nixl_reg_dlist_t reg_list(DRAM_SEG);
    reg_list.addDesc(nixlBlobDesc(addr, buffer.size(), 0, ""));
    ASSERT_EQ(agent.registerMem(reg_list), NIXL_SUCCESS);

    nixl_local_dlist_t descs(DRAM_SEG);
    descs.addDesc(nixlBasicDesc(addr, buffer.size(), 0));

    nixlMemViewH mvh = nullptr;
    ASSERT_EQ(agent.prepMemView(descs, mvh), NIXL_SUCCESS);
    ASSERT_NE(mvh, nullptr);

    agent.releaseMemView(mvh);
    EXPECT_EQ(agent.deregisterMem(reg_list), NIXL_SUCCESS);
}

TEST(ProxyBackendParamsTest, UnknownProxyParamRejected) {
    nixlAgentConfig cfg(false);
    nixlAgent agent("proxy_params_unknown", std::move(cfg));

    nixl_b_params_t params{{"device_proxy", "true"}, {"proxy_worker_count", "2"}};
    nixlBackendH *backend = nullptr;
    EXPECT_NE(agent.createBackend("UCX", params, backend), NIXL_SUCCESS);
}

} // namespace
