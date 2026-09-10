/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cstring>
#include <gtest/gtest.h>

#include "device/proxy/proxy_protocol.h"

TEST(ProxyProtocolTest, TokensAndOperandSurviveRecordCopy) {
    for (const auto opcode : {nixl_proxy_opcode_t::PUT, nixl_proxy_opcode_t::ATOMIC_ADD}) {
        nixlProxySubmission record{};
        record.opcode = opcode;
        record.src_view = 0xfedcba9876543210ULL;
        record.dst_view = 0x123456789abcdef0ULL;
        record.operand = 0xdeadbeef12345678ULL;
        nixlProxySubmission copy{};
        std::memcpy(&copy, &record, sizeof(record));
        EXPECT_EQ(copy.src_view, record.src_view);
        EXPECT_EQ(copy.dst_view, record.dst_view);
        EXPECT_EQ(copy.operand, record.operand);
        EXPECT_EQ(copy.opcode, opcode);
    }
}

TEST(ProxyProtocolTest, ViewHeaderAndTrailingPointers) {
    EXPECT_EQ(nixlProxyDeviceMemViewBytes(3), sizeof(nixlProxyDeviceMemView) + 3 * sizeof(void *));
    EXPECT_EQ(nixlProxyDeviceContextData{}.protocol_version, kProxyProtocolVersion);
}
