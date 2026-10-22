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

#ifndef NIXL_SRC_UTILS_THREAD_AFFINITY_H
#define NIXL_SRC_UTILS_THREAD_AFFINITY_H

#include "nixl_types.h"

namespace nixl {

/**
 * @brief Restrict the calling thread to the CPUs of NUMA node `node` within its current
 *        affinity, and prefer that node for its memory. Threads it creates inherit both.
 * @retval NIXL_ERR_NOT_FOUND No CPU of the node is in the thread's current affinity.
 * @retval NIXL_ERR_BACKEND A system call failed.
 */
[[nodiscard]] nixl_status_t
bindThreadToNumaNode(int node) noexcept;

} // namespace nixl

#endif // NIXL_SRC_UTILS_THREAD_AFFINITY_H
