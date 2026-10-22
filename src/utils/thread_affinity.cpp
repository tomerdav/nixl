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

#include "thread_affinity.h"

#include <linux/mempolicy.h>
#include <pthread.h>
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace nixl {

nixl_status_t
bindThreadToNumaNode(int node) noexcept {
    std::ifstream file("/sys/devices/system/node/node" + std::to_string(node) + "/cpulist");
    std::string cpulist;
    cpu_set_t allowed;
    if (node < 0 || !std::getline(file, cpulist) ||
        pthread_getaffinity_np(pthread_self(), sizeof(allowed), &allowed) != 0) {
        return NIXL_ERR_NOT_FOUND;
    }

    cpu_set_t cpus;
    CPU_ZERO(&cpus);
    std::istringstream ranges(cpulist);
    std::string range;
    while (std::getline(ranges, range, ',')) {
        unsigned first;
        unsigned last;
        const int fields = std::sscanf(range.c_str(), "%u-%u", &first, &last);
        if (fields == 1) {
            last = first;
        } else if (fields != 2) {
            continue;
        }
        for (unsigned cpu = first; cpu <= last && cpu < CPU_SETSIZE; ++cpu) {
            if (CPU_ISSET(cpu, &allowed)) {
                CPU_SET(cpu, &cpus);
            }
        }
    }
    if (CPU_COUNT(&cpus) == 0) {
        return NIXL_ERR_NOT_FOUND;
    }
    if (pthread_setaffinity_np(pthread_self(), sizeof(cpus), &cpus) != 0) {
        return NIXL_ERR_BACKEND;
    }

    constexpr unsigned bits = sizeof(unsigned long) * 8;
    std::vector<unsigned long> nodemask(node / bits + 1);
    nodemask[node / bits] = 1UL << (node % bits);
    const unsigned long max_node = nodemask.size() * bits + 1;
    if (syscall(SYS_set_mempolicy, MPOL_PREFERRED, nodemask.data(), max_node) != 0) {
        return NIXL_ERR_BACKEND;
    }
    return NIXL_SUCCESS;
}

} // namespace nixl
