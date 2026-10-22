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
#include "device/device_ops.h"

#include <cuda_runtime.h>
#include <linux/mempolicy.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <cctype>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "common/nixl_log.h"
#include "device/device_ops_cuda_gdrcopy.h"

namespace {

std::string
cudaFailureMsg(const char *operation, cudaError_t error) {
    return std::string(operation) + " failed: " + cudaGetErrorString(error);
}

nixl_status_t
cudaStatus(cudaError_t error, const char *operation) {
    if (error != cudaSuccess) {
        NIXL_ERROR << cudaFailureMsg(operation, error);
        return NIXL_ERR_BACKEND;
    }
    return NIXL_SUCCESS;
}

size_t
pageAligned(size_t size) noexcept {
    const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    return (size + page - 1) / page * page;
}

bool
preferNode(void *addr, size_t length, int node) noexcept {
    constexpr size_t bits = sizeof(unsigned long) * 8;
    std::vector<unsigned long> mask(node / bits + 1);
    mask[node / bits] = 1UL << (node % bits);
    const unsigned long max_node = mask.size() * bits + 1;
    return syscall(SYS_mbind, addr, length, MPOL_PREFERRED, mask.data(), max_node, 0) == 0;
}

/** Zeroed, registered pages preferring `node`, or nullptr; `length` is what to unmap. */
void *
mapOnNode(size_t size, int node, size_t &length) noexcept {
    length = pageAligned(size);
    void *pages = mmap(nullptr, length, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (pages == MAP_FAILED) {
        return nullptr;
    }
    if (!preferNode(pages, length, node)) {
        NIXL_DEBUG << "mbind to NUMA node " << node << " failed: " << std::strerror(errno);
    }
    std::memset(pages, 0, length);
    if (cudaHostRegister(pages, length, cudaHostRegisterMapped | cudaHostRegisterPortable) !=
        cudaSuccess) {
        cudaGetLastError();
        munmap(pages, length);
        return nullptr;
    }
    return pages;
}

/** The device's PCI numa_node from sysfs, or -1. */
int
pciNumaNode(int device_id) noexcept {
    char bus_id[32];
    if (cudaDeviceGetPCIBusId(bus_id, sizeof(bus_id), device_id) != cudaSuccess) {
        cudaGetLastError();
        return -1;
    }
    std::string bdf(bus_id);
    for (char &c : bdf) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    if (bdf.size() > 12) {
        bdf = bdf.substr(bdf.size() - 12);
    }
    std::ifstream file("/sys/bus/pci/devices/" + bdf + "/numa_node");
    int numa_node = -1;
    return (file >> numa_node) ? numa_node : -1;
}

class cudaDeviceOps final : public nixl::deviceOps {
public:
    nixl_status_t
    doAllocDeviceMem(void *&ptr, size_t size) noexcept override {
        void *allocation;
        const nixl_status_t status = cudaStatus(cudaMalloc(&allocation, size), "cudaMalloc");
        if (status != NIXL_SUCCESS) {
            return status;
        }
        ptr = allocation;
        return NIXL_SUCCESS;
    }

    void
    doFreeDeviceMem(void *ptr) noexcept override {
        // cudaFree recovers the owning device and accepts nullptr.
        const cudaError_t error = cudaFree(ptr);
        if (error != cudaSuccess) {
            NIXL_WARN << cudaFailureMsg("cudaFree", error);
        }
    }

    nixl_status_t
    doAllocMappedHostMem(void *&host_ptr,
                         void *&dev_ptr,
                         size_t size,
                         int numa_node,
                         size_t &length) noexcept override {
        size_t mapped_length = 0;
        // Own pages: cudaHostAlloc suballocates from chunks placed by their first user.
        void *allocation = numa_node >= 0 ? mapOnNode(size, numa_node, mapped_length) : nullptr;
        if (allocation == nullptr) {
            if (numa_node >= 0) {
                NIXL_INFO << "Mapped host memory not placed on NUMA node " << numa_node;
            }
            mapped_length = 0;
            // Mapped guarantees cudaHostGetDevicePointer; portable permits use from other GPUs.
            const nixl_status_t status = cudaStatus(
                cudaHostAlloc(&allocation, size, cudaHostAllocMapped | cudaHostAllocPortable),
                "cudaHostAlloc");
            if (status != NIXL_SUCCESS) {
                return status;
            }
        }
        void *device_alias;
        const nixl_status_t status = cudaStatus(
            cudaHostGetDevicePointer(&device_alias, allocation, 0), "cudaHostGetDevicePointer");
        if (status != NIXL_SUCCESS) {
            doFreeMappedHostMem(allocation, mapped_length);
            return status;
        }
        host_ptr = allocation;
        dev_ptr = device_alias;
        length = mapped_length;
        return NIXL_SUCCESS;
    }

    void
    doFreeMappedHostMem(void *host_ptr, size_t length) noexcept override {
        if (length == 0) {
            // cudaFreeHost is a no-op on nullptr, matching doFreeDeviceMem.
            const cudaError_t error = cudaFreeHost(host_ptr);
            if (error != cudaSuccess) {
                NIXL_WARN << cudaFailureMsg("cudaFreeHost", error);
            }
            return;
        }
        const cudaError_t error = cudaHostUnregister(host_ptr);
        if (error != cudaSuccess) {
            NIXL_WARN << cudaFailureMsg("cudaHostUnregister", error) << "; leaking the pages";
            return;
        }
        if (munmap(host_ptr, length) != 0) {
            NIXL_WARN << "munmap failed: " << std::strerror(errno);
        }
    }

    nixl_status_t
    copy(void *dst, const void *src, size_t size, copyDirection direction) noexcept override {
        if (size == 0 || dst == nullptr || src == nullptr) {
            NIXL_ERROR << "Invalid device copy: dst=" << dst << " src=" << src << " size=" << size;
            return NIXL_ERR_INVALID_PARAM;
        }
        cudaMemcpyKind kind;
        switch (direction) {
        case copyDirection::HostToDevice:
            kind = cudaMemcpyHostToDevice;
            break;
        case copyDirection::DeviceToHost:
            kind = cudaMemcpyDeviceToHost;
            break;
        default:
            NIXL_ERROR << "Invalid device copy direction: " << static_cast<int>(direction);
            return NIXL_ERR_INVALID_PARAM;
        }
        return cudaStatus(cudaMemcpy(dst, src, size, kind),
                          direction == copyDirection::HostToDevice ? "cudaMemcpy host to device" :
                                                                     "cudaMemcpy device to host");
    }

    nixl_status_t
    memsetDeviceMem(void *ptr, int value, size_t size) noexcept override {
        if (size == 0 || ptr == nullptr) {
            NIXL_ERROR << "Invalid device memset: ptr=" << ptr << " size=" << size;
            return NIXL_ERR_INVALID_PARAM;
        }
        return cudaStatus(cudaMemset(ptr, value, size), "cudaMemset");
    }

    nixl_status_t
    synchronize() noexcept override {
        return cudaStatus(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    }

    nixl_status_t
    getActiveDevice(int &device_id) noexcept override {
        return cudaStatus(cudaGetDevice(&device_id), "cudaGetDevice");
    }

    nixl_status_t
    setActiveDevice(int device_id) noexcept override {
        return cudaStatus(cudaSetDevice(device_id), "cudaSetDevice");
    }

    nixl_status_t
    getHostNumaNode(int device_id, int &node) noexcept override {
        int numa_node = -1;
#if CUDART_VERSION >= 12020
        if (cudaDeviceGetAttribute(&numa_node, cudaDevAttrHostNumaId, device_id) != cudaSuccess) {
            cudaGetLastError();
            numa_node = -1;
        }
#endif
        if (numa_node < 0) {
            numa_node = pciNumaNode(device_id);
        }
        if (numa_node < 0) {
            return NIXL_ERR_NOT_SUPPORTED;
        }
        node = numa_node;
        return NIXL_SUCCESS;
    }

    nixl_status_t
    doAllocHostPublishedMem(size_t size,
                            std::unique_ptr<nixl::hostPublishedDeviceMem> &out) noexcept override {
        return nixl::allocGdrCopyPublishedMem(size, out);
    }
};

} // namespace

extern "C" NIXL_DEVICE_OPS_EXPORT nixl::deviceOps *
nixlCreateCudaDeviceOps() noexcept {
    int device_count = 0;
    const cudaError_t error = cudaGetDeviceCount(&device_count);
    if (error == cudaErrorNoDevice) {
        NIXL_INFO << "No CUDA-capable GPU is available";
        return nullptr;
    }
    if (cudaStatus(error, "cudaGetDeviceCount") != NIXL_SUCCESS) {
        return nullptr;
    }
    if (device_count == 0) {
        NIXL_INFO << "CUDA reported zero available GPUs";
        return nullptr;
    }

    static cudaDeviceOps ops;
    return &ops;
}
