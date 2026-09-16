# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

mock_device_allocator_valid = shared_library('mock_device_allocator_valid',
    'mock_device_allocator_plugin.cpp',
    cpp_args: [
        '-DMOCK_DEVICE_ALLOCATOR_API_VERSION=1',
        '-DMOCK_DEVICE_ALLOCATOR_RUNTIME=nixlDeviceRuntime::CUDA',
        '-DMOCK_DEVICE_ALLOCATOR_PROBE_SUCCESS',
    ],
    include_directories: [nixl_inc_dirs, utils_inc_dirs],
    dependencies: [device_utils_interface],
)

mock_device_allocator_wrong_abi = shared_library('mock_device_allocator_wrong_abi',
    'mock_device_allocator_plugin.cpp',
    cpp_args: [
        '-DMOCK_DEVICE_ALLOCATOR_API_VERSION=999',
        '-DMOCK_DEVICE_ALLOCATOR_RUNTIME=nixlDeviceRuntime::CUDA',
    ],
    include_directories: [nixl_inc_dirs, utils_inc_dirs],
    dependencies: [device_utils_interface],
)

mock_device_allocator_wrong_runtime = shared_library('mock_device_allocator_wrong_runtime',
    'mock_device_allocator_plugin.cpp',
    cpp_args: [
        '-DMOCK_DEVICE_ALLOCATOR_API_VERSION=1',
        '-DMOCK_DEVICE_ALLOCATOR_RUNTIME=nixlDeviceRuntime::HIP',
    ],
    include_directories: [nixl_inc_dirs, utils_inc_dirs],
    dependencies: [device_utils_interface],
)

mock_device_allocator_failed_probe = shared_library('mock_device_allocator_failed_probe',
    'mock_device_allocator_plugin.cpp',
    cpp_args: [
        '-DMOCK_DEVICE_ALLOCATOR_API_VERSION=1',
        '-DMOCK_DEVICE_ALLOCATOR_RUNTIME=nixlDeviceRuntime::CUDA',
    ],
    include_directories: [nixl_inc_dirs, utils_inc_dirs],
    dependencies: [device_utils_interface],
)

device_allocator_unit_test_dep = declare_dependency(
    sources: ['device_allocator.cpp'],
    include_directories: [nixl_inc_dirs, utils_inc_dirs, gtest_inc_dirs],
    compile_args: [
        '-DMOCK_DEVICE_ALLOCATOR_VALID="' + mock_device_allocator_valid.full_path() + '"',
        '-DMOCK_DEVICE_ALLOCATOR_WRONG_ABI="' + mock_device_allocator_wrong_abi.full_path() + '"',
        '-DMOCK_DEVICE_ALLOCATOR_WRONG_RUNTIME="' + mock_device_allocator_wrong_runtime.full_path() + '"',
        '-DMOCK_DEVICE_ALLOCATOR_FAILED_PROBE="' + mock_device_allocator_failed_probe.full_path() + '"',
    ],
    dependencies: [nixl_dep, nixl_common_dep, device_utils_interface, gpu_dependencies],
)
