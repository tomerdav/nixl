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

device_utils_lib = shared_library('device_utils',
           ['device_allocator.cpp', 'device_allocator.h', 'device_allocator_plugin.h', 'utility_services.h'],
           include_directories: [nixl_inc_dirs, utils_inc_dirs],
           dependencies: [nixl_common_dep],
           install: true)

device_utils_interface = declare_dependency(link_with: device_utils_lib)

if cuda_dep.found()
    device_allocator_cuda_plugin = shared_library('utility_device_allocator_cuda',
               'device_allocator_cuda.cpp',
               include_directories: [nixl_inc_dirs, utils_inc_dirs],
               dependencies: [cuda_dep, nixl_common_dep, device_utils_interface],
               install: true,
               install_dir: plugin_install_dir,
               install_rpath: '$ORIGIN/..')

    if get_option('buildtype') == 'debug'
        run_command('sh', '-c',
                    'echo "UTILITY_DEVICE_ALLOCATOR_CUDA=' + device_allocator_cuda_plugin.full_path() + '" >> ' + plugin_build_dir + '/pluginlist',
                    check: true)
    endif
endif

if rocm_dep.found()
    device_allocator_hip_source = hipify_src.process('device_allocator_cuda.cpp',
                                                     extra_args: ['-o'])
    device_allocator_hip_plugin = shared_library('utility_device_allocator_hip',
               device_allocator_hip_source,
               include_directories: [nixl_inc_dirs, utils_inc_dirs],
               dependencies: [rocm_dep, nixl_common_dep, device_utils_interface],
               cpp_args: ['-DNIXL_DEVICE_ALLOCATOR_HIP'],
               install: true,
               install_dir: plugin_install_dir,
               install_rpath: '$ORIGIN/..')

    if get_option('buildtype') == 'debug'
        run_command('sh', '-c',
                    'echo "UTILITY_DEVICE_ALLOCATOR_HIP=' + device_allocator_hip_plugin.full_path() + '" >> ' + plugin_build_dir + '/pluginlist',
                    check: true)
    endif
endif
