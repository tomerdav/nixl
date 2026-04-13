# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

if not cuda_dep.found()
    warning('CUDA not found, skipping device pingpong benchmark')
    subdir_done()
endif

if not ucx_gpu_device_api_available
    warning('UCX GPU device API not available, skipping device pingpong benchmark')
    subdir_done()
endif

nixl_dep = declare_dependency(
    link_with: nixl_lib,
    include_directories: nixl_inc_dirs,
)
nixl_lib_dir = join_paths(meson.project_build_root(), 'src', 'core')

pingpong_sources = [
    'bench_main.cpp',
    'bench_host.cpp',
    'bench_kernel.cu',
]

pingpong_inc_dirs = [
    nixl_inc_dirs,
    nixl_gpu_inc_dirs,
    include_directories('.'),
]

pingpong_cpp_args = [
    '-DHAVE_CUDA',
    '-Wno-deprecated-declarations',
]

pingpong_cuda_args = [
    '-DHAVE_CUDA',
    '--expt-relaxed-constexpr',
    '-Xcompiler', '-Wno-deprecated-declarations',
    '-Xcompiler', '-Wno-unused-variable',
]

# NVCC 12.9 workaround (same bug as in ep/meson.build)
nvcc = meson.get_compiler('cuda')
if nvcc.version().version_compare('>=12.9') and nvcc.version().version_compare('<12.10')
    pingpong_cpp_args  += ['-D_LIBCUDACXX_ATOMIC_UNSAFE_AUTOMATIC_STORAGE']
    pingpong_cuda_args += ['-D_LIBCUDACXX_ATOMIC_UNSAFE_AUTOMATIC_STORAGE']
endif

executable('nixl_device_pingpong',
    pingpong_sources,
    dependencies: [nixl_dep, cuda_dep] + (ucx_dep.found() ? [ucx_dep] : []),
    include_directories: pingpong_inc_dirs,
    cpp_args: pingpong_cpp_args,
    cuda_args: pingpong_cuda_args,
    build_rpath: nixl_lib_dir,
    install: false,
)
