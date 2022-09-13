# Copyright (c) 2017 - 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#
# Sources for CUTLASS Profiler Tool
#

set(CUTLASS_TOOLS_PROFILER_SOURCES
  src/main.cpp
  src/cutlass_profiler.cu
  src/options.cu
  src/performance_report.cpp
  src/enumerated_types.cpp
  src/gpu_timer.cpp
  src/device_allocation.cu
  src/device_context.cu
  src/cublas_helpers.cu             
  src/cudnn_helpers.cpp                   
  src/problem_space.cpp
  src/operation_profiler.cu
  src/gemm_operation_profiler.cu
  src/rank_k_operation_profiler.cu
  src/rank_2k_operation_profiler.cu
  src/trmm_operation_profiler.cu
  src/symm_operation_profiler.cu
  src/conv2d_operation_profiler.cu          
  src/conv3d_operation_profiler.cu          
  src/sparse_gemm_operation_profiler.cu
)

#
# Build target
#

cutlass_add_executable(
  cutlass_profiler
  ${CUTLASS_TOOLS_PROFILER_SOURCES}
)
add_executable(nvidia::cutlass::profiler ALIAS cutlass_profiler)
set_target_properties(cutlass_profiler PROPERTIES EXPORT_NAME profiler)

#
# Include paths
#

target_include_directories(
  cutlass_profiler
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/src
  )

#
# Library dependencies
#

target_link_libraries(
  cutlass_profiler
  PRIVATE 
  cutlass_lib
  cutlass_tools_util_includes
  $<$<BOOL:${CUTLASS_ENABLE_CUBLAS}>:nvidia::cublas>
  $<$<BOOL:${CUTLASS_ENABLE_CUDNN}>:nvidia::cudnn>
  cudart
  )

install(
  TARGETS cutlass_profiler
  EXPORT NvidiaCutlass
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  )

set(CUTLASS_PROFILER_TEST_COMMAND_OPTIONS_GEMM   --operation=Gemm       --providers=cutlass --verification-providers=cublas,device      --junit-output=test_cutlass_profiler_gemm)
set(CUTLASS_PROFILER_TEST_COMMAND_OPTIONS_CONV2D --operation=Conv2d     --providers=cutlass --verification-providers=cudnn,device       --junit-output=test_cutlass_profiler_conv2d)
set(CUTLASS_PROFILER_TEST_COMMAND_OPTIONS_CONV3D --operation=Conv3d     --providers=cutlass --verification-providers=cudnn,device,host  --junit-output=test_cutlass_profiler_conv3d)
set(CUTLASS_PROFILER_TEST_COMMAND_OPTIONS_SPGEMM --operation=SparseGemm --providers=cutlass --verification-providers=cublas,device,host --junit-output=test_cutlass_profiler_spgemm)
set(CUTLASS_PROFILER_TEST_COMMAND_OPTIONS_RANK_K   --operation=RankK       --providers=cutlass --verification-providers=cublas        --junit-output=test_cutlass_profiler_rank_k)
set(CUTLASS_PROFILER_TEST_COMMAND_OPTIONS_RANK_2K   --operation=Rank2K       --providers=cutlass --verification-providers=cublas        --junit-output=test_cutlass_profiler_rank_2k)
set(CUTLASS_PROFILER_TEST_COMMAND_OPTIONS_TRMM   --operation=Trmm       --providers=cutlass --verification-providers=device,host        --junit-output=test_cutlass_profiler_trmm)
set(CUTLASS_PROFILER_TEST_COMMAND_OPTIONS_SYMM   --operation=Symm       --providers=cutlass --verification-providers=cublas,host        --junit-output=test_cutlass_profiler_symm)

cutlass_add_executable_tests(
  test_profiler cutlass_profiler
  DEPENDEES test_all
  TEST_COMMAND_OPTIONS 
    CUTLASS_PROFILER_TEST_COMMAND_OPTIONS_GEMM
    CUTLASS_PROFILER_TEST_COMMAND_OPTIONS_CONV2D
    CUTLASS_PROFILER_TEST_COMMAND_OPTIONS_CONV3D
    CUTLASS_PROFILER_TEST_COMMAND_OPTIONS_SPGEMM
    CUTLASS_PROFILER_TEST_COMMAND_OPTIONS_RANK_K
    CUTLASS_PROFILER_TEST_COMMAND_OPTIONS_RANK_2K
    CUTLASS_PROFILER_TEST_COMMAND_OPTIONS_TRMM
    CUTLASS_PROFILER_TEST_COMMAND_OPTIONS_SYMM
  DISABLE_EXECUTABLE_INSTALL_RULE
  )
