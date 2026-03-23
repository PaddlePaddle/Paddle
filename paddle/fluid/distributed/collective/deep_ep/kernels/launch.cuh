// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// The file has been adapted from DeepSeek DeepEP project
// Copyright (c) 2025 DeepSeek
// Licensed under the MIT License -
// https://github.com/deepseek-ai/DeepEP/blob/main/LICENSE

#pragma once

#include "paddle/fluid/distributed/collective/deep_ep/kernels/configs.cuh"

// Include dispatch macros from separate files to reduce compilation memory
#include "paddle/fluid/distributed/collective/deep_ep/kernels/dispatch_experts.cuh"
#include "paddle/fluid/distributed/collective/deep_ep/kernels/dispatch_hidden.cuh"
#include "paddle/fluid/distributed/collective/deep_ep/kernels/dispatch_misc.cuh"
#include "paddle/fluid/distributed/collective/deep_ep/kernels/dispatch_ranks.cuh"

#ifndef SETUP_LAUNCH_CONFIG
#define SETUP_LAUNCH_CONFIG(num_sms, num_threads, stream)                     \
  cudaLaunchConfig_t cfg = {(num_sms), (num_threads), 0, stream, nullptr, 0}; \
  cudaLaunchAttribute attr[2];                                                \
  attr[0].id = cudaLaunchAttributeCooperative;                                \
  attr[0].val.cooperative = 1;                                                \
  attr[1].id = cudaLaunchAttributeClusterDimension;                           \
  attr[1].val.clusterDim.x = (num_sms % 2 == 0 ? 2 : 1);                      \
  attr[1].val.clusterDim.y = 1;                                               \
  attr[1].val.clusterDim.z = 1;                                               \
  cfg.attrs = attr;                                                           \
  cfg.numAttrs = 2
#endif

#ifndef LAUNCH_KERNEL
#define LAUNCH_KERNEL(config, kernel, ...) \
  CUDA_CHECK(cudaLaunchKernelEx(config, kernel, ##__VA_ARGS__))
#endif
