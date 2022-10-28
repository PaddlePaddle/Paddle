// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include "paddle/phi/kernels/conv_kernel.h"

#include "paddle/fluid/platform/cudnn_workspace_helper.h"
#include "paddle/fluid/platform/profiler.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cpu/conv_util.h"
#include "paddle/phi/kernels/funcs/batch_norm_utils.h"
#include "paddle/phi/kernels/funcs/padding.h"
#include "paddle/phi/kernels/impl/conv_cudnn_impl.h"

#include "paddle/phi/backends/dynload/cudnn_frontend.h"

namespace phi {

template <typename T, typename Context>
void ConvCudnnKernelImplV8(const Context& ctx,
                           const DenseTensor& input,
                           const DenseTensor& filter,
                           const std::vector<int>& strides,
                           const std::vector<int>& paddings_t,
                           const std::string& padding_algorithm,
                           int groups,
                           const std::vector<int>& dilations_t,
                           const std::string& data_format,
                           bool use_addto,
                           int workspace_size_MB,
                           bool exhaustive_search_t,
                           DenseTensor* output) {
  // TODO(tizheng): Implement CUDNNv8 kernel
  return;
}

}  // namespace phi
