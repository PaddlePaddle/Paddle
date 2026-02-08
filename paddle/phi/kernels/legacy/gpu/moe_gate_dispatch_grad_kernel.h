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

#pragma once

#include <vector>
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/contiguous_kernel.h"
#include "paddle/phi/kernels/legacy/gpu/moe_fuse_bwd_op.h"
#include "paddle/phi/kernels/transpose_kernel.h"

namespace phi {

template <typename T, typename Context>
void MoeGateDispatchGradKernel(const Context& dev_ctx,
                               const DenseTensor& combine_weights,
                               const DenseTensor& scatter_index,
                               const DenseTensor& expert_id,
                               const DenseTensor& y_grad,
                               const DenseTensor& combine_weights_grad,
                               const int64_t k,
                               const int64_t capacity,
                               const bool use_pad,
                               DenseTensor* x_grad,
                               DenseTensor* gate_logits_grad);

}  // namespace phi
