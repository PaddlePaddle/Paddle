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
#include <algorithm>
#include <mutex>
#include <unordered_map>
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"
#include "paddle/phi/kernels/funcs/reduce_function.h"
#include "paddle/phi/kernels/impl/matmul_grad_kernel_impl.h"
#include "paddle/phi/kernels/reduce_sum_kernel.h"

#include "paddle/common/flags.h"
#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/dense_tensor.h"

namespace phi {

template <typename T, typename Context>
void LinearV2GradKernel(const Context& dev_ctx,
                        const DenseTensor& input,
                        const DenseTensor& weight,
                        const DenseTensor& bias,
                        const DenseTensor& out_grad,
                        const bool transpose_weight,
                        DenseTensor* input_grad,
                        DenseTensor* weight_grad,
                        DenseTensor* bias_grad) {
  phi::MatmulGradKernel<T, Context>(dev_ctx,
                                    input,
                                    weight,
                                    out_grad,
                                    false,
                                    transpose_weight,
                                    input_grad,
                                    weight_grad);

  if (bias_grad && bias.numel() != 0) {
    if (out_grad.numel() != bias_grad->numel()) {
      dev_ctx.template Alloc<T>(bias_grad);
      std::vector<int> reduce_dims =
          funcs::GetReduceDim(bias.dims(), out_grad.dims(), -1);
      phi::SumKernel<T, Context>(
          dev_ctx, out_grad, reduce_dims, out_grad.dtype(), false, bias_grad);
      bias_grad->Resize(bias.dims());
    } else {
      phi::Copy(dev_ctx, out_grad, dev_ctx.GetPlace(), false, bias_grad);
      bias_grad->Resize(bias.dims());
    }
  }
}

}  // namespace phi
