// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/activation_kernel.h"
#include "paddle/phi/kernels/elementwise_divide_kernel.h"
#include "paddle/phi/kernels/elementwise_subtract_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/reduce_sum_kernel.h"
#include "paddle/phi/kernels/scale_kernel.h"

namespace phi {

// self_grad = sum_to(out_grad / (1 + exp(other - self)), self.shape)
//
// Mirrors torch's autograd formula op by op (sub, exp, add-scalar, div, then
// a single sum over the broadcast dims), each step rounded to T, so the result
// is bitwise identical to torch.
template <typename T, typename Context>
void LogAddExpGradOneSide(const Context& dev_ctx,
                          const DenseTensor& self,
                          const DenseTensor& other,
                          const DenseTensor& out_grad,
                          DenseTensor* self_grad) {
  if (self_grad == nullptr) {
    return;
  }
  if (out_grad.numel() == 0) {
    if (self_grad->numel() == 0) {
      dev_ctx.template Alloc<T>(self_grad);
    } else {
      phi::Full<T, Context>(dev_ctx, self_grad->dims(), 0, self_grad);
    }
    return;
  }

  // Same reduce dims as torch's sum_to_size: all leading dims, plus the dims
  // where self has size 1 but out_grad does not.
  const auto& out_dims = out_grad.dims();
  const DDim self_dims = self_grad->dims();
  const int lead = out_dims.size() - self_dims.size();
  std::vector<int64_t> reduce_axes;
  std::vector<int64_t> keep_shape(out_dims.size());
  for (int i = 0; i < out_dims.size(); ++i) {
    const bool reduced =
        i < lead || (self_dims[i - lead] == 1 && out_dims[i] != 1);
    if (reduced) {
      reduce_axes.push_back(i);
    }
    keep_shape[i] = reduced ? 1 : out_dims[i];
  }

  // All four steps are elementwise over out_dims, so each one overwrites its
  // input in place: computed directly in self_grad when nothing is reduced,
  // otherwise in a single out-size buffer that is then summed into self_grad.
  DenseTensor buf;
  DenseTensor* work = reduce_axes.empty() ? self_grad : &buf;
  work->Resize(out_dims);
  SubtractKernel<T, Context>(dev_ctx, other, self, work);
  ExpKernel<T, Context>(dev_ctx, *work, work);
  ScaleKernel<T, Context>(dev_ctx, *work, 1.0f, 1.0f, true, work);
  DivideKernel<T, Context>(dev_ctx, out_grad, *work, work);
  if (reduce_axes.empty()) {
    return;
  }
  self_grad->Resize(common::make_ddim(keep_shape));
  SumKernel<T, Context>(dev_ctx,
                        buf,
                        IntArray(reduce_axes),
                        DataType::UNDEFINED,
                        true,
                        self_grad);
  self_grad->Resize(self_dims);
}

template <typename T, typename Context>
void LogAddExpGradKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         const DenseTensor& y,
                         const DenseTensor& out_grad,
                         DenseTensor* x_grad,
                         DenseTensor* y_grad) {
  LogAddExpGradOneSide<T, Context>(dev_ctx, x, y, out_grad, x_grad);
  LogAddExpGradOneSide<T, Context>(dev_ctx, y, x, out_grad, y_grad);
}

}  // namespace phi
