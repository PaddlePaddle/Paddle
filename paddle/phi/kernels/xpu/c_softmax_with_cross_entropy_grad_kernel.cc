// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/axis_utils.h"
#include "paddle/phi/kernels/funcs/cross_entropy.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/softmax.h"
#include "paddle/phi/kernels/funcs/softmax_impl.h"
#include "paddle/phi/kernels/xpu/elementwise.h"
#include "paddle/phi/kernels/xpu/reduce.h"
#include "paddle/utils/string/string_helper.h"

namespace phi {

template <typename T, typename Context>
void CSoftmaxWithCrossEntropyGradKernel(const Context& dev_ctx,
                                        const DenseTensor& softmax_in,
                                        const DenseTensor& label_in,
                                        const DenseTensor& loss_grad_in,
                                        int64_t ignore_index,
                                        int rank,
                                        int nranks,
                                        DenseTensor* logits_grad) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  const phi::DenseTensor* labels = &label_in;
  const phi::DenseTensor* loss_grad = &loss_grad_in;
  phi::DenseTensor* logit_grad = logits_grad;
  const phi::DenseTensor* softmax = &softmax_in;

  if (logit_grad != softmax) {
    phi::Copy(dev_ctx, *softmax, dev_ctx.GetPlace(), false, logit_grad);
  }
  const auto softmax_dims = softmax->dims();
  const int axis = softmax_dims.size() - 1;
  const int64_t N = phi::funcs::SizeToAxis(axis, softmax_dims);
  const int64_t D = phi::funcs::SizeFromAxis(axis, softmax_dims);

  const int64_t start_index = rank * D;
  const int64_t end_index = start_index + D;
  const auto& label_type = labels->dtype();

  int ret = 0;
  if (label_type == phi::DataType::INT32) {
    ret = xpu::mask_label_by_index_grad<XPUType, int32_t>(
        dev_ctx.x_context(),
        reinterpret_cast<const XPUType*>(loss_grad->data<T>()),
        labels->data<int32_t>(),
        reinterpret_cast<XPUType*>(logit_grad->data<T>()),
        start_index,
        end_index,
        N,
        D,
        ignore_index);
  } else if (label_type == phi::DataType::INT64) {
    ret = xpu::mask_label_by_index_grad<XPUType, int64_t>(
        dev_ctx.x_context(),
        reinterpret_cast<const XPUType*>(loss_grad->data<T>()),
        labels->data<int64_t>(),
        reinterpret_cast<XPUType*>(logit_grad->data<T>()),
        start_index,
        end_index,
        N,
        D,
        ignore_index);
  }
  PADDLE_ENFORCE_XDNN_SUCCESS(ret, "mask_label_by_index_grad");
}

}  // namespace phi

PD_REGISTER_KERNEL(c_softmax_with_cross_entropy_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::CSoftmaxWithCrossEntropyGradKernel,
                   float,
                   phi::dtype::bfloat16) {}
