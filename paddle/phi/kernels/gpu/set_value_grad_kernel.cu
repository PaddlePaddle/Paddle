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

#include "paddle/phi/kernels/set_value_grad_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/complex.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/impl/set_value_grad_kernel_impl.h"
#include "paddle/phi/kernels/impl/set_value_kernel_impl.h"
#include "paddle/phi/kernels/reduce_sum_kernel.h"
#include "paddle/phi/kernels/strided_slice_kernel.h"

namespace phi {

phi::IntArray ComputeReduceDims(const DDim& first, const DDim& second) {
  std::vector<int> reduce_axes;
  int i = 0, j = 0;
  int n1 = first.size(), n2 = second.size();

  while (i < n1 && j < n2) {
    if (first[i] == second[j]) {
      i++;
      j++;
    } else {
      reduce_axes.push_back(i);
      i++;
    }
  }

  while (i < n1) {
    reduce_axes.push_back(i);
    i++;
  }

  if (j != n2) {
    PADDLE_THROW(
        errors::InvalidArgument("The shape %d must can be reduce be shape %d.",
                                second.to_str(),
                                first.to_str()));
  }

  return IntArray(reduce_axes);
}

template <typename T, typename Context>
void SetValueGradKernelV2(const Context& dev_ctx,
                          const DenseTensor& out_grad,
                          const IntArray& starts,
                          const IntArray& ends,
                          const IntArray& steps,
                          const std::vector<int64_t>& axes,
                          const std::vector<int64_t>& decrease_axes,
                          const std::vector<int64_t>& none_axes,
                          DenseTensor* x_grad,
                          DenseTensor* value_grad) {
  const int rank = out_grad.dims().size();
  std::vector<int64_t> starts_local = starts.GetData();
  std::vector<int64_t> ends_local = ends.GetData();
  std::vector<int64_t> steps_local = steps.GetData();

  bool ellipsis_flag = true;
  for (size_t i = 0; i < axes.size(); i++) {
    auto idx = axes[i];
    if (!(starts_local[i] == 0 && ends_local[i] == out_grad.dims()[idx] &&
          steps_local[i] == 1)) {
      ellipsis_flag = false;
    }
  }

  if (ellipsis_flag) {
    if (x_grad) {
      FullKernel<T, Context>(dev_ctx,
                             common::vectorize(x_grad->dims()),
                             Scalar(0),
                             x_grad->dtype(),
                             x_grad);
    }
    if (value_grad) {
      if (value_grad->numel() == out_grad.numel()) {
        if (value_grad->dims() != out_grad.dims()) {
          DenseTensor out_grad_temp;
          ShareDataKernel<T, Context>(dev_ctx, out_grad, &out_grad_temp);
          ReshapeKernel<Context>(dev_ctx,
                                 out_grad_temp,
                                 IntArray(vectorize(value_grad->dims())),
                                 &out_grad_temp);
          Copy(dev_ctx, out_grad_temp, dev_ctx.GetPlace(), false, value_grad);
        } else {
          Copy(dev_ctx, out_grad, dev_ctx.GetPlace(), false, value_grad);
        }
      } else {
        SumKernel<T, Context>(dev_ctx,
                              out_grad,
                              IntArray(vectorize(value_grad->dims())),
                              out_grad.dtype(),
                              false,
                              value_grad);
      }
    }
    return;
  }

  if (x_grad) {
    // Set gradient of `Input`
    Copy(dev_ctx, out_grad, dev_ctx.GetPlace(), false, x_grad);
    SetValueKernelV2<T, Context>(dev_ctx,
                                 *x_grad,
                                 starts,
                                 ends,
                                 steps,
                                 axes,
                                 decrease_axes,
                                 none_axes,
                                 {1},
                                 std::vector<Scalar>({Scalar(0)}),
                                 x_grad);
  }

  if (value_grad) {
    DenseTensor value_grad_orig;
    std::vector<int> infer_flags(axes.size(), 1);
    std::vector<int> axes_int32(axes.begin(), axes.end());
    std::vector<int> decrease_axes_int32(decrease_axes.begin(),
                                         decrease_axes.end());
    StridedSliceRawKernel<T, Context>(dev_ctx,
                                      out_grad,
                                      axes_int32,
                                      starts,
                                      ends,
                                      steps,
                                      infer_flags,
                                      decrease_axes_int32,
                                      &value_grad_orig);
    if (value_grad_orig.dims() == value_grad->dims()) {
      value_grad = &value_grad_orig;
    } else {
      auto reduce_dims =
          ComputeReduceDims(value_grad_orig.dims(), value_grad->dims());
      SumKernel<T, Context>(dev_ctx,
                            value_grad_orig,
                            reduce_dims,
                            value_grad->dtype(),
                            false,
                            value_grad);
    }
  }
}

template <typename T, typename Context>
void SetValueWithScalarGradKernelV2(const Context& dev_ctx,
                                    const DenseTensor& out_grad,
                                    const IntArray& starts,
                                    const IntArray& ends,
                                    const IntArray& steps,
                                    const std::vector<int64_t>& axes,
                                    const std::vector<int64_t>& decrease_axes,
                                    const std::vector<int64_t>& none_axes,
                                    DenseTensor* x_grad) {
  SetValueGradKernelV2<T, Context>(dev_ctx,
                                   out_grad,
                                   starts,
                                   ends,
                                   steps,
                                   axes,
                                   decrease_axes,
                                   none_axes,
                                   x_grad,
                                   nullptr);
}

}  // namespace phi
PD_REGISTER_KERNEL(set_value_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::SetValueGradKernelV2,
                   float,
                   double,
                   int,
                   int64_t,
                   bool,
                   int16_t,
                   uint8_t,
                   int8_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}

PD_REGISTER_KERNEL(set_value_with_scalar_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::SetValueWithScalarGradKernelV2,
                   float,
                   double,
                   int,
                   int64_t,
                   bool,
                   int16_t,
                   uint8_t,
                   int8_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
