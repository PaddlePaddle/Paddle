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
#include "paddle/phi/infermeta/unary.h"
#include "paddle/phi/kernels/funcs/common_shape.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/impl/share_data_kernel_impl.h"
#include "paddle/phi/kernels/reduce_sum_kernel.h"
#include "paddle/phi/kernels/set_value_kernel.h"
#include "paddle/phi/kernels/shape_kernel.h"
#include "paddle/phi/kernels/strided_slice_kernel.h"

namespace phi {
template <typename T, typename Context>
void SetValueGradKernel(const Context& dev_ctx,
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
      dev_ctx.template Alloc<T>(x_grad);
      phi::funcs::set_constant(dev_ctx, x_grad, static_cast<float>(0.0));
    }
    if (value_grad) {
      if (value_grad->numel() == out_grad.numel()) {
        if (value_grad->dims() != out_grad.dims()) {
          DenseTensor out_grad_temp;
          ShareDataKernel<T, Context>(dev_ctx, out_grad, &out_grad_temp);
          out_grad_temp.Resize(value_grad->dims());
          Copy(dev_ctx, out_grad_temp, dev_ctx.GetPlace(), false, value_grad);
        } else {
          Copy(dev_ctx, out_grad, dev_ctx.GetPlace(), false, value_grad);
        }
      } else {
        auto reduce_dim = phi::funcs::GetReduceDims(out_grad, *value_grad);
        SumKernel<T, Context>(
            dev_ctx, out_grad, reduce_dim, out_grad.dtype(), false, value_grad);
      }
    }
    return;
  }

  if (x_grad) {
    Copy(dev_ctx, out_grad, dev_ctx.GetPlace(), false, x_grad);
    SetValueKernel<T, Context>(dev_ctx,
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
    MetaTensor meta_out(&value_grad_orig);
    MetaTensor meta_in(out_grad);
    std::vector<int> infer_flags(axes.size(), 1);
    std::vector<int> axes_int32(axes.begin(), axes.end());
    std::vector<int> decrease_axes_int32(decrease_axes.begin(),
                                         decrease_axes.end());
    phi::StridedSliceRawInferMeta(meta_in,
                                  axes_int32,
                                  starts,
                                  ends,
                                  steps,
                                  infer_flags,
                                  decrease_axes_int32,
                                  &meta_out,
                                  MetaConfig(true, false));
    if (value_grad_orig.dims() != value_grad->dims()) {
      StridedSliceRawKernel<T, Context>(dev_ctx,
                                        out_grad,
                                        axes_int32,
                                        starts,
                                        ends,
                                        steps,
                                        infer_flags,
                                        decrease_axes_int32,
                                        &value_grad_orig);

      if (value_grad->numel() == value_grad_orig.numel()) {
        value_grad_orig.Resize(value_grad->dims());
        Copy(dev_ctx, value_grad_orig, dev_ctx.GetPlace(), false, value_grad);
      } else {
        auto reduce_dim =
            phi::funcs::GetReduceDims(value_grad_orig, *value_grad);
        SumKernel<T, Context>(dev_ctx,
                              value_grad_orig,
                              reduce_dim,
                              value_grad->dtype(),
                              false,
                              value_grad);
      }
    } else {
      StridedSliceRawKernel<T, Context>(dev_ctx,
                                        out_grad,
                                        axes_int32,
                                        starts,
                                        ends,
                                        steps,
                                        infer_flags,
                                        decrease_axes_int32,
                                        value_grad);
      // 0-dim will change to 1 dim so we need to set meta
      value_grad->set_meta(value_grad_orig.meta());
    }
  }
}

template <typename T, typename Context>
void SetValueWithScalarGradKernel(const Context& dev_ctx,
                                  const DenseTensor& out_grad,
                                  const IntArray& starts,
                                  const IntArray& ends,
                                  const IntArray& steps,
                                  const std::vector<int64_t>& axes,
                                  const std::vector<int64_t>& decrease_axes,
                                  const std::vector<int64_t>& none_axes,
                                  DenseTensor* x_grad) {
  SetValueGradKernel<T, Context>(dev_ctx,
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
                   phi::SetValueGradKernel,
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
                   phi::SetValueWithScalarGradKernel,
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
