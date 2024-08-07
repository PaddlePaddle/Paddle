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

#include "paddle/phi/kernels/rmsprop_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/impl/rmsprop_kernel_impl.h"

namespace phi {
template <typename T>
struct RmsFunctor<T, phi::GPUContext> {
  RmsFunctor(const phi::GPUContext &ctx,
             const DenseTensor &param,
             const DenseTensor &mean_square,
             const DenseTensor &grad,
             const DenseTensor &moment,
             const DenseTensor &learning_rate,
             const paddle::optional<DenseTensor> &mean_grad_opt,
             const paddle::optional<DenseTensor> &master_param,
             float epsilon_t,
             float decay_t,
             float momentum_t,
             bool centered,
             bool multi_precision,
             DenseTensor *param_out,
             DenseTensor *moment_out,
             DenseTensor *mean_square_out,
             DenseTensor *mean_grad_out,
             DenseTensor *master_param_outs) {
    auto &p_tensor = param;
    auto &ms_tensor = mean_square;
    auto &lr_tensor = learning_rate;
    auto &mom_tensor = moment;
    auto &grad_tensor = grad;
    size_t limit = static_cast<size_t>(ms_tensor.numel());
    DenseRmspropGradFunctor<T> grad_func(grad_tensor.data<T>());
    funcs::ForRange<phi::GPUContext> for_range(ctx, limit);
    using MPDType = typename phi::dtype::MPTypeTrait<T>::Type;
    MPDType *master_out_data =
        multi_precision ? ctx.template Alloc<MPDType>(master_param_outs)
                        : nullptr;

    if (centered) {
      auto mg_tensor = mean_grad_opt.get_ptr();
      if (mg_tensor) {
        PADDLE_ENFORCE_EQ(
            mg_tensor->Holder(),
            mean_grad_out->Holder(),
            common::errors::InvalidArgument(
                "MeanGrad and MeanGradOut must be the same Tensor"));
      } else {
        PADDLE_ENFORCE_EQ(
            mg_tensor,
            mean_grad_out,
            common::errors::InvalidArgument(
                "MeanGrad and MeanGradOut must be the same Tensor"));
      }

      for_range(CenteredRmspropFunctor<T, MPDType, DenseRmspropGradFunctor<T>>(
          ctx.template Alloc<T>(param_out),
          ctx.template Alloc<MPDType>(mean_square_out),
          ctx.template Alloc<MPDType>(moment_out),
          ctx.template Alloc<MPDType>(mean_grad_out),
          lr_tensor.data<MPDType>(),
          master_out_data,
          static_cast<MPDType>(decay_t),
          static_cast<MPDType>(epsilon_t),
          static_cast<MPDType>(momentum_t),
          grad_func));
    } else {
      for_range(
          UncenteredRmspropFunctor<T, MPDType, DenseRmspropGradFunctor<T>>(
              ctx.template Alloc<T>(param_out),
              ctx.template Alloc<MPDType>(mean_square_out),
              ctx.template Alloc<MPDType>(moment_out),
              lr_tensor.data<MPDType>(),
              master_out_data,
              static_cast<MPDType>(decay_t),
              static_cast<MPDType>(epsilon_t),
              static_cast<MPDType>(momentum_t),
              grad_func));
    }
  }
};
template struct RmsFunctor<phi::GPUContext, float>;
template struct RmsFunctor<phi::GPUContext, double>;
template struct RmsFunctor<phi::GPUContext, phi::dtype::float16>;
}  // namespace phi

PD_REGISTER_KERNEL(rmsprop,
                   GPU,
                   ALL_LAYOUT,
                   phi::RmspropDenseKernel,
                   float,
                   double,
                   phi::dtype::float16) {}

PD_REGISTER_KERNEL(rmsprop_dense_param_sparse_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::RmspropSparseKernel,
                   float,
                   double,
                   phi::dtype::float16) {}
