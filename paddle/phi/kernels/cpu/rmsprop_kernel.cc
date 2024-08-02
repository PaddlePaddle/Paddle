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

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/impl/rmsprop_kernel_impl.h"
namespace phi {
template <typename T>
struct RmsFunctor<T, phi::CPUContext> {
  RmsFunctor(const phi::CPUContext &ctx,
             const DenseTensor &param,
             const DenseTensor &mean_square,
             const DenseTensor &grad,
             const DenseTensor &moment,
             const DenseTensor &learning_rate,
             const paddle::optional<DenseTensor> &mean_grad_opt,
             const paddle::optional<DenseTensor> &master_param UNUSED,
             float epsilon_t,
             float decay_t,
             float momentum_t,
             bool centered,
             bool multi_precision UNUSED,
             DenseTensor *param_out,
             DenseTensor *moment_out,
             DenseTensor *mean_square_out,
             DenseTensor *mean_grad_out,
             DenseTensor *master_param_outs UNUSED) {
    auto epsilon = static_cast<T>(epsilon_t);
    auto rho = static_cast<T>(decay_t);
    auto momentum = static_cast<T>(momentum_t);

    auto &p_tensor = param;
    auto &ms_tensor = mean_square;
    auto &lr_tensor = learning_rate;
    auto &mom_tensor = moment;

    PADDLE_ENFORCE_EQ(p_tensor.IsSharedBufferWith(*param_out),
                      true,
                      common::errors::InvalidArgument(
                          "Param and ParamOut must be the same Tensor"));
    PADDLE_ENFORCE_EQ(mom_tensor.IsSharedBufferWith(*moment_out),
                      true,
                      common::errors::InvalidArgument(
                          "Moment and MomentOut must be the same Tensor"));
    PADDLE_ENFORCE_EQ(
        ms_tensor.IsSharedBufferWith(*mean_square_out),
        true,
        common::errors::InvalidArgument(
            "MeanSquare and MeanSquareOut must be the same Tensor"));

    auto &grad_tensor = grad;
    auto &place = *ctx.eigen_device();
    auto lr_value = lr_tensor.data<T>()[0];

    auto p = EigenVector<T>::Flatten(p_tensor);
    auto ms = EigenVector<T>::Flatten(ms_tensor);
    auto g = EigenVector<T>::Flatten(grad_tensor);
    auto mom = EigenVector<T>::Flatten(mom_tensor);

    auto p_out = EigenVector<T>::Flatten(*param_out);
    auto mom_out = EigenVector<T>::Flatten(*moment_out);
    auto ms_out = EigenVector<T>::Flatten(*mean_square_out);

    ms_out.device(place) = rho * ms + (1 - rho) * g * g;
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
      auto mg = EigenVector<T>::Flatten(*mg_tensor);
      auto mg_out = EigenVector<T>::Flatten(*mean_grad_out);

      mg_out.device(place) = rho * mg + (1 - rho) * g;
      mom_out.device(place) =
          momentum * mom +
          lr_value * g / (ms_out - mg_out.square() + epsilon).sqrt();
    } else {
      mom_out.device(place) =
          momentum * mom + lr_value * g / (ms_out + epsilon).sqrt();
    }
    p_out.device(place) = p - mom_out;
  }
};

}  // namespace phi
PD_REGISTER_KERNEL(
    rmsprop, CPU, ALL_LAYOUT, phi::RmspropDenseKernel, float, double) {}

PD_REGISTER_KERNEL(rmsprop_dense_param_sparse_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::RmspropSparseKernel,
                   float,
                   double) {}
