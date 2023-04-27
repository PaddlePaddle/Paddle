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

#include "glog/logging.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_dnn.h"
#include "paddle/phi/common/layout.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/flags.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/batch_norm_kernel.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/funcs/batch_norm_grad_functor.cu.h"

namespace phi {

template <typename T, typename Context>
void BatchNormGradKernel(const Context &dev_ctx,
                         const DenseTensor &x,
                         const DenseTensor &scale,
                         const DenseTensor &bias,
                         const paddle::optional<DenseTensor> &mean,
                         const paddle::optional<DenseTensor> &variance,
                         const DenseTensor &saved_mean,
                         const DenseTensor &saved_variance,
                         const paddle::optional<DenseTensor> &reserve_space,
                         const DenseTensor &y_grad,
                         float momentum,
                         float epsilon,
                         const std::string &data_layout,
                         bool is_test,
                         bool use_global_stats,
                         bool trainable_statistics,
                         DenseTensor *x_grad,
                         DenseTensor *scale_grad,
                         DenseTensor *bias_grad) {
  funcs::BatchNormGradFunctor<T, Context>(dev_ctx,
                                          x,
                                          scale,
                                          bias,
                                          mean,
                                          variance,
                                          saved_mean,
                                          saved_variance,
                                          reserve_space,
                                          y_grad,
                                          momentum,
                                          epsilon,
                                          data_layout,
                                          is_test,
                                          use_global_stats,
                                          trainable_statistics,
                                          false,
                                          x_grad,
                                          scale_grad,
                                          bias_grad);
}

template <typename T, typename Context>
void BatchNormDoubleGradKernel(
    const Context &ctx,
    const DenseTensor &x,
    const DenseTensor &scale,
    const paddle::optional<DenseTensor> &mean,
    const paddle::optional<DenseTensor> &variance,
    const DenseTensor &saved_mean,
    const DenseTensor &saved_variance,
    const DenseTensor &y_grad,
    const paddle::optional<DenseTensor> &x_grad_grad,
    const paddle::optional<DenseTensor> &scale_grad_grad,
    const paddle::optional<DenseTensor> &bias_grad_grad,
    float momentum,
    float epsilon,
    const std::string &data_layout_str,
    bool is_test,
    bool use_global_stats,
    bool trainable_statistics,
    DenseTensor *x_grad,
    DenseTensor *scale_grad,
    DenseTensor *y_grad_grad) {
  PADDLE_ENFORCE_EQ(is_test,
                    false,
                    phi::errors::InvalidArgument(
                        "`is_test = True` CANNOT be used in train program. If "
                        "you want to use global status in pre_train model, "
                        "please set `use_global_stats = True`"));

  const DataLayout data_layout = phi::StringToDataLayout(data_layout_str);

  const DenseTensor *running_mean = nullptr;
  const DenseTensor *running_variance = nullptr;
  if (use_global_stats) {
    running_mean = mean.get_ptr();
    running_variance = variance.get_ptr();
  }
  phi::funcs::NormDoubleGradFunctor<Context, T>(ctx,
                                                data_layout,
                                                &x,
                                                &scale,
                                                &y_grad,
                                                &saved_mean,
                                                &saved_variance,
                                                running_mean,
                                                running_variance,
                                                epsilon,
                                                use_global_stats,
                                                x_grad_grad.get_ptr(),
                                                scale_grad_grad.get_ptr(),
                                                bias_grad_grad.get_ptr(),
                                                x_grad,
                                                scale_grad,
                                                y_grad_grad);
}

}  // namespace phi

#ifdef PADDLE_WITH_HIP
PD_REGISTER_KERNEL(batch_norm_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::BatchNormGradKernel,
                   float,
                   phi::dtype::float16) {}

#else
#if CUDNN_VERSION_MIN(8, 1, 0)

PD_REGISTER_KERNEL(batch_norm_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::BatchNormGradKernel,
                   float,
                   double,
                   phi::dtype::bfloat16,
                   phi::dtype::float16) {
  if (kernel_key.dtype() == phi::DataType::FLOAT16 ||
      kernel_key.dtype() == phi::DataType::BFLOAT16) {
    kernel->OutputAt(0).SetDataType(phi::DataType::FLOAT32);  // x_grad
    kernel->OutputAt(1).SetDataType(phi::DataType::FLOAT32);  // scale_grad
    kernel->OutputAt(2).SetDataType(phi::DataType::FLOAT32);  // bias_grad
  }
}

#else
PD_REGISTER_KERNEL(batch_norm_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::BatchNormGradKernel,
                   float,
                   double,
                   phi::dtype::float16) {
  if (kernel_key.dtype() == phi::DataType::FLOAT16) {
    kernel->OutputAt(0).SetDataType(phi::DataType::FLOAT32);  // x_grad
    kernel->OutputAt(1).SetDataType(phi::DataType::FLOAT32);  // scale_grad
    kernel->OutputAt(2).SetDataType(phi::DataType::FLOAT32);  // bias_grad
  }
}
#endif
#endif

#ifdef PADDLE_WITH_HIP
PD_REGISTER_KERNEL(batch_norm_double_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::BatchNormDoubleGradKernel,
                   float,
                   double) {}
#else
PD_REGISTER_KERNEL(batch_norm_double_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::BatchNormDoubleGradKernel,
                   float,
                   double) {}
#endif
