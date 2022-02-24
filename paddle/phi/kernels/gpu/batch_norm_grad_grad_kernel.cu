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

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/batch_norm_kernel.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"

#include "paddle/fluid/platform/device/gpu/gpu_dnn.h"

#include "paddle/fluid/operators/norm_utils.cu.h"
#include "paddle/fluid/operators/norm_utils.h"

#include "paddle/fluid/framework/data_layout.h"
#include "paddle/fluid/operators/layout_utils.h"
#include "paddle/fluid/platform/enforce.h"

#include "paddle/fluid/platform/flags.h"
#include "paddle/phi/kernels/gpu/batch_norm_utils.h"

#ifdef __HIPCC__
#define LAUNCH_BOUNDS(BlockDim) __launch_bounds__(BlockDim)
#else
#define LAUNCH_BOUNDS(BlockDim)
#endif

namespace phi {

template <typename T, typename Context>
void BatchNormGradGradKernel(const Context& ctx,
                             const DenseTensor& x_grad_grad,
                             const DenseTensor& scale_grad_grad,
                             const DenseTensor& bias_grad_grad,
                             const DenseTensor& y_grad,
                             const DenseTensor& x,
                             const DenseTensor& scale,
                             const DenseTensor& saved_mean,
                             const DenseTensor& saved_variance,
                             paddle::optional<const DenseTensor&> mean,
                             paddle::optional<const DenseTensor&> variance,
                             float momentum,
                             float epsilon,
                             const std::string& data_layout_str,
                             bool is_test,
                             bool use_global_stats,
                             bool trainable_statistics,
                             bool fuse_with_relu,
                             DenseTensor* x_grad,
                             DenseTensor* scale_grad,
                             DenseTensor* y_grad_grad) {
  PADDLE_ENFORCE_EQ(is_test,
                    false,
                    phi::errors::InvalidArgument(
                        "`is_test = True` CANNOT be used in train program. If "
                        "you want to use global status in pre_train model, "
                        "please set `use_global_stats = True`"));

  const DataLayout data_layout =
      paddle::framework::StringToDataLayout(data_layout_str);

  const DenseTensor* running_mean = nullptr;
  const DenseTensor* running_variance = nullptr;
  if (use_global_stats) {
    running_mean = mean.get_ptr();
    running_variance = variance.get_ptr();
  }
  paddle::operators::NormDoubleGradFunctor<Context, T>(ctx,
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
                                                       &x_grad_grad,
                                                       &scale_grad_grad,
                                                       &bias_grad_grad,
                                                       x_grad,
                                                       scale_grad,
                                                       y_grad_grad);
}
}  // namespace phi

#ifdef PADDLE_WITH_HIP
PD_REGISTER_KERNEL(batch_norm_grad_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::BatchNormGradGradKernel,
                   float,
                   double) {}

#else
PD_REGISTER_KERNEL(batch_norm_grad_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::BatchNormGradGradKernel,
                   float,
                   double) {}
#endif
