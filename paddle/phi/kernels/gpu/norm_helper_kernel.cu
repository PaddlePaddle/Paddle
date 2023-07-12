// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/norm_helper_kernel.h"
#include <assert.h>
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/fusion/gpu/attention_layer.norm.h"
#include "paddle/phi/kernels/fusion/gpu/fused_dropout_helper.h"
#include "paddle/phi/kernels/rms_norm_kernel.h"

namespace phi {

template <typename T>
class NormHelper {
 public:
  NormHelper(const phi::GPUContext& dev_ctx,
             const std::string& norm_type,
             const int rows,
             const int cols,
             const float epsilon,
             const float residual_alpha)
      : dev_ctx_(dev_ctx),
        norm_type_(norm_type),
        rows_(rows),
        cols_(cols),
        epsilon_(epsilon),
        residual_alpha_(residual_alpha),
        layernorm_helper_(dev_ctx_, epsilon_, rows_, cols_) {
    phi::fusion::DropoutParam dropout_param(
        true, 0, true, true, 0.0, nullptr, 0);
    residual_bias_add_layernorm_helper_ =
        phi::fusion::FusedDropoutLayerNormHelper<T, uint8_t>(
            dev_ctx, rows_, cols_, dropout_param, epsilon_, residual_alpha_);
  }

  /*
  Note(Zhengzekang):
  Since input `X` and `Residual` in FusedMT will be swaped by preallocated
  buffer, I have no choice but to pass the data pointer instead of
  phi::DenseTensor.
  */

  // dst = Norm(x + residual + bias(optional))
  void NormResidualBias(const phi::DenseTensor& x,
                        const paddle::optional<DenseTensor>& residual,
                        const paddle::optional<DenseTensor>& bias,
                        const phi::DenseTensor& norm_weight,
                        const paddle::optional<DenseTensor>& norm_bias,
                        phi::DenseTensor* mean,
                        phi::DenseTensor* var,
                        phi::DenseTensor* bias_residual_out,
                        phi::DenseTensor* output) {
    using U = phi::funcs::LayerNormParamType<T>;
    const T* x_data = x.data<T>();
    const T* residual_data = residual.get().data<T>();
    const T* bias_data = bias ? bias.get().data<T>() : nullptr;
    U* mean_data = mean ? mean->data<U>() : nullptr;
    U* var_data = var ? var->data<U>() : nullptr;
    T* bias_residual_out_data = bias_residual_out->data<T>();
    T* output_data = output->data<T>();

    if (norm_type_ == "layernorm") {
      // For layernorm, it use FP32 type weight and bias.
      const U* norm_weight_data = norm_weight.data<U>();
      const U* norm_bias_data = norm_bias ? norm_bias.get().data<U>() : nullptr;
      residual_bias_add_layernorm_helper_.LayernormResidualDropoutBias(
          dev_ctx_,
          x_data,
          residual_data,
          bias_data,
          norm_weight_data,
          norm_bias_data,
          bias_residual_out_data,
          nullptr,
          output_data,
          mean_data,
          var_data);
    } else if (norm_type_ == "rmsnorm") {
      // For rmsnorm, it use Input's type weight and bias.
      // Currently, it only used in inference, so we do not save intermediate
      // result for backward.
      const T* norm_weight_data = norm_weight.data<T>();
      const T* norm_bias_data = norm_bias ? norm_bias.get().data<T>() : nullptr;
      phi::ResidualAddRmsNormWrapper<T, phi::GPUContext>(dev_ctx_,
                                                         x_data,
                                                         residual_data,
                                                         bias_data,
                                                         norm_weight_data,
                                                         norm_bias_data,
                                                         epsilon_,
                                                         rows_,
                                                         cols_,
                                                         bias_residual_out_data,
                                                         output_data);
    } else {
      PADDLE_THROW(phi::errors::Unimplemented(
          "Currently NormHelper only support `layernorm`, `rmsnorm`. "));
    }
  }

  // dst = Norm(x)
  void Norm(const phi::DenseTensor& x,
            const phi::DenseTensor& norm_weight,
            const paddle::optional<DenseTensor>& norm_bias,
            phi::DenseTensor* mean,
            phi::DenseTensor* var,
            phi::DenseTensor* output) {
    using U = phi::funcs::LayerNormParamType<T>;
    const T* x_data = x.data<T>();
    U* mean_data = mean ? mean->data<U>() : nullptr;
    U* var_data = var ? var->data<U>() : nullptr;
    T* output_data = output->data<T>();

    if (norm_type_ == "layernorm") {
      // For layernorm, it use FP32 type weight and bias.
      const U* norm_weight_data = norm_weight.data<U>();
      const U* norm_bias_data = norm_bias ? norm_bias.get().data<U>() : nullptr;
      layernorm_helper_.ComputeForward(x_data,
                                       norm_weight_data,
                                       norm_bias_data,
                                       output_data,
                                       mean_data,
                                       var_data);

      VLOG(0) << "+======";

    } else if (norm_type_ == "rmsnorm") {
      // For rmsnorm, it use Input's type weight and bias.
      // Currently, it only used in inference, so we do not save intermediate
      // result for backward.
      const T* norm_weight_data = norm_weight.data<T>();
      const T* norm_bias_data = norm_bias ? norm_bias.get().data<T>() : nullptr;
      phi::RmsNormWrapper<T, phi::GPUContext>(dev_ctx_,
                                              x_data,
                                              norm_weight_data,
                                              norm_bias_data,
                                              epsilon_,
                                              rows_,
                                              cols_,
                                              output_data);
    } else {
      PADDLE_THROW(phi::errors::Unimplemented(
          "Currently NormHelper only support `layernorm`, `rmsnorm`. "));
    }
  }

 private:
  const phi::GPUContext& dev_ctx_;
  std::string norm_type_;
  int rows_;
  int cols_;
  float epsilon_;
  float residual_alpha_;
  phi::fusion::FusedDropoutLayerNormHelper<T, uint8_t>
      residual_bias_add_layernorm_helper_;
  phi::fusion::AttnLayerNorm<T> layernorm_helper_;
};

template <typename T, typename Context>
void NormHelperKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const paddle::optional<DenseTensor>& residual,
                      const paddle::optional<DenseTensor>& bias,
                      const DenseTensor& norm_weight,
                      const paddle::optional<DenseTensor>& norm_bias,
                      float epsilon,
                      float residual_alpha,
                      const std::string& norm_type,
                      const int begin_norm_axis,
                      DenseTensor* mean,
                      DenseTensor* variance,
                      DenseTensor* residual_out,
                      DenseTensor* out) {
#if defined(PADDLE_WITH_HIP)
  LOG(ERROR) << "Please compile with CUDA, ROCM platform isn't support it";
#else
  using ComputeType = typename phi::dtype::MPTypeTrait<T>::Type;

  T* out_data = dev_ctx.template Alloc<T>(out);
  ComputeType* mean_data = dev_ctx.template Alloc<ComputeType>(mean);
  ComputeType* variance_data = dev_ctx.template Alloc<ComputeType>(variance);

  int32_t rows = 1;
  int32_t cols = 1;
  for (int i = 0; i < begin_norm_axis; i++) {
    rows *= x.dims()[i];
  }

  for (int i = begin_norm_axis; i < x.dims().size(); i++) {
    cols *= x.dims()[i];
  }

  NormHelper<T> norm_helper(
      dev_ctx, norm_type, rows, cols, epsilon, residual_alpha);
  if (residual) {
    T* residual_out_data = dev_ctx.template Alloc<T>(residual_out);

    // Do residual+biasAdd+Norm fusion.
    norm_helper.NormResidualBias(x,
                                 residual,
                                 bias, /*skip_bias*/
                                 norm_weight,
                                 norm_bias,
                                 mean,
                                 variance,
                                 residual_out,
                                 out);
  } else {
    // Do norm.
    norm_helper.Norm(x, norm_weight, norm_bias, mean, variance, out);
  }

#endif
}

}  // namespace phi

PD_REGISTER_KERNEL(norm_helper,
                   GPU,
                   ALL_LAYOUT,
                   phi::NormHelperKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
