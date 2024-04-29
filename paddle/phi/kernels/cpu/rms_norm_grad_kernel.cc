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

#include "paddle/phi/kernels/rms_norm_grad_kernel.h"
#include "glog/logging.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void RmsNormGradKernel(const Context& dev_ctx,
                       const DenseTensor& x,
                       const paddle::optional<DenseTensor>& bias,
                       const paddle::optional<DenseTensor>& residual,
                       const DenseTensor& norm_weight,
                       const paddle::optional<DenseTensor>& norm_bias,
                       const DenseTensor& inv_var,
                       const DenseTensor& out_grad,
                       const float epsilon,
                       const int begin_norm_axis,
                       const float quant_scale,
                       DenseTensor* x_grad,
                       DenseTensor* norm_weight_grad,
                       DenseTensor* norm_bias_grad) {
  if (bias || residual) {
    PADDLE_THROW(phi::errors::Unimplemented(
        "bias or residual is not supported in CPU rms_norm_grad yet"));
  }
  if (quant_scale > 0.0f) {
    PADDLE_THROW(phi::errors::Unimplemented(
        "quantization is not supported in CPU rms_norm_grad yet"));
  }

  const T* x_data = x.data<T>();
  const T* norm_weight_data = norm_weight.data<T>();
  const float* inv_var_data = inv_var.data<float>();
  const T* out_grad_data = out_grad.data<T>();

  dev_ctx.template Alloc<T>(x_grad);
  T* x_grad_data = x_grad->data<T>();

  T* norm_weight_grad_data = nullptr;
  if (norm_weight_grad) {
    dev_ctx.template Alloc<T>(norm_weight_grad);
    norm_weight_grad_data = norm_weight_grad->data<T>();
  }

  T* norm_bias_grad_data = nullptr;
  if (norm_bias && norm_bias_grad) {
    dev_ctx.template Alloc<T>(norm_bias_grad);
    norm_bias_grad_data = norm_bias_grad->data<T>();
  }

  int32_t rows = 1;
  int32_t cols = 1;
  for (int i = 0; i < begin_norm_axis; i++) {
    rows *= x.dims()[i];
  }

  for (int i = begin_norm_axis; i < x.dims().size(); i++) {
    cols *= x.dims()[i];
  }

  VLOG(1) << "[RMS Norm Grad Kernel], rows: " << rows << ", cols: " << cols;
  VLOG(1) << "[RMS Norm Grad Kernel], x.numel(): " << x.numel();
  VLOG(1) << "[RMS Norm Grad Kernel], out_grad.numel(): " << out_grad.numel();
  VLOG(1) << "[RMS Norm Grad Kernel], norm_weight.numel(): "
          << norm_weight.numel();
  if (norm_bias)
    VLOG(1) << "[RMS Norm Grad Kernel], norm_bias.numel(): "
            << norm_bias->numel();
  VLOG(1) << "[RMS Norm Grad Kernel], inv_var.numel(): " << inv_var.numel();
  VLOG(1) << "[RMS Norm Grad Kernel], x_grad.numel(): " << x_grad->numel();
  if (norm_weight_grad)
    VLOG(1) << "[RMS Norm Grad Kernel], norm_weight_grad.numel(): "
            << norm_weight_grad->numel();
  if (norm_bias_grad)
    VLOG(1) << "[RMS Norm Grad Kernel], norm_bias_grad.numel(): "
            << norm_bias_grad->numel();
  // norm_weight_grad_data[0] = static_cast<T>(0.0);

  PADDLE_ENFORCE_EQ(
      x.numel(),
      out_grad.numel(),
      phi::errors::InvalidArgument(
          "The number of elements in input tensor x(%d)"
          "must be equal to the number of elements in output_grad tensor(%d).",
          x.numel(),
          out_grad.numel()));

  PADDLE_ENFORCE_EQ(
      cols,
      norm_weight.dims()[0],
      phi::errors::InvalidArgument(
          "The product from begin_norm_axis to last_axis of input tensor x, "
          "i.e., cols(%d)"
          "must be equal to the norm_weight tensor's dimension(%d).",
          cols,
          norm_weight.dims()[0]));

  if (norm_bias) {
    PADDLE_ENFORCE_EQ(
        cols,
        norm_bias.get().dims()[0],
        phi::errors::InvalidArgument(
            "The product from begin_norm_axis to the last axis of input tensor "
            "x, "
            "i.e., cols(%d) "
            "must be equal to the norm_bias tensor's dimension(%d). ",
            cols,
            norm_bias.get().dims()[0]));
  }

  PADDLE_ENFORCE_EQ(rows,
                    inv_var.numel(),
                    phi::errors::InvalidArgument(
                        "The product from begin_norm_axis to the last axis of "
                        "input tensor x, "
                        "i.e., rows(%d) "
                        "must be equal to the inv_var tensor's numel(%d). ",
                        rows,
                        inv_var.numel()));

  std::vector<float> var_eps(rows);
  std::vector<float> rsqrt_var(rows);
  for (int i = 0; i < rows; i++) {
    var_eps[i] = inv_var_data[i] + epsilon;
    rsqrt_var[i] = std::sqrt(1.0 / var_eps[i]);
  }
  VLOG(1) << "[RMS Norm Grad Kernel], var_eps: " << var_eps[0]
          << ", rsqrt_var: " << rsqrt_var[0];

  // cal norm_weight_grad
  if (norm_weight_grad_data) {
    for (int j = 0; j < cols; j++) {
      float temp_sum = 0.0;
      VLOG(1) << "[RMS Norm Grad Kernel], j = " << j;
      for (int i = 0; i < rows; i++) {
        int index = i * cols + j;
        VLOG(1) << "[RMS Norm Grad Kernel], index: " << index
                << ", x_data[index]: " << static_cast<float>(x_data[index]);
        // temp_sum += static_cast<float>(x_data[index]) * rsqrt_var[i] *
        // static_cast<float>(out_grad_data[index]);
        temp_sum += static_cast<float>(x_data[index]) * rsqrt_var[i];
      }
      VLOG(1) << "[RMS Norm Grad Kernel], temp_sum: " << temp_sum;
      norm_weight_grad_data[j] = static_cast<T>(temp_sum);
    }
    VLOG(1) << "[RMS Norm Grad Kernel], norm_weight_grad: "
            << norm_weight_grad_data[0];
  }

  // cal norm_bias_grad
  if (norm_bias_grad_data) {
    for (int j = 0; j < cols; j++) {
      float temp_sum = 0.0;
      for (int i = 0; i < rows; i++) {
        temp_sum += static_cast<float>(out_grad_data[i * cols + j]);
      }
      norm_bias_grad_data[j] = static_cast<T>(temp_sum);
    }
    VLOG(1) << "[RMS Norm Grad Kernel], norm_bias_grad: "
            << norm_bias_grad_data[0];
  }

  // cal x_grad
  std::vector<float> grad_x_end(rows * cols);
  std::vector<float> grad_std(rows * cols);
  for (int i = 0; i < rows; i++) {
    float temp_sum = 0.0;
    for (int j = 0; j < cols; j++) {
      int index = i * cols + j;
      grad_x_end[index] = static_cast<float>(out_grad_data[index]) *
                          static_cast<float>(norm_weight_data[j]) *
                          rsqrt_var[i];
      temp_sum += (-1.0 * static_cast<float>(x_data[index]) / var_eps[i] *
                   static_cast<float>(out_grad_data[index]) *
                   static_cast<float>(norm_weight_data[j]));
    }
    for (int j = 0; j < cols; j++) {
      int index = i * cols + j;
      grad_std[index] =
          temp_sum * (rsqrt_var[i] / cols * static_cast<float>(x_data[index]));
    }
  }
  VLOG(1) << "[RMS Norm Grad Kernel], grad_x_end: " << grad_x_end[0];
  VLOG(1) << "[RMS Norm Grad Kernel], grad_std: " << grad_std[0];
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      int index = i * cols + j;
      x_grad_data[index] = static_cast<T>(grad_x_end[index] + grad_std[index]);
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(rms_norm_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::RmsNormGradKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
