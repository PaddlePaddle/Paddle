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

#include "paddle/phi/kernels/fake_quantize_grad_kernel.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"

namespace phi {

template <typename T, typename Context>
void QuantizeGradFunc(const Context& dev_ctx,
                      const DenseTensor& dout,
                      DenseTensor* dx) {
  PADDLE_ENFORCE_NOT_NULL(dx,
                          common::errors::PreconditionNotMet(
                              "The QuantizeGradFunc output dx is nullptr"));
  // Initialize dx as same as d_out
  dev_ctx.template Alloc<T>(dx);
  phi::Copy(dev_ctx, dout, dev_ctx.GetPlace(), false, dx);
}

template <typename T, typename Context>
void FakeChannelWiseQuantizeDequantizeAbsMaxGradKernel(const Context& dev_ctx,
                                                       const DenseTensor& dout,
                                                       int bit_length,
                                                       int round_type,
                                                       int quant_axis,
                                                       DenseTensor* dx) {
  QuantizeGradFunc<T, Context>(dev_ctx, dout, dx);
}

template <typename T, typename Context>
void FakeQuantizeDequantizeAbsMaxGradKernel(const Context& dev_ctx,
                                            const DenseTensor& dout,
                                            int bit_length,
                                            int round_type,
                                            DenseTensor* dx) {
  QuantizeGradFunc<T, Context>(dev_ctx, dout, dx);
}

template <typename T, typename Context>
void FakeQuantizeDequantizeMovingAverageAbsMaxGradKernel(
    const Context& dev_ctx,
    const DenseTensor& dout,
    float moving_rate,
    int bit_length,
    bool is_test,
    int round_type,
    DenseTensor* dx) {
  QuantizeGradFunc<T, Context>(dev_ctx, dout, dx);
}

}  // namespace phi

PD_REGISTER_KERNEL(fake_channel_wise_quantize_dequantize_abs_max_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::FakeChannelWiseQuantizeDequantizeAbsMaxGradKernel,
                   float) {
  kernel->InputAt(0).SetBackend(phi::Backend::ALL_BACKEND);
}

PD_REGISTER_KERNEL(fake_quantize_dequantize_abs_max_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::FakeQuantizeDequantizeAbsMaxGradKernel,
                   float) {
  kernel->InputAt(0).SetBackend(phi::Backend::ALL_BACKEND);
}

PD_REGISTER_KERNEL(fake_quantize_dequantize_moving_average_abs_max_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::FakeQuantizeDequantizeMovingAverageAbsMaxGradKernel,
                   float) {
  kernel->InputAt(0).SetBackend(phi::Backend::ALL_BACKEND);
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PD_REGISTER_KERNEL(fake_channel_wise_quantize_dequantize_abs_max_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::FakeChannelWiseQuantizeDequantizeAbsMaxGradKernel,
                   float) {}
PD_REGISTER_KERNEL(fake_quantize_dequantize_abs_max_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::FakeQuantizeDequantizeAbsMaxGradKernel,
                   float,
                   phi::dtype::float16) {}
PD_REGISTER_KERNEL(fake_quantize_dequantize_moving_average_abs_max_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::FakeQuantizeDequantizeMovingAverageAbsMaxGradKernel,
                   float,
                   phi::dtype::float16) {}
#endif
