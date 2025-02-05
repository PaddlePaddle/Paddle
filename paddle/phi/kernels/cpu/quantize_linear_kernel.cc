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

#include <string>

#include "paddle/phi/kernels/quantize_linear_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/type_traits.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/impl/quantize_linear_impl.h"

namespace phi {

template <typename T>
struct DequantizeFunctor<phi::CPUContext, T> {
  void operator()(const phi::CPUContext& dev_ctx,
                  const phi::DenseTensor* in,
                  const phi::DenseTensor* scale,
                  T max_range,
                  phi::DenseTensor* out) {
    auto in_e = phi::EigenVector<T>::Flatten(*in);
    const T* scale_factor = scale->data<T>();
    auto out_e = phi::EigenVector<T>::Flatten(*out);

    auto& dev = *dev_ctx.eigen_device();
    out_e.device(dev) = in_e * scale_factor[0] / max_range;
  }
};

template <typename T>
struct ChannelDequantizeFunctorV2<phi::CPUContext, T> {
  void operator()(const phi::CPUContext& dev_ctx,
                  const phi::DenseTensor* in,
                  const phi::DenseTensor* scale,
                  T max_range,
                  const int quant_axis,
                  phi::DenseTensor* out) {
    // Dequant op is before quantized op
    // Dequantize the weight of quantized op
    auto in_dims = in->dims();
    const int64_t channel = in_dims[quant_axis];
    const T* scale_factor = scale->data<T>();
    if (quant_axis == 0) {
      for (int64_t i = 0; i < channel; i++) {
        T s = scale_factor[i];
        phi::DenseTensor one_channel_in = in->Slice(i, i + 1);
        phi::DenseTensor one_channel_out = out->Slice(i, i + 1);
        auto in_e = phi::EigenVector<T>::Flatten(one_channel_in);
        auto out_e = phi::EigenVector<T>::Flatten(one_channel_out);
        auto& dev = *dev_ctx.eigen_device();
        out_e.device(dev) = in_e * s / max_range;
      }
    } else if (quant_axis == 1) {
      int64_t out_iter = 1;
      for (int i = 0; i < quant_axis; i++) {
        out_iter *= in_dims[i];
      }
      int64_t step_i = in->numel() / out_iter;
      int64_t step_j = in->numel() / (out_iter * channel);
      auto* in_data = in->data<T>();
      auto* out_data = dev_ctx.Alloc<T>(out, out->numel() * sizeof(T));
      for (int64_t i = 0; i < out_iter; i++) {
        for (int64_t j = 0; j < channel; j++) {
          auto* cur_in = in_data + i * step_i + j * step_j;
          auto* cur_out = out_data + i * step_i + j * step_j;
          T s = scale_factor[j];
          for (int64_t k = 0; k < step_j; k++) {
            *cur_out = (*cur_in) * s / max_range;
            ++cur_in;
            ++cur_out;
          }
        }
      }
    }
  }
};

template struct DequantizeFunctor<phi::CPUContext, phi::dtype::float16>;
template struct DequantizeFunctor<phi::CPUContext, float>;
template struct DequantizeFunctor<phi::CPUContext, double>;
template struct ChannelDequantizeFunctorV2<phi::CPUContext,
                                           phi::dtype::float16>;
template struct ChannelDequantizeFunctorV2<phi::CPUContext, float>;
template struct ChannelDequantizeFunctorV2<phi::CPUContext, double>;

}  // namespace phi

PD_REGISTER_KERNEL(
    quantize_linear, CPU, ALL_LAYOUT, phi::QuantizeLinearKernel, float) {}

PD_REGISTER_KERNEL(dequantize_linear,
                   CPU,
                   ALL_LAYOUT,
                   phi::DeQuantizeLinearKernel,
                   float,
                   int8_t,
                   double) {
  kernel->OutputAt(0).SetDataType(phi::DataType::UNDEFINED);
}

PD_REGISTER_KERNEL(quantize_linear_deprecated_train,
                   CPU,
                   ALL_LAYOUT,
                   phi::QuantizeLinearDeprecatedTrainKernel,
                   float) {}

PD_REGISTER_KERNEL(quantize_linear_deprecated_infer,
                   CPU,
                   ALL_LAYOUT,
                   phi::QuantizeLinearDeprecatedInferKernel,
                   float) {}

PD_REGISTER_KERNEL(dequantize_linear_deprecated,
                   CPU,
                   ALL_LAYOUT,
                   phi::DeQuantizeLinearDeprecatedKernel,
                   float,
                   int8_t,
                   double) {
  kernel->OutputAt(0).SetDataType(phi::DataType::UNDEFINED);
}
