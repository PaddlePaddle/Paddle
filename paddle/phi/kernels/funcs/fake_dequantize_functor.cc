/* Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/kernels/funcs/fake_dequantize_functor.h"

namespace phi {
namespace funcs {

template <typename Context, typename T>
void DequantizeFunctor<Context, T>::operator()(const Context& dev_ctx,
                                               const DenseTensor* in,
                                               const DenseTensor* scale,
                                               T max_range,
                                               DenseTensor* out) {
  auto in_e = phi::EigenVector<T>::Flatten(*in);
  const T* scale_factor = scale->data<T>();
  auto out_e = phi::EigenVector<T>::Flatten(*out);

  auto& dev = *dev_ctx.eigen_device();
  out_e.device(dev) = in_e * scale_factor[0] / max_range;
}

template <typename Context, typename T>
void ChannelDequantizeFunctor<Context, T>::operator()(
    const Context& dev_ctx,
    const DenseTensor* in,
    const DenseTensor** scales,
    const int scale_num,
    T max_range,
    const int quant_axis,
    const int x_num_col_dims,
    DenseTensor* out) {
  if (scale_num == 1) {
    // Dequant op is before quantized op
    // Dequantize the weight of quantized op
    auto in_dims = in->dims();
    const int64_t channel = in_dims[quant_axis];
    const T* scale_factor = scales[0]->data<T>();
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
      auto* out_data = dev_ctx.template Alloc<T>(out);
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
  } else if (scale_num == 2) {
    // Dequant op is after quantized op
    // Dequantize the output tensor of quantized op
    if (x_num_col_dims > 1) {
      auto in_dims = in->dims();
      const int64_t channel = in_dims[x_num_col_dims];
      const T* scale_one = scales[0]->data<T>();
      const T* scale_two = scales[1]->data<T>();
      int64_t out_iter = 1;
      for (int i = 0; i < x_num_col_dims; i++) {
        out_iter *= in_dims[i];
      }
      int64_t step_i = in->numel() / out_iter;
      int64_t step_j = in->numel() / (out_iter * channel);
      auto* in_data = in->data<T>();
      auto* out_data = dev_ctx.template Alloc<T>(out);
      for (int64_t i = 0; i < out_iter; i++) {
        for (int64_t j = 0; j < channel; j++) {
          auto* cur_in = in_data + i * step_i + j * step_j;
          auto* cur_out = out_data + i * step_i + j * step_j;
          T s = scale_one[j];
          for (int64_t k = 0; k < step_j; k++) {
            *cur_out = (*cur_in) * s * scale_two[0] / max_range;
            ++cur_in;
            ++cur_out;
          }
        }
      }
    } else {
      int batch_size = static_cast<int>(in->dims()[0]);
      int channel = static_cast<int>(in->dims()[1]);
      const T* scale_one = scales[0]->data<T>();
      const T* scale_two = scales[1]->data<T>();
      for (int i = 0; i < batch_size; i++) {
        phi::DenseTensor one_batch_in = in->Slice(i, i + 1).Resize(
            common::slice_ddim(in->dims(), 1, in->dims().size()));
        phi::DenseTensor one_batch_out = out->Slice(i, i + 1).Resize(
            common::slice_ddim(out->dims(), 1, out->dims().size()));
        for (int j = 0; j < channel; j++) {
          T s = scale_one[j];
          phi::DenseTensor one_channel_in = one_batch_in.Slice(j, j + 1);
          phi::DenseTensor one_channel_out = one_batch_out.Slice(j, j + 1);
          auto in_e = phi::EigenVector<T>::Flatten(one_channel_in);
          auto out_e = phi::EigenVector<T>::Flatten(one_channel_out);
          auto& dev = *dev_ctx.eigen_device();
          out_e.device(dev) = in_e * s * scale_two[0] / max_range;
        }
      }
    }
  }
}

template class ChannelDequantizeFunctor<CPUContext, float>;
template class ChannelDequantizeFunctor<CPUContext, double>;
template class DequantizeFunctor<CPUContext, float>;
template class DequantizeFunctor<CPUContext, double>;

}  // namespace funcs
}  // namespace phi
