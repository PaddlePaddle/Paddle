/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/kernels/funcs/maxouting.h"

#include "paddle/phi/backends/cpu/cpu_context.h"

namespace phi::funcs {

// All tensors are in NCHW or NHWC format, and the groups must be greater than 1
template <typename DeviceContext, typename T>
void MaxOutFunctor<DeviceContext, T>::operator()(const DeviceContext& dev_ctx,
                                                 const DenseTensor& input,
                                                 DenseTensor* output,
                                                 const int groups,
                                                 const int axis) {
  const int batch_size = static_cast<int>(input.dims()[0]);
  const int input_height =
      static_cast<int>(axis == 1 ? input.dims()[2] : input.dims()[1]);
  const int input_width =
      static_cast<int>(axis == 1 ? input.dims()[3] : input.dims()[2]);
  const int output_channels = static_cast<int>(output->dims()[axis]);
  int64_t fea_size = static_cast<int64_t>(input_height) * input_width;
  // c_size means the output size of each sample
  int64_t c_size = static_cast<int64_t>(fea_size) * output_channels;
  const T* input_data = input.data<T>();
  T* output_data = dev_ctx.template Alloc<T>(output);
  for (int i = 0; i < batch_size; ++i) {
    int64_t new_bindex = static_cast<int64_t>(c_size) * i;
    for (int c = 0; c < output_channels; ++c) {
      int64_t new_cindex = static_cast<int64_t>(fea_size) * c;
      for (int64_t f = 0; f < fea_size; ++f) {
        T ele = static_cast<T>(-FLT_MAX);
        int64_t input_idx = 0, output_idx = 0;
        for (int ph = 0; ph < groups; ++ph) {
          if (axis == 1) {
            input_idx = (new_bindex + new_cindex) * groups + ph * fea_size + f;
          } else {
            input_idx = (new_bindex + f * output_channels + c) * groups + ph;
          }
          T x = input_data[input_idx];
          ele = ele > x ? ele : x;
        }
        if (axis == 1) {
          output_idx = new_bindex + new_cindex + f;
        } else {
          output_idx = new_bindex + f * output_channels + c;
        }
        output_data[output_idx] = ele;
      }
    }
  }
}

template <typename DeviceContext, typename T>
void MaxOutGradFunctor<DeviceContext, T>::operator()(
    const DeviceContext& dev_ctx,
    const DenseTensor& input,
    DenseTensor* input_grad,
    const DenseTensor& output,
    const DenseTensor& output_grad,
    const int groups,
    const int axis) {
  const int batch_size = static_cast<int>(input.dims()[0]);
  const int input_height =
      static_cast<int>(axis == 1 ? input.dims()[2] : input.dims()[1]);
  const int input_width =
      static_cast<int>(axis == 1 ? input.dims()[3] : input.dims()[2]);
  const int output_channels = static_cast<int>(output.dims()[axis]);
  int64_t fea_size = static_cast<int64_t>(input_height) * input_width;
  const T* input_data = input.data<T>();
  const T* output_data = output.data<T>();
  const T* output_grad_data = output_grad.data<T>();
  T* input_grad_data = dev_ctx.template Alloc<T>(input_grad);
  for (int i = 0; i < batch_size; ++i) {
    int64_t blen = static_cast<int64_t>(fea_size) * output_channels * i;
    for (int c = 0; c < output_channels; ++c) {
      int64_t clen = static_cast<int64_t>(fea_size) * c;
      for (int64_t f = 0; f < fea_size; ++f) {
        int64_t input_idx0 = 0, output_idx = 0;
        bool continue_match = true;
        if (axis == 1) {
          input_idx0 = (blen + clen) * groups + f;
          output_idx = blen + clen + f;
        } else {
          input_idx0 = (blen + f * output_channels + c) * groups;
          output_idx = blen + f * output_channels + c;
        }
        for (int g = 0; g < groups && continue_match; ++g) {
          int64_t idx_offset = (axis == 1 ? fea_size * g : g);
          int64_t input_idx = input_idx0 + idx_offset;
          if (input_data[input_idx] == output_data[output_idx]) {
            input_grad_data[input_idx] += output_grad_data[output_idx];
            continue_match = false;
          }
        }
      }
    }
  }
}

template class MaxOutGradFunctor<CPUContext, float>;
template class MaxOutGradFunctor<CPUContext, double>;
template class MaxOutFunctor<CPUContext, float>;
template class MaxOutFunctor<CPUContext, double>;

}  // namespace phi::funcs
