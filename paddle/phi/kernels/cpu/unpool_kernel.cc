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

#include "paddle/phi/kernels/unpool_kernel.h"

#include <algorithm>
#include <vector>

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {

template <typename T, typename Context>
void UnpoolKernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  const DenseTensor& indices,
                  const std::vector<int>& ksize,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings,
                  const IntArray& output_size,
                  const std::string& data_format,
                  DenseTensor* out) {
  T* output_data = dev_ctx.template Alloc<T>(out);
  if (output_data) {
    phi::funcs::SetConstant<Context, T> set_zero;
    set_zero(dev_ctx, out, static_cast<T>(0));
  }
  const int batch_size = x.dims()[0];
  const int input_height = x.dims()[2];
  const int input_width = x.dims()[3];
  const int output_channels = out->dims()[1];
  const int output_height = out->dims()[2];
  const int output_width = out->dims()[3];
  int input_feasize = input_height * input_width;
  int output_feasize = output_height * output_width;
  const T* input_data = x.data<T>();
  const int* indices_data = indices.data<int>();
  for (int b = 0; b < batch_size; ++b) {
    for (int c = 0; c < output_channels; ++c) {
      for (int i = 0; i < input_feasize; ++i) {
        int index = indices_data[i];
        PADDLE_ENFORCE_LT(
            index,
            output_feasize,
            phi::errors::InvalidArgument(
                "index should less than output tensor height * output tensor "
                "width. Expected %ld < %ld, but got "
                "%ld >= %ld. Please check input value.",
                index,
                output_feasize,
                index,
                output_feasize));
        output_data[index] = input_data[i];
      }
      input_data += input_feasize;
      indices_data += input_feasize;
      output_data += output_feasize;
    }
  }
}

template <typename T, typename Context>
void Unpool3dKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    const DenseTensor& indices,
                    const std::vector<int>& ksize,
                    const std::vector<int>& strides,
                    const std::vector<int>& paddings,
                    const std::vector<int>& output_size,
                    const std::string& data_format,
                    DenseTensor* out) {
  T* output_data = dev_ctx.template Alloc<T>(out);
  if (output_data) {
    phi::funcs::SetConstant<Context, T> set_zero;
    set_zero(dev_ctx, out, static_cast<T>(0));
  }
  const int batch_size = x.dims()[0];
  const int input_depth = x.dims()[2];
  const int input_height = x.dims()[3];
  const int input_width = x.dims()[4];
  const int output_channels = out->dims()[1];
  const int output_depth = out->dims()[2];
  const int output_height = out->dims()[3];
  const int output_width = out->dims()[4];
  int input_feasize = input_depth * input_height * input_width;
  int output_feasize = output_depth * output_height * output_width;
  const T* input_data = x.data<T>();
  const int* indices_data = indices.data<int>();
  for (int b = 0; b < batch_size; ++b) {
    for (int c = 0; c < output_channels; ++c) {
      for (int i = 0; i < input_feasize; ++i) {
        int index = indices_data[i];
        PADDLE_ENFORCE_LT(
            index,
            output_feasize,
            phi::errors::InvalidArgument(
                "index should less than output tensor depth * output tensor "
                "height "
                "* output tensor width. Expected %ld < %ld, but got "
                "%ld >= %ld. Please check input value.",
                index,
                output_feasize,
                index,
                output_feasize));
        output_data[index] = input_data[i];
      }
      input_data += input_feasize;
      indices_data += input_feasize;
      output_data += output_feasize;
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(unpool, CPU, ALL_LAYOUT, phi::UnpoolKernel, float, double) {}

PD_REGISTER_KERNEL(
    unpool3d, CPU, ALL_LAYOUT, phi::Unpool3dKernel, float, double) {}
