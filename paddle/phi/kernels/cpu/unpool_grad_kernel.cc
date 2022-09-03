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

#include "paddle/phi/kernels/unpool_grad_kernel.h"

#include <algorithm>
#include <string>
#include <vector>

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {

template <typename T, typename Context>
void UnpoolGradKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const DenseTensor& indices,
                      const DenseTensor& out,
                      const DenseTensor& out_grad,
                      const std::vector<int>& ksize,
                      const std::vector<int>& strides,
                      const std::vector<int>& paddings,
                      const IntArray& output_size,
                      const std::string& data_format,
                      DenseTensor* x_grad) {
  T* input_grad_data = dev_ctx.template Alloc<T>(x_grad);
  const T* output_grad_data = out_grad.data<T>();
  phi::funcs::SetConstant<Context, T> zero;
  zero(dev_ctx, x_grad, static_cast<T>(0));
  const int batch_size = x.dims()[0];
  const int input_height = x.dims()[2];
  const int input_width = x.dims()[3];
  const int output_channels = out.dims()[1];
  const int output_height = out.dims()[2];
  const int output_width = out.dims()[3];
  int input_feasize = input_height * input_width;
  int output_feasize = output_height * output_width;
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
        input_grad_data[i] = output_grad_data[index];
      }
      input_grad_data += input_feasize;
      indices_data += input_feasize;
      output_grad_data += output_feasize;
    }
  }
}

template <typename T, typename Context>
void Unpool3dGradKernel(const Context& dev_ctx,
                        const DenseTensor& x,
                        const DenseTensor& indices,
                        const DenseTensor& out,
                        const DenseTensor& out_grad,
                        const std::vector<int>& ksize,
                        const std::vector<int>& strides,
                        const std::vector<int>& paddings,
                        const std::vector<int>& output_size,
                        const std::string& data_format,
                        DenseTensor* x_grad) {
  T* input_grad_data = dev_ctx.template Alloc<T>(x_grad);
  const T* output_grad_data = out_grad.data<T>();
  phi::funcs::SetConstant<Context, T> zero;
  zero(dev_ctx, x_grad, static_cast<T>(0));

  const int batch_size = x.dims()[0];
  const int input_depth = x.dims()[2];
  const int input_height = x.dims()[3];
  const int input_width = x.dims()[4];
  const int output_channels = out.dims()[1];
  const int output_depth = out.dims()[2];
  const int output_height = out.dims()[3];
  const int output_width = out.dims()[4];
  int input_feasize = input_depth * input_height * input_width;
  int output_feasize = output_depth * output_height * output_width;
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
        input_grad_data[i] = output_grad_data[index];
      }
      input_grad_data += input_feasize;
      indices_data += input_feasize;
      output_grad_data += output_feasize;
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(
    unpool_grad, CPU, ALL_LAYOUT, phi::UnpoolGradKernel, float, double) {}

PD_REGISTER_KERNEL(
    unpool3d_grad, CPU, ALL_LAYOUT, phi::Unpool3dGradKernel, float, double) {}
