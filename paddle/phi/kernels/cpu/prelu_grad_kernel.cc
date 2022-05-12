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

#include "paddle/phi/kernels/prelu_grad_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void PReluGradKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const DenseTensor& alpha,
                     const DenseTensor& out_grad,
                     const std::string& data_format,
                     const std::string& mode,
                     DenseTensor* x_grad,
                     DenseTensor* alpha_grad) {
  const T* alpha_ptr = alpha.data<T>();
  const T* x_ptr = x.data<T>();
  const T* out_grad_ptr = out_grad.data<T>();
  int numel = x.numel();
  auto dim = x.dims();
  int index = 0;
  int i = 0;
  if (x_grad) {
    T* x_grad_ptr = dev_ctx.template Alloc<T>(x_grad);
    if (mode == "channel") {
      if (data_format == "NCHW") {
        int temp = 1;
        for (int j = 2; j < dim.size(); j++) {
          temp *= dim[j];
        }
        for (i = 0; i < numel; i++) {
          index = (i / temp) % dim[1];
          x_grad_ptr[i] = x_ptr[i] > 0 ? out_grad_ptr[i]
                                       : alpha_ptr[index] * out_grad_ptr[i];
        }
      } else {
        for (i = 0; i < numel; i++) {
          index = i % dim[dim.size() - 1];
          x_grad_ptr[i] = x_ptr[i] > 0 ? out_grad_ptr[i]
                                       : alpha_ptr[index] * out_grad_ptr[i];
        }
      }
    } else if (mode == "element") {
      int temp = 1;
      for (int j = 1; j < dim.size(); j++) {
        temp *= dim[j];
      }
      for (i = 0; i < numel; i++) {
        index = i % temp;
        x_grad_ptr[i] =
            x_ptr[i] > 0 ? out_grad_ptr[i] : alpha_ptr[index] * out_grad_ptr[i];
      }
    } else {
      for (i = 0; i < numel; i++) {
        x_grad_ptr[i] =
            x_ptr[i] > 0 ? out_grad_ptr[i] : alpha_ptr[0] * out_grad_ptr[i];
      }
    }
  }

  index = 0;
  if (alpha_grad) {
    T* alpha_grad_ptr = dev_ctx.template Alloc<T>(alpha_grad);
    memset(alpha_grad_ptr, 0, sizeof(T) * alpha_grad->numel());

    if (mode == "channel") {
      if (data_format == "NCHW") {
        int temp = 1;
        for (int j = 2; j < dim.size(); j++) {
          temp *= dim[j];
        }
        for (i = 0; i < numel; i++) {
          index = (i / temp) % dim[1];
          alpha_grad_ptr[index] +=
              x_ptr[i] > 0 ? 0 : x_ptr[i] * out_grad_ptr[i];
        }
      } else {
        for (i = 0; i < numel; i++) {
          index = i % dim[dim.size() - 1];
          alpha_grad_ptr[index] +=
              x_ptr[i] > 0 ? 0 : x_ptr[i] * out_grad_ptr[i];
        }
      }
    } else if (mode == "element") {
      int temp = 1;
      for (int j = 1; j < dim.size(); j++) {
        temp *= dim[j];
      }
      for (i = 0; i < numel; i++) {
        index = i % temp;
        alpha_grad_ptr[index] += x_ptr[i] > 0 ? 0 : x_ptr[i] * out_grad_ptr[i];
      }
    } else {
      for (i = 0; i < numel; i++) {
        alpha_grad_ptr[0] += x_ptr[i] > 0 ? 0 : x_ptr[i] * out_grad_ptr[i];
      }
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(
    prelu_grad, CPU, ALL_LAYOUT, phi::PReluGradKernel, float, double) {}
