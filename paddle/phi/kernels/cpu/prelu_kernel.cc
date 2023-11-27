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

#include "paddle/phi/kernels/prelu_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void PReluKernel(const Context& dev_ctx,
                 const DenseTensor& x,
                 const DenseTensor& alpha,
                 const std::string& data_format,
                 const std::string& mode,
                 DenseTensor* out) {
  const T* x_ptr = x.data<T>();
  const T* alpha_ptr = alpha.data<T>();
  T* o_ptr = dev_ctx.template Alloc<T>(out);

  int numel = static_cast<int>(x.numel());
  auto dim = x.dims();
  int index = 0;
  int i = 0;
  if (mode == "channel") {
    if (data_format == "NCHW") {
      int temp = 1;
      for (int j = 2; j < dim.size(); j++) {
        temp *= static_cast<int>(dim[j]);
      }
      for (i = 0; i < numel; i++) {
        index = static_cast<int>((i / temp) % dim[1]);
        o_ptr[i] = x_ptr[i] > 0 ? x_ptr[i] : alpha_ptr[index] * x_ptr[i];
      }
    } else {
      for (i = 0; i < numel; i++) {
        index = static_cast<int>(i % dim[dim.size() - 1]);
        o_ptr[i] = x_ptr[i] > 0 ? x_ptr[i] : alpha_ptr[index] * x_ptr[i];
      }
    }
  } else if (mode == "element") {
    int temp = 1;
    for (int j = 1; j < dim.size(); j++) {
      temp *= static_cast<int>(dim[j]);
    }
    for (i = 0; i < numel; i++) {
      index = i % temp;
      o_ptr[i] = x_ptr[i] > 0 ? x_ptr[i] : alpha_ptr[index] * x_ptr[i];
    }
  } else {
    for (i = 0; i < numel; i++) {
      o_ptr[i] = x_ptr[i] > 0 ? x_ptr[i] : alpha_ptr[0] * x_ptr[i];
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(prelu, CPU, ALL_LAYOUT, phi::PReluKernel, float, double) {}
