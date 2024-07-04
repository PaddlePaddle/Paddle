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
#include <vector>

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/jit/kernels.h"

namespace phi::fusion {

template <typename T>
static void fc_relu(const T* x,
                    const T* w,
                    const T* b,
                    T* y,
                    const phi::jit::matmul_attr_t& attr) {
  auto matmul =
      phi::jit::KernelFuncs<phi::jit::MatMulTuple<T>, phi::CPUPlace>::Cache()
          .At(attr);
  auto addbias_relu =
      phi::jit::KernelFuncs<phi::jit::VAddReluTuple<T>, phi::CPUPlace>::Cache()
          .At(attr.n);
  matmul(x, w, y, &attr);
  T* dst = y;
  for (int i = 0; i < attr.m; ++i) {
    addbias_relu(b, dst, dst, attr.n);
    dst += attr.n;
  }
}

template <typename T, typename Context>
void FusionRepeatedFCReluKernel(const Context& dev_ctx,
                                const DenseTensor& x,
                                const std::vector<const DenseTensor*>& w,
                                const std::vector<const DenseTensor*>& bias,
                                std::vector<DenseTensor*> relu_out,
                                DenseTensor* out) {
  int weight_sz = static_cast<int>(w.size());

  auto i_dims = common::vectorize<int>(x.dims());
  const auto& w_dims = w[0]->dims();
  phi::jit::matmul_attr_t attr;
  attr.m = i_dims[0];
  attr.n = static_cast<int>(w_dims[1]);
  attr.k = static_cast<int>(w_dims[0]);
  relu_out[0]->Resize({attr.m, attr.n});
  auto* relu_out_temp = dev_ctx.template Alloc<T>(relu_out[0]);
  fc_relu(
      x.data<T>(), w[0]->data<T>(), bias[0]->data<T>(), relu_out_temp, attr);

  for (int i = 1; i < weight_sz - 1; ++i) {
    const auto& i_dims = relu_out[i - 1]->dims();
    const auto& w_dims = w[i]->dims();
    attr.m = static_cast<int>(i_dims[0]);
    attr.n = static_cast<int>(w_dims[1]);
    attr.k = static_cast<int>(w_dims[0]);
    relu_out[i]->Resize({attr.m, attr.n});
    auto* relu_out_tmp = dev_ctx.template Alloc<T>(relu_out[i]);
    fc_relu(relu_out[i - 1]->data<T>(),
            w[i]->data<T>(),
            bias[i]->data<T>(),
            relu_out_tmp,
            attr);
  }

  const auto& i_dims_last = relu_out[weight_sz - 2]->dims();
  const auto& w_dims_last = w[weight_sz - 1]->dims();
  attr.m = static_cast<int>(i_dims_last[0]);
  attr.n = static_cast<int>(w_dims_last[1]);
  attr.k = static_cast<int>(w_dims_last[0]);
  auto* out_data = dev_ctx.template Alloc<T>(out);
  fc_relu(relu_out[weight_sz - 2]->data<T>(),
          w[weight_sz - 1]->data<T>(),
          bias[weight_sz - 1]->data<T>(),
          out_data,
          attr);
}

}  // namespace phi::fusion

PD_REGISTER_KERNEL(fusion_repeated_fc_relu,
                   CPU,
                   ALL_LAYOUT,
                   phi::fusion::FusionRepeatedFCReluKernel,
                   float,
                   double) {}
