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

namespace phi {
namespace fusion {

template <typename T, typename Context>
void FusionSquaredMatSubKernel(const Context& dev_ctx,
                               const DenseTensor& x,
                               const DenseTensor& y,
                               const float scalar,
                               DenseTensor* squared_x,
                               DenseTensor* squared_y,
                               DenseTensor* squared_xy,
                               DenseTensor* out) {
  T scalar_t = static_cast<T>(scalar);

  auto x_dims = x.dims();
  auto y_dims = y.dims();
  phi::jit::matmul_attr_t attr;
  attr.m = static_cast<int>(x_dims[0]);
  attr.k = static_cast<int>(x_dims[1]);
  attr.n = static_cast<int>(y_dims[1]);
  int o_numel = attr.m * attr.n;

  auto vsquare_x =
      phi::jit::KernelFuncs<phi::jit::VSquareTuple<T>, phi::CPUPlace>::Cache()
          .At(attr.m * attr.k);
  auto vsquare_y =
      phi::jit::KernelFuncs<phi::jit::VSquareTuple<T>, phi::CPUPlace>::Cache()
          .At(attr.k * attr.n);
  auto vsquare_xy =
      phi::jit::KernelFuncs<phi::jit::VSquareTuple<T>, phi::CPUPlace>::Cache()
          .At(o_numel);
  auto vsub =
      phi::jit::KernelFuncs<phi::jit::VSubTuple<T>, phi::CPUPlace>::Cache().At(
          o_numel);
  auto vscal =
      phi::jit::KernelFuncs<phi::jit::VScalTuple<T>, phi::CPUPlace>::Cache().At(
          o_numel);
  auto matmul =
      phi::jit::KernelFuncs<phi::jit::MatMulTuple<T>, phi::CPUPlace>::Cache()
          .At(attr);

  const T* x_data = x.data<T>();
  const T* y_data = y.data<T>();
  T* squared_x_data = dev_ctx.template Alloc<T>(squared_x);
  T* squared_y_data = dev_ctx.template Alloc<T>(squared_y);
  T* squared_xy_data = dev_ctx.template Alloc<T>(squared_xy);
  T* o_data = dev_ctx.template Alloc<T>(out);

  matmul(x_data, y_data, squared_xy_data, &attr);
  vsquare_xy(squared_xy_data, squared_xy_data, o_numel);

  vsquare_x(x_data, squared_x_data, attr.m * attr.k);
  vsquare_y(y_data, squared_y_data, attr.k * attr.n);
  matmul(squared_x_data, squared_y_data, o_data, &attr);

  vsub(squared_xy_data, o_data, o_data, o_numel);
  vscal(&scalar_t, o_data, o_data, o_numel);
}

}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(fusion_squared_mat_sub,
                   CPU,
                   ALL_LAYOUT,
                   phi::fusion::FusionSquaredMatSubKernel,
                   float,
                   double) {}
