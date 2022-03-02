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

#include "paddle/phi/kernels/reduce_sum_grad_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
namespace phi {

struct SumGradFunctor {
  template <typename DeviceContext,
            typename X,
            typename Y,
            typename DX,
            typename DY,
            typename Dim>
  void operator()(const DeviceContext& place,
                  X* x,
                  Y* y,
                  DX* dx,
                  DY* dy,
                  const Dim& dim,
                  int size) {
    dx->device(place) = dy->broadcast(dim);
  }
};

template <typename T, typename Context>
void ReduceSumGradKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         const DenseTensor& out_grad,
                         const std::vector<int64_t>& dims,
                         bool keep_dim,
                         bool reduce_all,
                         DataType in_dtype,
                         DataType out_dtype,
                         DenseTensor* x_grad) {}

}  // namespace phi

PD_REGISTER_KERNEL(sum_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::ReduceSumGradKernel,
                   bool,
                   float,
                   double,
                   phi::dtype::float16,
                   int,
                   int64_t,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
