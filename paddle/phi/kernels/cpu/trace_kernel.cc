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

#include "paddle/phi/kernels/trace_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/diagonal.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"

namespace phi {

template <typename T, typename Context>
void TraceKernel(const Context& dev_ctx,
                 const DenseTensor& x,
                 int offset,
                 int axis1,
                 int axis2,
                 DenseTensor* out) {
  auto* out_data = dev_ctx.template Alloc<T>(out);

  const DenseTensor diag =
      funcs::Diagonal<T, Context>(dev_ctx, &x, offset, axis1, axis2);
  if (diag.numel() > 0) {
    auto x = phi::EigenMatrix<T>::Reshape(diag, diag.dims().size() - 1);
    auto output = phi::EigenVector<T>::Flatten(*out);
    auto reduce_dim = Eigen::array<int, 1>({1});
    output.device(*dev_ctx.eigen_device()) = x.sum(reduce_dim);
    out->Resize(out->dims());
  } else {
    std::fill(out_data, out_data + out->numel(), static_cast<T>(0));
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(trace,
                   CPU,
                   ALL_LAYOUT,
                   phi::TraceKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
