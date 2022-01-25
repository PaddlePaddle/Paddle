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

#include "paddle/pten/kernels/trace_kernel.h"
#include "paddle/pten/backends/cpu/cpu_context.h"
#include "paddle/pten/core/kernel_registry.h"
#include "paddle/pten/kernels/impl/trace_kernel_impl.h"

namespace pten {

template <typename T, typename Context>
void TraceKernel(const Context& ctx,
                 const DenseTensor& x,
                 int offset,
                 int axis1,
                 int axis2,
                 DenseTensor* out) {
  auto output_dims = out->dims();

  T* out_data = out->mutable_data<T>(ctx.GetPlace());

  const DenseTensor diag = Diagonal<T, Context>(ctx, &x, offset, axis1, axis2);
  if (diag.numel() > 0) {
    auto x = paddle::framework::EigenMatrix<T>::Reshape(diag,
                                                        diag.dims().size() - 1);
    auto output = paddle::framework::EigenVector<T>::Flatten(*out);
    auto reduce_dim = Eigen::array<int, 1>({1});
    output.device(*ctx.eigen_device()) = x.sum(reduce_dim);
    out->Resize(output_dims);
  } else {
    std::fill(out_data, out_data + out->numel(), static_cast<T>(0));
  }
}

}  // namespace pten

PT_REGISTER_KERNEL(trace,
                   CPU,
                   ALL_LAYOUT,
                   pten::TraceKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   paddle::platform::float16,
                   paddle::platform::complex<float>,
                   paddle::platform::complex<double>) {}
