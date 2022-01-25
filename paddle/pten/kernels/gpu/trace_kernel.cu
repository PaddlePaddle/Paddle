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

#include "paddle/pten/backends/cpu/cpu_context.h"
#include "paddle/pten/core/kernel_registry.h"
#include "paddle/pten/kernels/impl/trace_kernel_impl.h"
#include "paddle/pten/kernels/trace_kernel.h"

namespace pten {

template <typename T, typename Context>
void ScaleKernel(const Context& ctx,
                 const DenseTensor& x,
                 int offset,
                 int axis1,
                 int axis2,
                 DenseTensor* out) {
  T* out_data = out->mutable_data<T>(ctx.GetPlace());
  auto diag = Diagonal<DeviceContext, T>(context, input, offset, dim1, dim2);
  if (diag.numel() > 0) {
    auto stream = context.cuda_device_context().stream();
    std::vector<int> reduce_dims;
    reduce_dims.push_back(out->dims().size());
    kernels::TensorReduceFunctorImpl<Tx,
                                     Ty,
                                     kps::AddFunctor,
                                     kps::IdentityFunctor<T>>(
        input, out, kps::IdentityFunctor<T>(), reduce_dims, stream);
  } else {
    math::SetConstant<DeviceContext, T> functor;
    functor(context.device_context<DeviceContext>(), out, static_cast<T>(0));
  }
}

}  // namespace pten

PT_REGISTER_KERNEL(trace,
                   GPU,
                   ALL_LAYOUT,
                   pten::TraceKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   paddle::platform::complex<float>,
                   paddle::platform::complex<double>) {}
