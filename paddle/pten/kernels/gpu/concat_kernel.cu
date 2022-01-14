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

#include "paddle/pten/kernels/concat_kernel.h"

#include "paddle/fluid/platform/complex.h"
#include "paddle/pten/core/kernel_registry.h"
#include "paddle/pten/kernels/gpu/concat_and_split.h"

#include "paddle/pten/backends/gpu/gpu_context.h"
#include "paddle/pten/kernels/funcs/concat_funcs.h"

namespace pten {

template <typename T, typename Context>
void ConcatKernel(const Context& dev_ctx,
                  const std::vector<DenseTensor>& x,
                  const Scalar& axis_scalar,
                  DenseTensor* out) {
  out->mutable_data<T>();
  int64_t axis = axis_scalar.to<int64_t>();

  axis = pten::funcs::ComputeAxis(axis, x[0].dims().size());

  ConcatImpl<T, Context>(dev_ctx, x, axis, out);
}

}  // namespace pten

PT_REGISTER_CTX_KERNEL(concat,
                       GPU,
                       ALL_LAYOUT,
                       pten::ConcatKernel,
                       float,
                       double,
                       bool,
                       int64_t,
                       int,
                       uint8_t,
                       paddle::platform::float16,
                       paddle::platform::complex<float>,
                       paddle::platform::complex<double>) {}
