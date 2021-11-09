// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/pten/kernels/cpu/nn.h"
#include "paddle/pten/core/kernel_registry.h"
#include "paddle/pten/kernels/cpu/funcs/elementwise.h"
#include "paddle/pten/kernels/functions/blas/elementwise.h"
#include "paddle/pten/kernels/functions/eigen/elementwise.h"
#include "paddle/pten/kernels/functions/general/elementwise_functor.h"

namespace pten {

template <typename T>
void ElementwiseAdd(const CPUContext& dev_ctx,
                    const DenseTensor& x,
                    const DenseTensor& y,
                    int axis,
                    DenseTensor* out) {
  if (x.dims() == y.dims()) {
    SameDimsElementwiseCompute<general::SameDimsAddFunctor<CPUContext, T>>()(
        dev_ctx, x, y, out);
  } else {
    auto x_dims = x.dims();
    auto y_dims = y.dims();
    if (x_dims.size() >= y_dims.size()) {
      ElementwiseCompute<general::AddFunctor<T>, T>(
          dev_ctx, x, y, axis, general::AddFunctor<T>(), out);
    } else {
      ElementwiseCompute<general::InverseAddFunctor<T>, T>(
          dev_ctx, x, y, axis, general::InverseAddFunctor<T>(), out);
    }
  }
}

// TODO(YuanRisheng) Some Check need to be done when args is
// SelectedRows(refer to InferShape in elementwise_op.h).

}  // namespace pten

PT_REGISTER_MODULE(NnCPU);

using complex64 = ::paddle::platform::complex<float>;
using complex128 = ::paddle::platform::complex<double>;

PT_REGISTER_KERNEL("elementwise_add",
                   CPU,
                   ANY,
                   pten::ElementwiseAdd,
                   float,
                   double,
                   int,
                   int64_t,
                   complex64,
                   complex128) {}
