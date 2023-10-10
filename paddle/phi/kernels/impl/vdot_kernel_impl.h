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

#pragma once

#include "paddle/phi/common/complex.h"
#include "paddle/phi/kernels/complex_kernel.h"
#include "paddle/phi/kernels/dot_kernel.h"
#include "paddle/phi/kernels/vdot_kernel.h"

namespace phi {
template <typename T, typename Context>
void VdotKernel(const Context& dev_ctx,
                const DenseTensor& x,
                const DenseTensor& y,
                DenseTensor* out) {
  if (std::is_same<T, phi::dtype::complex<float>>::value ||
      std::is_same<T, phi::dtype::complex<double>>::value) {
    DenseTensor* conj = new DenseTensor();
    conj->Resize(x.dims());
    ConjKernel<T, Context>(dev_ctx, x, conj);
    DotKernel<T, Context>(dev_ctx, *conj, y, out);
  } else {
    DotKernel<T, Context>(dev_ctx, x, y, out);
  }
}
}  // namespace phi
