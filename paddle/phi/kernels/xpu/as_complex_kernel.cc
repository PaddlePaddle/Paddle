// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#ifdef PADDLE_WITH_XPU_FFT
#include "paddle/phi/kernels/as_complex_kernel.h"

#include "paddle/phi/common/complex.h"
#include "paddle/phi/common/type_traits.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"

namespace phi {
template <typename T, typename Context>
void AsComplexKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     DenseTensor* out) {
  dev_ctx.template Alloc<phi::dtype::complex<T>>(out);
  auto out_dims_original = out->dims();
  Copy(dev_ctx, x, dev_ctx.GetPlace(), false, out);
  out->Resize(out_dims_original);  // restored the shape.
  out->set_type(
      phi::CppTypeToDataType<phi::dtype::complex<T>>::Type());  // restored the
                                                                // dtype.
}

}  // namespace phi

PD_REGISTER_KERNEL(as_complex, XPU, ALL_LAYOUT, phi::AsComplexKernel, float) {
  kernel->OutputAt(0).SetDataType(phi::dtype::ToComplex(kernel_key.dtype()));
}
#endif  // PADDLE_WITH_XPU_FFT
