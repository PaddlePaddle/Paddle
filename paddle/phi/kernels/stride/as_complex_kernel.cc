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
#include "paddle/phi/kernels/as_complex_kernel.h"
#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/common/complex.h"
#include "paddle/phi/common/type_traits.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void AsComplexStridedKernel(const Context& dev_ctx,
                            const DenseTensor& x,
                            DenseTensor* out) {
  out->set_strides(DenseTensorMeta::calc_strides(out->dims()));
  if (x.dtype() == DataType::FLOAT32) {
    out->set_type(DataType::COMPLEX64);
  } else if (x.dtype() == DataType::FLOAT64) {
    out->set_type(DataType::COMPLEX128);
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "as_complex is not supported data type (%s).",
        DataTypeToString(x.dtype())));
  }
  out->set_offset(x.offset());
  out->ResetHolder(x.Holder());
  out->ShareInplaceVersionCounterWith(x);
}

}  // namespace phi

PD_REGISTER_KERNEL(
    as_complex, CPU, STRIDED, phi::AsComplexStridedKernel, float, double) {
  kernel->OutputAt(0).SetDataType(phi::dtype::ToReal(kernel_key.dtype()));
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PD_REGISTER_KERNEL(
    as_complex, GPU, STRIDED, phi::AsComplexStridedKernel, float, double) {
  kernel->OutputAt(0).SetDataType(phi::dtype::ToReal(kernel_key.dtype()));
}
#endif
