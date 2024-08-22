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
#include "paddle/phi/kernels/as_real_kernel.h"
#include "paddle/common/flags.h"
#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/kernel_registry.h"

COMMON_DECLARE_bool(use_stride_kernel);

namespace phi {

template <typename T, typename Context>
void AsRealStridedKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         DenseTensor* out) {
  if (!FLAGS_use_stride_kernel) {
    PADDLE_THROW(common::errors::Fatal(
        "FLAGS_use_stride_kernel is closed. Strided kernel "
        "be called, something wrong has happened!"));
  }
  auto out_stride_v = common::vectorize(x.strides());
  for (auto& v : out_stride_v) {
    v *= 2;
  }
  out_stride_v.push_back(1);
  out->set_strides(common::make_ddim(out_stride_v));

  if (x.dtype() == DataType::COMPLEX64) {
    out->set_type(DataType::FLOAT32);
  } else if (x.dtype() == DataType::COMPLEX128) {
    out->set_type(DataType::FLOAT64);
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "as_real is not supported data type (%s).",
        DataTypeToString(x.dtype())));
  }
  out->set_offset(x.offset());
  out->ResetHolder(x.Holder());
  out->ShareInplaceVersionCounterWith(x);
}

}  // namespace phi
PD_REGISTER_KERNEL(as_real,
                   CPU,
                   STRIDED,
                   phi::AsRealStridedKernel,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {
  kernel->OutputAt(0).SetDataType(phi::DataType::UNDEFINED);
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PD_REGISTER_KERNEL(as_real,
                   GPU,
                   STRIDED,
                   phi::AsRealStridedKernel,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {
  kernel->OutputAt(0).SetDataType(phi::DataType::UNDEFINED);
}
#endif

#ifdef PADDLE_WITH_CUSTOM_DEVICE
PD_REGISTER_KERNEL(as_real,
                   Custom,
                   STRIDED,
                   phi::AsRealStridedKernel,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {
  kernel->OutputAt(0).SetDataType(phi::DataType::UNDEFINED);
}
#endif
