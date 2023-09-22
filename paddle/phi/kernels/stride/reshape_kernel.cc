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

#include "paddle/phi/kernels/reshape_kernel.h"
#include <algorithm>
#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/contiguous_kernel.h"
#include "paddle/phi/kernels/funcs/strided_reshape_utils.h"

namespace phi {
template <typename Context>
void ReshapeStridedKernel(const Context& dev_ctx,
                          const DenseTensor& x,
                          const IntArray& shape,
                          DenseTensor* out,
                          DenseTensor* xshape) {
  DDim x_dims = x.dims();
  DDim x_stride = x.strides();
  size_t x_offset = x.offset();
  if (xshape) {
    x_dims = DDim(xshape->dims().Get() + 1, xshape->dims().size() - 1);
    x_stride = xshape->strides();
  }
  MetaTensor meta_out(out);
  InferMetaFromVecValue(x, shape.GetData(), &meta_out);

  DDim stride;
  if (ReshapeStride(x_dims, x_stride, out->dims(), stride)) {
    out->set_offset(x_offset);
    out->set_strides(stride);
    out->ResetHolder(x.Holder());
  } else {
    DenseTensor tmp;
    DenseTensor tmp_x = x;
    tmp_x.Resize(x_dims);
    tmp_x.set_strides(x_stride);
    tmp.set_meta(tmp_x.meta());
    PD_VISIT_ALL_TYPES(x.dtype(), "ReshapeStridedKernel", ([&] {
                         phi::ContiguousKernel<data_t, Context>(
                             dev_ctx, tmp_x, &tmp);
                       }));
    out->set_strides(DenseTensorMeta::calc_strides(out->dims()));
    out->set_offset(0);
    out->ResetHolder(tmp.Holder());
  }
}

}  // namespace phi
PD_REGISTER_KERNEL_FOR_ALL_BACKEND_DTYPE_EXCEPT_CUSTOM(
    reshape, STRIDED, phi::ReshapeStridedKernel) {}
