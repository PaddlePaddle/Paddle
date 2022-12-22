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

#include <algorithm>

#include "paddle/phi/core/ddim.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/reshape_kernel.h"
#include "paddle/phi/kernels/transpose_kernel.h"

namespace phi {
namespace fusion {

template <typename T, typename Context>
void TransposeReshapeFusionKernel(const Context& dev_ctx,
                                  const DenseTensor& x,
                                  bool trans_first,
                                  const std::vector<int>& axis,
                                  const std::vector<int>& shape,
                                  DenseTensor* out) {
  auto* x_data = x.data<T>();
  auto out_dims = out->dims();
  phi::DenseTensor tmp;
  if (trans_first) {
    size_t x_rank = x.dims().size();
    size_t axis_size = axis.size();
    phi::DDim transpose_dims(x.dims());
    for (size_t i = 0; i < axis_size; i++) {
      transpose_dims[i] = x.dims()[axis[i]];
    }
    tmp.Resize(transpose_dims);
    phi::TransposeKernel<T>(dev_ctx, x, axis, &tmp);
    out->ShareDataWith(tmp);
    out->Resize(out_dims);
  } else {
    // MetaTensor meta_out(tmp);
    // std::vector<int64_t> shapes(shape.size());
    // for (size_t i = 0; i < shape.size(); ++i) {
    //   shapes[i] = shape[i];
    // }
    // InferMetaFromVecValue(x, shapes, &meta_out);
    tmp.ShareBufferWith(x);
    phi::ReshapeKernel(dev_ctx, x, shape, &tmp);
    dev_ctx.template Alloc<T>(out);
    phi::TransposeKernel<T>(dev_ctx, tmp, axis, out);
  }
}

}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(fused_transpose_reshape,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::TransposeReshapeFusionKernel,
                   float,
                   double,
                   int32_t,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
