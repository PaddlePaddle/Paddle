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
  std::cout << "x.dims() = " << x.dims() << ", x.stride() = " << x.stride()
            << ", x.offset() = " << x.offset() << ", x.dtype() = " << x.dtype()
            << ", x.numel() = " << x.numel()
            << ", x.holder() = " << x.Holder()->ptr()
            << ", x.holder().size() = " << x.Holder()->size() << std::endl;
  if (out->Holder()) {
    std::cout << "out.dims() = " << out->dims()
              << ", out.stride() = " << out->stride()
              << ", out.offset() = " << out->offset()
              << ", out.dtype() = " << out->dtype()
              << ", out.numel() = " << out->numel()
              << ", out.holder() = " << out->Holder()->ptr()
              << ", out.holder().size() = " << out->Holder()->size()
              << std::endl;
  } else {
    std::cout << "out.dims() = " << out->dims()
              << ", out.stride() = " << out->stride()
              << ", out.offset() = " << out->offset()
              << ", out.dtype() = " << out->dtype()
              << ", out.numel() = " << out->numel() << std::endl;
  }
  if (xshape) {
    std::cout << "xshape.dims() = " << xshape->dims()
              << ", xshape.stride() = " << xshape->stride()
              << ", xshape.offset() = " << xshape->offset()
              << ", xshape.dtype() = " << xshape->dtype()
              << ", xshape.numel() = " << xshape->numel() << std::endl;
  }
  MetaTensor meta_out(out);
  InferMetaFromVecValue(x, shape.GetData(), &meta_out);
  DDim x_dims = x.dims();
  DDim x_stride = x.stride();
  size_t x_offset = x.offset();
  DDim stride;
  if (ReshapeStride(x_dims, x_stride, out->dims(), stride)) {
    out->set_offset(x_offset);
    out->set_stride(stride);
    std::cout << "2 x.dims() = " << x.dims() << ", x.stride() = " << x.stride()
              << ", x.offset() = " << x.offset()
              << ", x.dtype() = " << x.dtype() << ", x.numel() = " << x.numel()
              << ", x.holder() = " << x.Holder()->ptr()
              << ", x.holder().size() = " << x.Holder()->size() << std::endl;
    if (out->Holder()) {
      std::cout << "2 out.dims() = " << out->dims()
                << ", out.stride() = " << out->stride()
                << ", out.offset() = " << out->offset()
                << ", out.dtype() = " << out->dtype()
                << ", out.numel() = " << out->numel()
                << ", out.holder() = " << out->Holder()->ptr()
                << ", out.holder().size() = " << out->Holder()->size()
                << std::endl;
    } else {
      std::cout << "2 out.dims() = " << out->dims()
                << ", out.stride() = " << out->stride()
                << ", out.offset() = " << out->offset()
                << ", out.dtype() = " << out->dtype()
                << ", out.numel() = " << out->numel() << std::endl;
    }
    if (xshape) {
      std::cout << "2 xshape.dims() = " << xshape->dims()
                << ", xshape.stride() = " << xshape->stride()
                << ", xshape.offset() = " << xshape->offset()
                << ", xshape.dtype() = " << xshape->dtype()
                << ", xshape.numel() = " << xshape->numel() << std::endl;
    }
    out->ResetHolder(x.Holder());
  } else {
    std::cout << "reshape else!!!" << std::endl;
    DenseTensor tmp;
    tmp.set_meta(x.meta());
    PD_VISIT_ALL_TYPES(x.dtype(), "ReshapeStridedKernel", ([&] {
                         phi::ContiguousKernel<data_t, Context>(
                             dev_ctx, x, &tmp);
                       }));
    out->set_stride(DenseTensorMeta::calc_stride(out->dims()));
    out->ResetHolder(tmp.Holder());
  }
}

}  // namespace phi
PD_REGISTER_KERNEL_FOR_ALL_BACKEND_DTYPE_EXCEPT_CUSTOM(
    reshape, STRIDED, phi::ReshapeStridedKernel) {}
