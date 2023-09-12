//   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/infermeta/unary.h"
#include "paddle/phi/kernels/funcs/common_shape.h"
#ifdef PADDLE_WITH_XPU
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#endif

namespace phi {

template <typename Context>
void ReshapeInferKernel(const Context& dev_ctx,
                        const DenseTensor& x,
                        const IntArray& shape,
                        DenseTensor* out) {
  MetaTensor meta_out(out);
  InferMetaFromVecValue(x, shape.GetData(), &meta_out);
  // Zero-Size Tensor
  if (x.numel() == 0) {
    return;
  }
  if (x.initialized() && x.Holder() == out->Holder()) {
    dev_ctx.Alloc(out, x.dtype());
    return;
  }
  dev_ctx.Alloc(out, x.dtype());
  // TODO(chenweihang): the output dims are overwrite after copying,
  // here we need to use copy method that only copy data
  auto dims = out->dims();
  phi::Copy(dev_ctx, x, dev_ctx.GetPlace(), false, out);
  out->Resize(dims);
  out->ResetLoD(x.lod());
}

#ifdef PADDLE_WITH_XPU
template <>
void ReshapeInferKernel<phi::XPUContext>(const XPUContext& dev_ctx,
                                         const DenseTensor& x,
                                         const IntArray& shape,
                                         DenseTensor* out) {
  MetaTensor meta_out(out);
  InferMetaFromVecValue(x, shape.GetData(), &meta_out);
  if (x.numel() == 0) {
    return;
  }
  if (x.initialized() && x.Holder() == out->Holder()) {
    dev_ctx.Alloc(out, x.dtype());
    return;
  }
  dev_ctx.Alloc(out, x.dtype());
  auto dims = out->dims();
  auto* src_ptr = x.data();
  auto* dst_ptr = out->data();
  auto size = x.numel() * phi::SizeOf(x.dtype());
  int ret = xpu::copy(dev_ctx.x_context(),
                      reinterpret_cast<const int8_t*>(src_ptr),
                      reinterpret_cast<int8_t*>(dst_ptr),
                      size);
  PADDLE_ENFORCE_XDNN_SUCCESS(ret, "copy");
  out->Resize(dims);
  out->ResetLoD(x.lod());
}
#endif

template <typename Context>
void ReshapeKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   const IntArray& shape,
                   DenseTensor* out,
                   DenseTensor* xshape UNUSED) {
  ReshapeInferKernel(dev_ctx, x, shape, out);
}

}  // namespace phi

PD_REGISTER_KERNEL_FOR_ALL_BACKEND_DTYPE(reshape_infer,
                                         ALL_LAYOUT,
                                         phi::ReshapeInferKernel) {}
PD_REGISTER_KERNEL_FOR_ALL_BACKEND_DTYPE(reshape,
                                         ALL_LAYOUT,
                                         phi::ReshapeKernel) {
  kernel->OutputAt(1).SetBackend(phi::Backend::UNDEFINED);
}
