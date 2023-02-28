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

#include "paddle/phi/kernels/view_kernel.h"

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
void ViewInferKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const IntArray& shape,
                     DenseTensor* out) {
  MetaTensor meta_out(out);
  InferMetaFromVecValue(x, shape.GetData(), &meta_out);
  if (x.initialized() && x.Holder() == out->Holder()) {
    dev_ctx.Alloc(out, x.dtype());
    return;
  }
  out->ResetHolder(x.Holder());
}

#ifdef PADDLE_WITH_XPU
template <>
void ViewInferKernel<phi::XPUContext>(const XPUContext& dev_ctx,
                                      const DenseTensor& x,
                                      const IntArray& shape,
                                      DenseTensor* out) {
  MetaTensor meta_out(out);
  InferMetaFromVecValue(x, shape.GetData(), &meta_out);
  if (x.initialized() && x.Holder() == out->Holder()) {
    dev_ctx.Alloc(out, x.dtype());
    return;
  }
  out->ResetHolder(x.Holder())
}
#endif

#ifdef PADDLE_WITH_CUDA
template <>
void ViewInferKernel<phi::GPUContext>(const GPUContext& dev_ctx,
                                      const DenseTensor& x,
                                      const IntArray& shape,
                                      DenseTensor* out) {
  MetaTensor meta_out(out);
  InferMetaFromVecValue(x, shape.GetData(), &meta_out);
  if (x.initialized() && x.Holder() == out->Holder()) {
    dev_ctx.Alloc(out, x.dtype());
    return;
  }
  LOG(WARNING) << "use stride view kernel";
  out->ResetHolder(x.Holder());
}
#endif
template <typename Context>
void ViewKernel(const Context& dev_ctx,
                const DenseTensor& x,
                const IntArray& shape,
                DenseTensor* out) {
  ViewInferKernel(dev_ctx, x, shape, out);
}

}  // namespace phi

PD_REGISTER_GENERAL_KERNEL(view_infer,
                           CPU,
                           ALL_LAYOUT,
                           phi::ViewInferKernel<phi::CPUContext>,
                           ALL_DTYPE) {}
PD_REGISTER_GENERAL_KERNEL(
    view, CPU, ALL_LAYOUT, phi::ViewKernel<phi::CPUContext>, ALL_DTYPE) {}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PD_REGISTER_GENERAL_KERNEL(view_infer,
                           GPU,
                           ALL_LAYOUT,
                           phi::ViewInferKernel<phi::GPUContext>,
                           ALL_DTYPE) {}
PD_REGISTER_GENERAL_KERNEL(
    view, GPU, ALL_LAYOUT, phi::ViewKernel<phi::GPUContext>, ALL_DTYPE) {}
#endif

#ifdef PADDLE_WITH_XPU
PD_REGISTER_GENERAL_KERNEL(view_infer,
                           XPU,
                           ALL_LAYOUT,
                           phi::ViewInferKernel<phi::XPUContext>,
                           ALL_DTYPE) {}
PD_REGISTER_GENERAL_KERNEL(
    view, XPU, ALL_LAYOUT, phi::ViewKernel<phi::XPUContext>, ALL_DTYPE) {}
#endif
