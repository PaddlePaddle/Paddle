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
<<<<<<< HEAD
#ifdef PADDLE_WITH_XPU
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#endif
=======
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

namespace phi {

template <typename Context>
<<<<<<< HEAD
void ReshapeInferKernel(const Context& dev_ctx,
                        const DenseTensor& x,
                        const IntArray& shape,
                        DenseTensor* out) {
=======
void ReshapeKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   const IntArray& shape,
                   DenseTensor* out) {
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
  MetaTensor meta_out(out);
  InferMetaFromVecValue(x, shape.GetData(), &meta_out);
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

<<<<<<< HEAD
#ifdef PADDLE_WITH_XPU
template <>
void ReshapeInferKernel<phi::XPUContext>(const XPUContext& dev_ctx,
                                         const DenseTensor& x,
                                         const IntArray& shape,
                                         DenseTensor* out) {
  MetaTensor meta_out(out);
  InferMetaFromVecValue(x, shape.GetData(), &meta_out);
  if (x.initialized() && x.Holder() == out->Holder()) {
    dev_ctx.Alloc(out, x.dtype());
    return;
  }
  dev_ctx.Alloc(out, x.dtype());
  auto dims = out->dims();
  auto* src_ptr = x.data();
  auto* dst_ptr = out->data();
  auto size = x.numel() * paddle::experimental::SizeOf(x.dtype());
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
                   DenseTensor* xshape) {
  ReshapeInferKernel(dev_ctx, x, shape, out);
=======
template <typename Context>
void ReshapeWithXShape(const Context& dev_ctx,
                       const DenseTensor& x,
                       const IntArray& shape,
                       DenseTensor* out,
                       DenseTensor* xshape) {
  ReshapeKernel(dev_ctx, x, shape, out);
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
}

}  // namespace phi

<<<<<<< HEAD
PD_REGISTER_GENERAL_KERNEL(reshape_infer,
                           CPU,
                           ALL_LAYOUT,
                           phi::ReshapeInferKernel<phi::CPUContext>,
                           ALL_DTYPE) {}
PD_REGISTER_GENERAL_KERNEL(
    reshape, CPU, ALL_LAYOUT, phi::ReshapeKernel<phi::CPUContext>, ALL_DTYPE) {}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PD_REGISTER_GENERAL_KERNEL(reshape_infer,
                           GPU,
                           ALL_LAYOUT,
                           phi::ReshapeInferKernel<phi::GPUContext>,
                           ALL_DTYPE) {}
PD_REGISTER_GENERAL_KERNEL(
    reshape, GPU, ALL_LAYOUT, phi::ReshapeKernel<phi::GPUContext>, ALL_DTYPE) {}
#endif

#ifdef PADDLE_WITH_XPU
PD_REGISTER_GENERAL_KERNEL(reshape_infer,
                           XPU,
                           ALL_LAYOUT,
                           phi::ReshapeInferKernel<phi::XPUContext>,
                           ALL_DTYPE) {}
PD_REGISTER_GENERAL_KERNEL(
    reshape, XPU, ALL_LAYOUT, phi::ReshapeKernel<phi::XPUContext>, ALL_DTYPE) {}
=======
PD_REGISTER_GENERAL_KERNEL(
    reshape, CPU, ALL_LAYOUT, phi::ReshapeKernel<phi::CPUContext>, ALL_DTYPE) {}
PD_REGISTER_GENERAL_KERNEL(reshape_with_xshape,
                           CPU,
                           ALL_LAYOUT,
                           phi::ReshapeWithXShape<phi::CPUContext>,
                           ALL_DTYPE) {}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PD_REGISTER_GENERAL_KERNEL(
    reshape, GPU, ALL_LAYOUT, phi::ReshapeKernel<phi::GPUContext>, ALL_DTYPE) {}
PD_REGISTER_GENERAL_KERNEL(reshape_with_xshape,
                           GPU,
                           ALL_LAYOUT,
                           phi::ReshapeWithXShape<phi::GPUContext>,
                           ALL_DTYPE) {}
#endif

#ifdef PADDLE_WITH_XPU
PD_REGISTER_GENERAL_KERNEL(
    reshape, XPU, ALL_LAYOUT, phi::ReshapeKernel<phi::XPUContext>, ALL_DTYPE) {}
PD_REGISTER_GENERAL_KERNEL(reshape_with_xshape,
                           XPU,
                           ALL_LAYOUT,
                           phi::ReshapeWithXShape<phi::XPUContext>,
                           ALL_DTYPE) {}
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
#endif
