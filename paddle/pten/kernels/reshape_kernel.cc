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

#include "paddle/pten/kernels/reshape_kernel.h"
#include "paddle/pten/backends/all_context.h"
#include "paddle/pten/core/kernel_registry.h"
#include "paddle/pten/infermeta/unary.h"
#include "paddle/pten/kernels/copy_kernel.h"
#include "paddle/pten/kernels/funcs/common_shape.h"

namespace pten {

template <typename Context>
void ReshapeKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   const ScalarArray& shape,
                   DenseTensor* out) {
  MetaTensor meta_out(out);
  InferMetaFromVecValue(x, shape.GetData(), &meta_out);
  if (x.initialized() && x.Holder() == out->Holder()) {
    dev_ctx.Alloc(out);
    return;
  }
  dev_ctx.Alloc(out);
  // TODO(chenweihang): the output dims are overwrite after copying,
  // here we need to use copy method that only copy data
  auto dims = out->dims();
  pten::Copy(dev_ctx, x, false, out);
  out->Resize(dims);
  out->ResetLoD(x.lod());
}

template <typename Context>
void ReshapeWithXShape(const Context& dev_ctx,
                       const DenseTensor& x,
                       const ScalarArray& shape,
                       DenseTensor* xshape,
                       DenseTensor* out) {
  funcs::SetXShape(x, xshape);
  ReshapeKernel(dev_ctx, x, shape, out);
}

}  // namespace pten

PT_REGISTER_GENERAL_KERNEL(reshape,
                           CPU,
                           ALL_LAYOUT,
                           pten::ReshapeKernel<pten::CPUContext>,
                           ALL_DTYPE) {}
PT_REGISTER_GENERAL_KERNEL(reshape_with_xshape,
                           CPU,
                           ALL_LAYOUT,
                           pten::ReshapeWithXShape<pten::CPUContext>,
                           ALL_DTYPE) {}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PT_REGISTER_GENERAL_KERNEL(reshape,
                           GPU,
                           ALL_LAYOUT,
                           pten::ReshapeKernel<pten::GPUContext>,
                           ALL_DTYPE) {}
PT_REGISTER_GENERAL_KERNEL(reshape_with_xshape,
                           GPU,
                           ALL_LAYOUT,
                           pten::ReshapeWithXShape<pten::GPUContext>,
                           ALL_DTYPE) {}
#endif

#ifdef PADDLE_WITH_XPU
PT_REGISTER_GENERAL_KERNEL(reshape,
                           XPU,
                           ALL_LAYOUT,
                           pten::ReshapeKernel<pten::XPUContext>,
                           ALL_DTYPE) {}
PT_REGISTER_GENERAL_KERNEL(reshape_with_xshape,
                           XPU,
                           ALL_LAYOUT,
                           pten::ReshapeWithXShape<pten::XPUContext>,
                           ALL_DTYPE) {}
#endif
