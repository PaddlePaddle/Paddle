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

#include "paddle/pten/kernels/cpu/manipulation.h"
#include "paddle/pten/api/ext/dispatch.h"
#include "paddle/pten/infermeta/unary.h"
#include "paddle/pten/kernels/cpu/utils.h"
#include "paddle/pten/kernels/hybird/general/manipulation.h"

namespace pten {

void Reshape(const CPUContext& dev_ctx,
             const DenseTensor& x,
             const ScalarArray& shape,
             DenseTensor* out) {
  auto out_meta = InferMetaFromVecValue(x.meta(), shape.GetData());
  if (x.data() == out->data() && x.numel() == out->numel()) {
    out->Resize(out_meta.dims);
    return;
  }
  pten::Copy(dev_ctx, x, false, out);
  out->Resize(out_meta.dims);
  out->ResetLoD(x.lod());
}

void ReshapeWithXShape(const CPUContext& dev_ctx,
                       const DenseTensor& x,
                       const ScalarArray& shape,
                       DenseTensor* xshape,
                       DenseTensor* out) {
  general::SetXShape(x, xshape);
  Reshape(dev_ctx, x, shape, out);
}

}  // namespace pten

PT_REGISTER_NO_TEMPLATE_KERNEL(
    reshape, CPU, ALL_LAYOUT, pten::Reshape, ALL_DTYPE) {}
PT_REGISTER_NO_TEMPLATE_KERNEL(
    reshape_with_xshape, CPU, ALL_LAYOUT, pten::ReshapeWithXShape, ALL_DTYPE) {}
