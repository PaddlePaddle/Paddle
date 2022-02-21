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

#include "paddle/pten/kernels/split_kernel.h"

#include "paddle/fluid/operators/strided_memcpy.h"
#include "paddle/pten/common/float16.h"
#include "paddle/pten/core/kernel_registry.h"

#include "paddle/pten/kernels/gpu/concat_and_split.h"
namespace pten {

template <typename T, typename Context>
void SplitKernel(const Context& dev_ctx,
                 const DenseTensor& x,
                 const ScalarArray& num_or_sections,
                 const Scalar& axis_scalar,
                 std::vector<DenseTensor*> outs) {
  // need to infershape output
  if (num_or_sections.IsInitByTensor() || axis_scalar.IsInitByTensor()) {
    std::vector<MetaTensor> out_metas;
    for (size_t i = 0; i < outs.size(); ++i) {
      out_metas.push_back(outs[i]);
    }

    pten::SplitInferMeta(x, num_or_sections, axis_scalar, &out_metas, true);

    for (size_t i = 0; i < out_metas.size(); ++i) {
      outs[i]->Resize(out_metas[i].dims());
    }
  }

  std::vector<const DenseTensor*> shape_refer;
  for (size_t j = 0; j < outs.size(); ++j) {
    dev_ctx.Alloc(outs[j]);
    shape_refer.emplace_back(outs[j]);
  }

  int axis = axis_scalar.to<int>();
  // Sometimes direct copies will be faster, this maybe need deeply analysis.
  if (axis == 0 && outs.size() < 10) {
    paddle::operators::StridedMemcpyWithAxis0<T>(
        dev_ctx, x, shape_refer, &outs);
  } else {
    SplitImpl<T, Context>(dev_ctx, x, shape_refer, axis, &outs);
  }
}

}  // namespace pten

PT_REGISTER_KERNEL(split,
                   GPU,
                   ALL_LAYOUT,
                   pten::SplitKernel,
                   float,
                   double,
                   int64_t,
                   int,
                   bool,
                   pten::dtype::float16,
                   pten::dtype::bfloat16) {}
