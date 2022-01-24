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
#include "paddle/fluid/platform/float16.h"
#include "paddle/pten/core/kernel_registry.h"

#include "paddle/pten/infermeta/unary.h"
#include "paddle/pten/kernels/cpu/concat_and_split.h"
namespace pten {

template <typename T, typename Context>
void SplitKernel(const Context& dev_ctx,
                 const DenseTensor& x,
                 const ScalarArray& num_or_sections,
                 const Scalar& axis_scalar,
                 std::vector<DenseTensor*> outs) {
  auto ns = num_or_sections.GetData();
  for (size_t i = 0; i < ns.size(); ++i) {
    VLOG(0) << "num_or_sections[" << i << "]=" << ns[i];
  }
  VLOG(0) << "num_or_sections.IsInitByTensor() = "
          << num_or_sections.IsInitByTensor();
  VLOG(0) << "axis_scalar.IsInitByTensor() = " << axis_scalar.IsInitByTensor();
  // need to infershape output
  // if(num_or_sections.IsInitByTensor() || axis_scalar.IsInitByTensor()) {
  auto out_metas =
      pten::SplitInferMeta(x.meta(), num_or_sections, axis_scalar, true);

  VLOG(0) << "outs.size() = " << outs.size();
  VLOG(0) << "out_metas.size() = " << out_metas.size();
  for (size_t i = 0; i < out_metas.size(); ++i) {
    outs[i]->Resize(out_metas[i].dims);

    VLOG(0) << "outs[" << i << "].dims = " << out_metas[i].dims;
  }
  //}

  std::vector<const DenseTensor*> shape_refer;
  for (size_t j = 0; j < outs.size(); ++j) {
    VLOG(0) << "outs[" << j
            << "]->mutable_data begin, dims = " << outs[j]->dims();
    outs[j]->mutable_data<T>(dev_ctx.GetPlace());
    VLOG(0) << "outs[" << j
            << "]->mutable_data done, dims = " << outs[j]->dims();

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
                   CPU,
                   ALL_LAYOUT,
                   pten::SplitKernel,
                   float,
                   double,
                   int64_t,
                   int,
                   bool,
                   paddle::platform::float16) {}
