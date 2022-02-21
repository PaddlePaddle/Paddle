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

#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/infermeta/unary.h"
#include "paddle/phi/kernels/copy_kernel.h"
#include "paddle/phi/kernels/flatten_kernel.h"
#include "paddle/phi/kernels/funcs/common_shape.h"

namespace phi {

template <typename T, typename Context>
void UnbindKernel(const Context& ctx,
                  const DenseTensor& x,
                  int axis,
                  std::vector<DenseTensor*> outs) {
  auto x_dims = x.dims();
  axis = axis < 0 ? x_dims.size() + axis : axis;

  std::vector<const DenseTensor*> shape_refer;
  for (size_t j = 0; j < outs.size(); ++j) {
    ctx.template Alloc<T>(outs[j]);
    shape_refer.emplace_back(outs[j]);
  }

  SplitImpl<T, Context>(ctx, x, shape_refer, axis, &outs);
}

}  // namespace phi

PT_REGISTER_KERNEL(unbind,
                   CPU,
                   ALL_LAYOUT,
                   phi::UnbindKernel,
                   float,
                   double,
                   dtype::float16,
                   dtype::bfloat16,
                   int,
                   int64_t) {}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PT_REGISTER_KERNEL(unbind,
                   GPU,
                   ALL_LAYOUT,
                   phi::UnbindKernel,
                   float,
                   double,
                   dtype::float16,
                   dtype::bfloat16,
                   int,
                   int64_t) {}
#endif
