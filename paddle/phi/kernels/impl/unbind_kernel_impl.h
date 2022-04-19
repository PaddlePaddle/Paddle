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

#pragma once

#include "paddle/phi/kernels/funcs/concat_and_split_functor.h"
#include "paddle/phi/kernels/unbind_kernel.h"

namespace phi {

template <typename T, typename Context>
void UnbindKernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  int axis,
                  std::vector<DenseTensor*> outs) {
  auto x_dims = x.dims();
  axis = axis < 0 ? x_dims.size() + axis : axis;

  std::vector<const DenseTensor*> shape_refer;
  for (size_t j = 0; j < outs.size(); ++j) {
    dev_ctx.template Alloc<T>(outs[j]);
    shape_refer.emplace_back(outs[j]);
  }

  phi::funcs::SplitFunctor<Context, T> functor;
  functor(dev_ctx, x, shape_refer, axis, &outs);
}

}  // namespace phi
