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

#include "paddle/phi/kernels/funcs/maxouting.h"
#include "paddle/phi/kernels/maxout_kernel.h"

namespace phi {

template <typename T, typename Context>
void MaxOutKernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  int groups,
                  int axis,
                  DenseTensor* out) {
  if (axis < 0) {
    axis += x.dims().size();
  }

  phi::funcs::MaxOutFunctor<Context, T> maxout_forward;
  maxout_forward(dev_ctx, x, out, groups, axis);
}

}  // namespace phi
