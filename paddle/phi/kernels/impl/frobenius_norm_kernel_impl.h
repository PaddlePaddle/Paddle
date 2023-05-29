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

#include "paddle/phi/kernels/cpu/reduce.h"
#include "paddle/phi/kernels/frobenius_norm_kernel.h"
#include "paddle/phi/kernels/funcs/reduce_functor.h"

namespace phi {

template <typename T, typename Context>
void FrobeniusNormKernel(const Context& ctx,
                         const DenseTensor& x,
                         const std::vector<int64_t>& axis,
                         bool keep_dim,
                         bool reduce_all,
                         DenseTensor* out) {
  reduce_all = recompute_reduce_all(x, axis, reduce_all);
  Reduce<Context, T, funcs::FrobeniusNormFunctor>(
      ctx, x, reduce_all, axis, keep_dim, x.dtype(), out);
}

}  // namespace phi
