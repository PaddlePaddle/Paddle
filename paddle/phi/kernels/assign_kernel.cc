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

#include "paddle/phi/kernels/assign_kernel.h"

#include "paddle/phi/kernels/copy_kernel.h"

namespace phi {

template <typename Context>
void AssignKernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  DenseTensor* out) {
  if (x.numel() == 0) {
    return;
  }
  dev_ctx.Alloc(out, x.dtype());
  Copy<Context>(dev_ctx, src, false, out);
}

template <typename Context>
void AssignArrayKernel(const Context& dev_ctx,
                       const std::vector<DenseTensor*>& x,
                       std::vector<DenseTensor*> out) {
  for (size_t i = 0; i < x.size(); ++i) {
    AssignKernel<Context>(dev_ctx, *x[i], out.at(i));
  }
}

}  // namespace phi
