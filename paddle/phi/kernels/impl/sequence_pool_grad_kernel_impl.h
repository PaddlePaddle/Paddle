/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include "paddle/phi/kernels/funcs/sequence_pooling.h"

namespace phi {

template <typename T, typename Context>
void SequencePoolGradKernel(const Context& dev_ctx,
                            const DenseTensor& x UNUSED,
                            const paddle::optional<DenseTensor>& max_index,
                            const DenseTensor& out_grad,
                            bool is_test UNUSED,
                            const std::string& pooltype,
                            float pad_value UNUSED,
                            DenseTensor* x_grad) {
  const phi::DenseTensor* index = nullptr;
  if (pooltype == "MAX") {
    index = max_index.get_ptr();
  }
  dev_ctx.template Alloc<T>(x_grad);
  phi::funcs::SequencePoolGradFunctor<Context, T> pool;
  pool(dev_ctx, pooltype, out_grad, x_grad, index);
}

}  // namespace phi
