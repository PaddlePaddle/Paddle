/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/eigvalsh_kernel.h"

#include "paddle/phi/kernels/funcs/values_vectors_functor.h"

namespace phi {

template <typename T, typename Context>
void EigvalshKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    const std::string& uplo,
                    bool is_test,
                    DenseTensor* out_w,
                    DenseTensor* out_v) {
  bool is_lower = (uplo == "L");
  phi::funcs::MatrixEighFunctor<Context, T> functor;
  if (is_test) {
    functor(dev_ctx, x, out_w, nullptr, is_lower, false);
  } else {
    functor(dev_ctx, x, out_w, out_v, is_lower, true);
  }
}

}  // namespace phi
