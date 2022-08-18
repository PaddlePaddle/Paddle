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

#include "paddle/phi/kernels/funcs/blas/blas.h"

namespace phi {

template <typename T, typename Context>
void MvKernel(const Context& dev_ctx,
              const DenseTensor& x,
              const DenseTensor& vec,
              DenseTensor* out) {
  const auto& dim_x = x.dims();

  // get data ptr
  const T* x_data = x.data<T>();
  const T* vec_data = vec.data<T>();
  T* out_data = dev_ctx.template Alloc<T>(out);

  auto blas = phi::funcs::GetBlas<Context, T>(dev_ctx);

  blas.GEMV(false,
            dim_x[0],
            dim_x[1],
            static_cast<T>(1),
            x_data,
            vec_data,
            static_cast<T>(0),
            out_data);
}

}  // namespace phi
