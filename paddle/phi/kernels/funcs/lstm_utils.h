// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
#include <string>

#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/mixed_vector.h"
#include "paddle/phi/kernels/funcs/detail/activation_functions.h"
#include "paddle/phi/kernels/funcs/sequence2batch.h"

namespace phi {

template <typename Context, typename T>
inline void ReorderInitState(const Context& dev_ctx,
                             const phi::DenseTensor& src,
                             phi::Vector<size_t> index_lod,
                             phi::DenseTensor* dst,
                             bool indexed_src) {
  phi::funcs::CopyMatrixRowsFunctor<Context, T> row_shuffle;
  dst->Resize(src.dims());
  dev_ctx.template Alloc<T>(dst);
  row_shuffle(dev_ctx, src, index_lod, dst, indexed_src);
}
}  // namespace phi
