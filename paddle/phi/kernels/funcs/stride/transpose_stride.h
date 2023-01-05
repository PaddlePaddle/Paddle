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

#include "paddle/phi/core/kernel_registry.h"

namespace phi {
namespace funcs {

template <typename T, typename Context>
void TransposeStride(const Context& ctx,
                     const DenseTensor& x,
                     const std::vector<int>& axis,
                     DenseTensor* out) {
  auto meta = x.meta();
  int64_t* out_strides = meta.strides.GetMutable();
  const int64_t* in_strides = x.strides().Get();
  DDim in_dims = x.dims();
  for (size_t i = 0; i < axis.size(); i++) {
    out_strides[i] = in_strides[axis[i]];
    meta.dims[i] = in_dims[axis[i]];
  }
  meta.strides.set_valiable(true);
  meta.strides.set_contiguous(false);
  meta.strides.set_rank(in_dims.size());

  out->set_meta(meta);
  out->ResetHolder(x.Holder());
}

}  // namespace funcs
}  // namespace phi
