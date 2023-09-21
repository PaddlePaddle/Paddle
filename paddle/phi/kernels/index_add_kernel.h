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

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/infermeta/binary.h"

namespace phi {

template <typename T, typename Context>
void IndexAddKernel(const Context& ctx,
                    const DenseTensor& x,
                    const DenseTensor& index,
                    const DenseTensor& add_value,
                    int axis,
                    DenseTensor* output);

template <typename T, typename Context>
DenseTensor IndexAdd(const Context& dev_ctx,
                     const DenseTensor& x,
                     const DenseTensor& index,
                     const DenseTensor& add_value,
                     int axis) {
  DenseTensor dense_out;
  MetaTensor meta_out(&dense_out);
  MetaTensor meta_x(&x);
  MetaTensor meta_index(&index);
  MetaTensor meta_add_value(&add_value);
  IndexAddInferMeta(meta_x, meta_index, meta_add_value, axis, &meta_out);
  IndexAddKernel<T, Context>(dev_ctx, x, index, add_value, axis, &dense_out);
  return dense_out;
}

}  // namespace phi
