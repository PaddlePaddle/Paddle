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
#include <vector>
#include "paddle/phi/core/dense_tensor.h"
namespace phi {
namespace funcs {
template <typename Context, typename RepeatsT = int>
void RepeatsTensor2IndexTensor(const Context& ctx,
                               const DenseTensor& repeats,
                               DenseTensor* index) {
  DenseTensor repeats_cpu_copy;
  if (repeats.place().GetType() != phi::AllocationType::CPU) {
    phi::Copy(ctx, repeats, phi::CPUPlace(), true, &repeats_cpu_copy);
  }
  const RepeatsT* repeats_data =
      repeats.place().GetType() == phi::AllocationType::CPU
          ? repeats.data<RepeatsT>()
          : repeats_cpu_copy.data<RepeatsT>();

  int64_t index_size = 0;
  for (int i = 0; i < repeats.dims()[0]; i++) {
    PADDLE_ENFORCE_GE(repeats_data[i],
                      0,
                      phi::errors::InvalidArgument(
                          "repeats must grater or equal than 0, but got %d",
                          repeats_data[i]));
    index_size += repeats_data[i];
  }
  std::vector<RepeatsT> index_vec(index_size);
  int offset = 0;
  for (int i = 0; i < repeats.dims()[0]; i++) {
    std::fill_n(index_vec.begin() + offset, repeats_data[i], i);
    offset += repeats_data[i];
  }
  index->Resize(common::make_ddim({index_size}));

  phi::TensorFromVector<RepeatsT>(index_vec, ctx, index);
}
}  // namespace funcs
}  // namespace phi
