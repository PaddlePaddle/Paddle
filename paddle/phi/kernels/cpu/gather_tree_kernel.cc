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

#include "paddle/phi/kernels/gather_tree_kernel.h"

#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void GatherTreeKernel(const Context &dev_ctx,
                      const DenseTensor &ids,
                      const DenseTensor &parents,
                      DenseTensor *out) {
  const auto *ids_data = ids.data<T>();
  const auto *parents_data = parents.data<T>();

  T *out_data = dev_ctx.template Alloc<T>(out);

  auto &ids_dims = ids.dims();
  int64_t max_length = ids_dims[0];
  auto batch_size = ids_dims[1];
  auto beam_size = ids_dims[2];

  PADDLE_ENFORCE_NOT_NULL(ids_data,
                          common::errors::InvalidArgument(
                              "Input(Ids) of gather_tree should not be null."));

  PADDLE_ENFORCE_NOT_NULL(
      parents_data,
      common::errors::InvalidArgument(
          "Input(Parents) of gather_tree should not be null."));

  for (int batch = 0; batch < batch_size; batch++) {
    for (int beam = 0; beam < beam_size; beam++) {
      auto idx =
          (max_length - 1) * batch_size * beam_size + batch * beam_size + beam;
      out_data[idx] = ids_data[idx];
      auto parent = parents_data[idx];
      for (int64_t step = max_length - 2; step >= 0; step--) {
        PADDLE_ENFORCE_LT(
            parent,
            beam_size,
            common::errors::InvalidArgument(
                "The parents must be less than beam size, but received "
                "parents %d is greater than or equal to beam size %d. ",
                parent,
                beam_size));

        PADDLE_ENFORCE_GE(
            parent,
            0,
            common::errors::InvalidArgument(
                "The parents must be greater than or equal to 0, but received "
                "parents %d is less than 0. ",
                parent));

        idx = step * batch_size * beam_size + batch * beam_size;
        out_data[idx + beam] = ids_data[idx + parent];
        parent = parents_data[idx + parent];
      }
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(
    gather_tree, CPU, ALL_LAYOUT, phi::GatherTreeKernel, int, int64_t) {}
