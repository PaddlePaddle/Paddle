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

#include "paddle/phi/core/kernel_registry.h"

template <typename T, typename Context>
void GatherTreeKernel(const Context &dev_ctx,
                      const DenseTensor &x,
                      int ids,
                      int parents,
                      int axis2,
                      DenseTensor *out) {
  auto *ids = ctx.Input<Tensor>("Ids");
  auto *parents = ctx.Input<Tensor>("Parents");
  auto *out = ctx.Output<Tensor>("Out");

  const auto *ids_data = ids->data<T>();
  const auto *parents_data = parents->data<T>();
  auto *out_data = out->mutable_data<T>(ctx.GetPlace());

  auto &ids_dims = ids->dims();
  auto max_length = ids_dims[0];
  auto batch_size = ids_dims[1];
  auto beam_size = ids_dims[2];

  PADDLE_ENFORCE_NOT_NULL(ids_data,
                          platform::errors::InvalidArgument(
                              "Input(Ids) of gather_tree should not be null."));

  PADDLE_ENFORCE_NOT_NULL(
      parents_data,
      platform::errors::InvalidArgument(
          "Input(Parents) of gather_tree should not be null."));

  for (int batch = 0; batch < batch_size; batch++) {
    for (int beam = 0; beam < beam_size; beam++) {
      auto idx =
          (max_length - 1) * batch_size * beam_size + batch * beam_size + beam;
      out_data[idx] = ids_data[idx];
      auto parent = parents_data[idx];
      for (int step = max_length - 2; step >= 0; step--) {
        idx = step * batch_size * beam_size + batch * beam_size;
        out_data[idx + beam] = ids_data[idx + parent];
        parent = parents_data[idx + parent];
      }
    }
  }
}

PD_REGISTER_KERNEL(abs_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::AbsGradKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   complex<float>,
                   complex<double>) {}
PD_REGISTER_KERNEL(abs_double_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::AbsDoubleGradKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   complex<float>,
                   complex<double>) {}
