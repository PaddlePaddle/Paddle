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
#include <utility>
#include <vector>

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"

namespace phi {

template <typename T, typename Context>
void PartialSumKernel(const Context& dev_ctx,
                      const std::vector<const DenseTensor*>& x,
                      int start_index,
                      int length,
                      DenseTensor* out) {
  PADDLE_ENFORCE_EQ(
      x.size() > 0,
      true,
      phi::errors::InvalidArgument("The input should not be null."));

  auto* out_t = dev_ctx.template Alloc<T>(out);
  auto batch_size = x[0]->dims()[0];
  if (length == -1) {
    length = x[0]->dims()[1] - start_index;
  }

  memset(out_t, 0, sizeof(T) * batch_size * length);

  for (size_t i = 0; i < x.size(); ++i) {
    auto* in_t = x[i]->data<T>();
    auto total_len = x[i]->dims()[1];
    for (auto bs_id = 0; bs_id < batch_size; ++bs_id) {
      for (auto k = 0; k < length; ++k) {
        out_t[bs_id * length + k] += in_t[bs_id * total_len + start_index + k];
      }
    }
  }
}

template <typename T, typename Context>
void PartialSumGradientOpKernel(const Context& dev_ctx,
                                const std::vector<const DenseTensor*>& x,
                                const DenseTensor& out_grad,
                                int start_index,
                                int length,
                                std::vector<DenseTensor*> x_grad) {
  PADDLE_ENFORCE_EQ(
      x.size() > 0,
      true,
      phi::errors::InvalidArgument("The input should not be null."));
  auto batch_size = x[0]->dims()[0];
  if (length == -1) {
    length = x[0]->dims()[1] - start_index;
  }

  // initialize
  auto& place = *dev_ctx.eigen_device();
  for (size_t i = 0; i < x_grad.size(); ++i) {
    dev_ctx.template Alloc<T>(x_grad[i]);
    auto dxt = phi::EigenVector<T>::Flatten(*x_grad[i]);
    dxt.device(place) = dxt.constant(static_cast<T>(0));
  }

  auto* out_grad_t = out_grad.data<T>();
  for (size_t i = 0; i < x_grad.size(); ++i) {
    auto* out_t = x_grad[i]->data<T>();
    auto total_len = x[i]->dims()[1];
    for (auto bs_id = 0; bs_id < batch_size; ++bs_id) {
      for (int len = 0; len < length; ++len) {
        out_t[start_index + bs_id * total_len + len] =
            out_grad_t[bs_id * length + len] * static_cast<T>(1);
      }
    }
  }
}

}  // namespace phi
