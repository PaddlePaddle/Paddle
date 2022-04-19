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

#include "paddle/phi/kernels/dropout_kernel.h"
#include "paddle/fluid/framework/generator.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"

namespace phi {

template <typename T, typename Context>
void DropoutRawKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      paddle::optional<const DenseTensor&> seed_tensor,
                      float p,
                      bool is_test,
                      const std::string& mode,
                      int seed,
                      bool fix_seed,
                      DenseTensor* out,
                      DenseTensor* mask) {
  auto* y = out;
  const auto* x_data = x.data<T>();
  auto* y_data = y->mutable_data<T>(dev_ctx.GetPlace());
  float dropout_prob = p;

  auto& dropout_implementation = mode;
  bool upscale_in_train = (dropout_implementation == "upscale_in_train");
  if (!is_test) {
    auto* mask_data = mask->mutable_data<uint8_t>(dev_ctx.GetPlace());
    size_t size = phi::product(mask->dims());

    // Special case when dropout_prob is 1.0
    if (dropout_prob == 1.0f) {
      std::memset(y_data, 0, size * sizeof(*y_data));        // NOLINT
      std::memset(mask_data, 0, size * sizeof(*mask_data));  // NOLINT
      return;
    }
    // std::minstd_rand engine;
    // NOTE: fixed seed should only be used in unittest or for debug.
    // Guarantee to use random seed in training.
    int seed_data = 0;
    if (seed_tensor.get_ptr() != nullptr) {
      seed_data = *(seed_tensor->data<int>());
    } else {
      seed_data = fix_seed ? seed : 0;
    }
    auto engine = paddle::framework::GetCPURandomEngine(seed_data);

    std::uniform_real_distribution<float> dist(0, 1);

    for (size_t i = 0; i < size; ++i) {
      if (dist(*engine) < dropout_prob) {
        mask_data[i] = 0;
        y_data[i] = 0;
      } else {
        mask_data[i] = 1;
        if (upscale_in_train) {
          y_data[i] = x_data[i] / static_cast<T>(1.0f - dropout_prob);
        } else {
          y_data[i] = x_data[i];
        }
      }
    }
  } else {
    if (upscale_in_train) {
      const auto* X_data = x.data<T>();
      auto* Y_data = y->mutable_data<T>(dev_ctx.GetPlace());
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
      for (int i = 0; i < x.numel(); i++) {
        Y_data[i] = X_data[i];
      }
    } else {
      auto X = EigenMatrix<T>::Reshape(x, 1);
      auto Y = EigenMatrix<T>::Reshape(*y, 1);
      auto& place = *dev_ctx.eigen_device();
      Y.device(place) = X * static_cast<T>(1.0f - dropout_prob);
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(dropout,
                   CPU,
                   ALL_LAYOUT,
                   phi::DropoutRawKernel,
                   float,
                   double,
                   phi::dtype::bfloat16) {}
