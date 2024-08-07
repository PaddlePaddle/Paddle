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

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/generator.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/expand_kernel.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"

namespace phi {

template <typename T, typename Context>
void ComputeDropoutInference(const Context& ctx,
                             const DenseTensor& x,
                             const Scalar& dropout_prob,
                             bool upscale_in_train,
                             DenseTensor* y) {
  if (upscale_in_train) {
    const auto* X_data = x.data<T>();
    T* Y_data = ctx.template Alloc<T>(y);
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
    for (int i = 0; i < x.numel(); i++) {
      Y_data[i] = X_data[i];
    }
  } else {
    auto X = EigenMatrix<T>::Reshape(x, 1);
    auto Y = EigenMatrix<T>::Reshape(*y, 1);
    auto& place = *ctx.eigen_device();
    Y.device(place) = X * static_cast<T>(1.0f - dropout_prob.to<float>());
  }
}

template <typename T, typename Context>
void DropoutRawKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const paddle::optional<DenseTensor>& seed_tensor,
                      const Scalar& p,
                      bool is_test,
                      const std::string& mode,
                      int seed,
                      bool fix_seed,
                      DenseTensor* out,
                      DenseTensor* mask) {
  auto* y = out;
  const auto* x_data = x.data<T>();
  T* y_data = dev_ctx.template Alloc<T>(y);
  float dropout_prob = p.to<float>();

  auto& dropout_implementation = mode;
  bool upscale_in_train = (dropout_implementation == "upscale_in_train");
  if (!is_test && mask) {
    auto* mask_data = dev_ctx.template Alloc<uint8_t>(mask);
    size_t size = common::product(mask->dims());

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
    std::shared_ptr<std::mt19937_64> engine;
    if (seed_data) {
      engine = std::make_shared<std::mt19937_64>();
      engine->seed(seed_data);
    } else {
      engine = dev_ctx.GetGenerator()->GetCPUEngine();
    }

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
    ComputeDropoutInference<T, Context>(
        dev_ctx, x, dropout_prob, upscale_in_train, y);
  }
}

template <typename T, typename Context>
void DropoutNdKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const paddle::optional<DenseTensor>& seed_tensor,
                     const Scalar& p,
                     bool is_test,
                     const std::string& mode,
                     int seed,
                     bool fix_seed,
                     const std::vector<int>& axis,
                     DenseTensor* out,
                     DenseTensor* mask) {
  auto* y = out;
  const auto* x_data = x.data<T>();
  T* y_data = dev_ctx.template Alloc<T>(y);
  float dropout_prob = p.to<float>();

  auto& dropout_implementation = mode;
  bool upscale_in_train = (dropout_implementation == "upscale_in_train");
  if (!is_test && mask) {
    DenseTensor t_mask;
    t_mask.Resize(mask->dims());
    T* t_mask_data = dev_ctx.template Alloc<T>(&t_mask);
    auto* mask_data = dev_ctx.template Alloc<uint8_t>(mask);
    size_t size = common::product(mask->dims());

    // Special case when dropout_prob is 1.0
    if (dropout_prob == 1.0f) {
      std::memset(y_data, 0, size * sizeof(*y_data));            // NOLINT
      std::memset(t_mask_data, 0, size * sizeof(*t_mask_data));  // NOLINT
      std::memset(mask_data, 0, size * sizeof(*mask_data));      // NOLINT
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
    std::shared_ptr<std::mt19937_64> engine;
    if (seed_data) {
      engine = std::make_shared<std::mt19937_64>();
      engine->seed(seed_data);
    } else {
      engine = dev_ctx.GetGenerator()->GetCPUEngine();
    }

    std::uniform_real_distribution<float> dist(0, 1);

    for (size_t i = 0; i < size; ++i) {
      if (dist(*engine) < dropout_prob) {
        t_mask_data[i] = 0;
        mask_data[i] = 0;
      } else {
        t_mask_data[i] = 1;
        mask_data[i] = 1;
      }
    }
    auto& x_dims = x.dims();
    DenseTensor broadcast_mask;
    broadcast_mask.Resize(x_dims);
    T* broadcast_mask_data = dev_ctx.template Alloc<T>(&broadcast_mask);

    std::vector<int64_t> mask_bst_dims_vec;
    for (int i = 0; i < x_dims.size(); i++) {
      mask_bst_dims_vec.emplace_back(x_dims[i]);
    }
    IntArray mask_bst_dims(mask_bst_dims_vec);
    ExpandKernel<T, Context>(dev_ctx, t_mask, mask_bst_dims, &broadcast_mask);

    for (auto i = 0; i < x.numel(); i++) {
      if (broadcast_mask_data[i] == static_cast<T>(1)) {
        if (upscale_in_train) {
          y_data[i] = x_data[i] / static_cast<T>(1.0f - dropout_prob);
        } else {
          y_data[i] = x_data[i];
        }
      } else {
        y_data[i] = 0;
      }
    }
  } else {
    ComputeDropoutInference<T, Context>(
        dev_ctx, x, dropout_prob, upscale_in_train, y);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(dropout,
                   CPU,
                   ALL_LAYOUT,
                   phi::DropoutRawKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {
  kernel->OutputAt(1).SetDataType(phi::DataType::UINT8);
}

PD_REGISTER_KERNEL(
    dropout_nd, CPU, ALL_LAYOUT, phi::DropoutNdKernel, float, double) {
  kernel->OutputAt(1).SetDataType(phi::DataType::UINT8);
}
