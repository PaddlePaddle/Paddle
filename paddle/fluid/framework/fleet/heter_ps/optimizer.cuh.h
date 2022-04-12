/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
#ifdef PADDLE_WITH_HETERPS

#if defined(PADDLE_WITH_CUDA)
#include <curand_kernel.h>
#endif
#include <vector>
#include "paddle/fluid/framework/fleet/heter_ps/feature_value.h"
#include "paddle/fluid/framework/fleet/heter_ps/optimizer_conf.h"

namespace paddle {
namespace framework {

#if defined(PADDLE_WITH_CUDA)
template <typename ValType, typename GradType>
class Optimizer {
 public:
  Optimizer() {}

  ~Optimizer() {}

  void initialize() {}

  __device__ void update_lr(float& w, float& g2sum, float g,  // NOLINT
                            float scale) {
    double add_g2sum = 0;
    double ratio = optimizer_config::learning_rate *
                   sqrt(optimizer_config::initial_g2sum /
                        (optimizer_config::initial_g2sum + g2sum));
    double scaled_grad = g / scale;

    w += scaled_grad * ratio;

    if (w < optimizer_config::min_bound) w = optimizer_config::min_bound;
    if (w > optimizer_config::max_bound) w = optimizer_config::max_bound;

    add_g2sum += scaled_grad * scaled_grad;

    g2sum += add_g2sum;
  }

  __device__ void update_mf(int n, float* w, float& g2sum,  // NOLINT
                            const float* g, float scale) {
    double add_g2sum = 0;
    double ratio = optimizer_config::mf_learning_rate *
                   sqrt(optimizer_config::mf_initial_g2sum /
                        (optimizer_config::mf_initial_g2sum + g2sum));
    for (int i = 0; i < n; ++i) {
      double scaled_grad = g[i] / scale;

      w[i] += scaled_grad * ratio;

      if (w[i] < optimizer_config::mf_min_bound)
        w[i] = optimizer_config::mf_min_bound;
      if (w[i] > optimizer_config::mf_max_bound)
        w[i] = optimizer_config::mf_max_bound;
      add_g2sum += scaled_grad * scaled_grad;
    }

    g2sum += add_g2sum / n;
  }

  __device__ void update_value(ValType& val, const GradType& grad) {  // NOLINT
    val.slot = grad.slot;
    val.show += grad.show;
    val.clk += grad.clk;
    val.delta_score += optimizer_config::nonclk_coeff * (grad.show - grad.clk) +
                       optimizer_config::clk_coeff * grad.clk;

    update_lr(val.lr, val.lr_g2sum, grad.lr_g, grad.show);

    if (val.mf_size == 0) {
      if (optimizer_config::mf_create_thresholds <=
          optimizer_config::nonclk_coeff * (val.show - val.clk) +
              optimizer_config::clk_coeff * val.clk) {
        val.mf_size = MF_DIM + 1;
        val.mf[0] = 0;
        int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
        curandState state;
        curand_init(clock64(), tid_x, 0, &state);
        for (int i = 0; i < MF_DIM; ++i) {
          val.mf[i + 1] =
              (curand_uniform(&state)) * optimizer_config::mf_initial_range;
        }
      }
    } else {
      update_mf(MF_DIM, &val.mf[1], val.mf[0], grad.mf_g, grad.show);
    }
  }

  __device__ void dy_mf_update_value(ValType* ptr, const GradType& grad) {
    ptr->slot = grad.slot;
    ptr->show += grad.show;
    ptr->clk += grad.clk;
    ptr->delta_score +=
        optimizer_config::nonclk_coeff * (grad.show - grad.clk) +
        optimizer_config::clk_coeff * grad.clk;

    update_lr(ptr->lr, ptr->lr_g2sum, grad.lr_g, grad.show);
    // use MF_DIM temporarily
    // ptr->mf_dim = grad.mf_dim;

    if (ptr->mf_size == 0) {
      if (optimizer_config::mf_create_thresholds <=
          optimizer_config::nonclk_coeff * (ptr->show - ptr->clk) +
              optimizer_config::clk_coeff * ptr->clk) {
        // ptr->mf_size = ptr->mf_dim + 1;

        ptr->mf_size = MF_DIM + 1;
        ptr->mf[0] = 0;
        int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
        curandState state;
        curand_init(clock64(), tid_x, 0, &state);
        for (int i = 0; i < MF_DIM; ++i) {
          ptr->mf[i + 1] =
              (curand_uniform(&state)) * optimizer_config::mf_initial_range;
        }
      }
    } else {
      update_mf(MF_DIM, &(ptr->mf[1]), ptr->mf[0], grad.mf_g,
                grad.show);  // for local test
    }
  }
};

#endif
}  // end namespace framework
}  // end namespace paddle
#endif
