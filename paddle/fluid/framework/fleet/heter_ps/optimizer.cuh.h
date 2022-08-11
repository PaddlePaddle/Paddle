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

template <typename GPUAccessor>
class SparseAdagradOptimizer {
 public:
  SparseAdagradOptimizer() {}
  SparseAdagradOptimizer(GPUAccessor gpu_accessor) {
    gpu_accessor_ = gpu_accessor;
    _lr_embedding_dim = 1;
    _embedding_dim = gpu_accessor_.common_feature_value.EmbedWDim();
  }

  ~SparseAdagradOptimizer() {}

  __device__ void update_value_work(const OptimizerConfig& optimizer_config,
                                    int n,
                                    float* w,
                                    float* sgd,  // NOLINT
                                    const float* g,
                                    float scale,
                                    float slot) {
    float& g2sum = sgd[G2SumIndex()];
    double add_g2sum = 0;

    float learning_rate = optimizer_config.mf_learning_rate;
    if (slot != optimizer_config.nodeid_slot) {
      learning_rate = optimizer_config.feature_learning_rate;
    }
    double ratio =
        learning_rate * sqrt(optimizer_config.mf_initial_g2sum /
                             (optimizer_config.mf_initial_g2sum + g2sum));
    for (int i = 0; i < n; ++i) {
      double scaled_grad = g[i] / scale;

      w[i] += scaled_grad * ratio;

      if (w[i] < optimizer_config.mf_min_bound)
        w[i] = optimizer_config.mf_min_bound;
      if (w[i] > optimizer_config.mf_max_bound)
        w[i] = optimizer_config.mf_max_bound;
      add_g2sum += scaled_grad * scaled_grad;
    }

    g2sum += add_g2sum / n;
  }

  __device__ void update_value(const OptimizerConfig& optimizer_config,
                               float& val,  // NOLINT
                               const float& grad) {
    printf(
        "Warning: update_value will not used. Please use dy_mf_update_value\n");
  }
  __device__ void dy_mf_update_value(const OptimizerConfig& optimizer_config,
                                     float* ptr,
                                     const float* grad) {
    float g_show = grad[gpu_accessor_.common_push_value.ShowIndex()];
    float g_click = grad[gpu_accessor_.common_push_value.ClickIndex()];

    ptr[gpu_accessor_.common_feature_value.SlotIndex()] =
        grad[gpu_accessor_.common_push_value.SlotIndex()];
    ptr[gpu_accessor_.common_feature_value.ShowIndex()] += g_show;
    ptr[gpu_accessor_.common_feature_value.ClickIndex()] += g_click;
    ptr[gpu_accessor_.common_feature_value.DeltaScoreIndex()] +=
        optimizer_config.nonclk_coeff * (g_show - g_click) +
        optimizer_config.clk_coeff * g_click;
    float slot = ptr[gpu_accessor_.common_feature_value.SlotIndex()];

    update_value_work(
        optimizer_config,
        1,
        ptr + gpu_accessor_.common_feature_value.EmbedWIndex(),
        ptr + gpu_accessor_.common_feature_value.EmbedG2SumIndex(),
        grad + gpu_accessor_.common_push_value.EmbedGIndex(),
        g_show,
        slot);

    int mf_dim = int(ptr[gpu_accessor_.common_feature_value.MfDimIndex()]);
    if (ptr[gpu_accessor_.common_feature_value.MfSizeIndex()] == 0) {
      if (optimizer_config.mf_create_thresholds <=
          optimizer_config.nonclk_coeff *
                  (ptr[gpu_accessor_.common_feature_value.ShowIndex()] -
                   ptr[gpu_accessor_.common_feature_value.ClickIndex()]) +
              optimizer_config.clk_coeff *
                  ptr[gpu_accessor_.common_feature_value.ClickIndex()]) {
        ptr[gpu_accessor_.common_feature_value.MfSizeIndex()] =
            gpu_accessor_.common_feature_value.MFSize(mf_dim) / sizeof(float);

        int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
        curandState state;
        curand_init(clock64(), tid_x, 0, &state);
        for (int i = 0; i < mf_dim; ++i) {
          ptr[gpu_accessor_.common_feature_value.EmbedxWIndex() + i] =
              (curand_uniform(&state)) * optimizer_config.mf_initial_range;
        }
      }
    } else {
      update_value_work(
          optimizer_config,
          mf_dim,
          ptr + gpu_accessor_.common_feature_value.EmbedxWIndex(),
          ptr + gpu_accessor_.common_feature_value.EmbedxG2SumIndex(),
          grad + gpu_accessor_.common_push_value.EmbedxGIndex(),
          g_show,
          slot);
    }
  }

  __host__ __device__ size_t Dim() { return EmbedDim() + EmbedxDim(); }
  __host__ __device__ size_t EmbedDim() { return _lr_embedding_dim; }
  __host__ __device__ size_t EmbedxDim() { return _embedding_dim; }
  __host__ __device__ size_t G2SumIndex() { return 0; }
  __host__ __device__ size_t EmbedxG2SumIndex() { return 0; }

 private:
  GPUAccessor gpu_accessor_;
  size_t _embedding_dim;
  size_t _lr_embedding_dim;
};

template <typename GPUAccessor>
class SparseAdamOptimizer {
 public:
  SparseAdamOptimizer() {}
  SparseAdamOptimizer(GPUAccessor gpu_accessor) {
    gpu_accessor_ = gpu_accessor;
    _lr_embedding_dim = 1;
    _embedding_dim = gpu_accessor_.common_feature_value.EmbedWDim();
  }

  ~SparseAdamOptimizer() {}

  __device__ void update_lr(const OptimizerConfig& optimizer_config,
                            int n,
                            float* w,
                            float* sgd,
                            const float* g,
                            float scale) {
    float* moment1 = sgd + GSumIndex();
    float* moment2 = sgd + G2SumIndex();
    float* beta1_pow = sgd + Beta1PowIndex();
    float* beta2_pow = sgd + Beta2PowIndex();

    float beta1_pow_ = *beta1_pow;
    float beta2_pow_ = *beta2_pow;

    float epsilon = 1e-08;
    double ratio = optimizer_config.learning_rate * sqrt(1.0 - beta2_pow_) /
                   (1.0 - beta1_pow_);
    for (int i = 0; i < n; ++i) {
      double scaled_grad = g[i] / scale;

      double new_moment1 =
          optimizer_config.beta1_decay_rate * moment1[i] +
          (1.0 - optimizer_config.beta1_decay_rate) * scaled_grad;
      double new_moment2 =
          optimizer_config.beta2_decay_rate * moment2[i] +
          (1.0 - optimizer_config.beta2_decay_rate) * scaled_grad * scaled_grad;
      w[i] += ratio * (new_moment1 / (sqrt(new_moment2) + epsilon));

      if (w[i] < optimizer_config.mf_min_bound)
        w[i] = optimizer_config.mf_min_bound;
      if (w[i] > optimizer_config.mf_max_bound)
        w[i] = optimizer_config.mf_max_bound;

      moment1[i] = new_moment1;
      moment2[i] = new_moment2;
    }
    (*beta1_pow) *= optimizer_config.beta1_decay_rate;
    (*beta2_pow) *= optimizer_config.beta2_decay_rate;
  }

  __device__ void update_mf(const OptimizerConfig& optimizer_config,
                            int n,
                            float* w,
                            float* sgd,
                            const float* g,
                            float scale) {
    float* moment1 = sgd + EmbedxGSumIndex();
    float* moment2 = sgd + EmbedxG2SumIndex();
    float* beta1_pow = sgd + EmbedxBeta1PowIndex();
    float* beta2_pow = sgd + EmbedxBeta2PowIndex();

    float beta1_pow_ = *beta1_pow;
    float beta2_pow_ = *beta2_pow;

    float epsilon = 1e-08;
    double ratio = optimizer_config.learning_rate * sqrt(1.0 - beta2_pow_) /
                   (1.0 - beta1_pow_);
    for (int i = 0; i < n; ++i) {
      double scaled_grad = g[i] / scale;

      double new_moment1 =
          optimizer_config.beta1_decay_rate * moment1[i] +
          (1.0 - optimizer_config.beta1_decay_rate) * scaled_grad;
      double new_moment2 =
          optimizer_config.beta2_decay_rate * moment2[i] +
          (1.0 - optimizer_config.beta2_decay_rate) * scaled_grad * scaled_grad;
      w[i] += ratio * (new_moment1 / (sqrt(new_moment2) + epsilon));

      if (w[i] < optimizer_config.mf_min_bound)
        w[i] = optimizer_config.mf_min_bound;
      if (w[i] > optimizer_config.mf_max_bound)
        w[i] = optimizer_config.mf_max_bound;

      moment1[i] = new_moment1;
      moment2[i] = new_moment2;
    }
    (*beta1_pow) *= optimizer_config.beta1_decay_rate;
    (*beta2_pow) *= optimizer_config.beta2_decay_rate;
  }

  __device__ void update_value(const OptimizerConfig& optimizer_config,
                               float& val,  // NOLINT
                               const float& grad) {
    printf(
        "Warning: update_value will not used. Please use dy_mf_update_value\n");
  }
  __device__ void dy_mf_update_value(const OptimizerConfig& optimizer_config,
                                     float* ptr,
                                     const float* grad) {
    float g_show = grad[gpu_accessor_.common_push_value.ShowIndex()];
    float g_click = grad[gpu_accessor_.common_push_value.ClickIndex()];

    ptr[gpu_accessor_.common_feature_value.SlotIndex()] =
        grad[gpu_accessor_.common_push_value.SlotIndex()];
    ptr[gpu_accessor_.common_feature_value.ShowIndex()] += g_show;
    ptr[gpu_accessor_.common_feature_value.ClickIndex()] += g_click;
    ptr[gpu_accessor_.common_feature_value.DeltaScoreIndex()] +=
        optimizer_config.nonclk_coeff * (g_show - g_click) +
        optimizer_config.clk_coeff * g_click;

    update_lr(optimizer_config,
              1,
              ptr + gpu_accessor_.common_feature_value.EmbedWIndex(),
              ptr + gpu_accessor_.common_feature_value.EmbedG2SumIndex(),
              grad + gpu_accessor_.common_push_value.EmbedGIndex(),
              g_show);
    int mf_dim = int(ptr[gpu_accessor_.common_feature_value.MfDimIndex()]);
    if (ptr[gpu_accessor_.common_feature_value.MfSizeIndex()] == 0) {
      if (optimizer_config.mf_create_thresholds <=
          optimizer_config.nonclk_coeff *
                  (ptr[gpu_accessor_.common_feature_value.ShowIndex()] -
                   ptr[gpu_accessor_.common_feature_value.ClickIndex()]) +
              optimizer_config.clk_coeff *
                  ptr[gpu_accessor_.common_feature_value.ClickIndex()]) {
        ptr[gpu_accessor_.common_feature_value.MfSizeIndex()] =
            gpu_accessor_.common_feature_value.MFSize(mf_dim) / sizeof(float);

        int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
        curandState state;
        curand_init(clock64(), tid_x, 0, &state);
        for (int i = 0; i < mf_dim; ++i) {
          ptr[gpu_accessor_.common_feature_value.EmbedxWIndex() + i] =
              (curand_uniform(&state)) * optimizer_config.mf_initial_range;
        }
        ptr[gpu_accessor_.common_feature_value.EmbedxG2SumIndex() +
            EmbedxBeta1PowIndex()] = optimizer_config.beta1_decay_rate;
        ptr[gpu_accessor_.common_feature_value.EmbedxG2SumIndex() +
            EmbedxBeta2PowIndex()] = optimizer_config.beta2_decay_rate;
      }
    } else {
      update_mf(optimizer_config,
                mf_dim,
                ptr + gpu_accessor_.common_feature_value.EmbedxWIndex(),
                ptr + gpu_accessor_.common_feature_value.EmbedxG2SumIndex(),
                grad + gpu_accessor_.common_push_value.EmbedxGIndex(),
                g_show);
    }
    // printf("EmbedxGIndex: %f, mf_gsum: %f, ",
    // gpu_accessor_.common_push_value.EmbedxGIndex(),
    //          ptr[gpu_accessor_.common_feature_value.EmbedxG2SumIndex()]);
  }

  __host__ __device__ size_t Dim() { return EmbedDim() + EmbedxDim(); }
  __host__ __device__ size_t EmbedDim() { return _lr_embedding_dim * 2 + 2; }
  __host__ __device__ size_t EmbedxDim() { return _embedding_dim * 2 + 2; }
  __host__ __device__ size_t GSumIndex() { return 0; }
  __host__ __device__ size_t G2SumIndex() {
    return GSumIndex() + _lr_embedding_dim;
  }
  __host__ __device__ size_t Beta1PowIndex() {
    return G2SumIndex() + _lr_embedding_dim;
  }
  __host__ __device__ size_t Beta2PowIndex() { return Beta1PowIndex() + 1; }
  __host__ __device__ size_t EmbedxGSumIndex() { return 0; }
  __host__ __device__ size_t EmbedxG2SumIndex() {
    return EmbedxGSumIndex() + _embedding_dim;
  }
  __host__ __device__ size_t EmbedxBeta1PowIndex() {
    return EmbedxG2SumIndex() + _embedding_dim;
  }
  __host__ __device__ size_t EmbedxBeta2PowIndex() {
    return EmbedxBeta1PowIndex() + 1;
  }

 private:
  GPUAccessor gpu_accessor_;
  size_t _embedding_dim;
  size_t _lr_embedding_dim;
};

template <typename GPUAccessor>
class SparseAdamSharedOptimizer {
 public:
  SparseAdamSharedOptimizer() {}
  SparseAdamSharedOptimizer(GPUAccessor gpu_accessor) {
    gpu_accessor_ = gpu_accessor;
    _lr_embedding_dim = 1;
    _embedding_dim = gpu_accessor_.common_feature_value.EmbedWDim();
  }

  ~SparseAdamSharedOptimizer() {}

  __device__ void update_value_work(const OptimizerConfig& optimizer_config,
                                    int n,
                                    float* w,
                                    float* sgd,
                                    const float* g,
                                    float scale) {
    float* moment1 = sgd + GSumIndex();
    float* moment2 = sgd + G2SumIndex();
    float* beta1_pow = sgd + Beta1PowIndex();
    float* beta2_pow = sgd + Beta2PowIndex();

    float beta1_pow_ = *beta1_pow;
    float beta2_pow_ = *beta2_pow;
    float moment1_ = *moment1;
    float moment2_ = *moment2;
    float epsilon = 1e-08;
    double ratio = optimizer_config.learning_rate * sqrt(1.0 - beta2_pow_) /
                   (1.0 - beta1_pow_);

    double sum_mom1 = 0.0;
    double sum_mom2 = 0.0;
    for (int i = 0; i < n; ++i) {
      double scaled_grad = g[i] / scale;

      double new_moment1 =
          optimizer_config.beta1_decay_rate * moment1_ +
          (1.0 - optimizer_config.beta1_decay_rate) * scaled_grad;
      double new_moment2 =
          optimizer_config.beta2_decay_rate * moment2_ +
          (1.0 - optimizer_config.beta2_decay_rate) * scaled_grad * scaled_grad;
      w[i] += ratio * (new_moment1 / (sqrt(new_moment2) + epsilon));

      if (w[i] < optimizer_config.mf_min_bound)
        w[i] = optimizer_config.mf_min_bound;
      if (w[i] > optimizer_config.mf_max_bound)
        w[i] = optimizer_config.mf_max_bound;

      sum_mom1 += new_moment1;
      sum_mom2 += new_moment2;
    }

    (*moment1) = sum_mom1 / n;
    (*moment2) = sum_mom2 / n;
    (*beta1_pow) *= optimizer_config.beta1_decay_rate;
    (*beta2_pow) *= optimizer_config.beta2_decay_rate;
  }

  __device__ void update_value(const OptimizerConfig& optimizer_config,
                               float& val,  // NOLINT
                               const float& grad) {
    printf(
        "Warning: update_value will not used. Please use dy_mf_update_value\n");
  }

  __device__ void dy_mf_update_value(const OptimizerConfig& optimizer_config,
                                     float* ptr,
                                     const float* grad) {
    float g_show = grad[gpu_accessor_.common_push_value.ShowIndex()];
    float g_click = grad[gpu_accessor_.common_push_value.ClickIndex()];

    ptr[gpu_accessor_.common_feature_value.SlotIndex()] =
        grad[gpu_accessor_.common_push_value.SlotIndex()];
    ptr[gpu_accessor_.common_feature_value.ShowIndex()] += g_show;
    ptr[gpu_accessor_.common_feature_value.ClickIndex()] += g_click;
    ptr[gpu_accessor_.common_feature_value.DeltaScoreIndex()] +=
        optimizer_config.nonclk_coeff * (g_show - g_click) +
        optimizer_config.clk_coeff * g_click;

    update_value_work(
        optimizer_config,
        1,
        ptr + gpu_accessor_.common_feature_value.EmbedWIndex(),
        ptr + gpu_accessor_.common_feature_value.EmbedG2SumIndex(),
        grad + gpu_accessor_.common_push_value.EmbedGIndex(),
        g_show);
    int mf_dim = int(ptr[gpu_accessor_.common_feature_value.MfDimIndex()]);
    if (ptr[gpu_accessor_.common_feature_value.MfSizeIndex()] == 0) {
      if (optimizer_config.mf_create_thresholds <=
          optimizer_config.nonclk_coeff *
                  (ptr[gpu_accessor_.common_feature_value.ShowIndex()] -
                   ptr[gpu_accessor_.common_feature_value.ClickIndex()]) +
              optimizer_config.clk_coeff *
                  ptr[gpu_accessor_.common_feature_value.ClickIndex()]) {
        ptr[gpu_accessor_.common_feature_value.MfSizeIndex()] =
            gpu_accessor_.common_feature_value.MFSize(mf_dim) / sizeof(float);

        int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
        curandState state;
        curand_init(clock64(), tid_x, 0, &state);
        for (int i = 0; i < mf_dim; ++i) {
          ptr[gpu_accessor_.common_feature_value.EmbedxWIndex() + i] =
              (curand_uniform(&state)) * optimizer_config.mf_initial_range;
        }
        ptr[gpu_accessor_.common_feature_value.EmbedxG2SumIndex() +
            EmbedxBeta1PowIndex()] = optimizer_config.beta1_decay_rate;
        ptr[gpu_accessor_.common_feature_value.EmbedxG2SumIndex() +
            EmbedxBeta2PowIndex()] = optimizer_config.beta2_decay_rate;
      }
    } else {
      update_value_work(
          optimizer_config,
          mf_dim,
          ptr + gpu_accessor_.common_feature_value.EmbedxWIndex(),
          ptr + gpu_accessor_.common_feature_value.EmbedxG2SumIndex(),
          grad + gpu_accessor_.common_push_value.EmbedxGIndex(),
          g_show);
    }
  }

  __host__ __device__ size_t Dim() { return EmbedDim() + EmbedxDim(); }
  __host__ __device__ size_t EmbedDim() { return 4; }
  __host__ __device__ size_t EmbedxDim() { return 4; }
  __host__ __device__ size_t GSumIndex() { return 0; }
  __host__ __device__ size_t G2SumIndex() { return GSumIndex() + 1; }
  __host__ __device__ size_t Beta1PowIndex() { return G2SumIndex() + 1; }
  __host__ __device__ size_t Beta2PowIndex() { return Beta1PowIndex() + 1; }
  __host__ __device__ size_t EmbedxGSumIndex() { return 0; }
  __host__ __device__ size_t EmbedxG2SumIndex() {
    return EmbedxGSumIndex() + 1;
  }
  __host__ __device__ size_t EmbedxBeta1PowIndex() {
    return EmbedxG2SumIndex() + 1;
  }
  __host__ __device__ size_t EmbedxBeta2PowIndex() {
    return EmbedxBeta1PowIndex() + 1;
  }

 private:
  GPUAccessor gpu_accessor_;
  size_t _embedding_dim;
  size_t _lr_embedding_dim;
};

#endif
}  // end namespace framework
}  // end namespace paddle
#endif
