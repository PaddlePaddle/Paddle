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

class Optimizer {
 public:
  __host__ Optimizer(CommonFeatureValueAccessor feature_value_accessor) {
    feature_value_accessor_ = feature_value_accessor;
  }
  __host__ ~Optimizer() {}
  // __host__ virtual void initialize(const OptimizerConfig& optimizer_config,
  //                         size_t emb_dim) {
    
  //   _lr_embedding_dim = 1;
  //   _embedding_dim = emb_dim;
  // }

  __device__ void update_value(const OptimizerConfig& optimizer_config,
                               float& val,  // NOLINT
                               const float& grad) {
  }

  __device__ void dy_mf_update_value(const OptimizerConfig& optimizer_config,
                                     float* ptr, const float* grad) {
  }
  
  // __host__ float& MinBound() { return _min_bound; }
  // float& MaxBound() { return _max_bound; }

  CommonFeatureValueAccessor feature_value_accessor_;

//  protected:
  // float _min_bound;
  // float _max_bound;
  // float _initial_range;
  size_t _embedding_dim;
  size_t _lr_embedding_dim;


};

class SparseAdagradOptimizer : public Optimizer {
 public:
  
  __host__ SparseAdagradOptimizer(CommonFeatureValueAccessor feature_value_accessor): Optimizer(feature_value_accessor) {
    _lr_embedding_dim = 1;
    _embedding_dim = feature_value_accessor_.common_feature_value.EmbedWDim();
  }

  // __host__ virtual void initialize(const OptimizerConfig& optimizer_config,
  //                         size_t emb_dim) {
  //   _lr_embedding_dim = 1;
  //   _embedding_dim = feature_value_accessor_.common_feature_value.EmbedWDim();
  //   // this->_learning_rate = optimizer_config.learning_rate;
  //   // this->_initial_g2sum = optimizer_config.initial_g2sum;
  //   // this->_initial_range = optimizer_config.initial_range;
  //   // this->_min_bound = optimizer_config.min_bound;
  //   // this->_max_bound = optimizer_config.max_bound;
  // }

  
  // __device__ void update_lr(const OptimizerConfig& optimizer_config,
  //                           float& w,               // NOLINT
  //                           float* sgd, float g,  // NOLINT
  //                           float scale) {
  //   float& g2sum = sgd[G2SumIndex()];
  //   double add_g2sum = 0;
  //   double ratio = optimizer_config.learning_rate *
  //                  sqrt(optimizer_config.initial_g2sum /
  //                       (optimizer_config.initial_g2sum + g2sum));
  //   double scaled_grad = g / scale;

  //   w += scaled_grad * ratio;

  //   if (w < optimizer_config.min_bound) w = optimizer_config.min_bound;
  //   if (w > optimizer_config.max_bound) w = optimizer_config.max_bound;

  //   add_g2sum += scaled_grad * scaled_grad;

  //   g2sum += add_g2sum;
  // }

  
  __device__ void update_value_work(const OptimizerConfig& optimizer_config, int n,
                            float* w,
                            float* sgd,  // NOLINT
                            const float* g, float scale) {
    float& g2sum = sgd[G2SumIndex()];
    double add_g2sum = 0;
    double ratio = optimizer_config.mf_learning_rate *
                   sqrt(optimizer_config.mf_initial_g2sum /
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

  // __device__ void update_value_work(const OptimizerConfig& optimizer_config,
  //                           float* w,
  //                           float* sgd,  // NOLINT
  //                           const float* g, float scale) {
  //   float& g2sum = sgd[G2SumIndex()];
  //   double add_g2sum = 0;
  //   double ratio = _learning_rate *
  //                  sqrt(_initial_g2sum /
  //                       (_initial_g2sum + g2sum));
  //   for (int i = 0; i < _embedding_dim; ++i) {
  //     double scaled_grad = g[i] / scale;

  //     w[i] += scaled_grad * ratio;

  //     if (w[i] < this->_min_bound)
  //       w[i] = this->_min_bound;
  //     if (w[i] > this->_max_bound)
  //       w[i] = this->_max_bound;
  //     add_g2sum += scaled_grad * scaled_grad;
  //   }

  //   g2sum += add_g2sum / _embedding_dim;
  // }

  __device__ void update_value(const OptimizerConfig& optimizer_config,
                               float& val,  // NOLINT
                               const float& grad) {
  }
  __device__ void dy_mf_update_value(const OptimizerConfig& optimizer_config,
                                     float* ptr, const float* grad) {
    float g_show = grad[feature_value_accessor_.common_push_value.ShowIndex()];
    float g_click = grad[feature_value_accessor_.common_push_value.ClickIndex()];


    ptr[feature_value_accessor_.common_feature_value.SlotIndex()] =
        grad[feature_value_accessor_.common_push_value.SlotIndex()];
    ptr[feature_value_accessor_.common_feature_value.ShowIndex()] += g_show;
    ptr[feature_value_accessor_.common_feature_value.ClickIndex()] += g_click;
    ptr[feature_value_accessor_.common_feature_value.DeltaScoreIndex()] +=
        optimizer_config.nonclk_coeff * (g_show - g_click) +
                        optimizer_config.clk_coeff * g_click;

    update_value_work(optimizer_config, 1,
              ptr + feature_value_accessor_.common_feature_value.EmbedWIndex(),
              ptr + feature_value_accessor_.common_feature_value.EmbedG2SumIndex(),
              grad + feature_value_accessor_.common_push_value.EmbedGIndex(),
              g_show);

    int mf_dim = int(ptr[feature_value_accessor_.common_feature_value.MfDimIndex()]);
    if (ptr[feature_value_accessor_.common_feature_value.MfSizeIndex()] == 0) {
      if (optimizer_config.mf_create_thresholds <=
          optimizer_config.nonclk_coeff * 
            (ptr[feature_value_accessor_.common_feature_value.ShowIndex()] - 
                ptr[feature_value_accessor_.common_feature_value.ClickIndex()]) +
              optimizer_config.clk_coeff * ptr[feature_value_accessor_.common_feature_value.ClickIndex()]) {
        ptr[feature_value_accessor_.common_feature_value.MfSizeIndex()] = mf_dim;

        
        // ptr->mf_size = MF_DIM + 1;
        int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
        curandState state;
        curand_init(clock64(), tid_x, 0, &state);
        for (int i = 0; i < mf_dim; ++i) {
          ptr[feature_value_accessor_.common_feature_value.EmbedxWIndex() + i] = 
              (curand_uniform(&state)) * optimizer_config.mf_initial_range;
        }
      }
    } else {
      update_value_work(optimizer_config, mf_dim,
          ptr + feature_value_accessor_.common_feature_value.EmbedxWIndex(),
          ptr + feature_value_accessor_.common_feature_value.EmbedxG2SumIndex(),
          grad + feature_value_accessor_.common_push_value.EmbedxGIndex(),
          g_show);
    }
  }
  
  __host__ __device__ size_t Dim() { return EmbedDim() + EmbedxDim();}
  __host__ __device__ size_t EmbedDim() { return _lr_embedding_dim; }
  __host__ __device__ size_t EmbedxDim() { return _embedding_dim; }
  __host__ __device__ size_t G2SumIndex() { return 0; }
  __host__ __device__ size_t EmbedxG2SumIndex() { return 0; }


//  private:
  // float _learning_rate;
  // float _initial_g2sum;
};

class SparseAdamOptimizer : public Optimizer {
 public:

  __host__ SparseAdamOptimizer(CommonFeatureValueAccessor feature_value_accessor): Optimizer(feature_value_accessor) {
    _lr_embedding_dim = 1;
    _embedding_dim = feature_value_accessor_.common_feature_value.EmbedWDim();
  }

  // __host__ virtual void initialize(const OptimizerConfig& optimizer_config,
  //                         size_t emb_dim) {
  //   _lr_embedding_dim = 1;
  //   _embedding_dim = feature_value_accessor_.common_feature_value.EmbedWDim();
  //   // this->_learning_rate = optimizer_config.learning_rate;
  //   // this->_initial_range = optimizer_config.initial_range;
  //   // this->_beta1_decay_rate = optimizer_config.beta1_decay_rate;
  //   // this->_beta2_decay_rate = optimizer_config.beta2_decay_rate;
  //   // this->_ada_epsilon = optimizer_config.ada_epsilon; //    float epsilon = 1e-08;
  //   // this->_min_bound = optimizer_config.min_bound;
  //   // this->_max_bound = optimizer_config.max_bound;
  // }

  // __device__ void update_lr(const OptimizerConfig& optimizer_config,
  //                           float& w,               // NOLINT
  //                           float* sgd, float g,  // NOLINT
  //                           float scale) {

  //   float* moment1 = sgd + GSumIndex();
  //   float* moment2 = sgd + G2SumIndex();
  //   float* beta1_pow = sgd + Beta1PowIndex();
  //   float* beta2_pow = sgd + Beta2PowIndex();

  //   float beta1_pow_ = *beta1_pow;
  //   float beta2_pow_ = *beta2_pow;
  //   float moment1_ = *moment1;
  //   float moment2_ = *moment2;

  //   float epsilon = 1e-08;
  //   double ratio = optimizer_config.learning_rate * sqrt(1.0 - beta2_pow_) / (1.0 - beta1_pow_);
  //   double scaled_grad = g / scale;
  //   double new_moment1 = optimizer_config.beta1_decay_rate * moment1_ + (1.0 - optimizer_config.beta1_decay_rate) * scaled_grad;
  //   double new_moment2 = optimizer_config.beta2_decay_rate * moment2_ + (1.0 - optimizer_config.beta2_decay_rate) * scaled_grad * scaled_grad;
  //   w += ratio * (new_moment1 / (sqrt(new_moment2) + epsilon));

  //   if (w < optimizer_config.min_bound) w = optimizer_config.min_bound;
  //   if (w > optimizer_config.max_bound) w = optimizer_config.max_bound;

  //   (*moment1) = new_moment1;
  //   (*moment2) = new_moment2;

  //   (*beta1_pow) *= optimizer_config.beta1_decay_rate;
  //   (*beta2_pow) *= optimizer_config.beta2_decay_rate;

  // }

  __device__ void update_value_work(const OptimizerConfig& optimizer_config, int n,
                            float* w,
                            float* sgd,
                            const float* g, float scale) {
    float* moment1 = sgd + GSumIndex();
    float* moment2 = sgd + G2SumIndex();
    float* beta1_pow = sgd + Beta1PowIndex();
    float* beta2_pow = sgd + Beta2PowIndex();

    float beta1_pow_ = *beta1_pow;
    float beta2_pow_ = *beta2_pow;

    float epsilon = 1e-08;
    double ratio = optimizer_config.learning_rate * sqrt(1.0 - beta2_pow_) / (1.0 - beta1_pow_);
    for (int i = 0; i < n; ++i) {
      double scaled_grad = g[i] / scale;

      double new_moment1 = optimizer_config.beta1_decay_rate * moment1[i] + (1.0 - optimizer_config.beta1_decay_rate) * scaled_grad;
      double new_moment2 = optimizer_config.beta2_decay_rate * moment2[i] + (1.0 - optimizer_config.beta2_decay_rate) * scaled_grad * scaled_grad;
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

  // __device__ void update_value_work(const OptimizerConfig& optimizer_config,
  //                           float* w,
  //                           float* sgd,  // NOLINT
  //                           const float* g, float scale) {
  //   float* moment1 = sgd + GSumIndex();
  //   float* moment2 = sgd + G2SumIndex();
  //   float* beta1_pow = sgd + Beta1PowIndex();
  //   float* beta2_pow = sgd + Beta2PowIndex();

  //   float beta1_pow_ = *beta1_pow;
  //   float beta2_pow_ = *beta2_pow;

  //   double ratio = _learning_rate * sqrt(1.0 - beta2_pow_) / (1.0 - beta1_pow_);

  //   for (int i = 0; i < this->_embedding_dim; ++i) {
  //     double scaled_grad = g[i] / scale;

  //     double new_moment1 = _beta1_decay_rate * moment1[i] + (1.0 - _beta1_decay_rate) * scaled_grad;
  //     double new_moment2 = _beta2_decay_rate * moment2[i] + (1.0 - _beta2_decay_rate) * scaled_grad * scaled_grad;
  //     w[i] += ratio * (new_moment1 / (sqrt(new_moment2) + _ada_epsilon));


  //     if (w[i] < this->_min_bound)
  //       w[i] = this->_min_bound;
  //     if (w[i] > this->_max_bound)
  //       w[i] = this->_max_bound;

  //     moment1[i] = new_moment1;
  //     moment2[i] = new_moment2;
  //   }
  //   (*beta1_pow) *= _beta1_decay_rate;
  //   (*beta2_pow) *= _beta2_decay_rate;
  // }

  __device__ void update_value(const OptimizerConfig& optimizer_config,
                               float& val,  // NOLINT
                               const float& grad) {
  }
  __device__ void dy_mf_update_value(const OptimizerConfig& optimizer_config,
                                     float* ptr, const float* grad) {
    
    float g_show = grad[feature_value_accessor_.common_push_value.ShowIndex()];
    float g_click = grad[feature_value_accessor_.common_push_value.ClickIndex()];


    ptr[feature_value_accessor_.common_feature_value.SlotIndex()] =
        grad[feature_value_accessor_.common_push_value.SlotIndex()];
    ptr[feature_value_accessor_.common_feature_value.ShowIndex()] += g_show;
    ptr[feature_value_accessor_.common_feature_value.ClickIndex()] += g_click;
    ptr[feature_value_accessor_.common_feature_value.DeltaScoreIndex()] +=
        optimizer_config.nonclk_coeff * (g_show - g_click) +
                        optimizer_config.clk_coeff * g_click;

    update_value_work(optimizer_config, 1,
              ptr + feature_value_accessor_.common_feature_value.EmbedWIndex(),
              ptr + feature_value_accessor_.common_feature_value.EmbedG2SumIndex(),
              grad + feature_value_accessor_.common_push_value.EmbedGIndex(),
              g_show);
    int mf_dim = int(ptr[feature_value_accessor_.common_feature_value.MfDimIndex()]);
    if (ptr[feature_value_accessor_.common_feature_value.MfSizeIndex()] == 0) {
      if (optimizer_config.mf_create_thresholds <=
          optimizer_config.nonclk_coeff * 
            (ptr[feature_value_accessor_.common_feature_value.ShowIndex()] - 
                ptr[feature_value_accessor_.common_feature_value.ClickIndex()]) +
              optimizer_config.clk_coeff * ptr[feature_value_accessor_.common_feature_value.ClickIndex()]) {
        ptr[feature_value_accessor_.common_feature_value.MfSizeIndex()] = mf_dim;

        
        // ptr->mf_size = MF_DIM + 1;
        int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
        curandState state;
        curand_init(clock64(), tid_x, 0, &state);
        for (int i = 0; i < mf_dim; ++i) {
          ptr[feature_value_accessor_.common_feature_value.EmbedxWIndex() + i] = 
              (curand_uniform(&state)) * optimizer_config.mf_initial_range;
        }
        ptr[feature_value_accessor_.common_feature_value.EmbedG2SumIndex() + Beta1PowIndex()] = 
          optimizer_config.beta1_decay_rate;
        ptr[feature_value_accessor_.common_feature_value.EmbedG2SumIndex() + Beta2PowIndex()] = 
          optimizer_config.beta2_decay_rate;
        ptr[feature_value_accessor_.common_feature_value.EmbedxG2SumIndex() + EmbedxBeta1PowIndex()] = 
          optimizer_config.beta1_decay_rate;
        ptr[feature_value_accessor_.common_feature_value.EmbedxG2SumIndex() + EmbedxBeta2PowIndex()] = 
          optimizer_config.beta2_decay_rate;
      }
    } else {
      update_value_work(optimizer_config, mf_dim,
          ptr + feature_value_accessor_.common_feature_value.EmbedxWIndex(),
          ptr + feature_value_accessor_.common_feature_value.EmbedxG2SumIndex(),
          grad + feature_value_accessor_.common_push_value.EmbedxGIndex(),
          g_show);
    }
  }
  
  __host__ __device__ size_t Dim() { return EmbedDim() + EmbedxDim(); }
  __host__ __device__ size_t EmbedDim() { return _lr_embedding_dim * 2 + 2; }
  __host__ __device__ size_t EmbedxDim() { return _embedding_dim * 2 + 2; }
  __host__ __device__ size_t GSumIndex() { return 0; }
  __host__ __device__ size_t G2SumIndex() { return GSumIndex() + _lr_embedding_dim; }
  __host__ __device__ size_t Beta1PowIndex() { return G2SumIndex() + _lr_embedding_dim; }
  __host__ __device__ size_t Beta2PowIndex() { return Beta1PowIndex() + 1; }
  __host__ __device__ size_t EmbedxGSumIndex() { return 0; }
  __host__ __device__ size_t EmbedxG2SumIndex() { return EmbedxGSumIndex() + _embedding_dim; }
  __host__ __device__ size_t EmbedxBeta1PowIndex() { return EmbedxG2SumIndex() + _embedding_dim; }
  __host__ __device__ size_t EmbedxBeta2PowIndex() { return EmbedxBeta1PowIndex() + 1; }


//  protected:
  // float _learning_rate;
  // float _beta1_decay_rate;
  // float _beta2_decay_rate;
  // float _ada_epsilon;
};


class SparseAdamSharedOptimizer : public Optimizer {
 public:

  __host__ SparseAdamSharedOptimizer(CommonFeatureValueAccessor feature_value_accessor): Optimizer(feature_value_accessor) {
    _lr_embedding_dim = 1;
    _embedding_dim = feature_value_accessor_.common_feature_value.EmbedWDim();
  }

  // virtual void initialize() {
  //   _lr_embedding_dim = 1;
  //   _embedding_dim = feature_value_accessor_.common_feature_value.EmbedWDim();
  //   // this->_learning_rate = optimizer_config.learning_rate;
  //   // this->_initial_range = optimizer_config.initial_range;
  //   // this->_beta1_decay_rate = optimizer_config.beta1_decay_rate;
  //   // this->_beta2_decay_rate = optimizer_config.beta2_decay_rate;
  //   // this->_ada_epsilon = optimizer_config.ada_epsilon; //    float epsilon = 1e-08;
  //   // this->_min_bound = optimizer_config.min_bound;
  //   // this->_max_bound = optimizer_config.max_bound;
  // }

  // __device__ void update_lr(const OptimizerConfig& optimizer_config,
  //                           float& w,               // NOLINT
  //                           float* sgd, float g,  // NOLINT
  //                           float scale) {
  //   float* moment1 = sgd + GSumIndex();
  //   float* moment2 = sgd + G2SumIndex();
  //   float* beta1_pow = sgd + Beta1PowIndex();
  //   float* beta2_pow = sgd + Beta2PowIndex();

  //   float beta1_pow_ = *beta1_pow;
  //   float beta2_pow_ = *beta2_pow;
  //   float moment1_ = *moment1;
  //   float moment2_ = *moment2;

  //   float epsilon = 1e-08;
  //   double ratio = optimizer_config.learning_rate * sqrt(1.0 - beta2_pow_) / (1.0 - beta1_pow_);
  //   double scaled_grad = g / scale;
  //   double new_moment1 = optimizer_config.beta1_decay_rate * moment1_ + (1.0 - optimizer_config.beta1_decay_rate) * scaled_grad;
  //   double new_moment2 = optimizer_config.beta2_decay_rate * moment2_ + (1.0 - optimizer_config.beta2_decay_rate) * scaled_grad * scaled_grad;
  //   w += ratio * (new_moment1 / (sqrt(new_moment2) + epsilon));

  //   if (w < optimizer_config.min_bound) w = optimizer_config.min_bound;
  //   if (w > optimizer_config.max_bound) w = optimizer_config.max_bound;

  //   (*moment1) = new_moment1;
  //   (*moment2) = new_moment2;

  //   (*beta1_pow) *= optimizer_config.beta1_decay_rate;
  //   (*beta2_pow) *= optimizer_config.beta2_decay_rate;
  // }

  __device__ void update_value_work(const OptimizerConfig& optimizer_config, int n,
                            float* w,
                            float* sgd,
                            const float* g, float scale) {
    float* moment1 = sgd + GSumIndex();
    float* moment2 = sgd + G2SumIndex();
    float* beta1_pow = sgd + Beta1PowIndex();
    float* beta2_pow = sgd + Beta2PowIndex();

    float beta1_pow_ = *beta1_pow;
    float beta2_pow_ = *beta2_pow;
    float moment1_ = *moment1;
    float moment2_ = *moment2;
    float epsilon = 1e-08;
    double ratio = optimizer_config.learning_rate * sqrt(1.0 - beta2_pow_) / (1.0 - beta1_pow_);

    double sum_mom1 = 0.0;
    double sum_mom2 = 0.0;
    for (int i = 0; i < n; ++i) {
      double scaled_grad = g[i] / scale;

      double new_moment1 = optimizer_config.beta1_decay_rate * moment1_ + (1.0 - optimizer_config.beta1_decay_rate) * scaled_grad;
      double new_moment2 = optimizer_config.beta2_decay_rate * moment2_ + (1.0 - optimizer_config.beta2_decay_rate) * scaled_grad * scaled_grad;
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
  }

  __device__ void dy_mf_update_value(const OptimizerConfig& optimizer_config,
                                     float* ptr, const float* grad) {
    
    float g_show = grad[feature_value_accessor_.common_push_value.ShowIndex()];
    float g_click = grad[feature_value_accessor_.common_push_value.ClickIndex()];


    ptr[feature_value_accessor_.common_feature_value.SlotIndex()] =
        grad[feature_value_accessor_.common_push_value.SlotIndex()];
    ptr[feature_value_accessor_.common_feature_value.ShowIndex()] += g_show;
    ptr[feature_value_accessor_.common_feature_value.ClickIndex()] += g_click;
    ptr[feature_value_accessor_.common_feature_value.DeltaScoreIndex()] +=
        optimizer_config.nonclk_coeff * (g_show - g_click) +
                        optimizer_config.clk_coeff * g_click;

    update_value_work(optimizer_config, 1,
              ptr + feature_value_accessor_.common_feature_value.EmbedWIndex(),
              ptr + feature_value_accessor_.common_feature_value.EmbedG2SumIndex(),
              grad + feature_value_accessor_.common_push_value.EmbedGIndex(),
              g_show);
    int mf_dim = int(ptr[feature_value_accessor_.common_feature_value.MfDimIndex()]);
    if (ptr[feature_value_accessor_.common_feature_value.MfSizeIndex()] == 0) {
      if (optimizer_config.mf_create_thresholds <=
          optimizer_config.nonclk_coeff * 
            (ptr[feature_value_accessor_.common_feature_value.ShowIndex()] - 
                ptr[feature_value_accessor_.common_feature_value.ClickIndex()]) +
              optimizer_config.clk_coeff * ptr[feature_value_accessor_.common_feature_value.ClickIndex()]) {
        ptr[feature_value_accessor_.common_feature_value.MfSizeIndex()] = mf_dim;

        
        // ptr->mf_size = MF_DIM + 1;
        int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
        curandState state;
        curand_init(clock64(), tid_x, 0, &state);
        for (int i = 0; i < mf_dim; ++i) {
          ptr[feature_value_accessor_.common_feature_value.EmbedxWIndex() + i] = 
              (curand_uniform(&state)) * optimizer_config.mf_initial_range;
        }
        ptr[feature_value_accessor_.common_feature_value.EmbedG2SumIndex() + Beta1PowIndex()] = 
          optimizer_config.beta1_decay_rate;
        ptr[feature_value_accessor_.common_feature_value.EmbedG2SumIndex() + Beta2PowIndex()] = 
          optimizer_config.beta2_decay_rate;
        ptr[feature_value_accessor_.common_feature_value.EmbedxG2SumIndex() + EmbedxBeta1PowIndex()] = 
          optimizer_config.beta1_decay_rate;
        ptr[feature_value_accessor_.common_feature_value.EmbedxG2SumIndex() + EmbedxBeta2PowIndex()] = 
          optimizer_config.beta2_decay_rate;
      }
    } else {
      update_value_work(optimizer_config, mf_dim,
          ptr + feature_value_accessor_.common_feature_value.EmbedxWIndex(),
          ptr + feature_value_accessor_.common_feature_value.EmbedxG2SumIndex(),
          grad + feature_value_accessor_.common_push_value.EmbedxGIndex(),
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
  __host__ __device__ size_t EmbedxG2SumIndex() { return EmbedxGSumIndex() + 1; }
  __host__ __device__ size_t EmbedxBeta1PowIndex() { return EmbedxG2SumIndex() + 1; }
  __host__ __device__ size_t EmbedxBeta2PowIndex() { return EmbedxBeta1PowIndex() + 1; }


//  protected:
//   float _learning_rate;
//   float _beta1_decay_rate;
//   float _beta2_decay_rate;
//   float _ada_epsilon;
};


#endif
}  // end namespace framework
}  // end namespace paddle
#endif
