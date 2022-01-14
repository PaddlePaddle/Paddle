// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/distributed/ps/table/sparse_sgd_rule.h"
#include <gflags/gflags.h>
#include "glog/logging.h"

DEFINE_bool(enable_show_scale_gradient, true, "enable show scale gradient");

namespace paddle {
namespace distributed {

void SparseNaiveSGDRule::load_config(const SparseCommonSGDRuleParameter& param,
                                     size_t emb_dim) {
  _embedding_dim = emb_dim;
  auto naive_param = param.naive();
  learning_rate_ = naive_param.learning_rate();
  _initial_range = naive_param.initial_range();
  if (naive_param.weight_bounds_size() == 0) {
    _min_bound = -std::numeric_limits<float>::max();
    _max_bound = std::numeric_limits<float>::max();
  } else {
    CHECK(naive_param.weight_bounds_size() >= 2)
        << "invalid repeated size for weight_bounds:"
        << naive_param.weight_bounds_size();
    _min_bound = naive_param.weight_bounds(0);
    _max_bound = naive_param.weight_bounds(1);
  }
}

void SparseNaiveSGDRule::update_value_work(float* w, float* sgd,
                                           const float* push_value,
                                           float scale) {
  for (size_t i = 0; i < _embedding_dim; ++i) {
    w[i] -= learning_rate_ * push_value[i];
    bound_value(w[i]);
  }
}

void SparseNaiveSGDRule::init_value_work(float* value, float* sgd,
                                         bool zero_init) {
  if (zero_init) {
    for (size_t i = 0; i < _embedding_dim; ++i) {
      value[i] = 0;
    }
  } else {
    for (size_t i = 0; i < _embedding_dim; ++i) {
      value[i] =
          (local_uniform_real_distribution<float>()(local_random_engine()) * 2 -
           1) *
          _initial_range;
      bound_value(value[i]);
    }
  }
}
void SparseAdaGradSGDRule::load_config(
    const SparseCommonSGDRuleParameter& param, size_t emb_dim) {
  _embedding_dim = emb_dim;
  auto adagrad_param = param.adagrad();
  learning_rate_ = adagrad_param.learning_rate();
  _initial_g2sum = adagrad_param.initial_g2sum();
  _initial_range = adagrad_param.initial_range();

  if (adagrad_param.weight_bounds_size() == 0) {
    _min_bound = -std::numeric_limits<float>::max();
    _max_bound = std::numeric_limits<float>::max();
  } else {
    CHECK(adagrad_param.weight_bounds_size() >= 2)
        << "invalid repeated size for weight_bounds:"
        << adagrad_param.weight_bounds_size();
    _min_bound = adagrad_param.weight_bounds(0);
    _max_bound = adagrad_param.weight_bounds(1);
  }
}

void SparseAdaGradSGDRule::update_value_work(float* w, float* sgd,
                                             const float* grad, float scale) {
  float& g2sum = sgd[g2sum_index()];
  double add_g2sum = 0;

  for (int i = 0; i < _embedding_dim; i++) {
    double scaled_grad = grad[i] / scale;
    w[i] -= learning_rate_ * scaled_grad *
            sqrt(_initial_g2sum / (_initial_g2sum + g2sum));
    bound_value(w[i]);
    add_g2sum += scaled_grad * scaled_grad;
  }

  g2sum += add_g2sum / _embedding_dim;
}

void SparseAdaGradSGDRule::init_value_work(float* value, float* sgd,
                                           bool zero_init) {
  for (int i = 0; i < _embedding_dim; ++i) {
    if (zero_init) {
      value[i] = 0.0;
      bound_value(value[i]);
    } else {
      value[i] =
          (local_uniform_real_distribution<double>()(local_random_engine()) *
               2 -
           1) *
          _initial_range;
      bound_value(value[i]);
    }
  }
  sgd[g2sum_index()] = 0;
}

void StdAdaGradSGDRule::load_config(const SparseCommonSGDRuleParameter& param,
                                    size_t emb_dim) {
  _embedding_dim = emb_dim;
  auto adagrad_param = param.adagrad();
  learning_rate_ = adagrad_param.learning_rate();
  _initial_g2sum = adagrad_param.initial_g2sum();
  _initial_range = adagrad_param.initial_range();

  if (adagrad_param.weight_bounds_size() == 0) {
    _min_bound = -std::numeric_limits<float>::max();
    _max_bound = std::numeric_limits<float>::max();
  } else {
    CHECK(adagrad_param.weight_bounds_size() >= 2)
        << "invalid repeated size for weight_bounds:"
        << adagrad_param.weight_bounds_size();
    _min_bound = adagrad_param.weight_bounds(0);
    _max_bound = adagrad_param.weight_bounds(1);
  }
}

void StdAdaGradSGDRule::update_value_work(float* w, float* sgd,
                                          const float* grad, float scale) {
  for (int i = 0; i < _embedding_dim; i++) {
    float& g2sum = sgd[g2sum_index() + i];
    double scaled_grad = grad[i] / scale;
    w[i] -= learning_rate_ * scaled_grad *
            sqrt(_initial_g2sum / (_initial_g2sum + g2sum));
    bound_value(w[i]);
    g2sum += scaled_grad * scaled_grad;
  }
}

void StdAdaGradSGDRule::init_value_work(float* value, float* sgd,
                                        bool zero_init) {
  for (int i = 0; i < _embedding_dim; ++i) {
    if (zero_init) {
      value[i] = 0.0;
      bound_value(value[i]);
    } else {
      value[i] =
          (local_uniform_real_distribution<double>()(local_random_engine()) *
               2 -
           1) *
          _initial_range;
      bound_value(value[i]);
    }
    sgd[g2sum_index() + i] = 0;
  }
}

void SparseAdamSGDRule::load_config(const SparseCommonSGDRuleParameter& param,
                                    size_t emb_dim) {
  _embedding_dim = emb_dim;
  auto adam_param = param.adam();
  learning_rate_ = adam_param.learning_rate();
  _initial_range = adam_param.initial_range();
  _beta1_decay_rate = adam_param.beta1_decay_rate();
  _beta2_decay_rate = adam_param.beta2_decay_rate();
  _ada_epsilon = adam_param.ada_epsilon();
  if (adam_param.weight_bounds_size() == 0) {
    _min_bound = -std::numeric_limits<float>::max();
    _max_bound = std::numeric_limits<float>::max();
  } else {
    CHECK(adam_param.weight_bounds_size() >= 2)
        << "invalid repeated size for weight_bounds:"
        << adam_param.weight_bounds_size();
    _min_bound = adam_param.weight_bounds(0);
    _max_bound = adam_param.weight_bounds(1);
  }
}

void SparseAdamSGDRule::update_value_work(float* w, float* sgd,
                                          const float* grad, float scale) {
  float* gsum = sgd + gsum_index();
  float* g2sum = sgd + g2sum_index();
  float* beta1_pow = sgd + beta1_pow_index();
  float* beta2_pow = sgd + beta2_pow_index();
  const float* g = grad;

  float lr = learning_rate_;
  float beta1_pow_ = *beta1_pow;
  float beta2_pow_ = *beta2_pow;

  // lr not change in one update
  lr *= sqrt(1 - beta2_pow_) / (1 - beta1_pow_);
  for (int i = 0; i < _embedding_dim; i++) {
    // Calculation
    gsum[i] = _beta1_decay_rate * gsum[i] + (1 - _beta1_decay_rate) * g[i];
    g2sum[i] =
        _beta2_decay_rate * g2sum[i] + (1 - _beta2_decay_rate) * g[i] * g[i];
    w[i] = w[i] - lr * (gsum[i] / (sqrt(g2sum[i]) + _ada_epsilon));
    bound_value(w[i]);
  }
  // update beta_pow_decay
  (*beta1_pow) *= _beta1_decay_rate;
  (*beta2_pow) *= _beta2_decay_rate;
}

void SparseAdamSGDRule::init_value_work(float* value, float* sgd,
                                        bool zero_init) {
  for (int i = 0; i < _embedding_dim; ++i) {
    if (zero_init) {
      value[i] = 0.0;
      bound_value(value[i]);
    } else {
      value[i] =
          (local_uniform_real_distribution<double>()(local_random_engine()) *
               2 -
           1) *
          _initial_range;
      bound_value(value[i]);
    }
  }
  // init rule gsum and g2sum
  for (int i = gsum_index(); i < beta1_pow_index(); i++) {
    sgd[i] = 0.0;
  }
  // init beta1_pow and beta2_pow
  *(sgd + beta1_pow_index()) = _beta1_decay_rate;
  *(sgd + beta2_pow_index()) = _beta2_decay_rate;
}
}  // namespace distributed
}  // namespace paddle
