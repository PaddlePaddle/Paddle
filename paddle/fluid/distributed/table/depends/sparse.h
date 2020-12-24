// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include <gflags/gflags.h>
#include <math.h>  // for sqrt in CPU and CUDA
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "paddle/fluid/distributed/common/utils.h"
#include "paddle/fluid/distributed/table/depends/large_scale_kv.h"

namespace paddle {
namespace distributed {

class SparseOptimizer {
 public:
  SparseOptimizer() {}
  explicit SparseOptimizer(const CommonAccessorParameter& common) {}
  virtual void update(const uint64_t* keys, const float* update_values,
                      size_t num, const std::vector<uint64_t>& offsets,
                      ValueBlock* block) = 0;
};

// sum calc for sparse tensor
class SSUM : public SparseOptimizer {
 public:
  SSUM(){};
  explicit SSUM(const CommonAccessorParameter& common) {
    auto& names = common.params();
    for (int x = 0; x < static_cast<int>(names.size()); ++x) {
      if (names[x] == "Param") {
        param_idx = x;
        update_numel = common.dims()[x];
      }
    }
  }

  void update(const uint64_t* keys, const float* update_values, size_t num,
              const std::vector<uint64_t>& offsets,
              ValueBlock* block) override {
    auto blas = GetBlas<float>();
    for (auto x : offsets) {
      auto id = keys[x];
      auto values = block->Get(id);
      float* param = values[param_idx]->data();

      std::vector<float> delta;
      delta.resize(update_numel);
      blas.VCOPY(update_numel, update_values + x * update_numel, delta.data());
      blas.VADD(update_numel, delta.data(), param, param);
    }
  }

  int param_idx;
  int update_numel;
};

// sgd optimzer for sparse tensor
class SSGD : public SparseOptimizer {
 public:
  SSGD(){};
  explicit SSGD(const CommonAccessorParameter& common) {
    auto& names = common.params();
    for (int x = 0; x < static_cast<int>(names.size()); ++x) {
      if (names[x] == "LearningRate") {
        learning_rate_idx = x;
      }
      if (names[x] == "Param") {
        param_idx = x;
        update_numel = common.dims()[x];
      }
    }
  }

  void update(const uint64_t* keys, const float* update_values, size_t num,
              const std::vector<uint64_t>& offsets,
              ValueBlock* block) override {
    auto blas = GetBlas<float>();
    for (auto x : offsets) {
      auto id = keys[x];
      auto values = block->Get(id);
      float* learning_rate = values[learning_rate_idx]->data();
      float* param = values[param_idx]->data();

      std::vector<float> grads;
      grads.resize(update_numel);
      blas.VCOPY(update_numel, update_values + x * update_numel, grads.data());
      blas.SCAL(update_numel, learning_rate[0], grads.data());
      blas.VSUB(update_numel, param, grads.data(), param);
    }
  }

  int learning_rate_idx;
  int param_idx;
  int update_numel;
};

// adam optimzer for sparse tensor
class SAdam : public SparseOptimizer {
 public:
  SAdam() {}
  explicit SAdam(const CommonAccessorParameter& common) {
    auto& names = common.params();
    for (int x = 0; x < static_cast<int>(names.size()); ++x) {
      if (names[x] == "LearningRate") {
        learning_rate_idx = x;
      }
      if (names[x] == "Param") {
        param_idx = x;
        update_numel = common.dims()[x];
      }
      if (names[x] == "Moment1") {
        moment1_idx = x;
      }
      if (names[x] == "Moment2") {
        moment2_idx = x;
      }
      if (names[x] == "Beta1Pow") {
        beta1_pow_idx = x;
      }
      if (names[x] == "Beta2Pow") {
        beta2_pow_idx = x;
      }
    }

    // add attr later
    beta1 = 0.9;
    beta2 = 0.999;
    epsilon = 1.0e-8;
  }

  void update(const uint64_t* keys, const float* update_values, size_t num,
              const std::vector<uint64_t>& offsets,
              ValueBlock* block) override {
    auto blas = GetBlas<float>();
    for (auto x : offsets) {
      auto id = keys[x];
      auto values = block->Get(id);
      float* learning_rate = values[learning_rate_idx]->data();
      float* param = values[param_idx]->data();
      float* moment1 = values[moment1_idx]->data();
      float* moment2 = values[moment2_idx]->data();
      float* beta1_pow = values[beta1_pow_idx]->data();
      float* beta2_pow = values[beta2_pow_idx]->data();

      beta1_pow[0] = beta1_pow[0] * beta1;
      beta2_pow[0] = beta2_pow[0] * beta2;

      float lr_ = learning_rate[0];
      lr_ *= sqrt(1 - beta2_pow[0]) / (1 - beta1_pow[0]);

      std::vector<float> grad, grad2, tmp;
      grad.resize(update_numel);
      grad2.resize(update_numel);
      tmp.resize(update_numel);

      blas.VCOPY(update_numel, update_values + x * update_numel, grad.data());
      blas.VCOPY(update_numel, update_values + x * update_numel, grad2.data());

      blas.SCAL(update_numel, 1 - beta1, grad.data());
      blas.VSQUARE(update_numel, grad2.data(), grad2.data());
      blas.SCAL(update_numel, 1 - beta2, grad2.data());

      blas.SCAL(update_numel, beta1, moment1);
      blas.VADD(update_numel, moment1, grad.data(), moment1);
      blas.SCAL(update_numel, beta2, moment2);
      blas.VADD(update_numel, moment2, grad2.data(), moment2);

      float* tmp_ = tmp.data();
      float eps_ = epsilon * sqrt(1 - beta2_pow[0]);

      SQRT<float>(update_numel, moment2, tmp_);
      ADD<float>(update_numel, tmp_, eps_, tmp_);

      blas.VDIV(update_numel, moment1, tmp_, tmp_);
      blas.SCAL(update_numel, lr_, tmp_);
      blas.VSUB(update_numel, param, tmp_, param);
    }
  }

  int learning_rate_idx;
  int param_idx;
  int moment1_idx;
  int moment2_idx;
  int beta1_pow_idx;
  int beta2_pow_idx;
  float beta1;
  float beta2;
  float epsilon;
  int update_numel;
};

}  // namespace distributed
}  // namespace paddle
