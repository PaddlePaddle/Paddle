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

namespace paddle {
namespace distributed {

// dense optimzier
// TODO(tangwei12) integrate with sparse optimzer later.
class DenseOptimizer {
 public:
  DenseOptimizer() {}
  explicit DenseOptimizer(const CommonAccessorParameter& accessor,
                          std::vector<std::vector<float>>* values) {}
  virtual void update(const float* update_values, size_t num, int begin,
                      int end) = 0;
};

// sum calc for dense tensor
class DSUM : public DenseOptimizer {
 public:
  explicit DSUM(const CommonAccessorParameter& accessor,
                std::vector<std::vector<float>>* values) {
    auto& names = accessor.params();
    for (int x = 0; x < static_cast<int>(names.size()); ++x) {
      if (names[x] == "Param") {
        param = (*values)[x].data();
      }
    }
  }

  void update(const float* update_values, size_t num, int begin,
              int end) override {
    auto update_numel = end - begin;
    GetBlas<float>().VADD(update_numel, update_values + begin, param + begin,
                          param + begin);
  }

  float* param;
};

// sgd optimizer for dense tensor
class DSGD : public DenseOptimizer {
 public:
  explicit DSGD(const CommonAccessorParameter& accessor,
                std::vector<std::vector<float>>* values) {
    auto& names = accessor.params();
    for (int x = 0; x < static_cast<int>(names.size()); ++x) {
      if (names[x] == "LearningRate") {
        learning_rate = (*values)[x].data();
      }
      if (names[x] == "Param") {
        param = (*values)[x].data();
      }
    }
  }

  void update(const float* update_values, size_t num, int begin,
              int end) override {
    auto update_numel = end - begin;
    std::vector<float> grads;
    grads.resize(update_numel);

    auto blas = GetBlas<float>();
    blas.VCOPY(update_numel, update_values + begin, grads.data());
    blas.SCAL(update_numel, *learning_rate, grads.data());
    blas.VSUB(update_numel, param + begin, grads.data(), param + begin);
  }

  float* learning_rate;
  float* param;
};

// adam optimizer for dense tensor
class DAdam : public DenseOptimizer {
 public:
  explicit DAdam(const CommonAccessorParameter& accessor,
                 std::vector<std::vector<float>>* values) {
    auto& names = accessor.params();
    for (int x = 0; x < static_cast<int>(names.size()); ++x) {
      if (names[x] == "LearningRate") {
        learning_rate = (*values)[x].data();
      }
      if (names[x] == "Param") {
        param = (*values)[x].data();
      }
      if (names[x] == "Moment1") {
        moment1 = (*values)[x].data();
      }
      if (names[x] == "Moment2") {
        moment2 = (*values)[x].data();
      }
      if (names[x] == "Beta1Pow") {
        beta1_pow = (*values)[x].data();
      }
      if (names[x] == "Beta2Pow") {
        beta2_pow = (*values)[x].data();
      }
    }

    // add attr later
    beta1 = 0.9;
    beta2 = 0.999;
    epsilon = 1.0e-8;
  }

  void update(const float* update_values, size_t num, int begin,
              int end) override {
    auto update_numel = end - begin;
    std::vector<float> grad, grad2, tmp;
    grad.resize(update_numel);
    grad2.resize(update_numel);
    tmp.resize(update_numel);

    auto blas = GetBlas<float>();
    blas.VCOPY(update_numel, update_values + begin, grad.data());
    blas.VCOPY(update_numel, update_values + begin, grad2.data());

    blas.SCAL(update_numel, 1 - beta1, grad.data());
    blas.VSQUARE(update_numel, grad2.data(), grad2.data());
    blas.SCAL(update_numel, 1 - beta2, grad2.data());

    blas.SCAL(update_numel, beta1, moment1 + begin);
    blas.VADD(update_numel, moment1 + begin, grad.data(), moment1 + begin);
    blas.SCAL(update_numel, beta2, moment2 + begin);
    blas.VADD(update_numel, moment2 + begin, grad2.data(), moment2 + begin);

    beta1_pow[0] = beta1_pow[0] * beta1;
    beta2_pow[0] = beta2_pow[0] * beta2;

    float lr_ = learning_rate[0];
    lr_ *= sqrt(1 - beta2_pow[0]) / (1 - beta1_pow[0]);

    float* tmp_ = tmp.data();
    float eps_ = epsilon * sqrt(1 - beta2_pow[0]);

    SQRT<float>(update_numel, moment2 + begin, tmp_);
    ADD<float>(update_numel, tmp_, eps_, tmp_);

    blas.VDIV(update_numel, moment1 + begin, tmp_, tmp_);
    blas.SCAL(update_numel, lr_, tmp_);
    blas.VSUB(update_numel, param + begin, tmp_, param + begin);
  }

  float* learning_rate;

  float* param;
  float* moment1;
  float* moment2;

  float* beta1_pow;
  float* beta2_pow;

  float beta1;
  float beta2;
  float epsilon;
};

}  // namespace distributed
}  // namespace paddle
