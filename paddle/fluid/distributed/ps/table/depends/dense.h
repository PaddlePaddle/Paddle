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

#include <math.h>  // for sqrt in CPU and CUDA
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "gflags/gflags.h"

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
  virtual void Update(const float* update_values, size_t num, int begin,
                      int end) = 0;
  virtual void SetGlobalLR(float* lr) { global_learning_rate_ = lr; }

 protected:
  float* global_learning_rate_;
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

  void Update(const float* update_values, size_t num, int begin,
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

  void Update(const float* update_values, size_t num, int begin,
              int end) override {
    auto update_numel = end - begin;
    std::vector<float> grads;
    grads.resize(update_numel);

    auto blas = GetBlas<float>();
    float lr = *(global_learning_rate_) * (*learning_rate);
    blas.VCOPY(update_numel, update_values + begin, grads.data());
    blas.SCAL(update_numel, lr, grads.data());
    blas.VSUB(update_numel, param + begin, grads.data(), param + begin);
  }

  float* learning_rate;
  float* param;
};

// adam optimizer for dense tensor
// TODO(zhaocaibei123): add CHECK(memory_dense_table.task_pool_size_) == 1
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

  // make sure memory_dense_table.task_pool_size_ == 1;
  // otherwise, task_pool_size_ times beta1_pow/beta2_pow multiplication
  void Update(const float* update_values, size_t num, int begin,
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

    float lr_ = *(global_learning_rate_)*learning_rate[0];
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

// adam optimizer for dense tensor
class DAdamD2Sum : public DenseOptimizer {
 public:
  explicit DAdamD2Sum(const CommonAccessorParameter& accessor,
                      std::vector<std::vector<float>>* values) {
    lr_hardcode = 5e-6;
    auto& names = accessor.params();
    for (int x = 0; x < static_cast<int>(names.size()); ++x) {
      if (names[x] == "LearningRate") {
        learning_rate = (*values)[x].data();
      } else if (names[x] == "Param") {
        param = (*values)[x].data();
      } else if (names[x] == "Moment") {
        mom_velocity = (*values)[x].data();
      } else if (names[x] == "G2Sum") {
        ada_g2sum = (*values)[x].data();
      } else if (names[x] == "D2Sum") {
        ada_d2sum = (*values)[x].data();
      } else if (names[x] == "MomentDecayRate") {
        mom_decay_rate = (*values)[x].data();
      } else if (names[x] == "AdaDecayRate") {
        ada_decay_rate = (*values)[x].data();
      } else if (names[x] == "AdaEpsilon") {
        ada_epsilon = (*values)[x].data();
      }
    }
  }

  void Update(const float* update_values, size_t num, int begin,
              int end) override {
    auto update_numel = end - begin;
    Eigen::Map<Eigen::MatrixXf> mat_ada_g2sum(ada_g2sum + begin, 1,
                                              update_numel);

    Eigen::Map<Eigen::MatrixXf> mat_ada_d2sum(ada_d2sum + begin, 1,
                                              update_numel);
    Eigen::Map<Eigen::MatrixXf> mat_mom_velocity(mom_velocity + begin, 1,
                                                 update_numel);
    Eigen::Map<Eigen::MatrixXf> mat_w(param + begin, 1, update_numel);

    Eigen::Map<const Eigen::MatrixXf> mat_grad(update_values + begin, 1,
                                               update_numel);

    mat_ada_d2sum = (mat_ada_d2sum * ada_decay_rate[0]).array() + 1;
    mat_ada_g2sum =
        (mat_ada_g2sum * ada_decay_rate[0]) + mat_grad.cwiseProduct(mat_grad);

    thread_local std::vector<float> scale_vec;
    scale_vec.resize(update_numel);
    Eigen::Map<Eigen::MatrixXf> scale(scale_vec.data(), 1, update_numel);
    memcpy(scale_vec.data(), mat_ada_d2sum.data(),
           sizeof(float) * update_numel);

    scale = scale.array() * ada_epsilon[0];
    scale = (mat_ada_d2sum + scale).cwiseQuotient(mat_ada_g2sum + scale);
    scale = scale.cwiseSqrt();
    mat_mom_velocity =
        (mat_mom_velocity - mat_grad) * mom_decay_rate[0] + mat_grad;

    mat_w -= learning_rate[0] * mat_mom_velocity.cwiseProduct(scale);
  }

  float* learning_rate;
  float lr_hardcode;

  float* param;
  float* mom_velocity;
  float* ada_g2sum;
  float* ada_d2sum;

  float* mom_decay_rate;
  float* ada_decay_rate;
  float* ada_epsilon;
};

// for data_norm
class DSummary : public DenseOptimizer {
 public:
  explicit DSummary(const CommonAccessorParameter& accessor,
                    std::vector<std::vector<float>>* values) {
    auto& names = accessor.params();
    for (int x = 0; x < static_cast<int>(names.size()); ++x) {
      if (names[x] == "Param") {
        param = (*values)[x].data();
      } else if (names[x] == "SummaryDecayRate") {
        summary_decay_rate = (*values)[x].data();
      }
    }
  }

  void Update(const float* update_values, size_t num, int begin,
              int end) override {
    auto update_numel = end - begin;
    Eigen::Map<Eigen::MatrixXf> mat_w(param + begin, 1, update_numel);
    Eigen::Map<const Eigen::MatrixXf> mat_grad(update_values + begin, 1,
                                               update_numel);
    mat_w = mat_w * summary_decay_rate_d + mat_grad;
  }

  float* summary_decay_rate;
  double summary_decay_rate_d = 0.999999;
  float* param;
};

}  // namespace distributed
}  // namespace paddle
