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
  virtual void update(const float* update_values, size_t num, int begin,
                      int end) = 0;
  virtual void set_global_lr(float* lr) { global_learning_rate_ = lr; }

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
    float lr = *(global_learning_rate_) * (*learning_rate);
    blas.VCOPY(update_numel, update_values + begin, grads.data());
    blas.SCAL(update_numel, lr, grads.data());
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
      }
      if (names[x] == "Param") {
        param = (*values)[x].data();
      }
      if (names[x] == "Moment") {
        mom_velocity = (*values)[x].data();
      }
      if (names[x] == "G2Sum") {
        ada_g2sum = (*values)[x].data();
      }
      if (names[x] == "D2Sum") {
        ada_d2sum = (*values)[x].data();
      }
      if (names[x] == "MomentDecayRate") {
        mom_decay_rate = (*values)[x].data();
      }
      if (names[x] == "AdaDecayRate") {
        ada_decay_rate = (*values)[x].data();
      }
      if (names[x] == "AdaEpsilon") {
        ada_epsilon = (*values)[x].data();
      }
    }
  }

  void update(const float* update_values, size_t num, int begin,
              int end) override {
    auto update_numel = end - begin;

    /*
    // for debug
    std::cout << "before update:\n";
    for (int i = 0; i < 3; ++ i) {
      std::cout << "param: " << i << " " << *(param+begin+i) <<
                   "grad: " << *(update_values+begin+i) << "\n";
    }*/

    std::vector<float> grad, grad2, scale;
    grad.resize(update_numel);
    grad2.resize(update_numel);
    scale.resize(update_numel);

    auto blas = GetBlas<float>();
    // copy grad
    blas.VCOPY(update_numel, update_values + begin, grad.data());
    blas.VCOPY(update_numel, update_values + begin, grad2.data());

    /*
    for (int i = 0; i < end-begin; ++ i) {
      std::cout << "copy grad: " << i << " " << *(grad.data()+begin+i) <<
                   "copy grad2: " << *(grad2.data()+begin+i) << "\n";
    }
    for (int i = 0; i < 3; ++ i) {
      std::cout << "d2sum before: " << i << " " << *(ada_d2sum+begin+i) << "\n";
    }*/

    // d2sum
    blas.SCAL(update_numel, ada_decay_rate[0], ada_d2sum + begin);
    ADD<float>(update_numel, ada_d2sum + begin, 1, ada_d2sum + begin);

    /*
    for (int i = 0; i < end-begin; ++ i) {
      std::cout << "d2sum update: " << i << " " << *(ada_d2sum+begin+i) << "\n";
    }
    for (int i = 0; i < 3; ++ i) {
      std::cout << "g2sum before: " << i << " " << *(ada_g2sum+begin+i) << "\n";
    }*/

    // g2sum
    blas.SCAL(update_numel, ada_decay_rate[0], ada_g2sum + begin);
    blas.VSQUARE(update_numel, grad2.data(), grad2.data());
    blas.VADD(update_numel, ada_g2sum + begin, grad2.data(), ada_g2sum + begin);

    /*
    for (int i = 0; i < end-begin; ++ i) {
      std::cout << "g2sum update: " << i << " " << *(ada_g2sum+begin+i) << "\n";
    }
    for (int i = 0; i < 3; ++ i) {
      std::cout << "mom before: " << i << " " << *(mom_velocity+begin+i) <<
    "\n";
    }*/

    // mom
    blas.SCAL(update_numel, mom_decay_rate[0], mom_velocity + begin);
    blas.SCAL(update_numel, 1 - mom_decay_rate[0], grad.data());
    blas.VADD(update_numel, mom_velocity + begin, grad.data(),
              mom_velocity + begin);

    /*
    for (int i = 0; i < end-begin; ++ i) {
      std::cout << "mom update: " << i << " " << *(mom_velocity+begin+i) <<
    "\n";
    }
    for (int i = 0; i < 3; ++ i) {
      std::cout << "scale before: " << i << " " << *(scale.data()+begin+i) <<
    "\n";
    }*/

    // scale
    float* scale_ = scale.data();
    blas.VDIV(update_numel, ada_g2sum + begin, ada_d2sum + begin, scale_);
    ADD<float>(update_numel, scale_, ada_epsilon[0], scale_);
    DIV<float>(update_numel, 1 + ada_epsilon[0], scale_, scale_);
    SQRT<float>(update_numel, scale_, scale_);

    /*
    for (int i = 0; i < 3; ++ i) {
      std::cout << "scale update: " << i << " " << *(scale.data()+begin+i) <<
    "\n";
    }*/

    blas.SCAL(update_numel, learning_rate[0], scale_);

    // TODO(zhaocaibei123): check if there exists elementwise_multiply in blas
    // TODO(zhaocaibei123): blas.VMUL
    ELE_MUL<float>(update_numel, scale_, mom_velocity + begin, scale_);

    /*
    for (int i = 0; i < 3; ++ i) {
      std::cout << "scale update2: " << i << " " << *(scale.data()+begin+i) <<
    "\n";
    }*/

    blas.VSUB(update_numel, param + begin, scale_, param + begin);

    /*
    for (int i = 0; i < end-begin; ++ i) {
      std::cout << "param update " << i << " " << *(param+begin+i) << "\n";
    }*/
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

}  // namespace distributed
}  // namespace paddle
