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

#pragma once
#include <math.h>

#include <thread>
#include <vector>

#include "glog/logging.h"                                  // for CHECK
#include "paddle/fluid/distributed/common/local_random.h"  // for local_uniform_real_distribution
#include "paddle/fluid/distributed/common/registerer.h"
#include "paddle/fluid/distributed/the_one_ps.pb.h"

namespace paddle {
namespace distributed {

class SparseValueSGDRule {
 public:
  SparseValueSGDRule() {}
  virtual ~SparseValueSGDRule() {}
  virtual void LoadConfig(const SparseCommonSGDRuleParameter& param,
                          size_t emb_dim) {
    _embedding_dim = emb_dim;
    _name = param.name();
  }
  virtual void UpdateValueWork(float* w,
                               float* sgd,
                               const float* push_value,
                               float scale) = 0;
  virtual void InitValueWork(float* value, float* sgd, bool zero_init) = 0;
  virtual size_t Dim() = 0;
  const std::string& GetName() const { return _name; }
  void InitValue(float* value, float* sgd, bool zero_init = true) {
    InitValueWork(value, sgd, zero_init);
  }
  void UpdateValue(float* w,
                   float* sgd,
                   const float* push_value,
                   float scale = 1) {
    UpdateValueWork(w, sgd, push_value, scale);
  }
  template <class T>
  void BoundValue(T& w) {  // NOLINT
    if (!(w >= _min_bound)) {
      w = (T)_min_bound;
    } else if (!(w <= _max_bound)) {
      w = (T)_max_bound;
    }
  }
  float& MinBound() { return _min_bound; }
  float& MaxBound() { return _max_bound; }

 protected:
  float _min_bound;
  float _max_bound;
  float _initial_range;
  size_t _embedding_dim;

 private:
  std::string _name;
};

REGISTER_PSCORE_REGISTERER(SparseValueSGDRule);

class SparseNaiveSGDRule : public SparseValueSGDRule {
 public:
  virtual void LoadConfig(const SparseCommonSGDRuleParameter& param,
                          size_t emb_dim);
  virtual void UpdateValueWork(float* w,
                               float* sgd,
                               const float* push_value,
                               float scale);
  virtual void InitValueWork(float* value, float* sgd, bool zero_init);
  virtual size_t Dim() { return 0; }

 private:
  float learning_rate_;
};

class SparseAdaGradSGDRule : public SparseValueSGDRule {
 public:
  virtual void LoadConfig(const SparseCommonSGDRuleParameter& param,
                          size_t emb_dim);
  virtual void UpdateValueWork(float* w,
                               float* sgd,
                               const float* push_value,
                               float scale);
  virtual void InitValueWork(float* value, float* sgd, bool zero_init);
  virtual size_t Dim() { return 1; }
  size_t G2SumIndex() { return 0; }

 private:
  float learning_rate_;
  float _initial_g2sum;
};

class SparseAdaGradV2SGDRule : public SparseValueSGDRule {
  // a new SparseAdaGradV2 use standard adagrad update rules.
  // g2sum = grad_x * grad_x + g2sum
  // x = x + lr * grad_x / sqrt(g2sum + epsilon)
 public:
  virtual void LoadConfig(const SparseCommonSGDRuleParameter& param,
                          size_t emb_dim);
  virtual void UpdateValueWork(float* w,
                               float* sgd,
                               const float* push_value,
                               float scale);
  virtual void InitValueWork(float* value, float* sgd, bool zero_init);
  virtual size_t Dim() { return 1; }
  size_t G2SumIndex() { return 0; }

 private:
  float learning_rate_;
  float _initial_g2sum;
};

class StdAdaGradSGDRule : public SparseValueSGDRule {
 public:
  virtual void LoadConfig(const SparseCommonSGDRuleParameter& param,
                          size_t emb_dim);
  virtual void UpdateValueWork(float* w,
                               float* sgd,
                               const float* push_value,
                               float scale);
  virtual void InitValueWork(float* value, float* sgd, bool zero_init);
  virtual size_t Dim() { return _embedding_dim; }
  size_t G2SumIndex() { return 0; }

 private:
  float learning_rate_;
  float _initial_g2sum;
};

class SparseAdamSGDRule : public SparseValueSGDRule {
 public:
  virtual void LoadConfig(const SparseCommonSGDRuleParameter& param,
                          size_t emb_dim);
  virtual void UpdateValueWork(float* w,
                               float* sgd,
                               const float* push_value,
                               float scale);
  virtual void InitValueWork(float* value, float* sgd, bool zero_init);
  virtual size_t Dim() { return _embedding_dim * 2 + 2; }
  size_t GSumIndex() { return 0; }
  size_t G2SumIndex() { return GSumIndex() + _embedding_dim; }
  size_t Beta1PowIndex() { return G2SumIndex() + _embedding_dim; }
  size_t Beta2PowIndex() { return Beta1PowIndex() + 1; }

 protected:
  float learning_rate_;
  float _beta1_decay_rate;
  float _beta2_decay_rate;
  float _ada_epsilon;
};

class SparseSharedAdamSGDRule : public SparseValueSGDRule {
 public:
  virtual void LoadConfig(const SparseCommonSGDRuleParameter& param,
                          size_t emb_dim);
  virtual void UpdateValueWork(float* w,
                               float* sgd,
                               const float* push_value,
                               float scale);
  virtual void InitValueWork(float* value, float* sgd, bool zero_init);
  virtual size_t Dim() { return 4; }
  size_t GSumIndex() { return 0; }
  size_t G2SumIndex() { return GSumIndex() + 1; }
  size_t Beta1PowIndex() { return G2SumIndex() + 1; }
  size_t Beta2PowIndex() { return Beta1PowIndex() + 1; }

 protected:
  float learning_rate_;
  float _beta1_decay_rate;
  float _beta2_decay_rate;
  float _ada_epsilon;
};

}  // namespace distributed
}  // namespace paddle
