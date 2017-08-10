/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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

#include "ParameterOptimizer.h"
#include "Regularizer.h"

namespace paddle {

// Plain SGD optimization.
class SgdOptimizer : public ParameterOptimizer {
public:
  explicit SgdOptimizer(const OptimizationConfig& optConfig)
      : ParameterOptimizer(optConfig) {
    addParameterType(PARAMETER_MOMENTUM);
  }

  virtual void startBatch(int64_t numSamplesProcessed) {
    learningRate_ = calcLearningRate(numSamplesProcessed, pass_);
  }
  virtual void update(const VectorPtr vecs[],
                      const ParameterConfig& paraConfig,
                      size_t sparseId) const {
    (void)sparseId;
    real torch_learningRate = optConfig_.learning_method() == "torch_momentum"
                                  ? 1.0 - paraConfig.momentum()
                                  : 1.0;
    vecs[PARAMETER_VALUE]->sgdUpdate(
        *vecs[PARAMETER_GRADIENT],
        *vecs[PARAMETER_MOMENTUM],
        learningRate_ * paraConfig.learning_rate() *
            (firstTime_ ? 1.0 : torch_learningRate),
        paraConfig.momentum(),
        applyDecay_ ? paraConfig.decay_rate() : 0);
  }
  virtual void finishBatch() { firstTime_ = false; }
};

// SGD optimization with sparse support.
class SparseMomentumParameterOptimizer : public ParameterOptimizer {
  /* sparse momentum optimizer

    update scheme:

    \alpha_t = \alpha_{t-1} / k
    \beta_t = \beta_{t-1} / (1 + \lambda\gamma_t)
    u_t = u_{t-1} - \alpha_t \gamma_t g_t
    v_t = v_{t-1} + \tau_{t-1} \alpha_t \gamma_t g_t
    \tau_t = \tau_{t-1} + \beta_t / \alpha_t

    where:
    k: momentum
    lambda: decay rate
    \gamma_t: learning rate at the t'th step
  */

public:
  explicit SparseMomentumParameterOptimizer(
      const OptimizationConfig& optConfig);
  virtual void init(size_t numRows, const ParameterConfig* config);
  virtual void startBatch(int64_t numSamplesProcessed);
  virtual void update(const VectorPtr vecs[],
                      const ParameterConfig& paraConfig,
                      size_t sparseId) const;
  virtual TraverseCallback needSpecialTraversal(
      const ParameterConfig& config) const;
  virtual void finishBatch();

private:
  real alpha_;
  real beta_;
  real tau_;
  real gamma_;
  real threshold_;
  real momentum_;
  real decayRate_;

protected:
  int64_t timer_;
  mutable std::vector<int64_t> t0Vec_;
  bool isParameterSparse_;
};

/*
 * AdaGrad optimization.
 * http://www.magicbroom.info/Papers/DuchiHaSi10.pdf
 */
class AdagradParameterOptimizer : public ParameterOptimizer {
public:
  explicit AdagradParameterOptimizer(const OptimizationConfig& optConfig)
      : ParameterOptimizer(optConfig) {
    addParameterType(PARAMETER_MOMENTUM);
    addParameterType(PARAMETER_GRADIENT_SQURESUM);
    addParameterType(PARAMETER_GRADIENT_SQURESUM1);
    addParameterType(PARAMETER_LEARNING_RATE);
    numUpdates_ = 0;
  }

  virtual void startBatch(int64_t numSamplesProcessed) {
    (void)numSamplesProcessed;
    ++numUpdates_;
  }
  virtual void update(const VectorPtr vecs[],
                      const ParameterConfig& config,
                      size_t sparseId) const;
  virtual TraverseCallback needSpecialTraversal(
      const ParameterConfig& config) const;

protected:
  int64_t numUpdates_;
  static const int64_t kMaxNumAccumulates = 16384;
};

/*
 * AdaDelta Optimization.
 * http://www.matthewzeiler.com/pubs/googleTR2012/googleTR2012.pdf
 */
class AdaDeltaParameterOptimizer : public ParameterOptimizer {
public:
  explicit AdaDeltaParameterOptimizer(const OptimizationConfig& optConfig)
      : ParameterOptimizer(optConfig) {
    addParameterType(PARAMETER_MOMENTUM);
    addParameterType(PARAMETER_GRADIENT_SQURESUM);
    addParameterType(PARAMETER_GRADIENT_SQURESUM1);
    addParameterType(PARAMETER_LEARNING_RATE);
    rou_ = optConfig.ada_rou();
    epsilon_ = optConfig.ada_epsilon();
  }

  virtual void startBatch(int64_t numSamplesProcessed) {
    learningRate_ = calcLearningRate(numSamplesProcessed, pass_);
  }

  virtual void update(const VectorPtr vecs[],
                      const ParameterConfig& config,
                      size_t sparseId) const;

protected:
  real rou_;
  real epsilon_;
};

// RMSProp Parameter Optimization.
class RMSPropParameterOptimizer : public ParameterOptimizer {
public:
  explicit RMSPropParameterOptimizer(const OptimizationConfig& optConfig)
      : ParameterOptimizer(optConfig) {
    addParameterType(PARAMETER_MOMENTUM);
    addParameterType(PARAMETER_GRADIENT_SQURESUM1);
    addParameterType(PARAMETER_GRADIENT_SQURESUM);
    addParameterType(PARAMETER_LEARNING_RATE);
    rou_ = optConfig.ada_rou();
    epsilon_ = optConfig.ada_epsilon();
  }

  virtual void init(size_t numRows, const ParameterConfig* config) {
    t0Vec_.resize(numRows);
    t0Vec_.assign(t0Vec_.size(), 0);
    timer_ = 0;
  }

  virtual void startBatch(int64_t numSamplesProcessed) {
    learningRate_ = calcLearningRate(numSamplesProcessed, pass_);
  }
  virtual void finishBatch() { timer_++; }

  virtual void update(const VectorPtr vecs[],
                      const ParameterConfig& config,
                      size_t sparseId) const;

protected:
  real rou_;
  real epsilon_;

  /**
   *  counting batches, donot need catch up with
   *  t(timer_) is current time,
   *  t0(t0Vec_) are last occur time of i rows.
   *  if one block is update by multi threads,
   *  caller should hash sparse ids to avoid write conflict in t0Vec_.
   */
  int64_t timer_;
  mutable std::vector<int64_t> t0Vec_;
};

// Decayed AdaGrad Optimization.
class DecayedAdagradParameterOptimizer : public ParameterOptimizer {
public:
  explicit DecayedAdagradParameterOptimizer(const OptimizationConfig& optConfig)
      : ParameterOptimizer(optConfig) {
    addParameterType(PARAMETER_MOMENTUM);
    addParameterType(PARAMETER_GRADIENT_SQURESUM);
    addParameterType(PARAMETER_LEARNING_RATE);
    rou_ = optConfig.ada_rou();
    epsilon_ = optConfig.ada_epsilon();
  }

  virtual void init(size_t numRows, const ParameterConfig* config) {
    t0Vec_.resize(numRows);
    t0Vec_.assign(t0Vec_.size(), 0);
    timer_ = 0;
  }

  virtual void startBatch(int64_t numSamplesProcessed) {
    learningRate_ = calcLearningRate(numSamplesProcessed, pass_);
  }
  virtual void finishBatch() { timer_++; }

  virtual void update(const VectorPtr vecs[],
                      const ParameterConfig& config,
                      size_t sparseId) const;

protected:
  real rou_;
  real epsilon_;

  /**
   *  counting batches, donot need catch up with
   *  t(timer_) is current time,
   *  t0(t0Vec_) are last occur time of i rows.
   *  if one block is update by multi threads,
   *  caller should hash sparse ids to avoid write conflict in t0Vec_.
   */
  int64_t timer_;
  mutable std::vector<int64_t> t0Vec_;
};

/**
 * Adam Optimizer.
 * Reference Paper: http://arxiv.org/abs/1412.6980 Algorithm 1
 */
class AdamParameterOptimizer : public ParameterOptimizer {
public:
  explicit AdamParameterOptimizer(const OptimizationConfig& optConfig)
      : ParameterOptimizer(optConfig),
        beta1_(optConfig.adam_beta1()),
        beta2_(optConfig.adam_beta2()),
        epsilon_(optConfig.adam_epsilon()),
        step_(1),
        learningRate_(optConfig.learning_rate()) {
    addParameterType(PARAMETER_MOMENTUM);
    addParameterType(PARAMETER_SECOND_MOMENTUM);
  }

  virtual void finishBatch() { ++step_; }

  virtual void update(const VectorPtr vecs[],
                      const ParameterConfig& config,
                      size_t sparseId) const;

protected:
  real beta1_;
  real beta2_;
  real epsilon_;
  int64_t step_;
  real learningRate_;
};

/**
 * AdaMax Optimizer.
 * Reference Paper: http://arxiv.org/abs/1412.6980 Algorithm 2
 */
class AdamaxParameterOptimizer : public ParameterOptimizer {
public:
  explicit AdamaxParameterOptimizer(const OptimizationConfig& optConfig)
      : ParameterOptimizer(optConfig),
        beta1_(optConfig.adam_beta1()),
        beta2_(optConfig.adam_beta2()),
        step_(1),
        learningRate_(optConfig.learning_rate()) {
    addParameterType(PARAMETER_MOMENTUM);
    addParameterType(PARAMETER_WEIGHTED_INFINITY_NORM);
  }

  virtual void finishBatch() { ++step_; }

  virtual void update(const VectorPtr vecs[],
                      const ParameterConfig& config,
                      size_t sparseId) const;

protected:
  real beta1_;
  real beta2_;
  int64_t step_;
  real learningRate_;
};

// Used in pserver,
// when PARAMETER_DELTA stores in PARAMETER_GRADIENT.
class AddOptimizer : public ParameterOptimizer {
public:
  explicit AddOptimizer(const OptimizationConfig& optConfig)
      : ParameterOptimizer(optConfig) {}

  virtual void startBatch(int64_t numSamplesProcessed) {
    // learningRate required by regularizer
    learningRate_ = calcLearningRate(numSamplesProcessed, pass_);
  }
  virtual void update(const VectorPtr vecs[],
                      const ParameterConfig& paraConfig,
                      size_t sparseId) const {
    vecs[PARAMETER_VALUE]->add(*vecs[PARAMETER_GRADIENT],
                               optConfig_.delta_add_rate());
  }
};

// A optimizer which does nothing.
class DummyOptimizer : public ParameterOptimizer {
public:
  explicit DummyOptimizer(const OptimizationConfig& optConfig)
      : ParameterOptimizer(optConfig) {}

  virtual void update(const VectorPtr vecs[],
                      const ParameterConfig& paraConfig,
                      size_t sparseId) const {}
};

// Do gradient clipping before sgd update
class OptimizerWithGradientClipping : public ParameterOptimizer {
public:
  OptimizerWithGradientClipping(const OptimizationConfig& optConfig,
                                ParameterOptimizer* optimizer)
      : ParameterOptimizer(optConfig), optimizer_(optimizer) {
    parameterTypes_ = optimizer_->getParameterTypes();
  }

  virtual void init(size_t numRows, const ParameterConfig* config) {
    optimizer_->init(numRows, config);
  }

  virtual void startPass() { optimizer_->startPass(); }
  virtual void finishPass() { optimizer_->finishPass(); }

  virtual void startBatch(int64_t numSamplesProcessed) {
    optimizer_->startBatch(numSamplesProcessed);
    learningRate_ = optimizer_->getLearningRate();
  }
  virtual void finishBatch() { optimizer_->finishBatch(); }

  virtual TraverseCallback needSpecialTraversal(
      const ParameterConfig& config) const {
    return optimizer_->needSpecialTraversal(config);
  }
  virtual void update(const VectorPtr vecs[],
                      const ParameterConfig& config,
                      size_t sparseId) const;

  virtual void setNoDecay() { optimizer_->setNoDecay(); }

protected:
  std::unique_ptr<ParameterOptimizer> optimizer_;
};

}  // namespace paddle
