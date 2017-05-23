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

#include "FirstOrderOptimizer.h"
#include "paddle/math/TrainingAlgorithmOp.h"
#include "paddle/utils/Flags.h"
#include "paddle/utils/Util.h"

#include <cmath>

DEFINE_bool(log_clipping, false, "enable log clipping or not");

namespace paddle {

SparseMomentumParameterOptimizer::SparseMomentumParameterOptimizer(
    const OptimizationConfig& optConfig)
    : ParameterOptimizer(optConfig) {
  addParameterType(PARAMETER_MOMENTUM);
  addParameterType(PARAMETER_MOMENTUM_UT);
  addParameterType(PARAMETER_MOMENTUM_VT);
  alpha_ = 1;
  beta_ = 1;
  tau_ = -1;
  threshold_ = 1e+06;
}

void SparseMomentumParameterOptimizer::init(size_t numRows,
                                            const ParameterConfig* config) {
  isParameterSparse_ = numRows != 0;
  t0Vec_.resize(numRows);
  t0Vec_.assign(t0Vec_.size(), 0);
  timer_ = 0;
  momentum_ = config->momentum();
  decayRate_ = config->decay_rate();
  gamma_ = config->learning_rate();
}

void SparseMomentumParameterOptimizer::startBatch(int64_t numSamplesProcessed) {
  learningRate_ = calcLearningRate(numSamplesProcessed, pass_);
  if (isParameterSparse_) {
    tau_ = tau_ + beta_ / alpha_;
    alpha_ = alpha_ / momentum_;
    beta_ = beta_ / (1 + decayRate_ * gamma_ * learningRate_);
  }
}

void SparseMomentumParameterOptimizer::update(const VectorPtr vecs[],
                                              const ParameterConfig& paraConfig,
                                              size_t sparseId) const {
  if (sparseId != -1LU) {
    CHECK_LT(sparseId, t0Vec_.size());
    if (t0Vec_[sparseId] == 0) {
      vecs[PARAMETER_MOMENTUM_VT]->assign(*vecs[PARAMETER_VALUE]);
      t0Vec_[sparseId] = 1;
    }
    vecs[PARAMETER_MOMENTUM_UT]->add(*vecs[PARAMETER_GRADIENT],
                                     -alpha_ * gamma_ * learningRate_);
    vecs[PARAMETER_MOMENTUM_VT]->add(*vecs[PARAMETER_GRADIENT],
                                     tau_ * alpha_ * gamma_ * learningRate_);
    vecs[PARAMETER_VALUE]->add(*vecs[PARAMETER_MOMENTUM_UT],
                               tau_ / beta_ + 1.0 / alpha_,
                               *vecs[PARAMETER_MOMENTUM_VT],
                               1.0 / beta_);

  } else {
    vecs[PARAMETER_VALUE]->sgdUpdate(*vecs[PARAMETER_GRADIENT],
                                     *vecs[PARAMETER_MOMENTUM],
                                     learningRate_ * paraConfig.learning_rate(),
                                     paraConfig.momentum(),
                                     applyDecay_ ? paraConfig.decay_rate() : 0);
  }
}

ParameterOptimizer::TraverseCallback
SparseMomentumParameterOptimizer::needSpecialTraversal(
    const ParameterConfig& config) const {
  if (alpha_ > threshold_ && isParameterSparse_) {
    //  Restart to avoid large value multiplication
    //  1. \alpha = 1, \beta = 1, \tau = 0
    //  2. Note that \tau * u_t + v_t = \beta \theta_t, therefore:
    //     u_t should be rescaled to u_t/alpha_
    //     v_t should be reset to \theta_t
    return [this](const VectorPtr vecs[],
                  const ParameterConfig& config,
                  size_t sparseId) {
      vecs[PARAMETER_MOMENTUM_UT]->divScalar(alpha_);
      vecs[PARAMETER_MOMENTUM_VT]->assign(*vecs[PARAMETER_VALUE]);
    };
  } else {
    return nullptr;
  }
}

void SparseMomentumParameterOptimizer::finishBatch() {
  timer_++;
  if (!isParameterSparse_) return;
  if (alpha_ > threshold_) {
    alpha_ = 1;
    beta_ = 1;
    tau_ = -1;
  }
}

void AdagradParameterOptimizer::update(const VectorPtr vecs[],
                                       const ParameterConfig& config,
                                       size_t sparseId) const {
  BaseMatrix& value = *vecs[PARAMETER_VALUE];
  BaseMatrix& grad = *vecs[PARAMETER_GRADIENT];
  BaseMatrix& mom = *vecs[PARAMETER_MOMENTUM];
  BaseMatrix& accum_buffer = *vecs[PARAMETER_GRADIENT_SQURESUM];
  BaseMatrix& accum = *vecs[PARAMETER_GRADIENT_SQURESUM1];
  BaseMatrix& lr = *vecs[PARAMETER_LEARNING_RATE];

  real epsilon = optConfig_.ada_epsilon();
  real learningRate = learningRate_ * config.learning_rate();
  real momentum = config.momentum();
  real decayRate = applyDecay_ ? config.decay_rate() : 0;

  adagradApply(value,
               grad,
               mom,
               accum_buffer,
               accum,
               lr,
               epsilon,
               learningRate,
               momentum,
               decayRate);
}

ParameterOptimizer::TraverseCallback
AdagradParameterOptimizer::needSpecialTraversal(
    const ParameterConfig& config) const {
  if (numUpdates_ % kMaxNumAccumulates == 0) {
    // Move the sum to a different buffer to avoid loss of precision
    // due to too many sums.
    return [this](const VectorPtr vecs[],
                  const ParameterConfig& config,
                  size_t sparseId) {
      vecs[PARAMETER_GRADIENT_SQURESUM]->add(
          *vecs[PARAMETER_GRADIENT_SQURESUM1]);
      vecs[PARAMETER_GRADIENT_SQURESUM1]->zeroMem();
    };
  } else {
    return nullptr;
  }
}

void AdaDeltaParameterOptimizer::update(const VectorPtr vecs[],
                                        const ParameterConfig& config,
                                        size_t sparseId) const {
  CHECK(sparseId == -1LU) << "Sparse update is not supported";

  BaseMatrix& value = *vecs[PARAMETER_VALUE];
  BaseMatrix& grad = *vecs[PARAMETER_GRADIENT];
  BaseMatrix& mom = *vecs[PARAMETER_MOMENTUM];
  BaseMatrix& accum = *vecs[PARAMETER_GRADIENT_SQURESUM];
  BaseMatrix& accum_update = *vecs[PARAMETER_GRADIENT_SQURESUM1];
  BaseMatrix& lr = *vecs[PARAMETER_LEARNING_RATE];

  real learningRate = learningRate_ * config.learning_rate();
  real momentum = config.momentum();
  real decayRate = applyDecay_ ? config.decay_rate() : 0;

  adadeltaApply(value,
                grad,
                mom,
                accum,
                accum_update,
                lr,
                rou_,
                epsilon_,
                learningRate,
                momentum,
                decayRate);
}

void RMSPropParameterOptimizer::update(const VectorPtr vecs[],
                                       const ParameterConfig& config,
                                       size_t sparseId) const {
  BaseMatrix& value = *vecs[PARAMETER_VALUE];
  BaseMatrix& grad = *vecs[PARAMETER_GRADIENT];
  BaseMatrix& mom = *vecs[PARAMETER_MOMENTUM];
  BaseMatrix& sum = *vecs[PARAMETER_GRADIENT_SQURESUM];
  BaseMatrix& sum1 = *vecs[PARAMETER_GRADIENT_SQURESUM1];
  BaseMatrix& lr = *vecs[PARAMETER_LEARNING_RATE];

  real accumulatedRou = rou_;
  bool firstTime = timer_ == 0;
  if (sparseId != -1LU) {
    CHECK_LT(sparseId, t0Vec_.size());
    accumulatedRou = std::pow(rou_, timer_ + 1 - t0Vec_[sparseId]);
    firstTime = t0Vec_[sparseId] == 0;
    t0Vec_[sparseId] = timer_ + 1;
  }

  real epsilon = optConfig_.ada_epsilon();
  real learningRate = learningRate_ * config.learning_rate();
  real momentum = config.momentum();
  real decayRate = applyDecay_ ? config.decay_rate() : 0;

  rmspropApply(value,
               grad,
               mom,
               sum,
               sum1,
               lr,
               accumulatedRou,
               rou_,
               epsilon,
               learningRate,
               momentum,
               decayRate,
               firstTime);
}

void DecayedAdagradParameterOptimizer::update(const VectorPtr vecs[],
                                              const ParameterConfig& config,
                                              size_t sparseId) const {
  BaseMatrix& value = *vecs[PARAMETER_VALUE];
  BaseMatrix& grad = *vecs[PARAMETER_GRADIENT];
  BaseMatrix& mom = *vecs[PARAMETER_MOMENTUM];
  BaseMatrix& sum = *vecs[PARAMETER_GRADIENT_SQURESUM];
  BaseMatrix& lr = *vecs[PARAMETER_LEARNING_RATE];

  real accumulatedRou = rou_;
  bool firstTime = timer_ == 0;
  if (sparseId != -1LU) {
    CHECK_LT(sparseId, t0Vec_.size());
    accumulatedRou = std::pow(rou_, timer_ + 1 - t0Vec_[sparseId]);
    firstTime = t0Vec_[sparseId] == 0;
    t0Vec_[sparseId] = timer_ + 1;
  }

  real epsilon = optConfig_.ada_epsilon();
  real learningRate = learningRate_ * config.learning_rate();
  real momentum = config.momentum();
  real decayRate = applyDecay_ ? config.decay_rate() : 0;

  decayedAdagradApply(value,
                      grad,
                      mom,
                      sum,
                      lr,
                      accumulatedRou,
                      rou_,
                      epsilon,
                      learningRate,
                      momentum,
                      decayRate,
                      firstTime);
}

void AdamParameterOptimizer::update(const VectorPtr vecs[],
                                    const ParameterConfig& config,
                                    size_t sparseId) const {
  CHECK(sparseId == -1UL) << "Sparse update is not supported";

  real beta1_power = std::pow(beta1_, step_);
  real beta2_power = std::pow(beta2_, step_);
  real learningRate = config.learning_rate() * learningRate_;

  BaseMatrix& value = *vecs[PARAMETER_VALUE];
  BaseMatrix& grad = *vecs[PARAMETER_GRADIENT];
  BaseMatrix& mom = *vecs[PARAMETER_MOMENTUM];
  BaseMatrix& v = *vecs[PARAMETER_SECOND_MOMENTUM];

  adamApply(value,
            grad,
            mom,
            v,
            beta1_,
            beta2_,
            beta1_power,
            beta2_power,
            epsilon_,
            learningRate);
}

void AdamaxParameterOptimizer::update(const VectorPtr vecs[],
                                      const ParameterConfig& config,
                                      size_t sparseId) const {
  CHECK(sparseId == -1UL) << "Sparse update is not supported";
  real learningRate = config.learning_rate() * learningRate_;

  BaseMatrix& value = *vecs[PARAMETER_VALUE];
  BaseMatrix& grad = *vecs[PARAMETER_GRADIENT];
  BaseMatrix& mom = *vecs[PARAMETER_MOMENTUM];
  BaseMatrix& u = *vecs[PARAMETER_WEIGHTED_INFINITY_NORM];

  adamaxApply(value, grad, mom, u, beta1_, beta2_, step_, learningRate);
}

void OptimizerWithGradientClipping::update(const VectorPtr vecs[],
                                           const ParameterConfig& config,
                                           size_t sparseId) const {
  real globalThreshold = optConfig_.gradient_clipping_threshold();
  real localThreshold = config.gradient_clipping_threshold();

  // Use local gradient clipping threshold if it's enabled,
  // otherwise using the global one.
  real threshold = localThreshold > 0.0f ? localThreshold : globalThreshold;
  std::string field = localThreshold > 0.0f ? "local" : "global";

  real maxAbsGrad = vecs[PARAMETER_GRADIENT]->getAbsMax();
  if (maxAbsGrad > threshold) {
    if (FLAGS_log_clipping) {
      real avgAbsGrad = vecs[PARAMETER_GRADIENT]->getAbsSum() /
                        vecs[PARAMETER_GRADIENT]->getSize();
      LOG(INFO) << "parameter=" << config.name() << " need clipping by "
                << field << " threshold=" << threshold
                << ", max grad=" << maxAbsGrad << ", avg grad=" << avgAbsGrad;
    }
    vecs[PARAMETER_GRADIENT]->clip(-threshold, threshold);
  }
  optimizer_->update(vecs, config, sparseId);
}

}  // namespace paddle
