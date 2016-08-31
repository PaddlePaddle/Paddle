/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */


#include "paddle/utils/Util.h"
#include "paddle/utils/Flags.h"

#include "FirstOrderOptimizer.h"

#include <cmath>

P_DEFINE_bool(log_clipping, false, "enable log clipping or not");

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
                               *vecs[PARAMETER_MOMENTUM_VT], 1.0 / beta_);

  } else {
    vecs[PARAMETER_VALUE]->sgdUpdate(
        *vecs[PARAMETER_GRADIENT], *vecs[PARAMETER_MOMENTUM],
        learningRate_ * paraConfig.learning_rate(), paraConfig.momentum(),
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
    return [this](const VectorPtr vecs[], const ParameterConfig& config,
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
  vecs[PARAMETER_GRADIENT_SQURESUM1]->addSquare(*vecs[PARAMETER_GRADIENT],
                                                1.0f);
  vecs[PARAMETER_LEARNING_RATE]->add(*vecs[PARAMETER_GRADIENT_SQURESUM],
                                     *vecs[PARAMETER_GRADIENT_SQURESUM1]);
  vecs[PARAMETER_LEARNING_RATE]->add(optConfig_.ada_epsilon());
  vecs[PARAMETER_LEARNING_RATE]->invSqrt(*vecs[PARAMETER_LEARNING_RATE]);

  vecs[PARAMETER_VALUE]->sgdUpdate(
      *vecs[PARAMETER_GRADIENT], *vecs[PARAMETER_MOMENTUM],
      *vecs[PARAMETER_LEARNING_RATE], learningRate_ * config.learning_rate(),
      config.momentum(), applyDecay_ ? config.decay_rate() : 0);
}

ParameterOptimizer::TraverseCallback
AdagradParameterOptimizer::needSpecialTraversal(
    const ParameterConfig& config) const {
  if (numUpdates_ % kMaxNumAccumulates == 0) {
    // Move the sum to a different buffer to avoid loss of precision
    // due to too many sums.
    return [this](const VectorPtr vecs[], const ParameterConfig& config,
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
  // E(g_t^2) = \rou * E(g_{t-1}^2) + (1-\rou) * g^2
  vecs[PARAMETER_GRADIENT_SQURESUM]->decayAddSquare(*vecs[PARAMETER_GRADIENT],
                                                    rou_, 1.0f - rou_);

  // learn_rate = sqrt( ( E(dx_{t-1}^2) + epsilon ) / ( E(g_t^2) + epsilon ) )
  vecs[PARAMETER_LEARNING_RATE]->dotDiv(*vecs[PARAMETER_GRADIENT_SQURESUM1],
                                        *vecs[PARAMETER_GRADIENT_SQURESUM],
                                        epsilon_, epsilon_);
  vecs[PARAMETER_LEARNING_RATE]->sqrt();

  // E(dx_t^2) = \rou * E(dx_{t-1}^2) + (1-\rou) * (-g*learn_rate)^2
  vecs[PARAMETER_GRADIENT_SQURESUM1]->decayAddSquareMul(
      *vecs[PARAMETER_GRADIENT], *vecs[PARAMETER_LEARNING_RATE], rou_,
      1.0f - rou_);

  vecs[PARAMETER_VALUE]->sgdUpdate(
      *vecs[PARAMETER_GRADIENT], *vecs[PARAMETER_MOMENTUM],
      *vecs[PARAMETER_LEARNING_RATE], learningRate_ * config.learning_rate(),
      config.momentum(), applyDecay_ ? config.decay_rate() : 0);
}

void RMSPropParameterOptimizer::update(const VectorPtr vecs[],
                                       const ParameterConfig& config,
                                       size_t sparseId) const {
  real accumulatedRou = rou_;

  bool firstTime = timer_ == 0;
  if (sparseId != -1LU) {
    CHECK_LT(sparseId, t0Vec_.size());
    accumulatedRou = std::pow(rou_, timer_ + 1 - t0Vec_[sparseId]);
    firstTime = t0Vec_[sparseId] == 0;
    t0Vec_[sparseId] = timer_ + 1;
  }

  // E(g_t^2) = \rou * E(g_{t-1}^2) + (1-\rou) * g^2
  // For the first time update, make the sum be the current square
  // so that the initial estimation of E(g_t^2) will not be too small.
  vecs[PARAMETER_GRADIENT_SQURESUM]->decayAddSquare(
      *vecs[PARAMETER_GRADIENT], accumulatedRou,
      firstTime ? 1.0f : 1.0f - rou_);

  // E(g_t) = \rou * E(g_{t-1}) + (1-\rou) * g
  vecs[PARAMETER_GRADIENT_SQURESUM1]->add(*vecs[PARAMETER_GRADIENT],
                                          accumulatedRou, 1.0f - rou_);

  // learn_rate = 1/sqrt( ( E(g_t^2) - (E(g_t))^2 + epsilon )
  // Basiclly if the sign of the gradient changes more often,
  // the learning rate will be decreased.
  vecs[PARAMETER_LEARNING_RATE]->assign(*vecs[PARAMETER_GRADIENT_SQURESUM]);
  vecs[PARAMETER_LEARNING_RATE]->addSquare(*vecs[PARAMETER_GRADIENT_SQURESUM1],
                                           -1.0f);
  vecs[PARAMETER_LEARNING_RATE]->add(optConfig_.ada_epsilon());
  vecs[PARAMETER_LEARNING_RATE]->invSqrt(*vecs[PARAMETER_LEARNING_RATE]);

  vecs[PARAMETER_VALUE]->sgdUpdate(
      *vecs[PARAMETER_GRADIENT], *vecs[PARAMETER_MOMENTUM],
      *vecs[PARAMETER_LEARNING_RATE], learningRate_ * config.learning_rate(),
      config.momentum(), applyDecay_ ? config.decay_rate() : 0);
}

void DecayedAdagradParameterOptimizer::update(const VectorPtr vecs[],
                                              const ParameterConfig& config,
                                              size_t sparseId) const {
  real accumulatedRou = rou_;

  bool firstTime = timer_ == 0;
  if (sparseId != -1LU) {
    CHECK_LT(sparseId, t0Vec_.size());
    accumulatedRou = std::pow(rou_, timer_ + 1 - t0Vec_[sparseId]);
    firstTime = t0Vec_[sparseId] == 0;
    t0Vec_[sparseId] = timer_ + 1;
  }

  // E(g_t^2) = \rou * E(g_{t-1}^2) + (1-\rou) * g^2
  // For the first time update, make the sum be the current square
  // so that the initial estimation of E(g_t^2) will not be too small.
  vecs[PARAMETER_GRADIENT_SQURESUM]->decayAddSquare(
      *vecs[PARAMETER_GRADIENT], accumulatedRou,
      firstTime ? 1.0f : 1.0f - rou_);

  // learn_rate = 1/sqrt( ( E(g_t^2) + epsilon )
  // Basiclly if the bigger the magnitude gradient is,
  // the smaller the learning rate will be.
  vecs[PARAMETER_LEARNING_RATE]->assign(optConfig_.ada_epsilon());
  vecs[PARAMETER_LEARNING_RATE]->add(*vecs[PARAMETER_GRADIENT_SQURESUM]);
  vecs[PARAMETER_LEARNING_RATE]->invSqrt(*vecs[PARAMETER_LEARNING_RATE]);

  vecs[PARAMETER_VALUE]->sgdUpdate(
      *vecs[PARAMETER_GRADIENT], *vecs[PARAMETER_MOMENTUM],
      *vecs[PARAMETER_LEARNING_RATE], learningRate_ * config.learning_rate(),
      config.momentum(), applyDecay_ ? config.decay_rate() : 0);
}

void AdamParameterOptimizer::update(const VectorPtr vecs[],
                                    const ParameterConfig& config,
                                    size_t sparseId) const {
  CHECK(sparseId == -1UL) << "Sparse update is not supported";
  Vector* m = vecs[PARAMETER_MOMENTUM].get();
  Vector* g = vecs[PARAMETER_GRADIENT].get();
  Vector* v = vecs[PARAMETER_SECOND_MOMENTUM].get();
  Vector* theta = vecs[PARAMETER_VALUE].get();

  // m_t = \beta_1 * m_{t-1} + (1-\beta_1)* g_t;
  m->add(*g, beta1_, 1 - beta1_);

  // v_t = \beta_2 * v_{t-1} + (1-\beta_2)* g_{t-1}^2
  g->square();
  v->add(*g, beta2_, 1 - beta2_);

  // tmp = m_t / ( \sqrt{v_t} + \epsilon )
  // \theta_t = \theta_{t-1} - \alpha * \sqrt(1-\beta_2^t) / (1-\beta_1^t) * tmp
  g->sqrt(*v);
  g->dotDiv(*m, *g, 0., epsilon_);
  real alpha = config.learning_rate() * learningRate_;
  alpha = alpha * std::sqrt(1 - std::pow(beta2_, step_)) /
          (1 - std::pow(beta1_, step_));
  theta->add(*theta, 1.0, *g, -alpha);
}

void AdamaxParameterOptimizer::update(const VectorPtr vecs[],
                                      const ParameterConfig& config,
                                      size_t sparseId) const {
  CHECK(sparseId == -1UL) << "Sparse update is not supported";
  Vector* m = vecs[PARAMETER_MOMENTUM].get();
  Vector* g = vecs[PARAMETER_GRADIENT].get();
  Vector* u = vecs[PARAMETER_WEIGHTED_INFINITY_NORM].get();
  Vector* theta = vecs[PARAMETER_VALUE].get();

  // m_t = \beta_1 * m_{t-1} + (1-\beta_1)* g_t;
  m->add(*g, beta1_, 1 - beta1_);

  // u_t = max(\beta_2*u_{t-1}, abs(g_t))
  u->mulScalar(beta2_);
  g->abs();
  u->max(*u, *g);

  // \theta_t = \theta_{t-1} - (\alpha/(1-\beta_1^t))*m_t/u_t
  g->dotDiv(*m, *u);
  real learningRate = config.learning_rate() * learningRate_;
  learningRate /= (1 - std::pow(beta1_, step_));
  theta->add(*theta, 1.0, *g, -learningRate);
}


void OptimizerWithGradientClipping::update(const VectorPtr vecs[],
                                           const ParameterConfig& config,
                                           size_t sparseId) const {
  real maxAbsGrad = vecs[PARAMETER_GRADIENT]->getAbsMax();
  if (maxAbsGrad > config.gradient_clipping_threshold()) {
    if (FLAGS_log_clipping) {
      real avgAbsGrad = vecs[PARAMETER_GRADIENT]->getAbsSum() /
                        vecs[PARAMETER_GRADIENT]->getSize();
      LOG(INFO) << "parameter=" << config.name() << " need clipping,"
                << " max grad=" << maxAbsGrad << " avg grad=" << avgAbsGrad;
    }
    vecs[PARAMETER_GRADIENT]->clip(-config.gradient_clipping_threshold(),
                                   config.gradient_clipping_threshold());
  }

  optimizer_->update(vecs, config, sparseId);
}

}  // namespace paddle
