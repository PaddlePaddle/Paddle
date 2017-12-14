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

#include "OptimizerWithRegularizer.h"

namespace paddle {

ParameterOptimizer::TraverseCallback
OptimizerWithRegularizerEveryNumBatches::needSpecialTraversal(
    const ParameterConfig& config) const {
  TraverseCallbackVec callbacks;

  if (isRegularizationBatch(config)) {
    callbacks.emplace_back(
        [this](const VectorPtr vecs[],
               const ParameterConfig& config,
               size_t sparseId) { this->doTraversal(vecs, config); });
  }

  if (auto callback = optimizer_->needSpecialTraversal(config)) {
    callbacks.emplace_back(callback);
  }

  return composeCallbacks(callbacks);
}

void OptimizerWithRegularizerEveryNumBatches::doTraversal(
    const VectorPtr vecs[], const ParameterConfig& config) const {
  int32_t base =
      std::max(baseTimer_, (timer_ + 1 - config.num_batches_regularization()));
  regularizer_->update(
      vecs, config, optimizer_->getLearningRate(), base, timer_ + 1);
}

ParameterOptimizer::TraverseCallback
OptimizerWithRegularizerEveryNumBatches::startCatchUpWith() const {
  TraverseCallbackVec callbacks;

  if (auto callback = optimizer_->startCatchUpWith()) {
    callbacks.emplace_back(callback);
  }

  if (baseTimer_ < timer_) {
    callbacks.emplace_back(
        [this](const VectorPtr vecs[],
               const ParameterConfig& config,
               size_t sparseId) { this->catchUpWith(vecs, config, sparseId); });
  }

  return composeCallbacks(callbacks);
}

void OptimizerWithRegularizerEveryNumBatches::catchUpWith(
    const VectorPtr vecs[],
    const ParameterConfig& config,
    size_t sparseId) const {
  int32_t base = timer_ - timer_ % config.num_batches_regularization();
  regularizer_->update(vecs,
                       config,
                       optimizer_->getLearningRate(),
                       std::max(base, baseTimer_),
                       timer_);
}

void OptimizerWithRegularizerSparse::init(size_t numRows,
                                          const ParameterConfig* config) {
  OptimizerWithRegularizer::init(numRows, config);
  t0Vec_.resize(numRows);

  timer_ = 0;
  t0Vec_.assign(t0Vec_.size(), 0);
}

void OptimizerWithRegularizerSparse::update(const VectorPtr vecs[],
                                            const ParameterConfig& config,
                                            size_t sparseId) const {
  optimizer_->update(vecs, config, sparseId);
  // para W(t0) -> W(t+1)
  CHECK_LT(sparseId, t0Vec_.size());
  regularizer_->update(vecs,
                       config,
                       optimizer_->getLearningRate(),
                       t0Vec_[sparseId],
                       timer_ + 1);
  t0Vec_[sparseId] = timer_ + 1;
}

ParameterOptimizer::TraverseCallback
OptimizerWithRegularizerSparse::startCatchUpWith() const {
  TraverseCallbackVec callbacks;

  if (auto callback = optimizer_->startCatchUpWith()) {
    callbacks.emplace_back(callback);
  }

  if (timer_ > 0) {
    callbacks.emplace_back(
        [this](const VectorPtr vecs[],
               const ParameterConfig& config,
               size_t sparseId) { this->catchUpWith(vecs, config, sparseId); });
  }

  return composeCallbacks(callbacks);
}

void OptimizerWithRegularizerSparse::catchUpWith(const VectorPtr vecs[],
                                                 const ParameterConfig& config,
                                                 size_t sparseId) const {
  // para W(t0) -> W(t+1)
  CHECK_LT(sparseId, t0Vec_.size());
  regularizer_->update(
      vecs, config, optimizer_->getLearningRate(), t0Vec_[sparseId], timer_);
}

// factory method to create instance of OptimizerWithRegularizer
ParameterOptimizer* OptimizerWithRegularizer::create(
    const OptimizationConfig& optConfig,
    const ParameterConfig& paraConfig,
    bool isParameterSparse,
    bool inPserver) {
  ParameterOptimizer* optimizer =
      ParameterOptimizer::create(optConfig, inPserver);
  if ((optConfig.gradient_clipping_threshold() > 0.0f ||
       paraConfig.gradient_clipping_threshold() > 0.0f) &&
      !dynamic_cast<AddOptimizer*>(optimizer)) {
    optimizer = new OptimizerWithGradientClipping(optConfig, optimizer);
  }
  Regularizer* regularizer =
      Regularizer::get(optimizer->getParameterTypes(), paraConfig);
  if (!regularizer) {
    return optimizer;
  }

  if (paraConfig.num_batches_regularization() > 1) {
    if (optConfig.num_batches_per_send_parameter() > 1) {
      CHECK_EQ(optConfig.num_batches_per_send_parameter() %
                   paraConfig.num_batches_regularization(),
               0)
          << "regularization should be apply in sending batch";
    }
    CHECK(paraConfig.momentum() == 0.0f) << "Parameter cannot support momentum "
                                            "if num_batches_regularization set";

    if (optConfig.center_parameter_update_method() == "average" &&
        optConfig.num_batches_per_send_parameter() ==
            paraConfig.num_batches_regularization()) {
      LOG(INFO) << "decay in pserver and no decay in trainer";
      if (inPserver) {  // decay in pserver
        optimizer->setNoDecay();
        return new OptimizerWithRegularizer(optConfig, optimizer, regularizer);
      }
      // no decay in trainer
      optimizer->setNoDecay();
      return optimizer;
    }
    if (dynamic_cast<AddOptimizer*>(optimizer)) {
      return optimizer;  // normal average, no decay in pserver
    }
    // normal
    optimizer->setNoDecay();
    return new OptimizerWithRegularizerEveryNumBatches(
        optConfig, optimizer, regularizer);
  }
  if (isParameterSparse) {
    CHECK(paraConfig.momentum() == 0.0f)
        << "Parameter cannot support momentum if it's sparse.";
    optimizer->setNoDecay();
    return new OptimizerWithRegularizerSparse(
        optConfig, optimizer, regularizer);
  }
  // dense
  if (paraConfig.decay_rate_l1() == 0.0f ||
      dynamic_cast<AddOptimizer*>(optimizer)) {
    return optimizer;
  }
  CHECK(paraConfig.momentum() == 0.0f)
      << "Parameter cannot support momentum if it use L1 decay.";
  optimizer->setNoDecay();
  return new OptimizerWithRegularizer(optConfig, optimizer, regularizer);
}

}  // namespace paddle
