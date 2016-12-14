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

#include "FirstOrderOptimizer.h"

namespace paddle {

// add regularizer for objective function to do optimization
class OptimizerWithRegularizer : public ParameterOptimizer {
public:
  static ParameterOptimizer* create(const OptimizationConfig& optConfig,
                                    const ParameterConfig& paraConfig,
                                    bool isParameterSparse,
                                    bool inPserver);

  OptimizerWithRegularizer(const OptimizationConfig& optConfig,
                           ParameterOptimizer* optimizer,
                           Regularizer* regularizer)
      : ParameterOptimizer(optConfig),
        optimizer_(optimizer),
        regularizer_(regularizer) {
    parameterTypes_ = optimizer_->getParameterTypes();
  }

  virtual void init(size_t numRows, const ParameterConfig* config) {
    optimizer_->init(numRows, config);
  }

  virtual void startPass() {
    optimizer_->startPass();
    timer_ = 0;
  }

  virtual void finishPass() { optimizer_->finishPass(); }

  virtual void startBatch(int64_t numSamplesProcessed) {
    optimizer_->startBatch(numSamplesProcessed);
  }

  virtual void finishBatch() {
    optimizer_->finishBatch();
    ++timer_;
  }

  virtual TraverseCallback needSpecialTraversal(
      const ParameterConfig& config) const {
    return optimizer_->needSpecialTraversal(config);
  }

  virtual void update(const VectorPtr vecs[],
                      const ParameterConfig& config,
                      size_t sparseId) const {
    optimizer_->update(vecs, config, sparseId);
    regularizer_->update(vecs, config, optimizer_->getLearningRate(), 0, 1);
  }

protected:
  std::unique_ptr<ParameterOptimizer> optimizer_;
  Regularizer* regularizer_;

  /**
   *  counting batches, clear after catch up with
   *  t(timer_) is current time,
   *  t0(t0Vec_) are last occur time of i rows.
   *  if one block is update by multi threads,
   *  caller should hash sparse ids to avoid write conflict in t0Vec_.
   */
  int timer_;
};

// Regularized Loss function for every num of batches
class OptimizerWithRegularizerEveryNumBatches
    : public OptimizerWithRegularizer {
public:
  OptimizerWithRegularizerEveryNumBatches(const OptimizationConfig& optConfig,
                                          ParameterOptimizer* optimizer,
                                          Regularizer* regularizer)
      : OptimizerWithRegularizer(optConfig, optimizer, regularizer) {}

  virtual void startPass() {
    OptimizerWithRegularizer::startPass();
    baseTimer_ = 0;
  }

  virtual void update(const VectorPtr vecs[],
                      const ParameterConfig& config,
                      size_t sparseId) const {
    optimizer_->update(vecs, config, sparseId);
  }

  virtual TraverseCallback needSpecialTraversal(
      const ParameterConfig& config) const;
  void doTraversal(const VectorPtr vecs[], const ParameterConfig& config) const;

  void catchUpWith(const VectorPtr vecs[],
                   const ParameterConfig& config,
                   size_t sparseId) const;

  virtual TraverseCallback startCatchUpWith() const;
  virtual void finishCatchUpWith() { baseTimer_ = timer_; }

protected:
  bool isRegularizationBatch(const ParameterConfig& config) const {
    return ((timer_ + 1) % config.num_batches_regularization() == 0);
  }

  /**
   *  recored the timer_ value while catchUpWith called.
   */
  int baseTimer_;
};

// Regularized Loss function with Sparse support
class OptimizerWithRegularizerSparse : public OptimizerWithRegularizer {
public:
  OptimizerWithRegularizerSparse(const OptimizationConfig& optConfig,
                                 ParameterOptimizer* optimizer,
                                 Regularizer* regularizer)
      : OptimizerWithRegularizer(optConfig, optimizer, regularizer) {}

  virtual void init(size_t numRows, const ParameterConfig* config);

  virtual void update(const VectorPtr vecs[],
                      const ParameterConfig& config,
                      size_t sparseId) const;
  void catchUpWith(const VectorPtr vecs[],
                   const ParameterConfig& config,
                   size_t sparseId) const;
  virtual TraverseCallback startCatchUpWith() const;
  virtual void finishCatchUpWith() {
    timer_ = 0;
    t0Vec_.assign(t0Vec_.size(), 0);
  }

protected:
  /**
   *  t0Vec_ are last occur time of i rows
   *  if one block is update by multi threads,
   *  caller should hash sparse ids to avoid write conflict in t0Vec_.
   */
  mutable std::vector<int32_t> t0Vec_;
};

}  // namespace paddle
