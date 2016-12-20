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

// After Optimization, parameter values are further averaged within
// time range.
class AverageOptimizer : public ParameterOptimizer {
public:
  // if *useParameterApply* set, use PARAMETER_APPLY to store averaged parameter
  // else use PARAMETER_VALUE, and value backup in PARAMETER_GRADIENT
  AverageOptimizer(const OptimizationConfig& optConfig,
                   ParameterOptimizer* optimizer,
                   bool useParameterApply);

  static ParameterOptimizer* create(const OptimizationConfig& optConfig,
                                    ParameterOptimizer* optimizer,
                                    bool isParameterSparse = false,
                                    bool useParameterApply = false);

  virtual void init(size_t numRows, const ParameterConfig* config) {
    optimizer_->init(numRows, config);
  }

  virtual void startPass() { optimizer_->startPass(); }
  virtual void finishPass() {
    optimizer_->finishPass();
    updateAverageWindowLimit();
  }

  virtual void startBatch(int64_t numSamplesProcessed);
  virtual void finishBatch();
  virtual void update(const VectorPtr vecs[],
                      const ParameterConfig& paraConfig,
                      size_t sparseId) const {
    optimizer_->update(vecs, paraConfig, sparseId);
    vecs[PARAMETER_SUM1]->add(*vecs[PARAMETER_VALUE], 1.0f);
  }

  virtual TraverseCallback needSpecialTraversal(
      const ParameterConfig& config) const;

  virtual TraverseCallback startCatchUpWith() const {
    return optimizer_->startCatchUpWith();
  }
  virtual void finishCatchUpWith() { return optimizer_->finishCatchUpWith(); }

  virtual TraverseCallback apply();
  virtual TraverseCallback restore();

  virtual void setNoDecay() { optimizer_->setNoDecay(); }

protected:
  std::unique_ptr<ParameterOptimizer> optimizer_;
  bool useApply_;

  // should only be called from finishPass()
  void updateAverageWindowLimit() {
    if (!optConfig_.has_max_average_window()) {
      // use the number of batches in the last pass as maxAverageWindow_
      CHECK_GT(numUpdates_, prevNumUpdates_);
      maxAverageWindow_ = numUpdates_ - prevNumUpdates_;
      prevNumUpdates_ = numUpdates_;
    }
    minAverageWindow_ = std::min(minAverageWindow_, numUpdates_);
  }

  bool isAverageWindowTooLong() const {
    return numAccumulates_ >= minAverageWindow_ &&
           numAccumulates_ >=
               std::min<int64_t>(maxAverageWindow_,
                                 numUpdates_ * optConfig_.average_window());
  }

  static const int64_t kMaxNumAccumulates = 16384;
  int64_t numUpdates_;
  int64_t prevNumUpdates_;
  int64_t numAccumulates_;
  int64_t oldNumAccumulates_;
  int64_t minAverageWindow_;
  int64_t maxAverageWindow_;
};

// Average Optimizer with Sparse support.
class AverageSparseOptimizer : public AverageOptimizer {
public:
  AverageSparseOptimizer(const OptimizationConfig& optConfig,
                         ParameterOptimizer* optimizer,
                         bool useParameterApply)
      : AverageOptimizer(optConfig, optimizer, useParameterApply) {}

  virtual void init(size_t numRows, const ParameterConfig* config) {
    AverageOptimizer::init(numRows, config);

    t0Vec_.resize(numRows);

    timer_ = 0;
    t0Vec_.assign(t0Vec_.size(), 0);
  }
  virtual void finishBatch() {
    AverageOptimizer::finishBatch();
    timer_++;
  }
  virtual void update(const VectorPtr vecs[],
                      const ParameterConfig& paraConfig,
                      size_t sparseId) const;
  void catchUpWith(const VectorPtr vecs[],
                   const ParameterConfig& paraConfig,
                   size_t sparseId) const;
  virtual TraverseCallback startCatchUpWith() const;
  virtual void finishCatchUpWith() {
    optimizer_->finishCatchUpWith();

    timer_ = 0;
    t0Vec_.assign(t0Vec_.size(), 0);
  }

protected:
  /**
   *  counting batches, clear after catch up with
   *  t(timer_) is current time,
   *  t0(t0Vec_) are last occur time of i rows.
   *  if one block is update by multi threads,
   *  caller should hash sparse ids to avoid write conflict in t0Vec_.
   */
  int timer_;
  mutable std::vector<int32_t> t0Vec_;
};

}  // namespace paddle
