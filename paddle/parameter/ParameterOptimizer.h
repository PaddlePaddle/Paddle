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

#include "LearningRateScheduler.h"
#include "Parameter.h"

namespace paddle {

/**
 * Some member functions are set to const for two reasons:
 *
 * 1. For sparse update thread safe: update(), traverse callback(const this)
 *    may be called many times, each time one row, and these function
 *    can be called parallelly by multi worker, to speed up large block.
 *
 * 2. For predicate functions, needSpecialTraversal(), startCatchUpWith()
 *    may be called many times, should be no state change between calls.
 */
class ParameterOptimizer {
public:
  typedef std::function<void(
      const VectorPtr vecs[], const ParameterConfig& config, size_t sparseId)>
      TraverseCallback;

public:
  explicit ParameterOptimizer(const OptimizationConfig& optConfig)
      : applyDecay_(true),
        optConfig_(optConfig),
        parameterTypes_{PARAMETER_VALUE, PARAMETER_GRADIENT},
        learningRate_(optConfig.learning_rate()),
        learningRateScheduler_(LearningRateScheduler::create(optConfig)),
        pass_(0),
        firstTime_(true) {}

  real calcLearningRate(int64_t numSamplesProcessed, int64_t pass) {
    return learningRateScheduler_->calcLearningRate(numSamplesProcessed, pass);
  }

  virtual ~ParameterOptimizer() {}

  /**
   * For sparse update, optimizer can maintain numRows of timer(t0).
   * Some sparse optimizer depends on parameter config in functions
   * such as startBatch(). Optimizer can get it here. But notice that,
   * not all callers can pass config here, so the optimizer should check
   * config passed in is not null ptr.
   */
  virtual void init(size_t numRows, const ParameterConfig* config) {}

  virtual void startPass() {}
  virtual void finishPass() { ++pass_; }

  /// called by Trainer before forward() of a batch.
  virtual void startBatch(int64_t numSamplesProcessed) {
    (void)numSamplesProcessed;
  }

  /**
   * following hooks useful for sparse update,
   * because the traversal in block costs.
   * called by Trainer after update and before finishBatch
   * e.g. Trainer call like this:
   *
   * @code
   * startBatch();
   * if (dense) {
   *   update(blockVec);
   * } else {//sparse
   *   for (row : rows_in_block) {update(rowVec)}
   * }
   * auto callback = needSpecialTraversal();
   * if (callback) {
   *   // do traverse, maybe multi-thread
   *   if (dense) {
   *     callback();
   *   } else {//sparse
   *     for (row : all_rows_in_block) {callback();}
   *   }
   * }
   * finishBatch();
   * @endcode
   *
   * @return callback if need traverse,
   *         else return nullptr.
   *         It should be no state change.
   */
  virtual TraverseCallback needSpecialTraversal(
      const ParameterConfig& config) const {
    return nullptr;
  }

  /// called by Trainer after backward() of a batch
  virtual void finishBatch() {}

  /**
   * between startBatch() and finishBatch(), update() will be called
   * by the trainer multiple times, each time for updating one Parameter
   * with its gradient in PARAMETER_GRADIENT. sparseId is row id,
   * when sparseId set, update is sparse, each time one row.
   */
  virtual void update(const VectorPtr vecs[],
                      const ParameterConfig& config,
                      size_t sparseId = -1LU) const = 0;

  /**
   * following hooks catch up with current time for sparse update,
   * In the beginning, call startCatchUpWith() and check return.
   * In the end, call finishCatchUpWith() to finish state.
   * callback do the actual works, can call many times for sparse data.
   * e.g. Trainer call like this:
   *
   * @code
   * auto callback = startCatchUpWith();
   * if (callback) {
   *   // do catch up with, maybe multi-thread
   *   if (dense) {
   *     callback();
   *   } else {//sparse
   *     for (row : rows_in_block) {callback();}
   *   }
   *   // finish catch up with, main thread
   *   finishCatchUpWith();
   * }
   * @endcode
   *
   * @return callback if need catch up with,
   *         else return nullptr.
   *         It should be no state change.
   */
  virtual TraverseCallback startCatchUpWith() const { return nullptr; }
  virtual void finishCatchUpWith() {}

  /**
   * following two hooks used by averager,
   * apply to final parameter value (PARAMETER_VALUE or PARAMETER_APPLY).
   *
   * restore() will restore orginal value if it apply to PARAMETER_VALUE.
   * Caller must ensure it's catched up with current time before apply.
   *
   * Use returned callback same way as callback returned by
   * ParameterOptimizer::needSpecialTraversal()
   */
  virtual TraverseCallback apply() { return nullptr; }
  virtual TraverseCallback restore() { return nullptr; }

  /// return the parameter types used by this updater
  const std::vector<ParameterType>& getParameterTypes() const {
    return parameterTypes_;
  }

  void addParameterType(ParameterType type) {
    for (auto t : parameterTypes_) {
      if (t == type) return;
    }
    parameterTypes_.push_back(type);
  }

  real getLearningRate() const { return learningRate_; }

  virtual void setNoDecay() { applyDecay_ = false; }

  static ParameterOptimizer* create(const OptimizationConfig& optConfig,
                                    bool inPserver = false);

protected:
  typedef std::vector<ParameterOptimizer::TraverseCallback> TraverseCallbackVec;

  static TraverseCallback composeCallbacks(
      const TraverseCallbackVec& callbacks) {
    if (callbacks.size() > 1LU) {
      return [callbacks](const VectorPtr vecs[],
                         const ParameterConfig& config,
                         size_t sparseId) {
        for (auto callback : callbacks) {
          callback(vecs, config, sparseId);
        }
      };
    }
    return (callbacks.size() == 1LU) ? callbacks[0] : nullptr;
  }

  bool applyDecay_;
  const OptimizationConfig& optConfig_;
  std::vector<ParameterType> parameterTypes_;

  /**
   * global learning rate, init value is opt_config.learning_rate,
   * sparse regularizer get this value per batch, after StartBatch() called
   * so, if lr change in StartBatch, please assign to learningRate_
   */
  real learningRate_;

  std::unique_ptr<LearningRateScheduler> learningRateScheduler_;
  int64_t pass_;  // current training pass (starting from 0)
  bool firstTime_;
};

}  // namespace paddle
