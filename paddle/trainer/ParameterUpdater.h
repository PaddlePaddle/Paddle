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

#include "paddle/utils/Thread.h"
#include "paddle/utils/Util.h"

#include "paddle/parameter/AverageOptimizer.h"
#include "paddle/parameter/FirstOrderOptimizer.h"
#include "paddle/parameter/OptimizerFunctions.h"
#include "paddle/parameter/OptimizerWithRegularizer.h"
#include "paddle/parameter/Parameter.h"
#include "paddle/parameter/ParameterUpdaterBase.h"

#include "TrainerConfig.pb.h"
#include "paddle/gserver/layers/Layer.h"

#include <memory>
#include <vector>

namespace paddle {

/**
 * @brief Parameter Updater for SGD, and local(not cluster) run.
 */
class SgdLocalUpdater : public ParameterUpdater {
 public:
  /**
   * @brief Ctor. Initialize optimizer locally by optConfig.
   * @param optConfig optimization config.
   * @param withAverager with average optimizer or not, default is true.
   */
  explicit SgdLocalUpdater(const OptimizationConfig& optConfig,
                           bool withAverager = true)
      : numSamplesProcessed_(0) {
    auto baseOptimizer = ParameterOptimizer::create(optConfig);
    optimizer_.reset(withAverager
                         ? AverageOptimizer::create(optConfig, baseOptimizer)
                         : baseOptimizer);
    CHECK(optimizer_) << "fail to create optimizer: "
                      << optConfig.learning_method();
    auto types = optimizer_->getParameterTypes();
    for (auto type : types) {
      addParameterType(type);
    }
  }

  /**
   * @brief Initialize parameters and optimizer_.
   *        For example,
   *           If optimizer need hassien vector, then parameter's hassien will
   *           be initialized.
   * @param parameters The parameter need to be initialized.
   */
  virtual void init(const std::vector<ParameterPtr>& parameters) {
    ParameterUpdater::init(parameters);
    optimizer_->init(parameters_.size(), nullptr);
    // check no L1 decay in parameter configs
    CHECK(std::find_if(parameters.begin(),
                       parameters.end(),
                       [](const ParameterPtr& para) {
                         return para->getConfig().decay_rate_l1() > 0.0f;
                       }) == parameters.end())
        << "SgdLocalUpdater cannot support L1 decay in parameter";
  }

  /**
   * @brief Start a batch with current mini-batch size
   * @param current mini-batch size.
   * @return Always PASS_TRAIN.
   */
  virtual PassType startBatch(int64_t batchSize) {
    numSamplesProcessed_ += batchSize;
    optimizer_->startBatch(numSamplesProcessed_);
    return PASS_TRAIN;
  }

  /**
   * @brief finish a mini-batch.
   */
  virtual void finishBatch(real cost) { optimizer_->finishBatch(); }

  /**
   * @brief start a pass.
   */
  virtual void startPass() { optimizer_->startPass(); }

  /**
   * @brief finish a pass.
   * @param cost sum cost during one pass.
   * @return true if accept (used for owlqn).
   */
  virtual bool finishPass() {
    optimizer_->finishPass();
    return ParameterUpdater::finishPass();
  }

  /**
   * @brief apply model average.
   */
  virtual void apply() {
    if (auto callback = optimizer_->apply()) {
      for (auto para : parameters_) {
        SetDevice device(para->getDeviceId());
        callback(para->getBufs(), para->getConfig(), -1UL);
      }
    }
  }

  /**
   * @brief restore parameter value before model average
   */
  virtual void restore() {
    if (auto callback = optimizer_->restore()) {
      for (auto para : parameters_) {
        SetDevice device(para->getDeviceId());
        callback(para->getBufs(), para->getConfig(), -1UL);
      }
    }
  }

 protected:
  /**
   * @brief update method. Update value from gradient.
   * @param para parameter that will be updated.
   */
  virtual void updateImpl(Parameter* para) {
    optimizer_->update(para->getBufs(), para->getConfig());
    if (auto callback = optimizer_->needSpecialTraversal(para->getConfig())) {
      callback(para->getBufs(), para->getConfig(), -1UL);
    }

    para->setValueUpdated();
    para->getBuf(PARAMETER_GRADIENT)->zeroMem();
  }

  std::unique_ptr<ParameterOptimizer> optimizer_;

  /**
   * @brief total number of samples processed.
   */
  int64_t numSamplesProcessed_;
};

/**
 * @brief SgdCpuUpdater is used only in recursive neural network
 * @deprecated
 */
class SgdCpuUpdater : public SgdLocalUpdater, public Deprecated {
 public:
  explicit SgdCpuUpdater(const OptimizationConfig& optConfig)
      : SgdLocalUpdater(optConfig),
        Deprecated(
            "SgdCpuUpdater is used only in recursive neural network, "
            "and recursive neural network is deprecated in paddle. "
            "Use it all by your own.") {}

  /**
   * @brief update all parameter on finish batch.
   * @param cost
   */
  virtual void finishBatch(real cost) {
    for (auto para : parameters_) {
      SgdLocalUpdater::update(para.get());
    }
    optimizer_->finishBatch();
  }

 protected:
  /**
   * @brief do nothing.
   * @param para
   */
  virtual void updateImpl(Parameter* para) {}
};

/**
 * @brief Sgd Local Updater With average in cpu.
 *
 * It will do model average in cpu to reduce gpu memory comsuption.
 */
class SgdUpdaterWithCpuAverager : public SgdLocalUpdater {
 public:
  /**
   * @brief Ctor.
   *
   * SgdUpdaterWithCpuAverager will do everything as a
   * SgdLocalUpdater, then copy parameter from GPU to CPU, and do model
   * average in cpu.
   */
  explicit SgdUpdaterWithCpuAverager(const OptimizationConfig& optConfig);
  ~SgdUpdaterWithCpuAverager();

  /**
   * @brief init. Initialize cpu parameters, model average optimizer.
   * @param parameters
   */
  virtual void init(const std::vector<ParameterPtr>& parameters);

  virtual PassType startBatch(int64_t batchSize) {
    averager_->startBatch(-1UL);
    return SgdLocalUpdater::startBatch(batchSize);
  }
  virtual void finishBatch(real cost);

  virtual void startPass() {
    averager_->startPass();
    SgdLocalUpdater::startPass();
  }
  virtual bool finishPass() {
    averager_->finishPass();
    return SgdLocalUpdater::finishPass();
  }

  /// apply the averaged parameter to PARAMETER_VALUE
  /// use PARAETER_GRADIENT for backing up PARAMETER_VALUE
  virtual void apply();

  /**
   * @brief Restore parameter before apply().
   */
  virtual void restore();

 protected:
  virtual void updateImpl(Parameter* para);

  void updateFunc(Parameter* para);

 protected:
  std::unique_ptr<ParameterOptimizer> averager_;

  /**
   * @brief The thread worker which do model average.
   *
   * For each parameter, GPU->CPU parameter is async, and do model average in
   * another thread. Because the training process don't need model average while
   * training, and model average only used in evaluation stage and saving stage.
   * So the model average is totally async.
   */
  ThreadWorker updateWorker_;

  /**
   * @brief The parameter mirror in cpu.
   */
  std::vector<ParameterPtr> cpuParameters_;

  /**
   * @brief GPU -> CPU copy event. Model average will wait after copy done.
   */
  std::vector<hl_event_t> copyEvents_;
};

}  // namespace paddle
