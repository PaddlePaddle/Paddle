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

#include "AverageOptimizer.h"

namespace paddle {

// factory method to create an instance of AverageOptimizer
ParameterOptimizer* AverageOptimizer::create(
    const OptimizationConfig& optConfig,
    ParameterOptimizer* optimizer,
    bool isParameterSparse,
    bool useParameterApply) {
  if (optConfig.average_window() <= 0) {
    return optimizer;
  }
  // disable average for embeded local updater
  if (!useParameterApply && optConfig.num_batches_per_send_parameter() > 1) {
    return optimizer;
  }
  if (isParameterSparse) {
    return new AverageSparseOptimizer(optConfig, optimizer, useParameterApply);
  }
  return new AverageOptimizer(optConfig, optimizer, useParameterApply);
}

AverageOptimizer::AverageOptimizer(const OptimizationConfig& optConfig,
                                   ParameterOptimizer* optimizer,
                                   bool useParameterApply)
    : ParameterOptimizer(optConfig),
      optimizer_(optimizer),
      useApply_(useParameterApply),
      numUpdates_(0),
      prevNumUpdates_(0),
      numAccumulates_(0),
      oldNumAccumulates_(0),
      minAverageWindow_(
          std::min<int64_t>(10000L, optConfig_.max_average_window())),
      maxAverageWindow_(optConfig_.max_average_window()) {
  parameterTypes_ = optimizer_->getParameterTypes();
  addParameterType(PARAMETER_SUM1);
  addParameterType(PARAMETER_SUM2);
  addParameterType(PARAMETER_SUM3);
  if (useParameterApply) {
    addParameterType(PARAMETER_APPLY);
  }
}

void AverageOptimizer::startBatch(int64_t numSamplesProcessed) {
  optimizer_->startBatch(numSamplesProcessed);
  learningRate_ = optimizer_->getLearningRate();

  ++numUpdates_;
  ++numAccumulates_;
}

/*
  After traversal, the averaged parameter can be obtained by
  ((PARAMETER_SUM1 + PARAMETER_SUM2 + PARAMETER_SUM3)
  / (numAccumulates_ + oldNumAccumulates_))
*/
ParameterOptimizer::TraverseCallback AverageOptimizer::needSpecialTraversal(
    const ParameterConfig& config) const {
  TraverseCallbackVec callbacks;

  if (auto callback = optimizer_->needSpecialTraversal(config)) {
    callbacks.emplace_back(callback);
  }

  if (numUpdates_ % kMaxNumAccumulates == 0) {
    // Move the sum to a different buffer to avoid loss of precision
    // due to too many sums.
    callbacks.emplace_back([this](const VectorPtr vecs[],
                                  const ParameterConfig& config,
                                  size_t sparseId) {
      vecs[PARAMETER_SUM2]->add(*vecs[PARAMETER_SUM1]);
      vecs[PARAMETER_SUM1]->zeroMem();
    });
  }

  if (isAverageWindowTooLong()) {
    // Now the average window is too long, discard the old sum.
    if (auto callback = this->startCatchUpWith()) {
      callbacks.emplace_back(callback);
    }
    callbacks.emplace_back([this](const VectorPtr vecs[],
                                  const ParameterConfig& config,
                                  size_t sparseId) {
      vecs[PARAMETER_SUM3]->add(*vecs[PARAMETER_SUM1], *vecs[PARAMETER_SUM2]);
      vecs[PARAMETER_SUM1]->zeroMem();
      vecs[PARAMETER_SUM2]->zeroMem();
    });
  }

  return composeCallbacks(callbacks);
}

void AverageOptimizer::finishBatch() {
  optimizer_->finishBatch();
  if (isAverageWindowTooLong()) {
    this->finishCatchUpWith();
    oldNumAccumulates_ = numAccumulates_;
    numAccumulates_ = 0;
  }
}

ParameterOptimizer::TraverseCallback AverageOptimizer::apply() {
  if (numAccumulates_ + oldNumAccumulates_ == 0) {
    return nullptr;
  }

  real scale = 1. / (numAccumulates_ + oldNumAccumulates_);
  if (useApply_) {
    return [scale](const VectorPtr vecs[],
                   const ParameterConfig& config,
                   size_t sparseId) {
      vecs[PARAMETER_APPLY]->add3(*vecs[PARAMETER_SUM1],
                                  *vecs[PARAMETER_SUM2],
                                  *vecs[PARAMETER_SUM3],
                                  scale,
                                  scale,
                                  scale);
    };
  } else {
    return [scale](const VectorPtr vecs[],
                   const ParameterConfig& config,
                   size_t sparseId) {
      vecs[PARAMETER_GRADIENT]->copyFrom(*vecs[PARAMETER_VALUE]);
      vecs[PARAMETER_VALUE]->add3(*vecs[PARAMETER_SUM1],
                                  *vecs[PARAMETER_SUM2],
                                  *vecs[PARAMETER_SUM3],
                                  scale,
                                  scale,
                                  scale);
    };
  }
}

ParameterOptimizer::TraverseCallback AverageOptimizer::restore() {
  if (numAccumulates_ + oldNumAccumulates_ == 0) {
    return nullptr;
  }
  if (useApply_) {
    return nullptr;
  }

  return [](
      const VectorPtr vecs[], const ParameterConfig& config, size_t sparseId) {
    vecs[PARAMETER_VALUE]->copyFrom(*vecs[PARAMETER_GRADIENT]);
    vecs[PARAMETER_GRADIENT]->zeroMem();
  };
}

void AverageSparseOptimizer::update(const VectorPtr vecs[],
                                    const ParameterConfig& paraConfig,
                                    size_t sparseId) const {
  optimizer_->update(vecs, paraConfig, sparseId);

  CHECK_LT(sparseId, t0Vec_.size());
  int timediff = timer_ + 1 - t0Vec_[sparseId];
  if (timediff > 0) {
    vecs[PARAMETER_SUM1]->add(*vecs[PARAMETER_VALUE], timediff);
    t0Vec_[sparseId] = timer_ + 1;
  }
}

ParameterOptimizer::TraverseCallback AverageSparseOptimizer::startCatchUpWith()
    const {
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

void AverageSparseOptimizer::catchUpWith(const VectorPtr vecs[],
                                         const ParameterConfig& paraConfig,
                                         size_t sparseId) const {
  CHECK_LT(sparseId, t0Vec_.size());
  int timediff = timer_ - t0Vec_[sparseId];
  if (timediff > 0) {
    vecs[PARAMETER_SUM1]->add(*vecs[PARAMETER_VALUE], timediff);
  }
}

}  // namespace paddle
