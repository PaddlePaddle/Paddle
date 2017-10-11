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

#include "Parameter.h"

namespace paddle {

class ParameterOptimizer;

class ParameterUpdater {
public:
  ParameterUpdater() : parameterTypes_{PARAMETER_VALUE, PARAMETER_GRADIENT} {}
  virtual ~ParameterUpdater() {}

  void addParameterType(ParameterType type) {
    for (auto t : parameterTypes_) {
      if (t == type) return;
    }
    parameterTypes_.push_back(type);
  }

  virtual void init(const std::vector<ParameterPtr>& parameters);

  // called by Trainer when starting a new pass
  virtual void startPass() {}

  // called by Trainer then finishing a pass, ruturn true if pass accepted
  virtual bool finishPass() { return true; }

  // called by Trainer before backward() of a batch
  // Return the type of pass it needs. This pass type will be passed
  // to GradientMachine::forward() by the caller.
  virtual PassType startBatch(int64_t batchSize) {
    (void)batchSize;
    return PASS_TRAIN;
  }

  // called by Trainer after backward() of a batch
  // cost: the cost for this batch
  virtual void finishBatch(real cost) { (void)cost; }

  // between startBatch() and finishBatch(), update() will be called
  // by the trainer multiple times, each time for updating one Parameter
  // with its gradient in PARAMETER_GRADIENT
  void update(Parameter* para) {
    SetDevice setDevice(para->getDeviceId());
    para->updateHook();
    this->updateImpl(para);
  }

  // only get required sparse rows by default,
  // get full matrix parameter if *fullSize* set
  // get PARAMETER_APPLY on pserver if *apply* set
  virtual void getParametersRemote(bool fullSize = false, bool apply = false) {}

  virtual void loadParametersRemote(const std::string& dirName) {}
  virtual void saveParametersRemote(const std::string& dirName) {}
  virtual void randParametersRemote() {}

  // something like regularization may be delayed apply
  // trainer should catch up with before parameter is saved or sended.
  virtual void catchUpWith() {}

  // following two hooks used by averager
  // apply to final parameter value (PARAMETER_VALUE or PARAMETER_APPLY).
  // restore() will restore orginal value if it apply to PARAMETER_VALUE.
  virtual void apply() {}
  virtual void restore() {}

  // return the parameter types used by this updater
  const std::vector<ParameterType>& getParameterTypes() const {
    return parameterTypes_;
  }

#ifndef PADDLE_DISABLE_TIMER
  virtual void setForwardbackwardTime(uint64_t delta) {}
#endif

protected:
  virtual void updateImpl(Parameter* para) = 0;

  std::vector<ParameterType> parameterTypes_;
  std::vector<ParameterPtr> parameters_;
  std::map<size_t, size_t> nonStaticParaIDMap_;
};

// Composite of ParameterUpdaters, each ParameterUpdater handle
// part of all Parameters. It's useful when we need different
// update strategy for different Parameter.
class ParameterUpdaterComposite : public ParameterUpdater {
public:
  ParameterUpdaterComposite() {}
  virtual ~ParameterUpdaterComposite() {}

  virtual void init(const std::vector<ParameterPtr>& parameters) = 0;

  virtual void startPass() {
    syncThreadPool_->execPlusOwner(
        [&](int tid, size_t numThreads) { updaters_[tid]->startPass(); });
  }

  virtual bool finishPass() {
    syncThreadPool_->execPlusOwner(
        [&](int tid, size_t numThreads) { updaters_[tid]->finishPass(); });
    return true;
  }

  virtual PassType startBatch(int64_t batchSize) {
    syncThreadPool_->execPlusOwner([&](int tid, size_t numThreads) {
      updaters_[tid]->startBatch(batchSize);
    });
    return PASS_TRAIN;
  }

  virtual void finishBatch(real cost) {
    syncThreadPool_->execPlusOwner(
        [&](int tid, size_t numThreads) { updaters_[tid]->finishBatch(cost); });
  }

  virtual void getParametersRemote(bool fullSize, bool apply) {
    syncThreadPool_->execPlusOwner([&](int tid, size_t numThreads) {
      updaters_[tid]->getParametersRemote(fullSize, apply);
    });
  }
  virtual void loadParametersRemote(const std::string& dirName) {
    syncThreadPool_->execPlusOwner([&](int tid, size_t numThreads) {
      updaters_[tid]->loadParametersRemote(dirName);
    });
  }
  virtual void saveParametersRemote(const std::string& dirName) {
    syncThreadPool_->execPlusOwner([&](int tid, size_t numThreads) {
      updaters_[tid]->saveParametersRemote(dirName);
    });
  }
  virtual void randParametersRemote() {
    syncThreadPool_->execPlusOwner([&](int tid, size_t numThreads) {
      updaters_[tid]->randParametersRemote();
    });
  }

  virtual void catchUpWith() {
    syncThreadPool_->execPlusOwner(
        [&](int tid, size_t numThreads) { updaters_[tid]->catchUpWith(); });
  }

#ifndef PADDLE_DISABLE_TIMER
  virtual void setForwardbackwardTime(uint64_t delta) {
    for (auto& updater : updaters_) {
      updater->setForwardbackwardTime(delta);
    }
  }
#endif

  virtual void apply() {
    syncThreadPool_->execPlusOwner(
        [&](int tid, size_t numThreads) { updaters_[tid]->apply(); });
  }
  virtual void restore() {
    syncThreadPool_->execPlusOwner(
        [&](int tid, size_t numThreads) { updaters_[tid]->restore(); });
  }

protected:
  virtual void updateImpl(Parameter* para) {}
  std::vector<std::unique_ptr<ParameterUpdater>> updaters_;
  std::unique_ptr<SyncThreadPool> syncThreadPool_;
};

}  // namespace paddle
