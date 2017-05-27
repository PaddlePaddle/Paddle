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

#include "ThreadParameterUpdater.h"

#include "paddle/utils/Logging.h"

#include "paddle/math/SparseRowMatrix.h"
#include "paddle/parameter/ThreadLocalBuffer.h"
#include "paddle/utils/Thread.h"

DECLARE_int32(trainer_count);

namespace paddle {

SgdThreadUpdater::SgdThreadUpdater(const OptimizationConfig& optConfig)
    : config_(optConfig), numSamplesProcessed_(0) {
  // fill types
  auto types = sgdOptimizerGetTypes(optConfig, false /*inPserver*/);
  for (auto type : types) {
    addParameterType(type);
  }
}

void SgdThreadUpdater::init(const std::vector<ParameterPtr>& parameters) {
  ParameterUpdater::init(parameters);

  // calc max parameter id
  size_t maxId = 0;
  for (auto& para : parameters_) {
    maxId = std::max(maxId, para->getID());
  }

  optimizers_.resize(maxId + 1);
  for (auto& para : parameters_) {
    int pid = para->getID();
    optimizers_[pid].reset(sgdOptimizerCreate(config_,
                                              para->getConfig(),
                                              para->isGradSparseUpdate(),
                                              false /*inPserver*/));
    size_t numRows = para->isGradSparseUpdate() ? para->getConfig().dims(0) : 0;
    optimizers_[pid]->init(numRows, &para->getConfig());
    if (para->isGradSparseUpdate() && FLAGS_trainer_count == 1) {
      // For trainer_count=1, the gradient machine is NeuralNetwork, which does
      // not create parameter buf for PARAMETER_GRADIENT for sparse update in
      // Parameter::enableType(). But gradient parameter buf is still used
      // in SgdThreadUpdater. We need to explicitly create it.
      //
      // The AverageOptimizer::restore/apply method will use PARAMETER_GRADIENT
      // as a temp buffer.
      para->enableBufType(PARAMETER_GRADIENT);
    }
  }
}

void SgdThreadUpdater::startPass() {
  for (auto& para : parameters_) {
    int pid = para->getID();
    optimizers_[pid]->startPass();
  }
}

bool SgdThreadUpdater::finishPass() {
  catchUpWith();

  for (auto& para : parameters_) {
    int pid = para->getID();
    optimizers_[pid]->finishPass();
  }
  return true;
}

void SgdThreadUpdater::updateImpl(Parameter* para) {
  if (!para->useGpu()) return;
  SetDevice setDevice(para->getDeviceId());
  ParameterOptimizer* optimizer = optimizers_[para->getID()].get();
  optimizer->update(para->getBufs(), para->getConfig());
  if (auto callback = optimizer->needSpecialTraversal(para->getConfig())) {
    callback(para->getBufs(), para->getConfig(), -1LU);
  }

  para->setValueUpdated();
  para->clearGradient();
}

void SgdThreadUpdater::threadTraverse(
    const ParameterOptimizer::TraverseCallback& callback,
    int tid,
    size_t numThreads,
    Parameter* para) {
  VectorPtr* vecs = parameter::getThreadLocalBuffer();
  if (para->isGradSparseUpdate()) {
    size_t height = para->getConfig().dims(0);
    size_t width = para->getConfig().dims(1);
    for (size_t i = tid; i < height; i += numThreads) {
      // setup sub bufs
      for (auto type : parameterTypes_) {
        vecs[type]->subVecFrom(*para->getBuf(type), i * width, width);
      }
      callback(vecs, para->getConfig(), i);
    }
  } else {  // dense
    // setup sub bufs
    auto interval = calcSplitArrayInterval(
        para->getSize(), (size_t)tid, numThreads, 8LU /*for avx*/);
    for (auto type : parameterTypes_) {
      vecs[type]->subVecFrom(*para->getBuf(type), interval);
    }

    callback(vecs, para->getConfig(), -1LU);
  }
}

void SgdThreadUpdater::traverse(GetTraverseCallback getTraverseCallback) {
  bool hasCpuPara = false;
  bool hasGpuPara = false;
  for (auto& para : parameters_) {
    if (para->useGpu()) {
      hasGpuPara = true;
    } else {
      hasCpuPara = true;
    }
  }

  auto cpuTraverse = [&](int tid, size_t numThreads) {
    for (auto& para : parameters_) {
      if (auto callback = getTraverseCallback(para.get())) {
        threadTraverse(callback, tid, numThreads, para.get());
      }
    }
  };
  auto gpuTraverse = [&](int tid, size_t numThreads) {
    for (auto& para : parameters_) {
      if (para->useGpu()) {
        if (auto callback = getTraverseCallback(para.get())) {
          SetDevice setDevice(para->getDeviceId());
          callback(para->getBufs(), para->getConfig(), -1LU);
        }
      }
    }
  };

  if (hasCpuPara && hasGpuPara) {
    getGlobalSyncThreadPool()->exec(cpuTraverse, gpuTraverse);
  } else if (hasCpuPara) {
    getGlobalSyncThreadPool()->exec(cpuTraverse);
  } else if (hasGpuPara) {
    gpuTraverse(0, 0);
  }
}

void SgdThreadUpdater::catchUpWith() {
  traverse([this](Parameter* para) {
    return optimizers_[para->getID()]->startCatchUpWith();
  });

  for (auto& para : parameters_) {
    int pid = para->getID();
    optimizers_[pid]->finishCatchUpWith();
  }
}

void SgdThreadUpdater::apply() {
  catchUpWith();

  traverse(
      [this](Parameter* para) { return optimizers_[para->getID()]->apply(); });
}

void SgdThreadUpdater::restore() {
  traverse([this](Parameter* para) {
    return optimizers_[para->getID()]->restore();
  });
}

PassType SgdThreadUpdater::startBatch(int64_t batchSize) {
  numSamplesProcessed_ += batchSize;
  for (auto& para : parameters_) {
    int pid = para->getID();
    optimizers_[pid]->startBatch(numSamplesProcessed_);
  }
  return PASS_TRAIN;
}

void SgdThreadUpdater::finishBatch(real cost) {
  getGlobalSyncThreadPool()->exec([&](int tid, size_t numThreads) {
    for (auto& para : parameters_) {
      if (para->isGradSparseUpdate()) {
        threadUpdateSparse(tid, numThreads, para.get());
      } else if (!para->useGpu()) {
        threadUpdateDense(tid, numThreads, para.get());
      }
    }
  });

  for (auto& para : parameters_) {
    int pid = para->getID();
    optimizers_[pid]->finishBatch();
  }
}

void SgdThreadUpdater::threadUpdateSparse(int tid,
                                          size_t numThreads,
                                          Parameter* para) {
  int pid = para->getID();
  ParameterOptimizer* optimizer = optimizers_[pid].get();
  VectorPtr* vecs = parameter::getThreadLocalBuffer();

  size_t height = para->getConfig().dims(0);
  size_t width = para->getConfig().dims(1);

  if (dynamic_cast<SparseRowIdsCpuMatrix*>(
          para->getMat(PARAMETER_GRADIENT).get())) {
    // From MultiGradientMachine
    SparseRowIdsCpuMatrix* mainMat = dynamic_cast<SparseRowIdsCpuMatrix*>(
        para->getMat(PARAMETER_GRADIENT).get());
    std::vector<uint32_t>& sparseIds = mainMat->getIds(tid);

    for (auto id : sparseIds) {
      // setup sub bufs
      for (auto type : parameterTypes_) {
        vecs[type]->subVecFrom(*para->getBuf(type), id * width, width);
      }
      optimizer->update(vecs, para->getConfig(), id);
      vecs[PARAMETER_GRADIENT]->zeroMem();
    }
    sparseIds.clear();
  } else if (dynamic_cast<SparseRowCpuMatrix*>(
                 para->getMat(PARAMETER_GRADIENT).get())) {
    // From NeuralNetwork
    SparseRowCpuMatrix* mainMat = dynamic_cast<SparseRowCpuMatrix*>(
        para->getMat(PARAMETER_GRADIENT).get());

    std::vector<unsigned int>& localIndices =
        mainMat->getIndexDictHandle()->localIndices;

    auto interval =
        calcSplitArrayInterval(localIndices.size(), tid, numThreads);
    for (size_t i = interval.first; i < interval.second; ++i) {
      auto id = localIndices[i];
      real* row = mainMat->getLocalRow(i);
      // setup sub bufs
      for (auto type : parameterTypes_) {
        if (type == PARAMETER_GRADIENT) {
          vecs[type]->subVecFrom(row, 0, width);
        } else {
          vecs[type]->subVecFrom(*para->getBuf(type), id * width, width);
        }
      }
      optimizer->update(vecs, para->getConfig(), id);
      vecs[PARAMETER_GRADIENT]->zeroMem();
    }
    // For numThreads > 1, MultiGradientMachine is used, which goes
    // to the above branch.
    CHECK_EQ(numThreads, 1UL);
    mainMat->clearIndices();
  } else {
    auto& m = *para->getMat(PARAMETER_GRADIENT).get();
    LOG(FATAL) << "Internal error: " << para->getName() << " "
               << typeid(m).name();
  }

  if (auto callback = optimizer->needSpecialTraversal(para->getConfig())) {
    for (size_t i = tid; i < height; i += numThreads) {
      // setup sub bufs
      for (auto type : parameterTypes_) {
        vecs[type]->subVecFrom(*para->getBuf(type), i * width, width);
      }
      callback(vecs, para->getConfig(), i);
    }
  }
}

void SgdThreadUpdater::threadUpdateDense(int tid,
                                         size_t numThreads,
                                         Parameter* para) {
  int pid = para->getID();
  ParameterOptimizer* optimizer = optimizers_[pid].get();
  VectorPtr* vecs = parameter::getThreadLocalBuffer();

  auto interval = calcSplitArrayInterval(
      para->getSize(), (size_t)tid, numThreads, 8LU /*for avx*/);

  // setup sub bufs
  for (auto type : parameterTypes_) {
    vecs[type]->subVecFrom(*para->getBuf(type), interval);
  }

  // update
  optimizer->update(vecs, para->getConfig());
  vecs[PARAMETER_GRADIENT]->zeroMem();

  if (auto callback = optimizer->needSpecialTraversal(para->getConfig())) {
    callback(vecs, para->getConfig(), -1LU);
  }
}

}  // namespace paddle
