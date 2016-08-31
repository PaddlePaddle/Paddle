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

#include "PaddleAPI.h"

#include <stdlib.h>
#include <memory>
#include <atomic>

#include "paddle/trainer/ParamUtil.h"
#include "paddle/trainer/Trainer.h"
#include "paddle/gserver/gradientmachines/NeuralNetwork.h"
#include "paddle/trainer/TrainerInternal.h"
#include "paddle/utils/Flags.h"

using paddle::real;

P_DECLARE_string(config);
P_DECLARE_string(init_model_path);
P_DECLARE_int32(start_pass);

struct TrainPassContext {
  int64_t batchId;
  int32_t batchSize;
  real avgTestCost;
  int64_t numAvgTests;
  int passInnerId;
  paddle::DataBatch data;
  std::vector<paddle::Argument> forwardOutput;
};

struct TrainerPrivate : public paddle::Trainer {
  void startTrain();
  void finishTrain();

  void startTrainPass();
  void finishTrainPass();

  bool _trainOneBatch();

  bool _prepareBatchData();
  void _forwardOneBatch() throw(UnsupportError);

  TrainerPrivate() : paddle::Trainer() {}

  TrainPassContext trainPassContext;
};

Trainer::Trainer() : m(new TrainerPrivate()) {
  auto conf = paddle::TrainerConfigHelper::createFromFlags();
  if (conf != nullptr) {
    m->init(conf);
  }
}

Trainer::~Trainer() { delete m; }

Trainer* Trainer::createByCommandLine() throw(IOError) {
  auto retv = new Trainer();
  if (retv->m->getConfig().IsInitialized()) {
    return retv;
  } else {
    throw IOError();
  }
}

void Trainer::startTrain() { m->startTrain(); }

void TrainerPrivate::startTrain() {
  srand(this->config_->getConfig().start_pass() + 1);
  this->dataProvider_->reset();
  this->trainerInternal_.getGradientMachine()->start(*config_, dataProvider_);
}

void Trainer::finishTrain() { m->finishTrain(); }

void TrainerPrivate::finishTrain() {
  this->trainerInternal_.getGradientMachine()->finish();
}

void Trainer::startTrainPass() { m->startTrainPass(); }

void TrainerPrivate::startTrainPass() {
  this->stats_.reset();
  this->trainPassContext.batchId = 0;
  this->trainPassContext.batchSize = this->config_->getOptConfig().batch_size();
  this->trainPassContext.avgTestCost = 0;
  this->trainPassContext.numAvgTests = 0;
  this->trainPassContext.passInnerId = 0;
  this->trainerInternal_.getParameterUpdater()->startPass();
  this->evaluator_->start();
}

void Trainer::finishTrainPass() { m->finishTrainPass(); }

void TrainerPrivate::finishTrainPass() {
  this->trainerInternal_.getGradientMachine()->onPassEnd();
  this->trainerInternal_.getParameterUpdater()->finishPass();
  evaluator_->finish();
}

void Trainer::setBatchSize(size_t batchSize) {
  this->m->trainPassContext.batchSize = batchSize;
}

bool Trainer::trainOneBatch(size_t batchSize) {
  if (batchSize == -1UL) {
    this->setBatchSize(batchSize);
  }
  return m->_trainOneBatch();
}

bool TrainerPrivate::_trainOneBatch() {
  if (this->_prepareBatchData()) {
    return true;
  }
  this->trainerInternal_.trainOneBatch(this->trainPassContext.batchId,
                                       this->trainPassContext.data);
  return false;
}

Matrix* Trainer::getLayerOutput(const std::string& layerName) {
  auto nn = std::dynamic_pointer_cast<paddle::NeuralNetwork>(
          this->m->getGradientMachine());
  CHECK(nn) << "trainerInternal_.getGradientMachine() is not NeuralNetwork";
  auto m = nn->getLayerOutput(layerName);
  return Matrix::createByPaddleMatrixPtr(&m);
}

bool Trainer::prepareBatchData(size_t batchSize) {
  if (batchSize != -1UL) {
    this->setBatchSize(batchSize);
  }
  return this->m->_prepareBatchData();
}

bool TrainerPrivate::_prepareBatchData() {
  int num = dataProvider_->getNextBatch(this->trainPassContext.batchSize,
                                        &this->trainPassContext.data);
  return num == 0;
}

void Trainer::finishTrainOneBatch() { ++m->trainPassContext.batchId; }

void Trainer::forwardOneBatch() throw(UnsupportError) { m->_forwardOneBatch(); }

void TrainerPrivate::_forwardOneBatch() throw(UnsupportError) {
  auto& dataBatch = this->trainPassContext.data;

  int64_t actualBatchSize = dataBatch.getSize();
  if (actualBatchSize == 0) {
    return;
  }

  const std::vector<paddle::Argument>& inArgs = dataBatch.getStreams();
  std::vector<paddle::Argument>& outArgs = this->trainPassContext.forwardOutput;
  outArgs.clear();
  paddle::PassType passType =
      this->trainerInternal_.getParameterUpdater()->startBatch(actualBatchSize);

  if (config_->getOptConfig().use_sparse_remote_updater()) {
    this->trainerInternal_.getGradientMachine()->prefetch(inArgs);
    this->trainerInternal_.getParameterUpdater()->getParametersRemote();
  }
  this->trainerInternal_.getGradientMachine()->forward(
        inArgs, &outArgs, passType);
}

Arguments* Trainer::getNetworkOutput() {
  return Arguments::createByPaddleArgumentVector(
      &m->trainPassContext.forwardOutput);
}
