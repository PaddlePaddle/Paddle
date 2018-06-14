/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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
#include "PaddleAPIPrivate.h"

#include <stdlib.h>
#include <atomic>
#include <memory>

#include "paddle/gserver/gradientmachines/NeuralNetwork.h"
#include "paddle/trainer/ParamUtil.h"
#include "paddle/trainer/Trainer.h"
#include "paddle/trainer/TrainerInternal.h"
#include "paddle/utils/Flags.h"

using paddle::real;

DECLARE_string(config);
DECLARE_string(init_model_path);
DECLARE_int32(start_pass);

struct TrainerPrivate : public paddle::Trainer {
  bool _trainOneBatch(size_t batchSize);
  bool forwardOneBatch(size_t batchSize);
  void forwardOneDataBatch(const std::vector<paddle::Argument>& inArgs);
  void setBatchSize(size_t batchSize);
  std::vector<paddle::Argument>& getForwardOutput();

  void startTestPeriod();
  void finishTestPeriod();
  void testOneDataBatch(const paddle::DataBatch& dataBatch);
  TrainerPrivate() : paddle::Trainer() {}
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

Trainer::Trainer(TrainerConfig* config, GradientMachine* gm)
    : m(new TrainerPrivate()) {
  m->init(config->m->conf, /* testing= */ false, gm ? gm->m->machine : nullptr);
}

Trainer* Trainer::create(TrainerConfig* config,
                         GradientMachine* gm) throw(IOError) {
  auto retv = new Trainer(config, gm);
  if (retv->m->getConfig().IsInitialized()) {
    return retv;
  } else {
    retv->m->getConfig().CheckInitialized();
    throw IOError();
  }
}

void Trainer::startTrain() { m->startTrain(); }

void Trainer::finishTrain() { m->finishTrain(); }

void Trainer::startTrainPass() { m->startTrainPass(); }

void Trainer::finishTrainPass() { m->finishTrainPass(); }

void Trainer::trainOneDataBatch(size_t batchSize, const Arguments& inArgs) {
  paddle::DataBatch dataBatch;
  dataBatch.getStreams() = inArgs.m->outputs;
  dataBatch.setSize(batchSize);
  m->trainOneDataBatch(dataBatch);
}

bool Trainer::trainOneBatch(size_t batchSize) {
  return m->_trainOneBatch(batchSize);
}

bool TrainerPrivate::_trainOneBatch(size_t batchSize) {
  paddle::DataBatch dataBatch;
  CHECK(dataProvider_) << "data_provider is not specified";
  int num = dataProvider_->getNextBatch(batchSize, &dataBatch);
  if (num == 0) {
    return false;
  }
  trainOneDataBatch(dataBatch);
  return false;
}

void TrainerPrivate::startTestPeriod() {
  if (!tester_) {
    createTester();
  }
  tester_->startTestPeriod();
}

void Trainer::startTestPeriod() { m->startTestPeriod(); }

void TrainerPrivate::testOneDataBatch(const paddle::DataBatch& dataBatch) {
  tester_->testOneDataBatch(dataBatch, &forwardOutput_);
}

void Trainer::testOneDataBatch(size_t batchSize, const Arguments& args) {
  paddle::DataBatch dataBatch;
  dataBatch.getStreams() = args.m->outputs;
  dataBatch.setSize(batchSize);
  m->testOneDataBatch(dataBatch);
}

void TrainerPrivate::finishTestPeriod() { tester_->finishTestPeriod(); }
void Trainer::finishTestPeriod() { m->finishTestPeriod(); }

Arguments* Trainer::getLayerOutput(const std::string& layerName) const {
  auto nn = this->m->getGradientMachine();
  CHECK(nn) << "trainerInternal_.getGradientMachine() is not NeuralNetwork";
  auto arg = nn->getLayerOutput(layerName);
  return Arguments::createByPaddleArgument(&arg);
}

void Trainer::forwardOneBatch(size_t batchSize) {
  m->forwardOneBatch(batchSize);
}

bool TrainerPrivate::forwardOneBatch(size_t batchSize) {
  CHECK(dataProvider_) << "data_provider is not specified";
  paddle::DataBatch dataBatch;
  int num = dataProvider_->getNextBatch(batchSize, &dataBatch);
  if (num == 0) {
    return false;
  }

  forwardOneDataBatch(dataBatch.getStreams());
  return true;
}

void TrainerPrivate::forwardOneDataBatch(
    const std::vector<paddle::Argument>& inArgs) {
  std::vector<paddle::Argument>& outArgs = forwardOutput_;

  if (config_->getOptConfig().use_sparse_remote_updater()) {
    trainerInternal_.getGradientMachine()->prefetch(inArgs);
    trainerInternal_.getParameterUpdater()->getParametersRemote();
  }
  trainerInternal_.getGradientMachine()->forward(
      inArgs, &outArgs, paddle::PASS_TEST);
}

Arguments* Trainer::getForwardOutput() {
  return Arguments::createByPaddleArgumentVector(&m->getForwardOutput());
}

std::vector<paddle::Argument>& TrainerPrivate::getForwardOutput() {
  return forwardOutput_;
}
