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

#include "Tester.h"

#include <fenv.h>
#include <stdio.h>

#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>

#include <google/protobuf/text_format.h>

#include "paddle/utils/GlobalConstants.h"
#include "paddle/utils/PythonUtil.h"
#include "paddle/utils/Stat.h"
#include "paddle/utils/Util.h"

#include "TesterConfig.h"
#include "paddle/gserver/gradientmachines/GradientMachineMode.h"
#include "paddle/gserver/gradientmachines/NeuralNetwork.h"
#include "paddle/gserver/layers/ValidationLayer.h"

namespace paddle {

Tester::Tester(const std::shared_ptr<TrainerConfigHelper>& config,
               std::unique_ptr<TesterConfig>&& intconfig,
               const GradientMachinePtr& gradientMachine,
               const std::shared_ptr<ParameterUpdater>& parameterUpdater,
               std::shared_ptr<DataProvider> testDataProvider)
    : config_(config),
      intconfig_(std::move(intconfig)),
      gradientMachine_(gradientMachine),
      parameterUpdater_(parameterUpdater),
      testDataProvider_(testDataProvider) {
  if (config_->getOptConfig().use_sparse_remote_updater()) {
    LOG(FATAL) << "It's prohibited to set sparse_remote_update "
               << "when doing train and test jobs in the same "
               << "process. You could run paddle --job=test in "
               << "a separate process.";
  }
  testEvaluator_.reset(gradientMachine_->makeEvaluator());
  if (intconfig_->distributeTest) {
    testParameterClient_.reset(new ParameterClient2(true));
  }

  if (testParameterClient_) {
    testParameterClient_->init(gradientMachine_->getParameters());
  }

  std::unique_ptr<ParameterUtilConfig> paramConfig(
      new ParameterUtilConfig(intconfig_->saveOnlyOne,
                              intconfig_->savingPeriod,
                              intconfig_->loadsaveParametersInPserver,
                              intconfig_->config));

  paramUtil_.reset(new ParameterUtil(
      config_, std::move(paramConfig), gradientMachine_, parameterUpdater_));
}

void Tester::startTestPeriod() {
  if (testDataProvider_) {
    testDataProvider_->reset();
  }
  testEvaluator_->start();
  testContext_.cost = 0;
  testContext_.numSamples = 0;

  parameterUpdater_->apply();
  if (intconfig_->prevBatchState) {
    gradientMachine_->getState(*intconfig_->trainState);
    gradientMachine_->setState(*intconfig_->testState);
  }
}

void Tester::testOneDataBatch(const DataBatch& dataBatch,
                              std::vector<Argument>* outArgs) {
  testContext_.cost +=
      forwardOneBatch(dataBatch, testEvaluator_.get(), outArgs);
  testContext_.numSamples += dataBatch.getSize();
}

void Tester::testOnePeriod() {
  DataBatch dataBatch;
  int64_t batchSize = config_->getOptConfig().batch_size();
  std::vector<Argument> outArgs;
  startTestPeriod();
  while (testDataProvider_->getNextBatch(batchSize, &dataBatch) != 0) {
    testOneDataBatch(dataBatch, &outArgs);
  }
  finishTestPeriod();
}

void Tester::finishTestPeriod() {
  if (intconfig_->prevBatchState) {
    gradientMachine_->resetState();
  }
  testEvaluator_->finish();
  CHECK_GT(testContext_.numSamples, 0)
      << "There is no samples in your test batch. Possibly "
         "wrong implementation of DataProvidor.reset()";
  LOG(INFO) << " Test samples=" << testContext_.numSamples
            << " cost=" << testContext_.cost / testContext_.numSamples
            << " Eval: " << *testEvaluator_;
  parameterUpdater_->restore();
  if (intconfig_->prevBatchState) {
    gradientMachine_->getState(*intconfig_->testState);
    gradientMachine_->setState(*intconfig_->trainState);
  }
}

int64_t Tester::testOneBatchById(int64_t batchId) {
  DataBatch dataBatch;
  int32_t batchSize = config_->getOptConfig().batch_size();

  testDataProvider_->getNextBatch(batchSize, &dataBatch);

  int64_t actualBatchSize = dataBatch.getSize();
  if (actualBatchSize == 0) {
    return 0;
  }

  std::vector<Argument> outArgs;

  stats_ += std::pair<int64_t, real>{
      actualBatchSize,
      forwardOneBatch(dataBatch, testEvaluator_.get(), &outArgs)};

  if (((batchId + 1) % intconfig_->logPeriod) == 0) {
    LOG(INFO) << " Batch=" << batchId + 1 << " " << stats_.getStats(false);
  }

  return actualBatchSize;
}

real Tester::forwardOneBatch(const DataBatch& dataBatch,
                             Evaluator* evaluator,
                             std::vector<Argument>* pOutArgs) {
  auto& outArgs = *pOutArgs;
  const std::vector<Argument>& inArgs = dataBatch.getStreams();
  if (intconfig_->loadsaveParametersInPserver) {
    REGISTER_TIMER("prefetch");
    gradientMachine_->prefetch(inArgs);
    parameterUpdater_->getParametersRemote(false /*full parameter*/,
                                           true /*after apply*/);
  }

  gradientMachine_->forward(inArgs, &outArgs, PASS_TEST);

  // write features if set this flag and outArgs is not empty
  std::string featFile = intconfig_->featFile;
  if (!featFile.empty() && outArgs.empty()) {
    size_t numOutputs = outArgs.size();
    std::vector<MatrixPtr> featMatrices;
    featMatrices.resize(numOutputs);
    for (size_t i = 0; i < numOutputs; ++i) {
      featMatrices[i] = Matrix::create(outArgs[i].value->getHeight(),
                                       outArgs[i].value->getWidth(),
                                       false,
                                       false);  // CPU data buffer
      featMatrices[i]->copyFrom(*(outArgs[i].value), HPPL_STREAM_DEFAULT);
    }
    hl_stream_synchronize(HPPL_STREAM_DEFAULT);
    FILE* fp = fopen(featFile.c_str(), "ab+");
    CHECK(!ferror(fp)) << "Fail to open " << featFile;

    size_t sampleNum = featMatrices[0]->getHeight();
    for (size_t i = 0; i < sampleNum; ++i) {
      for (size_t j = 0; j < numOutputs; ++j) {
        size_t dim = featMatrices[j]->getWidth();
        fwrite(featMatrices[j]->getData() + i * dim, sizeof(real), dim, fp);
      }
    }
    fclose(fp);
  }
  if (evaluator) {
    gradientMachine_->eval(evaluator);
  }

  // Save the output layers if predict_output_dir is not empty
  std::string predictOutputDir = intconfig_->predictOutputDir;
  if (!predictOutputDir.empty() && !outArgs.empty()) {
    CHECK(intconfig_->testing) << "Only valid in test mode";
    if (!os_.is_open()) {
      // TODO(yuyang18): Refactor these lines.
      constexpr int kBufLen = 100;
      char buf[kBufLen];
      snprintf(buf, kBufLen, "rank-%05d", intconfig_->trainerId);
      mkDir(predictOutputDir.c_str());
      std::string filename = path::join(predictOutputDir, buf);
      os_.open(filename, std::ofstream::trunc);
      CHECK(os_.is_open()) << "Failed to open file " << filename;
    }
    printOutput(outArgs, os_);
    return 0.0;  // In this case, there is no meaning to calculate cost
  }

  return Argument::sum(outArgs);
}

void Tester::testOnePassBatch(int passId) {
  stats_.reset();
  const std::vector<Argument> inArgs;
  gradientMachine_->forward(inArgs, nullptr, PASS_TEST);
  int64_t num;
  real cost;
  gradientMachine_->getStats(cost, num);
  stats_ += std::pair<int64_t, real>{num, cost};
  gradientMachine_->onPassEnd();

  LOG(INFO) << " Pass=" << passId << " " << stats_.getStats(false);
}

void Tester::testOnePass(int passId) {
  stats_.reset();
  int64_t batchId = 0;
  int num = 0;
  if (intconfig_->prevBatchState) {
    gradientMachine_->resetState();
  }

  testEvaluator_->start();

  do {
    num = testOneBatchById(batchId);
    ++batchId;
  } while (num > 0);

  gradientMachine_->onPassEnd();
  testEvaluator_->finish();

  LOG(INFO) << " Pass=" << passId << " " << stats_.getStats(false)
            << " Eval: " << *testEvaluator_;

  if (intconfig_->distributeTest) {
    testEvaluator_->distributeEval(testParameterClient_.get());
    if (0 == intconfig_->trainerId) {
      LOG(INFO) << "distribute eval: " << *testEvaluator_;
    }
  }
}

void Tester::test() {
  CHECK(testDataProvider_) << "TestData is not specified";
  testDataProvider_->setSkipShuffle();
  testDataProvider_->reset();
  gradientMachine_->start();

  // For evaluation
  std::vector<std::string> modelList;
  std::string modelListFromConfig = intconfig_->modelList;
  std::string initModelPath = intconfig_->initModelPath;
  if (!modelListFromConfig.empty()) {
    loadFileList(modelListFromConfig, modelList);
    intconfig_->testPass = 0;
    intconfig_->numPasses = modelList.size();
    intconfig_->savingPeriod = 1;
    CHECK_EQ(intconfig_->testWait, 0) << "--test_wait must be 0 for evaluation";
  } else if (!initModelPath.empty()) {
    modelList.push_back(initModelPath);
    intconfig_->testPass = 0;
    intconfig_->numPasses = 1;
    intconfig_->savingPeriod = 1;
    CHECK_EQ(intconfig_->testWait, 0) << "--test_wait must be 0 for evaluation";
  }

  for (int i = intconfig_->testPass; i < intconfig_->numPasses; ++i) {
    int passId = i;
    if (passId % intconfig_->savingPeriod == 0) {
      if (intconfig_->testWait) {
        while (paramUtil_->loadParameters(
                   passId, true /*local*/, true /*remote*/) == false) {
          LOG(INFO) << "Waiting for parameters of pass " << passId;
          sleep(60);  // sleep 60s
        }
      } else {
        if (modelList.size() == 0) {
          CHECK_EQ(paramUtil_->loadParameters(
                       passId, true /*local*/, true /*remote*/),
                   true);
        } else {
          paramUtil_->loadParametersWithPath(
              modelList[i], true /*local*/, true /*remote*/);
        }
      }
      if (IGradientMachineMode::trainWholeDataInOneBatch(intconfig_->mode)) {
        testOnePassBatch(passId);
      } else {
        testOnePass(passId);
      }
      if (passId + intconfig_->savingPeriod < intconfig_->numPasses) {
        // if there is at least 1 more pass to test, then call reset,
        // otherwise not.
        testDataProvider_->reset();
      }
    }
  }

  gradientMachine_->finish();
}

void Tester::printOutput(const std::vector<Argument>& outArgs,
                         std::ostream& os) {
  size_t numOutputs = outArgs.size();
  size_t numIns = outArgs[0].getBatchSize();
  if (cpuMat_.size() != numOutputs || cpuVec_.size() != numOutputs) {
    cpuMat_.resize(numOutputs, nullptr);
    cpuVec_.resize(numOutputs, nullptr);
  }

  for (size_t i = 0; i < numOutputs; ++i) {
    if (outArgs[i].value != nullptr) {
      if (outArgs[i].value->useGpu()) {
        if (dynamic_cast<GpuMatrix*>(outArgs[i].value.get())) {
          size_t dim = outArgs[i].value->getWidth();
          Matrix::resizeOrCreate(cpuMat_[i], numIns, dim, false, false);
          cpuMat_[i]->copyFrom(*outArgs[i].value);
        } else if (dynamic_cast<GpuSparseMatrix*>(outArgs[i].value.get())) {
          auto sparseMat =
              dynamic_cast<GpuSparseMatrix*>(outArgs[i].value.get());
          cpuMat_[i] = Matrix::createSparseMatrix(sparseMat->getHeight(),
                                                  sparseMat->getWidth(),
                                                  sparseMat->getElementCnt(),
                                                  sparseMat->getValueType(),
                                                  sparseMat->format_,
                                                  false,  /* trans */
                                                  false); /* useGpu */
          hl_stream_t stream = HPPL_STREAM_DEFAULT;
          cpuMat_[i]->copyFrom(*sparseMat, stream);
        } else {
          LOG(WARNING) << "Not supported gpu matrix type";
        }
      }
    } else if (outArgs[i].ids != nullptr) {
      if (outArgs[i].ids->useGpu()) {
        IVector::resizeOrCreate(cpuVec_[i], outArgs[i].ids->getSize(), false);
        cpuVec_[i]->copyFrom(*outArgs[i].ids);
      }
    } else if (outArgs[i].strs != nullptr) {
      continue;
    } else {
      LOG(WARNING) << "outArgs[" << i << "] has no data to print";
    }
  }

  for (size_t i = 0; i < numIns; ++i) {
    for (size_t j = 0; j < numOutputs; ++j) {
      if (outArgs[j].value != nullptr) {
        if (outArgs[j].value->useGpu()) {
          cpuMat_[j]->printOneRow(os, i);
        } else {
          outArgs[j].value->printOneRow(os, i);
        }
      } else if (outArgs[j].ids != nullptr) {
        if (outArgs[j].ids->useGpu()) {
          cpuVec_[j]->printOneElement(os, i);
        } else {
          outArgs[j].ids->printOneElement(os, i);
        }
      } else if (outArgs[j].strs != nullptr) {
        os << (*outArgs[j].strs)[i] << ";";
      }
    }
    os << std::endl;
  }
}
}  // namespace paddle
