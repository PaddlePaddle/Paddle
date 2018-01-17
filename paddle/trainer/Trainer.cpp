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

#include "Trainer.h"

#include <stdio.h>

#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>

#include <google/protobuf/text_format.h>

#include "paddle/utils/Common.h"
#include "paddle/utils/GlobalConstants.h"
#include "paddle/utils/PythonUtil.h"
#include "paddle/utils/Stat.h"
#include "paddle/utils/Util.h"

#include "RemoteParameterUpdater.h"
#include "TesterConfig.h"
#include "ThreadParameterUpdater.h"
#include "TrainerConfigHelper.h"
#include "paddle/gserver/gradientmachines/GradientMachineMode.h"
#include "paddle/gserver/gradientmachines/NeuralNetwork.h"
#include "paddle/gserver/layers/ValidationLayer.h"

DEFINE_string(config, "", "Trainer config file");

DEFINE_int32(test_period,
             0,
             "if equal 0, do test on all test data at the end of "
             "each pass. While if equal non-zero, do test on all test "
             "data every test_period batches");
DEFINE_bool(test_all_data_in_one_period,
            false,
            "This option was deprecated, since we will always do "
            "test on all test set ");

DEFINE_bool(local, true, "Train in local mode or not");

DEFINE_int32(average_test_period,
             0,
             "Do test on average parameter every so"
             " many batches. MUST be devided by FLAGS_log_period."
             " Default 0 means do not test average parameter");

DEFINE_int32(saving_period, 1, "Save parameteres every so many passes");
DEFINE_int64(saving_period_by_batches,
             0,
             "Save parameters every so many batches in one pass");
DEFINE_string(save_dir, "", "Directory for saving model parameter");
DEFINE_int32(start_pass,
             0,
             "Start training from this pass. "
             "Will load parameter from the previous pass");
DEFINE_int32(test_pass, -1, "Will load parameter start from this pass to test");
DEFINE_int32(test_wait, 0, "Waiting for pass parameter if not exist");
DEFINE_bool(with_cost, true, "enable cost layer or not");
DEFINE_bool(distribute_test, false, "test in distribute mode");

DEFINE_int32(num_passes, 100, "train for so many passes");

DEFINE_string(config_args,
              "",
              "arguments passed to config file."
              "Format: key1=value1,key2=value2");

DEFINE_bool(save_only_one,
            false,
            "Save only parameters in last pass, remove previous.");

DEFINE_string(feat_file, "", "File name of extracted feature.");
DEFINE_string(predict_output_dir,
              "",
              "Directory that saves the predicted results of output layers");
DEFINE_string(model_list, "", "File that saves the model list when evaluation");

namespace paddle {

void Trainer::init(const std::shared_ptr<TrainerConfigHelper>& config,
                   bool testing,
                   const std::shared_ptr<GradientMachine>& gradientMachine,
                   const std::shared_ptr<DataProvider>& dataProvider,
                   const std::shared_ptr<DataProvider>& testDataProvider) {
  this->stats_ = std::make_shared<TrainerStats>();

  config_ = config;

  config_->updateConfigFromFlags();

  testing_ = testing;

  // in testing, mode_ may GradientMachine::kTesting or
  // GradientMachine::kSgdSparseCpuTraining

  if (FLAGS_local) {
    CHECK(!FLAGS_loadsave_parameters_in_pserver)
        << "local and loadsave_parameters_in_pserver can not both true";
    if (config_->getOptConfig().use_sparse_remote_updater()) {
      config_->disableRemoteSparseUpdaterForEachParams();
      LOG(INFO) << "ignore sparse_remote_update=true due to  --local=true";
    }
  }
  if (FLAGS_loadsave_parameters_in_pserver) {
    CHECK(config_->getOptConfig().use_sparse_remote_updater())
        << "no parameter to load from pserver, please check network config";
  }
  if (testing && !FLAGS_loadsave_parameters_in_pserver) {
    if (config_->getOptConfig().use_sparse_remote_updater()) {
      config_->disableRemoteSparseUpdater();
      LOG(INFO) << "because parameter is loaded local,"
                << "tester ignore sparse_remote_update flag";
    }
  }

  CHECK(TrainAlgorithm::isValid(config_->getOptConfig().algorithm()))
      << "invalid algorithm configuration: "
      << config_->getOptConfig().algorithm();

  bool useSparseUpdater = false;
  for (auto& paraConfig : config_->getModelConfig().parameters()) {
    if (paraConfig.sparse_update() || paraConfig.sparse_remote_update()) {
      useSparseUpdater = true;
    }
  }

  if (FLAGS_use_mkldnn) {
    CHECK_EQ(FLAGS_trainer_count, 1) << "MKLDNN only need 1 trainer";
  }

  if (testing) {
    LOG(INFO) << "trainer: in testing mode";
    if (config_->getOptConfig().use_sparse_remote_updater() ||
        FLAGS_trainer_count > 1) {
      mode_ = GradientMachine::kSgdSparseCpuTraining;
      LOG(INFO) << "trainer mode: SgdSparseCpuTraining";
    } else {
      mode_ = GradientMachine::kTesting;
      LOG(INFO) << "trainer mode: Testing";
    }
  } else if (IGradientMachineMode::tryGetMode(
                 (int*)&mode_,
                 config_->getOptConfig().algorithm(),
                 FLAGS_trainer_count,
                 FLAGS_local,
                 FLAGS_use_gpu)) {
    LOG(INFO) << "Custom trainer mode.";
  } else if ((config_->getOptConfig().algorithm() == TrainAlgorithm::SGD ||
              config_->getOptConfig().algorithm() ==
                  TrainAlgorithm::AsyncSGD) &&
             useSparseUpdater) {
    mode_ = GradientMachine::kSgdSparseCpuTraining;
    LOG(INFO) << "trainer mode: SgdSparseCpuTraining";
  } else {
    mode_ = GradientMachine::kNormal;
    LOG(INFO) << "trainer mode: Normal";
  }

  // initialize trainer internal
  trainerInternal_.init(config_,
                        gradientMachine,
                        TrainerInternalConfig::createFromMode(mode_),
                        stats_,
                        testing);
  std::unique_ptr<ParameterUtilConfig> paramConfig(
      new ParameterUtilConfig(FLAGS_save_only_one,
                              FLAGS_saving_period,
                              FLAGS_loadsave_parameters_in_pserver,
                              FLAGS_config));

  paramUtil_.reset(
      new paddle::ParameterUtil(config_,
                                std::move(paramConfig),
                                trainerInternal_.getGradientMachine(),
                                trainerInternal_.getParameterUpdater()));

  bool gpuData =
      FLAGS_use_gpu && (!FLAGS_parallel_nn) &&
      (!IGradientMachineMode::dataMustInCpu(mode_, FLAGS_trainer_count));

  dataProvider_ = dataProvider;
  if (!dataProvider_ && config_->hasDataConfig() && !testing_) {
    dataProvider_.reset(DataProvider::create(*config_, *config_, gpuData));
  }
  if (!testDataProvider_) {
    // No evaluator_ if there is testDataProvider but no dataProvider.
    evaluator_.reset(trainerInternal_.getGradientMachine()->makeEvaluator());
    currentEvaluator_.reset(
        trainerInternal_.getGradientMachine()->makeEvaluator());
    if (FLAGS_average_test_period > 0 && FLAGS_trainer_id == 0 &&
        config_->getOptConfig().average_window() > 0) {
      CHECK_EQ(FLAGS_average_test_period % FLAGS_log_period, 0)
          << "FLAGS_average_test_period must be divided by FALGS_log_period";
      averageEvaluator_.reset(
          trainerInternal_.getGradientMachine()->makeEvaluator());
    }
  }

  testDataProvider_ = testDataProvider;
  if (!testDataProvider_ && config_->hasTestDataConfig()) {
    testDataProvider_.reset(
        DataProvider::create(config_->getTestDataConfig(), *config_, gpuData));
  }
  if (testDataProvider_) {
    createTester();
  }

  if (!testing &&
      (trainerInternal_.getGradientMachine()->hasStaticParameters())) {
    CHECK(!FLAGS_loadsave_parameters_in_pserver)
        << "is_static and loadsave_parameters_in_pserver can not both true";
  }
  if (testing) {
    // will load per pass for tester
  } else if (paramUtil_->tryLoadParametersFromConfig()) {
    // load from config already.
  } else {
    trainerInternal_.getGradientMachine()->randParameters();
  }

  // Only non static parameters need to be updated
  std::vector<ParameterPtr>& parameters =
      trainerInternal_.getGradientMachine()->getNonStaticParameters();
  if (trainerInternal_.getParameterUpdater()) {
    trainerInternal_.getParameterUpdater()->init(parameters);

    if (FLAGS_loadsave_parameters_in_pserver && FLAGS_trainer_id == 0) {
      if (testing) {
        // will load per pass for tester
      } else if (!config_->getConfig().init_model_path().empty() &&
                 (FLAGS_local || FLAGS_trainer_id == 0)) {
        paramUtil_->loadParametersWithPath(
            config_->getConfig().init_model_path(),
            false /*local*/,
            true /*remote*/);
      } else if (config_->getConfig().start_pass() > 0 &&
                 (FLAGS_local || FLAGS_trainer_id == 0)) {
        CHECK(paramUtil_->loadParameters(config_->getConfig().start_pass() - 1,
                                         false /*local*/,
                                         true /*remote*/));
      } else {
        trainerInternal_.getParameterUpdater()->randParametersRemote();
      }
    }
  }

  // set current evaluator and evalutor
  trainerInternal_.setCurrentEvaluator(currentEvaluator_.get());
  trainerInternal_.setEvaluator(evaluator_.get());
}

void Trainer::train(size_t numPasses) {
  startTrain();
  for (size_t i = 0; i < numPasses; ++i) {
    if (IGradientMachineMode::trainWholeDataInOneBatch(mode_)) {
      trainOnePassBatch(config_->getConfig().start_pass() + i);
    } else {
      trainOnePass();
    }
    if (i < numPasses - 1) {
      dataProvider_->reset();
    }
  }

  finishTrain();
}

static double genPerturbation(real* d, real* grad, size_t dim) {
  auto& reng = ThreadLocalRandomEngine::get();
  std::uniform_real_distribution<double> dist(-1, 1);
  double gradNorm = 0, dNorm = 0;
  for (size_t i = 0; i < dim; ++i) {
    d[i] = dist(reng);
    dNorm += d[i] * d[i];
    gradNorm += grad[i] * grad[i];
  }
  if (gradNorm > 0) {
    real s = 0.5 * sqrt(gradNorm / dNorm);
    for (size_t i = 0; i < dim; ++i) {
      d[i] = s * d[i] + grad[i];
    }
  }
  double delta = 0;
  for (size_t i = 0; i < dim; ++i) {
    delta += grad[i] * d[i];
  }
  return delta;
}

real Trainer::checkGradient() {
  trainerInternal_.getGradientMachine()->start();
  std::vector<ParameterPtr>& parameters =
      trainerInternal_.getGradientMachine()->getNonStaticParameters();
  DataBatch dataBatch;
  int32_t batchSize = config_->getOptConfig().batch_size();

  dataProvider_->getNextBatch(batchSize, &dataBatch);

  CHECK(dataBatch.getSize()) << "No data from data provider";
  std::vector<Argument>& inArgs = dataBatch.getStreams();
  std::vector<Argument> outArgs;

  trainerInternal_.getGradientMachine()->forward(inArgs, &outArgs, PASS_GC);
  real cost = Argument::sum(outArgs);
  LOG(INFO) << "original cost=" << cost;
  trainerInternal_.getGradientMachine()->backward();

  real maxDiff = 0;
  char fill = ' ';
  for (auto& parameter : parameters) {
    CpuVector oldPara(parameter->getSize());
    CpuVector newPara(parameter->getSize());
    oldPara.copyFrom(*parameter->getBuf(PARAMETER_VALUE));
    real* newp = newPara.getData();
    real* oldp = oldPara.getData();
    CpuVector cpuGrad(*parameter->getBuf(PARAMETER_GRADIENT));
    real* grad = cpuGrad.getData();
    size_t dim = parameter->getSize();
    std::vector<real> d(dim);

    double delta = genPerturbation(d.data(), grad, dim);

    // use a step such that delta / cost is FLAGS_checkgrad_eps
    real step =
        (delta != 0) ? cost / delta * FLAGS_checkgrad_eps : FLAGS_checkgrad_eps;
    delta *= step;
    for (size_t i = 0; i < dim; ++i) {
      newp[i] = oldp[i] + step * d[i];
    }

    parameter->getBuf(PARAMETER_VALUE)->copyFrom(newPara);
    parameter->setValueUpdated();
    trainerInternal_.getGradientMachine()->forward(inArgs, &outArgs, PASS_GC);
    real newCost1 = Argument::sum(outArgs);

    for (size_t i = 0; i < dim; ++i) {
      newp[i] = oldp[i] - step * d[i];
    }

    parameter->getBuf(PARAMETER_VALUE)->copyFrom(newPara);
    parameter->setValueUpdated();
    trainerInternal_.getGradientMachine()->forward(inArgs, &outArgs, PASS_GC);
    real newCost2 = Argument::sum(outArgs);

    real trueDelta = 0.5 * (newCost1 - newCost2);
    real diff = (1e-20 + trueDelta) / (1e-20 + delta) - 1;
    LOG(INFO) << std::setiosflags(std::ios::left) << std::setfill(fill)
              << std::setw(20) << parameter->getName()
              << "step=" << std::setw(15) << step << "cost1=" << std::setw(10)
              << newCost1 << "cost2=" << std::setw(10) << newCost2
              << "true_delta=" << std::setw(15) << trueDelta
              << "analytic_delta=" << std::setw(15) << delta << "diff=" << diff
              << (std::abs(diff) > 0.01 ? " ***" : "");

    maxDiff = std::max(maxDiff, std::abs(diff));

    // restore parameter
    parameter->getBuf(PARAMETER_VALUE)->copyFrom(oldPara);
    parameter->setValueUpdated();

    fill = (fill == ' ') ? '.' : ' ';
  }
  return maxDiff;
}

void Trainer::startTrain() {
  trainPassContext_.passId = config_->getConfig().start_pass();
  srand(config_->getConfig().start_pass() + 1);
  if (dataProvider_) {
    dataProvider_->reset();
  }

  trainerInternal_.getGradientMachine()->start();
}

void Trainer::finishTrain() { trainerInternal_.getGradientMachine()->finish(); }

void Trainer::startTrainPass() {
  stats_->reset();
  trainPassContext_.batchId = 0;
  trainPassContext_.avgTestCost = 0;
  trainPassContext_.numAvgTests = 0;
  trainPassContext_.passInnerId = 1;

  trainerInternal_.getParameterUpdater()->startPass();
  evaluator_->start();
  if (FLAGS_prev_batch_state) {
    trainerInternal_.getGradientMachine()->resetState();
    trainerInternal_.getGradientMachine()->getState(testState_);
  }
}

void Trainer::trainOneDataBatch(DataBatch& dataBatch) {
  int num = dataBatch.getSize();
  if (averageEvaluator_) {
    int64_t mod = trainPassContext_.batchId % FLAGS_average_test_period;
    if (mod >= FLAGS_average_test_period - FLAGS_log_period) {
      if (mod == FLAGS_average_test_period - FLAGS_log_period) {
        averageEvaluator_->start();
      }
      trainerInternal_.getParameterUpdater()->apply();
      if (FLAGS_prev_batch_state) {
        trainerInternal_.getGradientMachine()->getState(trainState_);
      }
      trainPassContext_.avgTestCost += tester_->forwardOneBatch(
          dataBatch, averageEvaluator_.get(), &forwardOutput_);
      if (FLAGS_prev_batch_state) {
        trainerInternal_.getGradientMachine()->setState(trainState_);
      }
      trainPassContext_.numAvgTests += num;
      trainerInternal_.getParameterUpdater()->restore();
    }
  }
  {
    REGISTER_TIMER("TrainBatch");
    trainerInternal_.trainOneBatch(
        trainPassContext_.batchId, dataBatch, &forwardOutput_);
  }

  if (averageEvaluator_ &&
      trainPassContext_.batchId % FLAGS_average_test_period ==
          FLAGS_average_test_period - 1) {
    averageEvaluator_->finish();
    LOG(INFO) << " Averaged parameter:"
              << " cost="
              << trainPassContext_.avgTestCost / trainPassContext_.numAvgTests
              << " Eval: " << *averageEvaluator_;
    trainPassContext_.numAvgTests = 0;
    trainPassContext_.avgTestCost = 0;
  }

  ++trainPassContext_.batchId;

  if (trainPassContext_.batchId % FLAGS_log_period == 0) {
    FOR_TIMING(globalStat.setThreadInfo(true));
    FOR_TIMING(globalStat.printAllStatus());
    FOR_TIMING(globalStat.reset());
  }

  if (testDataProvider_ && FLAGS_test_period > 0 &&
      trainPassContext_.batchId % FLAGS_test_period == 0) {
    tester_->testOnePeriod();
  }

  if (FLAGS_saving_period_by_batches > 0 &&
      trainPassContext_.batchId >
          FLAGS_saving_period_by_batches * trainPassContext_.passInnerId &&
      0 == FLAGS_trainer_id) {
    trainerInternal_.getParameterUpdater()->catchUpWith();
    if (testDataProvider_) {
      tester_->testOnePeriod();
    }
    paramUtil_->saveParametersOnePass(trainPassContext_.passId,
                                      trainPassContext_.passInnerId);
    ++trainPassContext_.passInnerId;
  }
}

void Trainer::finishTrainPass() {
  if (trainPassContext_.batchId == 0) {
    // This means no more data from DataProvider
    return;
  }

  trainerInternal_.finishTrainPass(trainPassContext_.passId,
                                   trainPassContext_.batchId);

  FOR_TIMING(globalStat.setThreadInfo(true));
  FOR_TIMING(globalStat.printAllStatus());
  FOR_TIMING(globalStat.reset());

  if (testDataProvider_) {
    tester_->testOnePeriod();
  }

  if (trainPassContext_.passId % FLAGS_saving_period == 0 &&
      FLAGS_trainer_id == 0) {
    paramUtil_->saveParametersOnePass(trainPassContext_.passId);
  }
  ++trainPassContext_.passId;
}

void Trainer::trainOnePass() {
  startTrainPass();
  size_t batchSize = config_->getOptConfig().batch_size();
  while (true) {
    DataBatch dataBatch;

    int num = 0;
    {
      REGISTER_TIMER("getTrainBatch");
      num = dataProvider_->getNextBatch(batchSize, &dataBatch);
    }
    if (num == 0) break;
    CHECK_EQ(num, dataBatch.getSize());
    trainOneDataBatch(dataBatch);
  }

  finishTrainPass();
}

void Trainer::trainOnePassBatch(int passId) {
  this->stats_->reset();

  trainerInternal_.getParameterUpdater()->startPass();
  const std::vector<Argument> inArgs;
  {
    REGISTER_TIMER("onePass");
    trainerInternal_.getGradientMachine()->forwardBackward(
        inArgs, nullptr, PASS_TRAIN, nullptr);
  }

  real cost = .0;
  int64_t num = 0;
  trainerInternal_.getGradientMachine()->getStats(cost, num);
  *stats_ += {num, cost};

  trainerInternal_.getGradientMachine()->onPassEnd();

  bool accepted = trainerInternal_.getParameterUpdater()->finishPass();

  globalStat.setThreadInfo(true);
  globalStat.printAllStatus();
  globalStat.reset();

  LOG(INFO) << " Pass=" << passId
            << " AcceptedPass=" << (accepted ? acceptedPassId_ : -1)
            << stats_->getStats(false /*withCurrentCost*/);

  if (accepted) {
    if (acceptedPassId_ % FLAGS_saving_period == 0 && FLAGS_trainer_id == 0) {
      paramUtil_->saveParameters(acceptedPassId_);
    }
    acceptedPassId_++;
    if (FLAGS_save_only_one && acceptedPassId_ >= FLAGS_saving_period) {
      paramUtil_->deleteParameters(acceptedPassId_ - FLAGS_saving_period);
    }
  }
}

real Trainer::calcGradient(const DataBatch& dataBatch,
                           const Vector& value,
                           Vector& gradient) {
  CHECK_EQ(value.getSize(), gradient.getSize());
  std::vector<ParameterPtr>& parameters =
      trainerInternal_.getGradientMachine()->getParameters();

  clearGradient();

  size_t offset = 0;
  size_t valueSize = value.getSize();

  for (auto& para : parameters) {
    CHECK_LE(offset + para->getSize(), valueSize);
    VectorPtr val =
        Vector::create(para->getSize(), value.getMemoryHandle(), offset);
    para->getBuf(PARAMETER_VALUE)->copyFrom(*val);
    para->setValueUpdated();
    offset += para->getSize();
  }

  CHECK_EQ(offset, valueSize);

  std::vector<Argument> inArgs = dataBatch.getStreams();
  std::vector<Argument> outArgs;

  trainerInternal_.getGradientMachine()->forwardBackward(
      inArgs, &outArgs, PASS_TRAIN);
  real cost = Argument::sum(outArgs);

  offset = 0;
  for (auto& para : parameters) {
    VectorPtr grad =
        Vector::create(para->getSize(), gradient.getMemoryHandle(), offset);
    if (para->getBuf(PARAMETER_GRADIENT)) {
      grad->copyFrom(*para->getBuf(PARAMETER_GRADIENT));
    }
    offset += para->getSize();
  }

  return cost;
}

void Trainer::clearGradient() {
  std::vector<ParameterPtr>& parameters =
      trainerInternal_.getGradientMachine()->getNonStaticParameters();
  for (auto& parameter : parameters) {
    parameter->clearGradient();
  }
}

int Trainer::getBatchSize() { return config_->getOptConfig().batch_size(); }

void Trainer::createTester() {
  tester_.reset(new paddle::Tester(config_,
                                   createTesterConfig(),
                                   trainerInternal_.getGradientMachine(),
                                   trainerInternal_.getParameterUpdater(),
                                   testDataProvider_));
}

void Trainer::test() { tester_->test(); }

std::unique_ptr<TesterConfig> Trainer::createTesterConfig() {
  TesterConfig* conf = new TesterConfig;
  if (FLAGS_test_period) {
    LOG(WARNING) << "The meaning of --test_period is changed: "
                 << "if equal 0, do test on all test data at the end of "
                 << "each pass. While if equal non-zero, do test on all test "
                 << "data every test_period batches ";
  }
  if (FLAGS_test_all_data_in_one_period) {
    LOG(WARNING) << "--test_all_data_in_one_period was deprecated, since "
                 << "we will always do test on all test set ";
  }
  conf->testPeriod = FLAGS_test_period;
  conf->prevBatchState = FLAGS_prev_batch_state;
  conf->logPeriod = FLAGS_log_period;
  conf->loadsaveParametersInPserver = FLAGS_loadsave_parameters_in_pserver;
  conf->featFile = FLAGS_feat_file;
  conf->predictOutputDir = FLAGS_predict_output_dir;
  conf->trainerId = FLAGS_trainer_id;
  conf->distributeTest = FLAGS_distribute_test;
  conf->config = FLAGS_config;
  conf->modelList = FLAGS_model_list;
  conf->testPass = FLAGS_test_pass;
  conf->numPasses = FLAGS_num_passes;
  conf->savingPeriod = FLAGS_saving_period;
  conf->testWait = FLAGS_test_wait;
  conf->initModelPath = FLAGS_init_model_path;
  conf->saveOnlyOne = FLAGS_save_only_one;
  conf->testing = testing_;
  conf->mode = mode_;
  conf->trainState = &trainState_;
  conf->testState = &testState_;
  return std::unique_ptr<TesterConfig>(conf);
}

ParameterUtil* Trainer::getParameterUtilPtr() { return paramUtil_.get(); }
}  // namespace paddle
