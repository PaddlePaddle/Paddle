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

#include <paddle/utils/GlobalConstants.h>
#include <paddle/utils/PythonUtil.h>
#include "paddle/trainer/Trainer.h"
#include "paddle/trainer/TrainerInternal.h"

#include <gtest/gtest.h>
#include <paddle/pserver/ParameterServer2.h>

using namespace paddle;  // NOLINT
using namespace std;     // NOLINT

static const string& configFile1 = "trainer/tests/sample_trainer_config.conf";
static const string& configFile2 =
    "trainer/tests/sample_trainer_config_parallel.conf";

static const string& configFileSimpleSparse =
    "trainer/tests/simple_sparse_neural_network.py";

DECLARE_bool(use_gpu);
DECLARE_string(config);
DECLARE_int32(gpu_id);
DECLARE_int32(seed);
DECLARE_int32(num_passes);
DECLARE_int32(saving_period);

class TrainerForTest : public paddle::Trainer {
 public:
  inline const std::shared_ptr<ParameterUpdater>& getParameterUpdaterForTest() {
    return this->trainerInternal_.getParameterUpdater();
  }
};

int gNumDevices = 0;

void trainerOnePassTest(const string& configFile,
                        bool useGpu,
                        bool parallel,
                        int trainerCount = 1,
                        double averageWindow = 0.0f,
                        bool doAverageInCpu = false) {
  FLAGS_use_gpu = useGpu;
  FLAGS_parallel_nn = parallel;
  FLAGS_config = configFile;
  FLAGS_trainer_count = trainerCount;
  LOG(INFO) << " useGpu=" << useGpu << " trainerCount=" << trainerCount
            << " configFile=" << configFile;
  srand(FLAGS_seed);

  if (useGpu) {
    if (gNumDevices < trainerCount) {
      return;
    }
  }

  Trainer trainer;
  auto config = TrainerConfigHelper::createFromFlagConfig();
  if (averageWindow > 0) {
    config->getOptConfig().set_average_window(averageWindow);
    config->getOptConfig().set_do_average_in_cpu(doAverageInCpu);
  }
  trainer.init(config);
  trainer.train();
}

// 1. test trainer (cpu, gpu).
TEST(trainerOnePass, cpu) { trainerOnePassTest(configFile1, false, false); }

#ifdef PADDLE_WITH_CUDA
TEST(trainerOnePass, gpu) { trainerOnePassTest(configFile1, true, false); }

TEST(trainerOnePass, gpu2) { trainerOnePassTest(configFile1, true, false, 2); }

TEST(trainerOnePass, gpu4) { trainerOnePassTest(configFile1, true, false, 4); }

TEST(trainerOnePass, parallel) {
  if (hl_get_device_count() >= 2) {
    trainerOnePassTest(configFile2, true, true);
  }
}
#endif

// 2. test average_window.
#ifdef PADDLE_WITH_CUDA
TEST(average_window, gpu) {
  trainerOnePassTest(configFile1, true, false, 4, 0.01);
}

TEST(average_window, gpu2) {
  FLAGS_num_passes = 20;
  trainerOnePassTest(configFile1, true, false, 2, 0.01);
  FLAGS_num_passes = 1;
}

TEST(average_window, gpu4) {
  FLAGS_num_passes = 20;
  trainerOnePassTest(configFile1, true, false, 4, 0.01);
  FLAGS_num_passes = 1;
}

TEST(average_window_cpu, gpu2) {
  FLAGS_num_passes = 20;
  trainerOnePassTest(configFile1, true, false, 2, 0.01, true);
  FLAGS_num_passes = 1;
}

TEST(average_window_cpu, gpu4) {
  FLAGS_num_passes = 20;
  trainerOnePassTest(configFile1, true, false, 4, 0.01, true);
  FLAGS_num_passes = 1;
}
#endif

// 3. test trainer + pserver.
DECLARE_int32(num_gradient_servers);
DECLARE_int32(port);
DECLARE_bool(local);
DECLARE_bool(use_old_updater);

double checkRemoteParameterUpdater(TrainerForTest& trainer) {
  auto gradientMachine = trainer.getGradientMachine();
  auto parameterUpdater = trainer.getParameterUpdaterForTest();
  auto dataProvider = trainer.getDataProvider();
  auto& parameters = gradientMachine->getParameters();
  const TrainerConfig& config = trainer.getConfig();
  const string& alg = config.opt_config().algorithm();

  vector<ParameterPtr> parameterCheck;
  for (auto& parameter : parameters) {
    parameterCheck.emplace_back(
        new Parameter(parameter->getConfig(), /* useGpu= */ false));
    parameterCheck.back()
        ->getBuf(PARAMETER_VALUE)
        ->copyFrom(*parameter->getBuf(PARAMETER_VALUE));
    parameterCheck.back()
        ->getBuf(PARAMETER_GRADIENT)
        ->copyFrom(*parameter->getBuf(PARAMETER_GRADIENT));
  }

  std::unique_ptr<ParameterUpdater> parameterUpdaterCheck;
  if (alg == TrainAlgorithm::SGD) {
    parameterUpdaterCheck.reset(new SgdLocalUpdater(config.opt_config()));
  } else {
    LOG(INFO) << "unsupported algorithm in remote parameter check: " << alg;
    return -1.0;
  }
  parameterUpdaterCheck->init(parameterCheck);

  // gradientMachine->start(config, *dataProvider);
  DataBatch dataBatch;
  int32_t batchSize = config.opt_config().batch_size();
  dataProvider->getNextBatch(batchSize, &dataBatch);
  CHECK(dataBatch.getSize()) << "No data from data provider";
  int64_t actualBatchSize = dataBatch.getSize();
  const vector<Argument>& inArgs = dataBatch.getStreams();
  vector<Argument> outArgs;

  UpdateCallback updateCallback = [parameterUpdater,
                                   parameterCheck](Parameter* para) {
    parameterCheck[para->getID()]
        ->getBuf(PARAMETER_GRADIENT)
        ->copyFrom(*para->getBuf(PARAMETER_GRADIENT));
    parameterUpdater->update(para);
  };

  parameterUpdater->startPass();
  parameterUpdaterCheck->startPass();

  for (int i = 0; i < config.opt_config().num_batches_per_get_parameter() * 2;
       ++i) {
    PassType passType = parameterUpdater->startBatch(actualBatchSize);
    gradientMachine->forwardBackward(
        inArgs, &outArgs, passType, updateCallback);
    parameterUpdater->finishBatch(0);

    parameterUpdaterCheck->startBatch(actualBatchSize);
    for (auto& para : parameterCheck) {
      parameterUpdaterCheck->update(para.get());
    }
    parameterUpdaterCheck->finishBatch(0);
  }

  double sum = 0.0f;
  for (size_t i = 0; i != parameters.size(); ++i) {
    real *v1, *v2;
    CpuVector trainerPara(parameters[i]->getSize());
    trainerPara.copyFrom(*parameters[i]->getBuf(PARAMETER_VALUE));
    if (!FLAGS_use_gpu) {
      v1 = parameters[i]->getBuf(PARAMETER_VALUE)->getData();
    } else {
      v1 = trainerPara.getData();
    }
    v2 = parameterCheck[i]->getBuf(PARAMETER_VALUE)->getData();

    size_t size = parameters[i]->getSize();
    double diff = 0;
    for (size_t j = 0; j < size; ++j) {
      diff += fabs(v1[j] - v2[j]);
    }
    sum += diff;
    LOG(INFO) << setiosflags(ios::left) << setfill(' ') << setw(20)
              << parameters[i]->getName() << "diff=" << setw(15) << diff;
  }

  parameterUpdater->finishPass();
  parameterUpdaterCheck->finishPass();
  gradientMachine->finish();
  return sum;
}

void checkRemoteParameterUpdaterTest(const string& configFile,
                                     bool useGpu,
                                     bool parallel,
                                     int trainerCount = 1,
                                     bool useOldUpdater = false,
                                     int num_batches_per_get_parameter = 1) {
  FLAGS_use_gpu = useGpu;
  FLAGS_parallel_nn = parallel;
  FLAGS_config = configFile;
  FLAGS_trainer_count = trainerCount;
  FLAGS_use_old_updater = useOldUpdater;
  LOG(INFO) << " useGpu=" << useGpu << " trainerCount=" << trainerCount
            << " configFile=" << configFile;
  srand(FLAGS_seed);

  if (useGpu) {
    if (gNumDevices < trainerCount) {
      return;
    }
  }

  FLAGS_local = 0;
  std::shared_ptr<ParameterServer2> pserver;
  pserver.reset(new ParameterServer2(std::string(), FLAGS_port));
  pserver->init();
  pserver->start();

  TrainerForTest trainer;
  auto config = TrainerConfigHelper::createFromFlagConfig();
  config->getOptConfig().set_num_batches_per_get_parameter(
      num_batches_per_get_parameter);
  trainer.init(config);
  EXPECT_EQ(checkRemoteParameterUpdater(trainer), 0);

  FLAGS_local = 1;
}

TEST(checkRemoteUpdater, cpuTrainer) {
  checkRemoteParameterUpdaterTest(configFile1, false, false);
}

TEST(checkRemoteUpdater, cpuTrainerOldUpdater) {
  checkRemoteParameterUpdaterTest(configFile1, false, false, 1, true);
}

#ifdef PADDLE_WITH_CUDA
TEST(checkRemoteUpdater, gpuTrainer) {
  checkRemoteParameterUpdaterTest(configFile1, true, false);
}

TEST(checkRemoteUpdater, gpu2Trainer) {
  checkRemoteParameterUpdaterTest(configFile1, true, false, 2);
}

TEST(checkRemoteUpdater, gpu4Trainer) {
  checkRemoteParameterUpdaterTest(configFile1, true, false, 4);
}

TEST(checkRemoteUpdater, gpuTrainerOldUpdater) {
  checkRemoteParameterUpdaterTest(configFile1, true, false, 1, true);
}

TEST(checkRemoteUpdater, gpu2TrainerOldUpdater) {
  checkRemoteParameterUpdaterTest(configFile1, true, false, 2, true);
}

TEST(checkRemoteUpdater, gpu4TrainerOldUpdater) {
  checkRemoteParameterUpdaterTest(configFile1, true, false, 4, true);
}

#endif

TEST(checkRemoteUpdater, cpuDeltaTrainer) {
  checkRemoteParameterUpdaterTest(configFile1, false, false, 1, false, 10);
}

TEST(checkRemoteUpdater, cpuDeltaTrainerOldUpdater) {
  checkRemoteParameterUpdaterTest(configFile1, false, false, 1, true, 10);
}

TEST(SgdThreadUpdater, simpleSparseNN) {
  trainerOnePassTest(configFileSimpleSparse, false, false, 1, 0.5, true);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  initMain(argc, argv);
  initPython(argc, argv);
  gNumDevices = hl_get_device_count();

  FLAGS_num_passes = 1;          // train one pass
  FLAGS_saving_period = 100000;  // do not save parameteres
  return RUN_ALL_TESTS();
}
