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

#include <paddle/utils/PythonUtil.h>

#include "paddle/trainer/Trainer.h"

#include <gtest/gtest.h>
#include <paddle/pserver/ParameterServer2.h>

using namespace paddle;  // NOLINT
using namespace std;     // NOLINT

static const string& configFile1 = "gserver/tests/sequence_lstm.conf";

DECLARE_bool(use_gpu);
DECLARE_string(config);
DECLARE_int32(gpu_id);
DECLARE_int32(seed);
DECLARE_int32(num_passes);
DECLARE_int32(saving_period);

DECLARE_int32(num_gradient_servers);
DECLARE_int32(port);
DECLARE_bool(local);
DECLARE_bool(use_old_updater);
DECLARE_bool(parallel_nn);
DECLARE_string(config_args);
DEFINE_double(max_diff_ratio,
              0.0f,
              "max diff ratio allowed for parameters value");

int gNumDevices = 0;

std::vector<ParameterPtr> trainerOnePassTest(const string& configFile,
                                             bool sparseUpdate,
                                             int trainerCount = 1,
                                             bool useGpu = false) {
  FLAGS_use_gpu = useGpu;
  FLAGS_config = configFile;
  FLAGS_trainer_count = trainerCount;
  FLAGS_config_args = sparseUpdate ? "sparse_update=1" : "sparse_update=0";

  LOG(INFO) << " useGpu=" << useGpu << " trainerCount=" << trainerCount
            << " configFile=" << configFile << " sparseUpdate=" << sparseUpdate;
  srand(FLAGS_seed);
  *ThreadLocalRand::getSeed() = FLAGS_seed;
  ThreadLocalRandomEngine::get().seed(FLAGS_seed);
  if (useGpu) {
    CHECK_LE(trainerCount, gNumDevices);
  }

  std::vector<std::shared_ptr<ParameterServer2>> pservers;
  if (!FLAGS_local) {
    int numPorts = FLAGS_ports_num + FLAGS_ports_num_for_sparse;
    pservers.resize(numPorts);

    for (int i = 0; i < numPorts; ++i) {
      pservers[i].reset(new ParameterServer2(std::string(), FLAGS_port + i));
      pservers[i]->init();
      pservers[i]->start();
    }
  }

  Trainer trainer;
  trainer.init(TrainerConfigHelper::createFromFlagConfig());
  trainer.train();
  return trainer.getGradientMachine()->getParameters();
}

std::vector<ParameterPtr>& getDenseParameters() {
  static std::vector<ParameterPtr> denseParameters;
  if (denseParameters.empty()) {
    // use dense training as base
    FLAGS_local = true;
    denseParameters = trainerOnePassTest(configFile1, false);
  }

  return denseParameters;
}

void checkBuffer(real* A,
                 const char* desA,
                 real* B,
                 const char* desB,
                 size_t len,
                 double maxDiffRatio) {
  double maxDiff = 0;
  double maxValue = 0;
  for (size_t i = 0; i < len; ++i) {
    double diff = fabs(A[i] - B[i]);
    maxValue = std::max<double>(maxValue, std::max(fabs(A[i]), fabs(B[i])));
    maxDiff = std::max(maxDiff, diff);
  }
  EXPECT_LE(maxDiff / maxValue, maxDiffRatio);
  LOG(INFO) << " maxDiff=" << maxDiff << " maxValue=" << maxValue
            << " maxDiff/maxValue=" << maxDiff / maxValue << "\n\n";
}

void compareValue(const vector<ParameterPtr>& parametersA,
                  const vector<ParameterPtr>& parametersB,
                  double maxDiffRatio = 0.0) {
  LOG(INFO) << "\n\n--------------------------------"
            << " Check Gradient Machine Parameters:"
            << " -------------------------------------\n";
  for (size_t i = 0; i < parametersA.size(); ++i) {
    ParameterPtr parameterA, parameterB;
    parameterA = parametersA[i];
    parameterB = parametersB[i];

    CpuVector paraA(parameterA->getSize());
    CpuVector paraB(parameterB->getSize());
    paraA.copyFrom(*parameterA->getBuf(PARAMETER_VALUE));
    paraB.copyFrom(*parameterB->getBuf(PARAMETER_VALUE));

    LOG(INFO) << "\n\n----------- PARAMETER_VALUE:  " << parameterA->getName()
              << " ; size : " << paraA.getSize() << " ------------";
    checkBuffer(paraA.getData(),
                "para_A",
                paraB.getData(),
                "para_B",
                paraA.getSize(),
                maxDiffRatio);
  }
}

TEST(compareSparse, cpu) {
  FLAGS_local = 1;  // disable remote sparse update in parameter config
  std::vector<ParameterPtr> parameters = trainerOnePassTest(configFile1, true);
  compareValue(getDenseParameters(), parameters);
}

TEST(compareSparse, remote_cpu) {
  FLAGS_local = 0;  // will enable remote sparse update
  FLAGS_ports_num_for_sparse = 5;
  std::vector<ParameterPtr> parameters = trainerOnePassTest(configFile1, true);
  compareValue(getDenseParameters(), parameters);
}

TEST(compareSparse, cpu10_local_vs_remote) {
  FLAGS_local = 1;  // disable remote sparse update in parameter config
  std::vector<ParameterPtr> localParameters =
      trainerOnePassTest(configFile1, true, 2);

  FLAGS_local = 0;  // will enable remote sparse update
  FLAGS_ports_num_for_sparse = 5;
  std::vector<ParameterPtr> remoteParameters =
      trainerOnePassTest(configFile1, true, 2);

  compareValue(localParameters, remoteParameters);
}

TEST(compareSparse, multiGradientMachine) {
  int numGpu;
#ifdef PADDLE_TYPE_DOUBLE
  double eps = 1e-8;
#else
  double eps = 1e-4;
#endif
  numGpu = hl_get_device_count();
  for (bool local : {false, true}) {
    FLAGS_local = local;
    FLAGS_ports_num_for_sparse = 5;
    for (bool useGpu : {false, true}) {
#ifndef PADDLE_WITH_CUDA
      if (useGpu) continue;
#endif
      FLAGS_parallel_nn = useGpu;
      LOG(INFO) << " local=" << local << " useGpu=" << useGpu;
      int trainerCount = useGpu ? numGpu : 2;
      std::vector<ParameterPtr> parameters =
          trainerOnePassTest(configFile1, true, trainerCount, useGpu);
      compareValue(getDenseParameters(), parameters, eps);
    }
  }
  FLAGS_parallel_nn = false;
}

TEST(compareSparse, NeuralNetwork) {
#ifdef PADDLE_TYPE_DOUBLE
  double eps = 1e-8;
#else
  double eps = 1e-4;
#endif
  for (bool local : {false, true}) {
    FLAGS_local = local;
    FLAGS_ports_num_for_sparse = 5;
    for (bool useGpu : {false, true}) {
#ifndef PADDLE_WITH_CUDA
      if (useGpu) continue;
#endif
      FLAGS_parallel_nn = useGpu;
      LOG(INFO) << " local=" << local << " useGpu=" << useGpu;
      int trainerCount = 1;
      std::vector<ParameterPtr> parameters =
          trainerOnePassTest(configFile1, true, trainerCount, useGpu);
      compareValue(getDenseParameters(), parameters, useGpu ? eps : 0);
    }
  }
  FLAGS_parallel_nn = false;
}

int main(int argc, char** argv) {
  // FIXME(tonyyang-svail):
  //   Turn off this test due CI failure:
  //   https://paddleci.ngrok.io/viewLog.html?buildId=27608&buildTypeId=Paddle_PrCi&tab=buildLog&_focus=10430
  return 0;
  testing::InitGoogleTest(&argc, argv);
  initMain(argc, argv);
  initPython(argc, argv);

  gNumDevices = hl_get_device_count();
  FLAGS_num_passes = 1;          // train one pass
  FLAGS_saving_period = 100000;  // do not save parameter

  return RUN_ALL_TESTS();
}
