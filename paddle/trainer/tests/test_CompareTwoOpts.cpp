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

#include <gtest/gtest.h>
#include <paddle/utils/PythonUtil.h>
#include <algorithm>
#include <cstdlib>

#include "paddle/trainer/Trainer.h"

using namespace paddle;  // NOLINT
using namespace std;     // NOLINT

DECLARE_int32(gpu_id);

DECLARE_bool(local);
DECLARE_bool(use_gpu);

DECLARE_string(config);
DECLARE_string(nics);

DEFINE_string(config_file_a, "", "config of one network to compare");
DEFINE_string(config_file_b, "", "config of another network to compare");
DEFINE_bool(need_high_accuracy,
            true,
            "whether need to run in double accuracy (recommended)");
DEFINE_double(
    max_diff_ratio,
    0.0f,
    "max diff ratio allowed for outputs and parameters (value/gradient)");

struct ComData {
  vector<Argument> outArgs;
  vector<ParameterPtr> parameters;
};

void calcGradient(ComData& data, const string configFile) {
  FLAGS_config = configFile;

  FLAGS_local = true;
  FLAGS_use_gpu = false;

  FLAGS_nics = "";

  *ThreadLocalRand::getSeed() = 0;
  srand(0);

  Trainer trainer;
  trainer.init(TrainerConfigHelper::createFromFlagConfig(), false);

  data.parameters = trainer.getGradientMachine()->getParameters();
  trainer.getDataProvider()->setSkipShuffle();
  trainer.train();
}

void checkBuffer(real* A,
                 const char* desA,
                 real* B,
                 const char* desB,
                 size_t len,
                 size_t width = 1) {
  int nNum = 0;
  for (size_t i = 0; i < len; ++i) {
    real diff = fabs(A[i] - B[i]);
    if (diff > 0.0f &&
        diff / std::max(fabs(A[i]), fabs(B[i])) > FLAGS_max_diff_ratio) {
      nNum++;
      LOG(INFO) << "Row: " << i / width << ", " << desA << " : " << A[i]
                << "    " << desB << " : " << B[i];
    }
  }
  EXPECT_EQ(0, nNum);
  LOG(INFO) << "\n\n";
}

void compareGradient(ComData& comDataA, ComData& comDataB) {
  vector<Argument> outArgsA = comDataA.outArgs;
  vector<Argument> outArgsB = comDataB.outArgs;

  for (size_t i = 0; i < outArgsA.size(); ++i) {
    CpuMatrix matA(outArgsA[i].value->getHeight(),
                   outArgsA[i].value->getWidth());
    CpuMatrix matB(outArgsB[i].value->getHeight(),
                   outArgsB[i].value->getWidth());

    matA.copyFrom(*outArgsA[i].value);
    matB.copyFrom(*outArgsB[i].value);

    LOG(INFO) << "\n--------------------------------"
              << " Check Network Output_" << i << ":"
              << " -------------------------------------\n";
    checkBuffer(matA.getData(),
                "network A output",
                matB.getData(),
                "network B output",
                matA.getElementCnt(),
                matA.getWidth());
  }

  vector<ParameterPtr>& parametersA = comDataA.parameters;
  vector<ParameterPtr>& parametersB = comDataB.parameters;

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
                "Network A",
                paraB.getData(),
                "Network B",
                paraA.getSize());

    CpuVector gradA(*parameterA->getBuf(PARAMETER_GRADIENT));
    CpuVector gradB(*parameterB->getBuf(PARAMETER_GRADIENT));

    LOG(INFO) << "\n\n----------- PARAMETER_GRADIENT: " << parameterA->getName()
              << " ; size : " << gradA.getSize() << " -----------";
    checkBuffer(gradA.getData(),
                "Network A",
                gradB.getData(),
                "Network B",
                gradA.getSize());
  }
}

TEST(Trainer, create) {
  ComData dataA;
  calcGradient(dataA, FLAGS_config_file_a);
  LOG(INFO) << "\n\ntraining of Network A is finished\n\n";

  ComData dataB;
  calcGradient(dataB, FLAGS_config_file_b);
  LOG(INFO) << "\n\ntraining of the Network B is finished\n\n";

  compareGradient(dataA, dataB);
}

int main(int argc, char** argv) {
  paddle::initMain(argc, argv);
  testing::InitGoogleTest(&argc, argv);
  initPython(argc, argv);

#ifndef PADDLE_TYPE_DOUBLE
  if (FLAGS_need_high_accuracy) {
    LOG(INFO) << "skip test due to it's need high accuracy";
    return 0;
  }
  if (FLAGS_max_diff_ratio == 0.0f) {
    FLAGS_max_diff_ratio = 2e-4;
    LOG(INFO) << "auto set max_diff_ratio " << FLAGS_max_diff_ratio
              << " in low accuracy mode";
  }
#else
  if (FLAGS_max_diff_ratio == 0.0f) {
    FLAGS_max_diff_ratio = 2e-7;
    LOG(INFO) << "auto set max_diff_ratio " << FLAGS_max_diff_ratio
              << " in high accuracy mode";
  }
#endif
  int ret = RUN_ALL_TESTS();
  return ret;
}
