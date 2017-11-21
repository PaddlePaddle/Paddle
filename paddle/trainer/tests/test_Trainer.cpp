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

#include <paddle/utils/PythonUtil.h>
#include <paddle/utils/Version.h>
#include "paddle/trainer/Trainer.h"

#include <gtest/gtest.h>

using namespace paddle;  // NOLINT
using namespace std;     // NOLINT

static const string& configFile1 = "trainer/tests/sample_trainer_config.conf";
static const string& configFile2 =
    "trainer/tests/sample_trainer_config_hsigmoid.conf";
static const string& configFile4 =
    "trainer/tests/sample_trainer_config_parallel.conf";

DECLARE_bool(use_gpu);
DECLARE_string(config);
DECLARE_int32(gpu_id);
DECLARE_bool(allow_only_one_model_on_one_gpu);

void checkGradientTest(const string& configFile,
                       bool useGpu,
                       bool parallel,
                       int trainerCount = 1) {
  FLAGS_use_gpu = useGpu;
  FLAGS_parallel_nn = parallel;
  FLAGS_config = configFile;
  FLAGS_trainer_count = trainerCount;
  LOG(INFO) << " useGpu=" << useGpu << " trainerCount=" << trainerCount
            << " configFile=" << configFile;

  Trainer trainer;
  trainer.init(TrainerConfigHelper::createFromFlagConfig());
  EXPECT_LE(fabs(trainer.checkGradient()), 0.02);
}

TEST(checkGradient, cpu) { checkGradientTest(configFile1, false, false); }

#ifdef PADDLE_WITH_CUDA
TEST(checkGradient, gpu) { checkGradientTest(configFile1, true, false); }

TEST(checkGradient, multiGpu) {
  int numGpu;
  numGpu = hl_get_device_count();
  for (auto count : {2, 4}) {
    if (count <= numGpu) {
      checkGradientTest(configFile1, true, false, count);
    }
  }
}

TEST(checkGradient, parallel) {
  if (hl_get_device_count() >= 2) {
    checkGradientTest(configFile4, true, true);
  }
}

TEST(checkGradient, multiParallel) {
  FLAGS_allow_only_one_model_on_one_gpu = false;
  checkGradientTest(configFile4, true, true, 2);
  FLAGS_allow_only_one_model_on_one_gpu = true;
}

#endif

TEST(checkGradient, multi) {
  int numGpu;
  if (version::isWithGpu()) {
    numGpu = hl_get_device_count();
  } else {
    numGpu = 0;
  }
  for (bool useGpu : {false, true}) {
    for (auto count : {2, 4}) {
      if (useGpu && count > numGpu) continue;
      checkGradientTest(configFile1, useGpu, false, count);
    }
  }
}

TEST(checkGradient, hsigmoid) { checkGradientTest(configFile2, false, false); }

TEST(checkGradient, non_parallel) {
  checkGradientTest(configFile4, false, false);
}

int main(int argc, char** argv) {
  initMain(argc, argv);
  initPython(argc, argv);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
