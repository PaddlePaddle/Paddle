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

#include "paddle/trainer/Trainer.h"

#include <gtest/gtest.h>

DECLARE_string(config);
DECLARE_string(config_args);
DEFINE_string(merger,
              "./paddle_merge_model",
              "path to paddle_merge_model binary");

using namespace paddle;  // NOLINT
using namespace std;     // NOLINT

static const string& configFile = "trainer/tests/sample_trainer_config.conf";
static const string& mergedModelFile = "./test_model_file";
static const string& modelDir = "./test_model_dir";

void checkBuffer(real* vec1, real* vec2, size_t len) {
  for (size_t i = 0; i < len; i++) {
    EXPECT_EQ(vec1[i], vec2[i]) << "vec1:" << vec1[i] << " vec2:" << vec2[i];
  }
}

void checkParameters(vector<ParameterPtr> A, vector<ParameterPtr> B) {
  CHECK_EQ(B.size(), A.size()) << "parameter size not equal";
  for (size_t i = 0; i < A.size(); i++) {
    auto vec1 = A[i]->getBuf(PARAMETER_VALUE);
    auto vec2 = B[i]->getBuf(PARAMETER_VALUE);
    CHECK_EQ(vec1->useGpu_, vec2->useGpu_) << "use gpu not equal";
    CHECK_EQ(vec1->getSize(), vec2->getSize()) << "size not equal";

    if (vec1->useGpu_ == false) {
      checkBuffer(vec1->getData(), vec2->getData(), vec1->getSize());
    } else {
      VectorPtr cpuVec1 = Vector::create(vec1->getSize(), false);
      VectorPtr cpuVec2 = Vector::create(vec2->getSize(), false);
      cpuVec1->copyFrom(*vec1, HPPL_STREAM_DEFAULT);
      cpuVec2->copyFrom(*vec2, HPPL_STREAM_DEFAULT);
      hl_stream_synchronize(HPPL_STREAM_DEFAULT);
      checkBuffer(cpuVec1->getData(), cpuVec2->getData(), cpuVec1->getSize());
    }
  }
}

TEST(GradientMachine, create) {
#ifdef PADDLE_ONLY_CPU
  FLAGS_use_gpu = false;
#endif
  mkDir(modelDir.c_str());
  FLAGS_config = configFile;
  FLAGS_config_args = "with_cost=False";
  auto config = TrainerConfigHelper::createFromFlagConfig();

  // save model to directory
  unique_ptr<GradientMachine> gradientMachine1(
      GradientMachine::create(*config));
  gradientMachine1->saveParameters(modelDir);
  Trainer trainer;
  trainer.init(config);
  ParameterUtil* paramUtil = trainer.getParameterUtilPtr();
  if (paramUtil != NULL) {
    paramUtil->saveConfigWithPath(modelDir);
  }

  // create a different GradientMachine
  unique_ptr<GradientMachine> gradientMachine2(
      GradientMachine::create(*config));
  gradientMachine2->randParameters();

  // merge config and model to one file
  string cmd = FLAGS_merger + " --model_dir=" + modelDir +
               " --config_args=with_cost=False" + " --model_file=" +
               mergedModelFile;
  LOG(INFO) << cmd;
  int ret = system(cmd.c_str());
  EXPECT_EQ(0, ret);
  if (ret) {
    return;
  }

  // create GradientMachine from the merged model
  DataConfig dataConfig;
  unique_ptr<GradientMachine> gradientMachine3(
      GradientMachine::create(mergedModelFile, &dataConfig));
  CHECK(gradientMachine3);
  EXPECT_EQ(dataConfig.type(), "simple");
  EXPECT_EQ(dataConfig.feat_dim(), 3);

  // compare the parameters of GradientMachine and GradientMachine3
  std::vector<ParameterPtr> paraMachine1 = gradientMachine1->getParameters();
  std::vector<ParameterPtr> paraMachine3 = gradientMachine3->getParameters();
  checkParameters(paraMachine1, paraMachine3);

  // Test that the GradientMachine created from the merged model
  // is same as the orginnal one.
  vector<Argument> inArgs(1);
  vector<Argument> outArgs;

  int inputDim = 3;
  int numSamples = 2;
  CpuMatrix cpuInput(numSamples, inputDim);
  for (int i = 0; i < numSamples; ++i) {
    for (int j = 0; j < inputDim; ++j) {
      cpuInput.getData()[i * inputDim + j] =
          rand() / (real)RAND_MAX;  // NOLINT TODO(yuyang): use rand_r
    }
  }
  MatrixPtr input = Matrix::create(numSamples,
                                   inputDim,
                                   /* trans */ false,
                                   FLAGS_use_gpu);
  input->copyFrom(cpuInput);
  inArgs[0].value = input;
  gradientMachine1->forward(inArgs, &outArgs, PASS_TEST);
  EXPECT_EQ((size_t)1, outArgs.size());

  vector<Argument> outArgs2;
  gradientMachine2->forward(inArgs, &outArgs2, PASS_TEST);
  CpuMatrix out1(outArgs[0].value->getHeight(), outArgs[0].value->getWidth());
  CpuMatrix out2(outArgs2[0].value->getHeight(), outArgs2[0].value->getWidth());
  out1.copyFrom(*outArgs[0].value);
  out2.copyFrom(*outArgs2[0].value);
  for (size_t i = 0; i < out1.getHeight() * out1.getWidth(); i++) {
    EXPECT_NE(out1.getData()[i], out2.getData()[i]);
  }

  gradientMachine3->forward(inArgs, &outArgs2, PASS_TEST);
  out2.copyFrom(*outArgs2[0].value);
  checkBuffer(
      out1.getData(), out2.getData(), out2.getHeight() * out2.getWidth());

  cmd = " rm -rf " + modelDir + "/*";
  LOG(INFO) << "cmd " << cmd;
  ret = system(cmd.c_str());
  EXPECT_EQ(0, ret);
  if (ret) {
    return;
  }

  cmd = " rm -rf " + mergedModelFile;
  LOG(INFO) << "cmd " << cmd;
  ret = system(cmd.c_str());
  EXPECT_EQ(0, ret);
  if (ret) {
    return;
  }

  // clean up
  rmDir(modelDir.c_str());
  remove(mergedModelFile.c_str());
}

int main(int argc, char** argv) {
  initMain(argc, argv);
  initPython(argc, argv);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
