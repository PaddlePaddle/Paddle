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


#include <fstream>

#include <paddle/utils/PythonUtil.h>
#include <paddle/trainer/Trainer.h>

#include <gtest/gtest.h>

using namespace paddle;  // NOLINT
using namespace std;     // NOLINT

static const string& CONFIG_FILE = "trainer/tests/sample_trainer_rnn_gen.conf";
static const string& OUTPUT_DIR = "trainer/tests/dump_text.test";
static string modelDir = "trainer/tests/rnn_gen_test_model_dir/t1";  // NOLINT
static string expectFile =                                           // NOLINT
    "trainer/tests/rnn_gen_test_model_dir/r1.test";                  // NOLINT

P_DECLARE_string(config_args);

vector<float> readRetFile(const string& fname) {
  ifstream inFile(fname);
  float ret;
  vector<float> nums;
  while (inFile >> ret) {
    nums.push_back(ret);
  }
  return nums;
}

void checkOutput(const string& expRetFile) {
  vector<float> rets = readRetFile(OUTPUT_DIR);
  vector<float> expRets = readRetFile(expRetFile);
  EXPECT_EQ(rets.size(), expRets.size());
  for (size_t i = 0; i < rets.size(); i++) {
    EXPECT_FLOAT_EQ(rets[i], expRets[i]);
  }
}

void prepareInArgs(vector<Argument>& inArgs,
                   const size_t batchSize, bool useGpu) {
  inArgs.clear();
  // sentence id
  Argument sentId;
  sentId.value = nullptr;
  IVector::resizeOrCreate(sentId.ids, batchSize, useGpu);
  for (size_t i = 0; i < batchSize; ++i) sentId.ids->setElement(i, i);
  inArgs.emplace_back(sentId);

  // a dummy layer to decide batch size
  Argument dummyInput;
  dummyInput.value = Matrix::create(batchSize, 2, false, useGpu);
  dummyInput.value->randomizeUniform();
  inArgs.emplace_back(dummyInput);
}

void testGeneration(bool useGpu, const string& expRetFile) {
  FLAGS_use_gpu = useGpu;
  auto config = std::make_shared<TrainerConfigHelper>(CONFIG_FILE);
  unique_ptr<GradientMachine> gradientMachine(GradientMachine::create(*config));
  gradientMachine->loadParameters(modelDir);
  vector<Argument> inArgs(2);

  const size_t batchSize = 15;
  prepareInArgs(inArgs, batchSize, useGpu);
  vector<Argument> outArgs;
  unique_ptr<Evaluator> testEvaluator(gradientMachine->makeEvaluator());
  testEvaluator->start();
  gradientMachine->forward(inArgs, &outArgs, PASS_TEST);
  gradientMachine->eval(testEvaluator.get());
  testEvaluator->finish();
  checkOutput(expRetFile);
}

#ifndef PADDLE_TYPE_DOUBLE

TEST(RecurrentGradientMachine, test_generation) {
#ifdef PADDLE_ONLY_CPU
  const auto useGpuConfs = {false};
#else
  const auto useGpuConfs = {true, false};
#endif
  FLAGS_config_args = "beam_search=0";  // no beam search
  string expectRetFileNoBeam = expectFile + ".nobeam";
  for (auto useGpu : useGpuConfs) {
    testGeneration(useGpu, expectRetFileNoBeam);
  }
  FLAGS_config_args = "beam_search=1";  // no beam search
  string expectRetFileBeam = expectFile + ".beam";
  for (auto useGpu : useGpuConfs) {
    testGeneration(useGpu, expectRetFileBeam);
  }
}
#endif

int main(int argc, char** argv) {
  initMain(argc, argv);
  initPython(argc, argv);
  CHECK(argc == 1 || argc == 3);
  if (argc == 3) {
    modelDir = argv[1];
    expectFile = argv[2];
  }
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
