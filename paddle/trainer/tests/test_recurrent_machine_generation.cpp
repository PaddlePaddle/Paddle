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

#include <fstream>

#include <paddle/trainer/Trainer.h>
#include <paddle/utils/PythonUtil.h>

#include <gtest/gtest.h>

using namespace paddle;  // NOLINT
using namespace std;     // NOLINT

static const string& CONFIG_FILE = "trainer/tests/sample_trainer_rnn_gen.conf";
static const string& NEST_CONFIG_FILE =
    "trainer/tests/sample_trainer_nest_rnn_gen.conf";
static const string& OUTPUT_DIR = "trainer/tests/dump_text.test";
static string modelDir = "trainer/tests/rnn_gen_test_model_dir/t1";  // NOLINT
static string expectFile =                                           // NOLINT
    "trainer/tests/rnn_gen_test_model_dir/r1.test";                  // NOLINT

DECLARE_string(config_args);

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
                   const size_t batchSize,
                   bool useGpu,
                   bool hasSubseq) {
  inArgs.clear();
  // sentence id
  Argument sentId;
  sentId.value = nullptr;
  if (hasSubseq) {
    // as there is only one sequence, there is only one label.
    IVector::resizeOrCreate(sentId.ids, 1, useGpu);
    sentId.ids->setElement(0, 0);
  } else {
    // as there is batchSize word, there is batchSize label.
    IVector::resizeOrCreate(sentId.ids, batchSize, useGpu);
    for (size_t i = 0; i < batchSize; ++i) sentId.ids->setElement(i, i);
  }
  inArgs.emplace_back(sentId);

  // a dummy layer to decide batch size
  Argument dummyInput;
  dummyInput.value = Matrix::create(batchSize, 2, false, useGpu);
  dummyInput.value->randomizeUniform();
  if (hasSubseq) {
    // generate one sequence with batchSize subsequence,
    // and each subsequence has only one word.
    dummyInput.sequenceStartPositions = ICpuGpuVector::create(2, false);
    int* buf = dummyInput.sequenceStartPositions->getMutableData(false);
    dummyInput.subSequenceStartPositions =
        ICpuGpuVector::create(batchSize + 1, false);
    int* subBuf = dummyInput.subSequenceStartPositions->getMutableData(false);
    buf[0] = 0;
    buf[1] = batchSize;
    for (size_t i = 0; i < batchSize + 1; i++) subBuf[i] = i;
  }
  inArgs.emplace_back(dummyInput);
}

void testGeneration(const string& configFile,
                    bool useGpu,
                    bool hasSubseq,
                    const string& expRetFile) {
  FLAGS_use_gpu = useGpu;
  auto config = std::make_shared<TrainerConfigHelper>(configFile);
  unique_ptr<GradientMachine> gradientMachine(GradientMachine::create(*config));
  gradientMachine->loadParameters(modelDir);
  vector<Argument> inArgs(2);

  const size_t batchSize = 15;
  prepareInArgs(inArgs, batchSize, useGpu, hasSubseq);
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
#ifndef PADDLE_WITH_CUDA
  const auto useGpuConfs = {false};
#else
  const auto useGpuConfs = {true, false};
#endif
  auto testGen = [&](const string& configFile,
                     bool hasSubseq,
                     const string& expRetFile,
                     bool beam_search) {
    FLAGS_config_args = beam_search ? "beam_search=1" : "beam_search=0";
    for (auto useGpu : useGpuConfs) {
      LOG(INFO) << configFile << " useGpu=" << useGpu
                << " beam_search=" << beam_search;
      testGeneration(configFile, useGpu, hasSubseq, expRetFile);
    }
  };
  testGen(CONFIG_FILE, false, expectFile + ".nobeam", false);  // no beam search
  testGen(CONFIG_FILE, false, expectFile + ".beam", true);     // beam search
  // In hierarchical RNN, beam search and one way search are only in inner-RNN,
  // outer-RNN will concat the generated inner-results (first for beam search)
  // from inner-RNN. Thus, they have the same outer-results.
  testGen(NEST_CONFIG_FILE,
          true,
          expectFile + ".nest",
          false);  // no beam search
  testGen(NEST_CONFIG_FILE, true, expectFile + ".nest", true);  // beam search
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
