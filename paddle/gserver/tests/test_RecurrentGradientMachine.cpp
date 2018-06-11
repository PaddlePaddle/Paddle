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

#include <gtest/gtest.h>
#include <paddle/gserver/gradientmachines/GradientMachine.h>
#include <paddle/parameter/ParameterUpdateFunctions.h>
#include <paddle/trainer/Trainer.h>
#include <paddle/trainer/TrainerInternal.h>
#include <paddle/utils/PythonUtil.h>
#include <paddle/utils/Util.h>
#include <paddle/utils/Version.h>

DECLARE_int32(seed);

using namespace paddle;  // NOLINT
using namespace std;     // NOLINT
class TrainerForTest : public paddle::Trainer {
 public:
  void startTrain() {
    GradientMachine& gm = *this->trainerInternal_.getGradientMachine();
    gm.start();
  }

  void finishTrain() {
    GradientMachine& gm = *this->trainerInternal_.getGradientMachine();
    gm.finish();
  }

  /**
   * Get total dimension of all parameters.
   *
   * @return the total dimension of all parameters
   */
  size_t getTotalParameterSize() const {
    auto p = const_cast<TrainerForTest*>(this);
    auto& params = p->getGradientMachine()->getParameters();
    return std::accumulate(
        params.begin(), params.end(), 0UL, [](size_t a, const ParameterPtr& p) {
          return a + p->getSize();
        });
  }
};

void CalCost(const string& conf,
             const string& dir,
             real* cost,
             int num_passes) {
  auto config = std::make_shared<TrainerConfigHelper>(conf);
  TrainerForTest trainer;
  trainer.init(config);
  mkDir(dir.c_str());
  config->setSaveDir(dir);
  auto dataProvider = trainer.getDataProvider();
  int32_t batchSize = config->getOptConfig().batch_size();
  real learningRate = config->getOptConfig().learning_rate();
  real momentum = 0;
  real decayRate = 0;
  int64_t dim = trainer.getTotalParameterSize();
  CpuVector vecW(dim);
  CpuVector vecGradient(dim);
  CpuVector vecMomentum(dim);

  // vecW needs to be assigned, otherwise the variable is an uncertain value.

  *ThreadLocalRand::getSeed() = FLAGS_seed;
  vecW.randnorm(0, 0.1);
  vecMomentum.randnorm(0, 0.1);

  trainer.startTrain();
  for (int i = 0; i < num_passes; ++i) {
    real totalCost = 0;
    dataProvider->reset();
    while (true) {
      DataBatch dataBatch;
      int num = dataProvider->getNextBatch(batchSize, &dataBatch);
      if (num == 0) break;
      totalCost += trainer.calcGradient(dataBatch, vecW, vecGradient);
      sgdUpdate(
          learningRate, momentum, decayRate, &vecW, &vecGradient, &vecMomentum);
    }
    cost[i] = totalCost;
  }
  trainer.finishTrain();
  rmDir(dir.c_str());
}

void test(const string& conf1, const string& conf2, double eps, bool useGpu) {
  if (!paddle::version::isWithGpu() && useGpu) {
    return;
  }
  FLAGS_use_gpu = useGpu;
  int num_passes = 5;
  real* cost1 = new real[num_passes];
  const string dir1 = "gserver/tests/t1";
  CalCost(conf1, dir1, cost1, num_passes);

  real* cost2 = new real[num_passes];
  const string dir2 = "gserver/tests/t2";
  CalCost(conf2, dir2, cost2, num_passes);

  for (int i = 0; i < num_passes; i++) {
    LOG(INFO) << "num_passes: " << i << ", cost1=" << cost1[i]
              << ", cost2=" << cost2[i]
              << ", diff=" << std::abs(cost1[i] - cost2[i]);
    ASSERT_NEAR(cost1[i], cost2[i], eps);
  }
  delete[] cost1;
  delete[] cost2;
}

TEST(RecurrentGradientMachine, HasSubSequence) {
  for (bool useGpu : {false, true}) {
    test("gserver/tests/sequence_layer_group.conf",
         "gserver/tests/sequence_nest_layer_group.conf",
         1e-5,
         useGpu);
  }
}

TEST(RecurrentGradientMachine, rnn) {
  for (bool useGpu : {false, true}) {
    test("gserver/tests/sequence_rnn.conf",
         "gserver/tests/sequence_nest_rnn.conf",
         1e-6,
         useGpu);
  }
}

TEST(RecurrentGradientMachine, rnn_multi_input) {
  for (bool useGpu : {false, true}) {
    test("gserver/tests/sequence_rnn_multi_input.conf",
         "gserver/tests/sequence_nest_rnn_multi_input.conf",
         1e-6,
         useGpu);
  }
}

TEST(RecurrentGradientMachine, rnn_multi_unequalength_input) {
  for (bool useGpu : {false, true}) {
    test("gserver/tests/sequence_rnn_multi_unequalength_inputs.py",
         "gserver/tests/sequence_nest_rnn_multi_unequalength_inputs.py",
         1e-6,
         useGpu);
  }
}

TEST(RecurrentGradientMachine, rnn_mixed_input) {
  for (bool useGpu : {false, true}) {
    test("gserver/tests/sequence_rnn_mixed_inputs.py",
         "gserver/tests/sequence_rnn_matched_inputs.py",
         1e-6,
         useGpu);
  }
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);

  if (paddle::version::isWithPyDataProvider()) {
    if (!paddle::version::isWithGpu()) {
      FLAGS_use_gpu = false;
    }
    initMain(argc, argv);
    initPython(argc, argv);
    return RUN_ALL_TESTS();
  } else {
    return 0;
  }
}
