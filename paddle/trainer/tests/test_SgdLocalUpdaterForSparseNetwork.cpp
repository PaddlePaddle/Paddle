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
#include "paddle/pserver/ParameterServer2.h"
#include "paddle/trainer/Trainer.h"
#include "paddle/utils/PythonUtil.h"
#include "paddle/utils/Util.h"

P_DECLARE_bool(local);

static std::unique_ptr<paddle::Trainer> createTrainer(
    bool useGpu, size_t trainerCount, const std::string& configFilename) {
  FLAGS_use_gpu = useGpu;
  FLAGS_trainer_count = trainerCount;
  paddle::Trainer* trainer = new paddle::Trainer();

  trainer->init(paddle::TrainerConfigHelper::create(configFilename));
  return std::unique_ptr<paddle::Trainer>(trainer);
}

TEST(SgdLocalUpdater, RemoteSparseNNCpu) {
  FLAGS_ports_num_for_sparse = 1;
  FLAGS_num_passes = 1;
  FLAGS_local = false;
  std::vector<std::shared_ptr<paddle::ParameterServer2>> pservers;

  for (int i = 0; i < FLAGS_ports_num + FLAGS_ports_num_for_sparse; ++i) {
    auto pserver =
        std::make_shared<paddle::ParameterServer2>("127.0.0.1", FLAGS_port + i);
    pserver->init();
    pserver->start();
    pservers.push_back(pserver);
  }

  auto trainerPtr = createTrainer(false, 1, "sparse_updated_network.py");
  ASSERT_TRUE(trainerPtr != nullptr);
  paddle::Trainer& trainer = *trainerPtr;
  trainer.startTrain();
  trainer.train(1);
  trainer.finishTrain();
}

TEST(SgdLocalUpdater, LocalSparseNNCpu) {
  FLAGS_local = true;
  auto trainerPtr = createTrainer(false, 1, "sparse_updated_network.py");
  ASSERT_TRUE(trainerPtr != nullptr);
  paddle::Trainer& trainer = *trainerPtr;
  trainer.startTrain();
  trainer.train(1);
  trainer.finishTrain();
}
// TEST(SgdLocalUpdater, SparseNNGpu) {
//  auto trainerPtr = createTrainer(true, 1, "sparse_updated_network.py");
//  ASSERT_TRUE(trainerPtr != nullptr);
//  paddle::Trainer& trainer = *trainerPtr;
//  trainer.startTrain();
//  trainer.train(1);
//  trainer.finishTrain();
//}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  paddle::initMain(argc, argv);
  paddle::initPython(argc, argv);
  return RUN_ALL_TESTS();
}
