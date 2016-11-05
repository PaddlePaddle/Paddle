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

#include <gtest/gtest.h>
#include <vector>
#include <string>
#include "paddle/gserver/layers/DataLayer.h"
#include "ModelConfig.pb.h"
#include "paddle/trainer/Trainer.h"

#include "TestUtil.h"
#include "LayerGradUtil.h"

using namespace paddle;  // NOLINT
using namespace std;     // NOLINT

P_DECLARE_bool(use_gpu);
P_DECLARE_bool(thread_local_rand_use_global_seed);

void testActivation(const string& act) {
  LOG(INFO) << "test activation: " << act;
  size_t size = 10;
  TestConfig config;
  config.biasSize = 0;
  config.layerConfig.set_type("addto");
  config.layerConfig.set_size(size);
  config.layerConfig.set_active_type(act);
  config.inputDefs.push_back({INPUT_DATA, "layer_0", size, 0});
  config.layerConfig.add_inputs();
  for (auto useGpu : {false, true}) {
    testLayerGrad(config,
                  act + "_activation",
                  100,
                  /* trans= */false,
                  useGpu,
                  /* useWeight */true);
  }
}

TEST(Activation, activation) {
  auto types = ActivationFunction::getAllRegisteredTypes();
  std::set<string> excluded{"sequence_softmax"};
  for (auto type : types) {
    if (excluded.count(type)) continue;
    testActivation(type);
  }
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  initMain(argc, argv);
  FLAGS_thread_local_rand_use_global_seed = true;
  srand(1);
  return RUN_ALL_TESTS();
}
