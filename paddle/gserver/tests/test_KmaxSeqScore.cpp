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
#include <string>
#include <vector>
#include "ModelConfig.pb.h"
#include "paddle/gserver/layers/DataLayer.h"
#include "paddle/trainer/Trainer.h"
#include "paddle/utils/GlobalConstants.h"

#include "LayerGradUtil.h"
#include "paddle/testing/TestUtil.h"

using namespace paddle;  // NOLINT
using namespace std;     // NOLINT

DECLARE_bool(use_gpu);
DECLARE_int32(gpu_id);
DECLARE_bool(thread_local_rand_use_global_seed);

// Test that the batchNormLayer can be followed by a ConvLayer
TEST(Layer, kmaxSeqScoreLayer) {
  for (auto hasSubseq : {true, false}) {
    for (auto useGpu : {true, false}) {
      TestConfig config;
      config.layerConfig.set_type("kmax_seq_score");
      config.inputDefs.push_back(
          {hasSubseq ? INPUT_HASSUB_SEQUENCE_DATA : INPUT_SEQUENCE_DATA,
           "layer_0",
           1,
           0});
      config.layerConfig.add_inputs();

      // data layer initialize
      std::vector<DataLayerPtr> dataLayers;
      LayerMap layerMap;
      vector<Argument> datas;
      initDataLayer(config,
                    &dataLayers,
                    &datas,
                    &layerMap,
                    "kmax_seq_score",
                    100,
                    false,
                    useGpu);
      // test layer initialize
      std::vector<ParameterPtr> parameters;
      LayerPtr kmaxSeqScoreLayer;
      initTestLayer(config, &layerMap, &parameters, &kmaxSeqScoreLayer);
      kmaxSeqScoreLayer->forward(PASS_TRAIN);
    }
  }
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  initMain(argc, argv);
  FLAGS_thread_local_rand_use_global_seed = true;
  srand(1);
  return RUN_ALL_TESTS();
}
