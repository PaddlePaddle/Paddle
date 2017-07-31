/* Copyright (c) 2017 PaddlePaddle Authors. All Rights Reserve.

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
#include "MkldnnTester.h"
#include "ModelConfig.pb.h"

using namespace paddle;  // NOLINT

DECLARE_bool(thread_local_rand_use_global_seed);
DECLARE_bool(use_gpu);

struct testFCDesc {
  int bs;
  int ic;
  int oc;
  int ih, iw;  // oh == ow == 1
};

void testFcLayer(const testFCDesc& pm) {
  const std::string compareTypes[] = {"mkldnn_fc", "fc"};
  TestConfig cfg;
  cfg.layerConfig.set_type(compareTypes[0]);
  cfg.layerConfig.set_size(pm.oc);
  cfg.inputDefs.push_back(
      {INPUT_DATA,
       "layer_0",
       /* size of input layer= */ size_t(pm.ic * pm.ih * pm.iw),
       /* size of weight= */ size_t(pm.oc * pm.ic * pm.ih * pm.iw)});
  cfg.layerConfig.add_inputs();

  MkldnnTester tester;
  for (auto dnnWgt : {false, true}) {
    cfg.layerConfig.set_init_wgt_from_mkldnn(dnnWgt);
    for (auto biasSize : {pm.oc, 0}) {
      cfg.biasSize = biasSize;
      TestConfig ref = cfg;
      ref.layerConfig.set_type(compareTypes[1]);
      for (auto bs : {pm.bs, 1}) {
        tester.run(cfg, ref, bs, pm.ih, pm.iw);
      }
    }
  }
}

TEST(MkldnnLayer, fcLayer) {
  testFcLayer({2, 2, 3, 1, 1});
  testFcLayer({16, 32, 64, 1, 1});
  testFcLayer({8, 16, 32, 13, 13});
  testFcLayer({4, 12, 18, 13, 11});
  testFcLayer({2, 64, 32, 16, 16});
  testFcLayer({15, 3, 6, 16, 16});
}

// TODO(TJ): add branch test

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  FLAGS_use_gpu = false;
  initMain(argc, argv);
  FLAGS_thread_local_rand_use_global_seed = true;
  srand(1);
  return RUN_ALL_TESTS();
}
