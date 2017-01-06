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
#include "paddle/gserver/layers/ExpandConvTransLayer.h"
#include "paddle/trainer/Trainer.h"
#include "paddle/utils/GlobalConstants.h"

#include "LayerGradUtil.h"
#include "paddle/testing/TestUtil.h"

using namespace paddle;  // NOLINT
using namespace std;     // NOLINT

DECLARE_bool(use_gpu);
DECLARE_int32(gpu_id);
DECLARE_double(checkgrad_eps);
DECLARE_bool(thread_local_rand_use_global_seed);
DECLARE_bool(prev_batch_state);

// Test that the batchNormLayer can be followed by a ConvLayer
TEST(Layer, batchNorm) {
  FLAGS_use_gpu = false;
  TestConfig configBN;
  const int CHANNELS = 6272;
  const int IMG_SIZE = 1;
  configBN.layerConfig.set_type("batch_norm");
  configBN.layerConfig.set_name("bn");
  configBN.layerConfig.set_size(CHANNELS * IMG_SIZE * IMG_SIZE);
  configBN.layerConfig.set_active_type("relu");
  configBN.biasSize = CHANNELS;
  configBN.inputDefs.push_back({INPUT_DATA,
                                "layer_0",
                                /* dim= */ IMG_SIZE * IMG_SIZE * CHANNELS,
                                /* paraSize= */ CHANNELS});

  configBN.inputDefs.push_back(
      {INPUT_DATA, "layer_1_running_mean", 1, CHANNELS});
  configBN.inputDefs.back().isStatic = true;
  configBN.inputDefs.push_back(
      {INPUT_DATA, "layer_2_running_var", 1, CHANNELS});
  configBN.inputDefs.back().isStatic = true;

  LayerInputConfig* input = configBN.layerConfig.add_inputs();
  configBN.layerConfig.add_inputs();
  configBN.layerConfig.add_inputs();

  ImageConfig* img_conf = input->mutable_image_conf();
  img_conf->set_channels(CHANNELS);
  img_conf->set_img_size(IMG_SIZE);

  // Setting up conv-layer config
  TestConfig config;
  config.biasSize = 64;
  config.layerConfig.set_type("exconv");
  config.layerConfig.set_num_filters(64);
  config.layerConfig.set_partial_sum(1);
  config.layerConfig.set_shared_biases(true);

  config.inputDefs.push_back({INPUT_DATA, "bn", 6272, 204800});
  input = config.layerConfig.add_inputs();
  ConvConfig* conv = input->mutable_conv_conf();
  conv->set_filter_size(5);
  conv->set_filter_size_y(5);
  conv->set_channels(128);
  conv->set_padding(1);
  conv->set_padding_y(1);
  conv->set_stride(2);
  conv->set_stride_y(2);
  conv->set_groups(1);
  conv->set_filter_channels(conv->channels() / conv->groups());
  conv->set_img_size(7);
  conv->set_output_x(3);
  config.layerConfig.set_size(conv->output_x() * conv->output_x() *
                              config.layerConfig.num_filters());
  config.layerConfig.set_name("conv");

  // data layer initialize
  std::vector<DataLayerPtr> dataLayers;
  LayerMap layerMap;
  vector<Argument> datas;
  initDataLayer(configBN,
                &dataLayers,
                &datas,
                &layerMap,
                "batch_norm",
                100,
                false,
                false);
  // test layer initialize
  std::vector<ParameterPtr> parameters;
  LayerPtr bnLayer;
  initTestLayer(configBN, &layerMap, &parameters, &bnLayer);

  std::vector<ParameterPtr> parameters2;
  LayerPtr convLayer;
  initTestLayer(config, &layerMap, &parameters2, &convLayer);

  bnLayer->forward(PASS_GC);
  convLayer->forward(PASS_GC);

  CHECK_EQ(static_cast<int>(convLayer->getOutputValue()->getHeight()), 100);
  CHECK_EQ(static_cast<int>(convLayer->getOutputValue()->getWidth()), 576);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  initMain(argc, argv);
  FLAGS_thread_local_rand_use_global_seed = true;
  srand(1);
  return RUN_ALL_TESTS();
}
