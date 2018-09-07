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
#include <string>
#include <vector>
#include "ModelConfig.pb.h"
#include "paddle/gserver/layers/DataLayer.h"
#include "paddle/math/MathUtils.h"
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

// Test that the convTrans forward is the same as conv backward
TEST(Layer, convTransLayerFwd) {
  // Setting up conv-trans layer
  TestConfig configt;
  configt.biasSize = 3;
  configt.layerConfig.set_type("exconvt");
  configt.layerConfig.set_num_filters(3);
  configt.layerConfig.set_partial_sum(1);
  configt.layerConfig.set_shared_biases(true);

  configt.inputDefs.push_back({INPUT_DATA, "layer_0", 1024, 384});
  LayerInputConfig* input = configt.layerConfig.add_inputs();
  ConvConfig* conv = input->mutable_conv_conf();
  conv->set_filter_size(2);
  conv->set_filter_size_y(4);
  conv->set_channels(16);
  conv->set_padding(0);
  conv->set_padding_y(1);
  conv->set_stride(2);
  conv->set_stride_y(2);
  conv->set_groups(1);
  conv->set_filter_channels(3 / conv->groups());
  conv->set_img_size(16);
  conv->set_output_x(outputSize(conv->img_size(),
                                conv->filter_size(),
                                conv->padding(),
                                conv->stride(),
                                /* caffeMode */ true));
  configt.layerConfig.set_size(conv->img_size() * conv->img_size() *
                               configt.layerConfig.num_filters());
  configt.layerConfig.set_name("convTrans");

  // data layer initialize
  std::vector<DataLayerPtr> dataLayers;
  LayerMap layerMap;
  vector<Argument> datas;
  initDataLayer(
      configt, &dataLayers, &datas, &layerMap, "convTrans", 100, false, false);
  // test layer initialize
  std::vector<ParameterPtr> parameters;
  LayerPtr convtLayer;
  initTestLayer(configt, &layerMap, &parameters, &convtLayer);
  convtLayer->getBiasParameter()->zeroMem();
  convtLayer->forward(PASS_GC);

  // Setting up conv-layer config
  TestConfig config;
  config.biasSize = 16;
  config.layerConfig.set_type("exconv");
  config.layerConfig.set_num_filters(16);
  config.layerConfig.set_partial_sum(1);
  config.layerConfig.set_shared_biases(true);

  config.inputDefs.push_back({INPUT_DATA, "layer_1", 768, 384});
  input = config.layerConfig.add_inputs();
  conv = input->mutable_conv_conf();
  conv->set_filter_size(2);
  conv->set_filter_size_y(4);
  conv->set_channels(3);
  conv->set_padding(0);
  conv->set_padding_y(1);
  conv->set_stride(2);
  conv->set_stride_y(2);
  conv->set_groups(1);
  conv->set_filter_channels(conv->channels() / conv->groups());
  conv->set_img_size(16);
  conv->set_output_x(outputSize(conv->img_size(),
                                conv->filter_size(),
                                conv->padding(),
                                conv->stride(),
                                /* caffeMode */ true));
  config.layerConfig.set_size(conv->output_x() * conv->output_x() *
                              config.layerConfig.num_filters());
  config.layerConfig.set_name("conv");

  // data layer initialize
  std::vector<DataLayerPtr> dataLayers2;
  LayerMap layerMap2;
  vector<Argument> datas2;
  initDataLayer(
      config, &dataLayers2, &datas2, &layerMap2, "conv", 100, false, false);
  // test layer initialize
  std::vector<ParameterPtr> parameters2;
  LayerPtr convLayer;
  initTestLayer(config, &layerMap2, &parameters2, &convLayer);

  // Sync convLayer and convtLayer parameter
  convLayer->getBiasParameter()->zeroMem();
  convLayer->getParameters()[0]
      ->getBuf(PARAMETER_VALUE)
      ->copyFrom(*(convtLayer->getParameters()[0]->getBuf(PARAMETER_VALUE)));

  // Set convLayer outputGrad as convTransLayer input value
  convLayer->forward(PASS_GC);
  convLayer->getOutput().grad->copyFrom(*(dataLayers[0]->getOutputValue()));

  vector<int> callbackFlags(parameters2.size(), 0);
  auto callback = [&](Parameter* para) { ++callbackFlags[para->getID()]; };
  convLayer->backward(callback);

  // Check that the convLayer backward is the same as convTransLayer forward
  checkMatrixEqual(convtLayer->getOutputValue(),
                   dataLayers2[0]->getOutputGrad());
}

// Do one forward pass of convTrans layer and check to see if its output
// matches the given result
void doOneConvtTest(size_t imgSize,
                    size_t output_x,
                    size_t stride,
                    size_t padding,
                    size_t filter_size,
                    MatrixPtr& result) {
  TestConfig configt;
  configt.biasSize = 1;
  configt.layerConfig.set_type("exconvt");
  configt.layerConfig.set_num_filters(1);
  configt.layerConfig.set_partial_sum(1);
  configt.layerConfig.set_shared_biases(true);

  configt.inputDefs.push_back(
      {INPUT_DATA, "layer_0", output_x * output_x, filter_size * filter_size});
  LayerInputConfig* input = configt.layerConfig.add_inputs();
  ConvConfig* conv = input->mutable_conv_conf();
  conv->set_filter_size(filter_size);
  conv->set_filter_size_y(filter_size);
  conv->set_channels(1);
  conv->set_padding(padding);
  conv->set_padding_y(padding);
  conv->set_stride(stride);
  conv->set_stride_y(stride);
  conv->set_groups(1);
  conv->set_filter_channels(1);
  conv->set_img_size(imgSize);
  conv->set_output_x(output_x);

  configt.layerConfig.set_size(conv->img_size() * conv->img_size() *
                               configt.layerConfig.num_filters());
  configt.layerConfig.set_name("convTrans");

  std::vector<DataLayerPtr> dataLayers;
  LayerMap layerMap;
  vector<Argument> datas;
  initDataLayer(
      configt, &dataLayers, &datas, &layerMap, "convTrans", 1, false, false);
  dataLayers[0]->getOutputValue()->zeroMem();
  dataLayers[0]->getOutputValue()->add(1.0);

  // test layer initialize
  std::vector<ParameterPtr> parameters;
  LayerPtr convtLayer;
  initTestLayer(configt, &layerMap, &parameters, &convtLayer);
  convtLayer->getBiasParameter()->zeroMem();
  convtLayer->getParameters()[0]->zeroMem();
  convtLayer->getParameters()[0]->getBuf(PARAMETER_VALUE)->add(1.0);
  convtLayer->forward(PASS_GC);

  checkMatrixEqual(convtLayer->getOutputValue(), result);
}

TEST(Layer, convTransLayerFwd2) {
  MatrixPtr result;
  result = Matrix::create(1, 5 * 5, false, false);
  result->zeroMem();
  result->add(1.0);
  doOneConvtTest(/* imgSize */ 5,
                 /* output_x */ 1,
                 /* stride */ 1,
                 /* padding */ 0,
                 /* filter_size */ 5,
                 result);

  real resultData[] = {1, 2, 2, 2, 1, 2, 4, 4, 4, 2, 2, 4, 4,
                       4, 2, 2, 4, 4, 4, 2, 1, 2, 2, 2, 1};
  result->setData(resultData);
  doOneConvtTest(/* imgSize */ 5,
                 /* output_x */ 2,
                 /* stride */ 1,
                 /* padding */ 0,
                 /* filter_size */ 4,
                 result);

  real resultData2[] = {1, 2, 2, 2, 1, 2, 4, 4, 4, 2, 2, 4, 4,
                        4, 2, 2, 4, 4, 4, 2, 1, 2, 2, 2, 1};
  result->setData(resultData2);
  doOneConvtTest(/* imgSize */ 5,
                 /* output_x */ 2,
                 /* stride */ 2,
                 /* padding */ 1,
                 /* filter_size */ 5,
                 result);

  real resultData3[] = {1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 2, 2, 4,
                        2, 2, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1};
  result->setData(resultData3);
  doOneConvtTest(/* imgSize */ 5,
                 /* output_x */ 2,
                 /* stride */ 2,
                 /* padding */ 0,
                 /* filter_size */ 3,
                 result);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  initMain(argc, argv);
  FLAGS_thread_local_rand_use_global_seed = true;
  srand(1);
  return RUN_ALL_TESTS();
}
