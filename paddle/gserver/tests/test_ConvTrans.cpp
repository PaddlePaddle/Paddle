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
#include "paddle/utils/GlobalConstants.h"
#include "paddle/gserver/layers/ExpandConvTransLayer.h"

#include "TestUtil.h"
#include "LayerGradUtil.h"

using namespace paddle;  // NOLINT
using namespace std;     // NOLINT

P_DECLARE_bool(use_gpu);
P_DECLARE_int32(gpu_id);
P_DECLARE_double(checkgrad_eps);
P_DECLARE_bool(thread_local_rand_use_global_seed);
P_DECLARE_bool(prev_batch_state);

TEST(Layer, convTransLayerFwd) {
    TestConfig configt;
    configt.biasSize = 3;
    configt.layerConfig.set_type("exconvt");
    configt.layerConfig.set_num_filters(3);
    configt.layerConfig.set_partial_sum(1);
    configt.layerConfig.set_shared_biases(true);

    configt.inputDefs.push_back({INPUT_DATA, "layer_0", 1024, 288});
    LayerInputConfig* input = configt.layerConfig.add_inputs();
    ConvConfig* conv = input->mutable_conv_conf();
    conv->set_filter_size(2);
    conv->set_filter_size_y(3);
    conv->set_channels(16);
    conv->set_padding(0);
    conv->set_padding_y(1);
    conv->set_stride(2);
    conv->set_stride_y(2);
    conv->set_groups(1);
    conv->set_filter_channels(3 / conv->groups());
    conv->set_img_size(16);
    conv->set_output_x(
        (2 * conv->padding() + conv->img_size() - conv->filter_size()) /
            ((float)conv->stride()) +
        1.5);

    configt.layerConfig.set_size(conv->img_size() * conv->img_size() *
                                configt.layerConfig.num_filters());
    configt.layerConfig.set_name("convTrans");

    // data layer initialize
    std::vector<DataLayerPtr> dataLayers;
    LayerMap layerMap;
    vector<Argument> datas;
    initDataLayer(configt, &dataLayers, &datas, &layerMap, "convTrans",
                  100, false, useGpu);
    // test layer initialize
    std::vector<ParameterPtr> parameters;
    LayerPtr convtLayer;
    initTestLayer(configt, &layerMap, &parameters, &convtLayer);
    convtLayer->getBiasParameter()->zeroMem();
    convtLayer->forward(PASS_GC);

    TestConfig config;
    config.biasSize = 16;
    config.layerConfig.set_type("exconv");
    config.layerConfig.set_num_filters(16);
    config.layerConfig.set_partial_sum(1);
    config.layerConfig.set_shared_biases(true);

    config.inputDefs.push_back({INPUT_DATA, "layer_1", 768, 288});
    input = config.layerConfig.add_inputs();
    conv = input->mutable_conv_conf();
    conv->set_filter_size(2);
    conv->set_filter_size_y(3);
    conv->set_channels(3);
    conv->set_padding(0);
    conv->set_padding_y(1);
    conv->set_stride(2);
    conv->set_stride_y(2);
    conv->set_groups(1);
    conv->set_filter_channels(conv->channels() / conv->groups());
    conv->set_img_size(16);
    conv->set_output_x(
        (2 * conv->padding() + conv->img_size() - conv->filter_size()) /
            ((float)conv->stride()) +
        1.5);
    config.layerConfig.set_size(conv->output_x() * conv->output_x() *
                                config.layerConfig.num_filters());
    config.layerConfig.set_name("conv");

    // data layer initialize
    std::vector<DataLayerPtr> dataLayers2;
    LayerMap layerMap2;
    vector<Argument> datas2;
    initDataLayer(config, &dataLayers2, &datas2, &layerMap2, "conv",
                  100, false, useGpu);
    // test layer initialize
    std::vector<ParameterPtr> parameters2;
    LayerPtr convLayer;
    initTestLayer(config, &layerMap2, &parameters2, &convLayer);

    convLayer->getBiasParameter()->zeroMem();
    convLayer->getParameters()[0]->getBuf(PARAMETER_VALUE)->copyFrom(
            *(convtLayer->getParameters()[0]->getBuf(PARAMETER_VALUE)));

    convLayer->forward(PASS_GC);
    convLayer->getOutput().grad->copyFrom(*(dataLayers[0]->getOutputValue()));

    vector<int> callbackFlags(parameters2.size(), 0);
    auto callback = [&](Parameter* para) { ++callbackFlags[para->getID()]; };
    convLayer->backward(callback);

    checkMatrixEqual(convtLayer->getOutputValue(),
                     dataLayers2[0]->getOutputGrad());
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  initMain(argc, argv);
  FLAGS_thread_local_rand_use_global_seed = true;
  srand(1);
  return RUN_ALL_TESTS();
}
