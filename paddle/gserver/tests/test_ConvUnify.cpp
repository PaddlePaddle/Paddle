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
#include "paddle/math/MathUtils.h"
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

// Do one forward pass of convTrans layer and check to see if its output
// matches the given result
MatrixPtr doOneConvTest(size_t imgSize,
                        size_t output_x,
                        size_t stride,
                        size_t padding,
                        size_t filter_size,
                        size_t channel,
                        size_t numfilters,
                        size_t groups,
                        MatrixPtr& inputData,
                        real* param,
                        bool useGpu) {
  TestConfig config;
  config.biasSize = numfilters;
  if (useGpu) {
    config.layerConfig.set_type("cudnn_conv");
  } else {
    config.layerConfig.set_type("exconv");
  }
  config.layerConfig.set_num_filters(numfilters);
  config.layerConfig.set_partial_sum(1);
  config.layerConfig.set_shared_biases(true);

  size_t weightSize = channel * filter_size * filter_size *
                      config.layerConfig.num_filters() / groups;
  config.inputDefs.push_back(
      {INPUT_DATA, "layer_0", imgSize * imgSize * channel, weightSize});
  LayerInputConfig* input = config.layerConfig.add_inputs();
  ConvConfig* conv = input->mutable_conv_conf();
  conv->set_filter_size(filter_size);
  conv->set_filter_size_y(filter_size);
  conv->set_channels(channel);
  conv->set_padding(padding);
  conv->set_padding_y(padding);
  conv->set_stride(stride);
  conv->set_stride_y(stride);
  conv->set_groups(groups);
  conv->set_filter_channels(channel / groups);
  conv->set_img_size(imgSize);
  conv->set_output_x(output_x);

  config.layerConfig.set_size(conv->output_x() * conv->output_x() *
                              config.layerConfig.num_filters());
  config.layerConfig.set_name("conv");

  std::vector<DataLayerPtr> dataLayers;
  LayerMap layerMap;
  vector<Argument> datas;
  initDataLayer(
      config, &dataLayers, &datas, &layerMap, "conv", 1, false, useGpu);
  dataLayers[0]->getOutputValue()->zeroMem();
  dataLayers[0]->getOutputValue()->copyFrom(*inputData);

  // test layer initialize
  std::vector<ParameterPtr> parameters;
  LayerPtr convLayer;
  initTestLayer(config, &layerMap, &parameters, &convLayer);
  convLayer->getBiasParameter()->zeroMem();
  convLayer->getParameters()[0]->zeroMem();
  convLayer->getParameters()[0]
      ->getBuf(PARAMETER_VALUE)
      ->copyFrom(param, weightSize);
  convLayer->forward(PASS_GC);

  return convLayer->getOutputValue();
}

TEST(Layer, convParaUnified) {
#ifndef PADDLE_ONLY_CPU
  MatrixPtr input, resultCpu, resultGpu;
  input = Matrix::create(1, 4 * 4, false, false);
  real inputData[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  real param[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 8, 7, 6, 5, 4, 3, 2, 1};

  input->setData(inputData);

  resultCpu = doOneConvTest(/* imgSize */ 4,
                            /* output_x */ 2,
                            /* stride */ 1,
                            /* padding */ 0,
                            /* filter_size */ 3,
                            /*channel*/ 1,
                            /*numfilters*/ 2,
                            /*groups*/ 1,
                            input,
                            param,
                            false);

  resultGpu = doOneConvTest(/* imgSize */ 4,
                            /* output_x */ 2,
                            /* stride */ 1,
                            /* padding */ 0,
                            /* filter_size */ 3,
                            /*channel*/ 1,
                            /*numfilters*/ 2,
                            /*groups*/ 1,
                            input,
                            param,
                            true);
  checkMatrixEqual(resultCpu, resultGpu);

  input = Matrix::create(1, 3 * 3 * 2, false, false);
  real inputData2[] = {
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};
  real param2[] = {1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6, 5, 4, 3, 2, 1};

  input->setData(inputData2);

  resultCpu = doOneConvTest(/* imgSize */ 3,
                            /* output_x */ 2,
                            /* stride */ 1,
                            /* padding */ 0,
                            /* filter_size */ 2,
                            /*channel*/ 2,
                            /*numfilters*/ 2,
                            /*groups*/ 1,
                            input,
                            param2,
                            false);

  resultGpu = doOneConvTest(/* imgSize */ 3,
                            /* output_x */ 2,
                            /* stride */ 1,
                            /* padding */ 0,
                            /* filter_size */ 2,
                            /*channel*/ 2,
                            /*numfilters*/ 2,
                            /*groups*/ 1,
                            input,
                            param2,
                            true);
  checkMatrixEqual(resultCpu, resultGpu);

  real param3[] = {1, 2, 3, 4, 4, 3, 2, 1};

  resultCpu = doOneConvTest(/* imgSize */ 3,
                            /* output_x */ 2,
                            /* stride */ 1,
                            /* padding */ 0,
                            /* filter_size */ 2,
                            /*channel*/ 2,
                            /*numfilters*/ 2,
                            /*groups*/ 2,
                            input,
                            param3,
                            false);

  resultGpu = doOneConvTest(/* imgSize */ 3,
                            /* output_x */ 2,
                            /* stride */ 1,
                            /* padding */ 0,
                            /* filter_size */ 2,
                            /*channel*/ 2,
                            /*numfilters*/ 2,
                            /*groups*/ 2,
                            input,
                            param3,
                            true);
  checkMatrixEqual(resultCpu, resultGpu);
#endif
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  initMain(argc, argv);
  FLAGS_thread_local_rand_use_global_seed = true;
  srand(1);
  return RUN_ALL_TESTS();
}
