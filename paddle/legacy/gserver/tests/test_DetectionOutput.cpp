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

#include "LayerGradUtil.h"
#include "paddle/testing/TestUtil.h"

using namespace paddle;  // NOLINT
using namespace std;     // NOLINT

// Do one forward pass of priorBox layer and check to see if its output
// matches the given result
void doOneDetectionOutputTest(MatrixPtr& inputLoc,
                              MatrixPtr& inputConf,
                              MatrixPtr& inputPriorBox,
                              size_t feature_map_width,
                              size_t feature_map_height,
                              real nms_threshold,
                              bool use_gpu,
                              MatrixPtr& result) {
  // Setting up the detection output layer
  TestConfig configt;
  configt.layerConfig.set_type("detection_output");
  LayerInputConfig* input = configt.layerConfig.add_inputs();
  configt.layerConfig.add_inputs();
  configt.layerConfig.add_inputs();

  DetectionOutputConfig* detOutput = input->mutable_detection_output_conf();
  detOutput->set_width(feature_map_width);
  detOutput->set_height(feature_map_height);
  detOutput->set_nms_threshold(nms_threshold);
  detOutput->set_num_classes(2);
  detOutput->set_nms_top_k(20);
  detOutput->set_keep_top_k(10);
  detOutput->set_background_id(0);
  detOutput->set_confidence_threshold(0.01);
  detOutput->set_input_num(1);
  configt.inputDefs.push_back({INPUT_DATA_TARGET, "priorbox", 32, 0});
  configt.inputDefs.push_back({INPUT_DATA, "input_loc", 16, 0});
  configt.inputDefs.push_back({INPUT_DATA, "input_conf", 8, 0});

  // data layer initialize
  std::vector<DataLayerPtr> dataLayers;
  LayerMap layerMap;
  vector<Argument> datas;
  initDataLayer(
      configt, &dataLayers, &datas, &layerMap, "priorbox", 1, false, use_gpu);

  dataLayers[0]->getOutputValue()->copyFrom(*inputPriorBox);
  dataLayers[1]->getOutputValue()->copyFrom(*inputLoc);
  dataLayers[2]->getOutputValue()->copyFrom(*inputConf);

  // test layer initialize
  bool store_FLAGS_use_gpu = FLAGS_use_gpu;
  FLAGS_use_gpu = use_gpu;
  std::vector<ParameterPtr> parameters;
  LayerPtr detectionOutputLayer;
  initTestLayer(configt, &layerMap, &parameters, &detectionOutputLayer);
  FLAGS_use_gpu = store_FLAGS_use_gpu;
  detectionOutputLayer->forward(PASS_GC);
  checkMatrixEqual(detectionOutputLayer->getOutputValue(), result);
}

TEST(Layer, detectionOutputLayerFwd) {
  bool useGpu = false;
  // CPU case 1.
  MatrixPtr inputLoc;
  MatrixPtr inputConf;
  MatrixPtr inputPriorBox;
  MatrixPtr result, result2, result3, result4;
  real nmsTreshold = 0.01;
  real inputLocData[] = {0.1,
                         0.1,
                         0.1,
                         0.1,
                         0.1,
                         0.1,
                         0.1,
                         0.1,
                         0.1,
                         0.1,
                         0.1,
                         0.1,
                         0.1,
                         0.1,
                         0.1,
                         0.1};
  real inputConfData[] = {0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6};
  real inputPriorBoxData[] = {0.1, 0.1, 0.5, 0.5, 0.1, 0.1, 0.2, 0.2,
                              0.2, 0.2, 0.6, 0.6, 0.1, 0.1, 0.2, 0.2,
                              0.3, 0.3, 0.7, 0.7, 0.1, 0.1, 0.2, 0.2,
                              0.4, 0.4, 0.8, 0.8, 0.1, 0.1, 0.2, 0.2};
  real resultData[] = {
      0, 1, 0.68997443, 0.099959746, 0.099959746, 0.50804031, 0.50804031};
  inputLoc = Matrix::create(1, 16, false, useGpu);
  inputConf = Matrix::create(1, 8, false, useGpu);
  inputPriorBox = Matrix::create(1, 32, false, useGpu);
  result = Matrix::create(1, 7, false, useGpu);
  inputLoc->setData(inputLocData);
  inputConf->setData(inputConfData);
  inputPriorBox->setData(inputPriorBoxData);
  result->setData(resultData);
  doOneDetectionOutputTest(inputLoc,
                           inputConf,
                           inputPriorBox,
                           /* feature_map_width */ 1,
                           /* feature_map_height */ 1,
                           nmsTreshold,
                           useGpu,
                           result);

  // CPU case 2.
  nmsTreshold = 0.2;
  result2 = Matrix::create(2, 7, false, useGpu);
  real resultData2[] = {0,
                        1,
                        0.68997443,
                        0.099959746,
                        0.099959746,
                        0.50804031,
                        0.50804031,
                        0,
                        1,
                        0.59868765,
                        0.29995975,
                        0.29995975,
                        0.70804024,
                        0.70804024};
  result2->setData(resultData2);
  doOneDetectionOutputTest(inputLoc,
                           inputConf,
                           inputPriorBox,
                           /* feature_map_width */ 1,
                           /* feature_map_height */ 1,
                           nmsTreshold,
                           useGpu,
                           result2);

#ifdef PADDLE_WITH_CUDA
  // GPU case 1.
  useGpu = true;
  inputLoc = Matrix::create(1, 16, false, useGpu);
  inputConf = Matrix::create(1, 8, false, useGpu);
  inputPriorBox = Matrix::create(1, 32, false, useGpu);
  inputLoc->copyFrom(inputLocData, 16);
  inputConf->copyFrom(inputConfData, 8);
  inputPriorBox->copyFrom(inputPriorBoxData, 32);

  nmsTreshold = 0.01;
  result3 = Matrix::create(1, 7, false, useGpu);
  result3->copyFrom(resultData, 7);
  doOneDetectionOutputTest(inputLoc,
                           inputConf,
                           inputPriorBox,
                           /* feature_map_width */ 1,
                           /* feature_map_height */ 1,
                           nmsTreshold,
                           useGpu,
                           result3);

  // GPU case 2.
  nmsTreshold = 0.2;
  result4 = Matrix::create(2, 7, false, useGpu);
  result4->copyFrom(resultData2, 14);
  doOneDetectionOutputTest(inputLoc,
                           inputConf,
                           inputPriorBox,
                           /* feature_map_width */ 1,
                           /* feature_map_height */ 1,
                           nmsTreshold,
                           useGpu,
                           result4);
#endif
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  initMain(argc, argv);
  return RUN_ALL_TESTS();
}
