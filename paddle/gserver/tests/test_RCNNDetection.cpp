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

#include "LayerGradUtil.h"
#include "paddle/testing/TestUtil.h"

using namespace paddle;  // NOLINT
using namespace std;     // NOLINT

// Do one forward pass of priorBox layer and check to see if its output
// matches the given result
void doOneProposalTest(MatrixPtr& inputRoi,
                       MatrixPtr& inputLoc,
                       MatrixPtr& inputConf,
                       real nmsThreshold,
                       real confidenceThreshold,
                       size_t nmsTopk,
                       size_t keepTopK,
                       size_t numClasses,
                       size_t backgroundId,
                       bool use_gpu,
                       MatrixPtr& result) {
  // Setting up the detection output layer
  TestConfig configt;
  configt.layerConfig.set_type("rcnn_detection");
  LayerInputConfig* input = configt.layerConfig.add_inputs();
  configt.layerConfig.add_inputs();
  configt.layerConfig.add_inputs();

  RCNNDetectionConfig* detectionConf = input->mutable_rcnn_detection_conf();
  detectionConf->set_nms_threshold(nmsThreshold);
  detectionConf->set_confidence_threshold(confidenceThreshold);
  detectionConf->set_nms_top_k(nmsTopk);
  detectionConf->set_keep_top_k(keepTopK);
  detectionConf->set_num_classes(numClasses);
  detectionConf->set_background_id(backgroundId);

  configt.inputDefs.push_back({INPUT_DATA_TARGET, "input_roi", 5, 0});
  configt.inputDefs.push_back({INPUT_DATA, "input_loc", numClasses * 4, 0});
  configt.inputDefs.push_back({INPUT_DATA, "input_conf", numClasses, 0});

  // data layer initialize
  std::vector<DataLayerPtr> dataLayers;
  LayerMap layerMap;
  vector<Argument> datas;
  initDataLayer(configt,
                &dataLayers,
                &datas,
                &layerMap,
                "rcnn_detection",
                inputRoi->getHeight(),
                false,
                use_gpu);

  dataLayers[0]->getOutputValue()->copyFrom(*inputRoi);
  dataLayers[1]->getOutputValue()->copyFrom(*inputLoc);
  dataLayers[2]->getOutputValue()->copyFrom(*inputConf);

  // test layer initialize
  bool store_FLAGS_use_gpu = FLAGS_use_gpu;
  FLAGS_use_gpu = use_gpu;
  std::vector<ParameterPtr> parameters;
  LayerPtr rcnnDetectionLayer;
  initTestLayer(configt, &layerMap, &parameters, &rcnnDetectionLayer);
  FLAGS_use_gpu = store_FLAGS_use_gpu;
  rcnnDetectionLayer->forward(PASS_GC);
  checkMatrixEqual(rcnnDetectionLayer->getOutputValue(), result);
}

TEST(Layer, rcnnDetectionLayerFwd) {
  real nmsThreshold = 0.6;
  real confidenceThreshold = 0.3;
  size_t nmsTopk = 2;
  size_t keepTopK = 2;
  size_t numClasses = 3;
  size_t backgroundId = 0;

  MatrixPtr inputRoi;
  MatrixPtr inputLoc;
  MatrixPtr inputConf;
  MatrixPtr result;
  // The format of the RoI is:
  // | batch_idx | xmin | ymin | xmax | ymax |
  real inputRoiData[] = {0, 1, 1, 6, 6, 0, 1, 3, 6, 8, 0, 1, 2, 6, 7};
  real inputLocData[3 * 3 * 4];
  for (size_t i = 0; i < 3 * 3 * 4; ++i) {
    inputLocData[i] = 0.1;
  }
  real inputConfData[] = {0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.2};
  real resultData[] = {0,
                       1,
                       0.51389724,
                       1.28448725,
                       1.28448725,
                       7.91551256,
                       7.91551256,
                       0,
                       2,
                       0.36029661,
                       1.28448725,
                       3.28448725,
                       7.91551256,
                       9.91551304};
  // CPU case.
  bool useGpu = false;
  inputRoi = Matrix::create(3, 5, false, useGpu);
  inputLoc = Matrix::create(3, numClasses * 4, false, useGpu);
  inputConf = Matrix::create(3, numClasses, false, useGpu);
  result = Matrix::create(2, 7, false, useGpu);
  inputRoi->setData(inputRoiData);
  inputLoc->setData(inputLocData);
  inputConf->setData(inputConfData);
  result->setData(resultData);
  doOneProposalTest(inputRoi,
                    inputLoc,
                    inputConf,
                    nmsThreshold,
                    confidenceThreshold,
                    nmsTopk,
                    keepTopK,
                    numClasses,
                    backgroundId,
                    useGpu,
                    result);

#ifndef PADDLE_ONLY_CPU
  // GPU case.
  useGpu = true;
  inputRoi = Matrix::create(3, 5, false, useGpu);
  inputLoc = Matrix::create(3, numClasses * 4, false, useGpu);
  inputConf = Matrix::create(3, numClasses, false, useGpu);
  result = Matrix::create(2, 7, false, useGpu);
  inputRoi->copyFrom(inputLocData, 3 * 5);
  inputLoc->copyFrom(inputConfData, 3 * numClasses * 4);
  inputConf->copyFrom(inputAnchorData, 3 * numClasses);
  result->copyFrom(resultData, 2 * 7);
  doOneProposalTest(inputRoi,
                    inputLoc,
                    inputConf,
                    nmsThreshold,
                    confidenceThreshold,
                    nmsTopk,
                    keepTopK,
                    numClasses,
                    backgroundId,
                    useGpu,
                    result);
#endif
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  initMain(argc, argv);
  return RUN_ALL_TESTS();
}
