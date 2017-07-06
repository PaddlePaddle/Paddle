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
void doOneProposalTest(MatrixPtr& inputLoc,
                       MatrixPtr& inputConf,
                       MatrixPtr& inputAnchor,
                       size_t feature_map_width,
                       size_t feature_map_height,
                       real nms_threshold,
                       bool use_gpu,
                       MatrixPtr& result) {
  // Setting up the detection output layer
  TestConfig configt;
  configt.layerConfig.set_type("proposal");
  LayerInputConfig* input = configt.layerConfig.add_inputs();
  configt.layerConfig.add_inputs();
  configt.layerConfig.add_inputs();

  ProposalConfig* proposalConf = input->mutable_proposal_conf();
  proposalConf->set_width(feature_map_width);
  proposalConf->set_height(feature_map_height);
  proposalConf->set_nms_threshold(nms_threshold);
  proposalConf->set_confidence_threshold(0.01);
  proposalConf->set_nms_top_k(20);
  proposalConf->set_keep_top_k(10);
  proposalConf->set_min_width(0);
  proposalConf->set_min_height(0);
  configt.inputDefs.push_back({INPUT_DATA_TARGET, "anchors", 28, 0});
  configt.inputDefs.push_back({INPUT_DATA, "input_loc", 16, 0});
  configt.inputDefs.push_back({INPUT_DATA, "input_conf", 8, 0});

  // data layer initialize
  std::vector<DataLayerPtr> dataLayers;
  LayerMap layerMap;
  vector<Argument> datas;
  initDataLayer(
      configt, &dataLayers, &datas, &layerMap, "anchors", 1, false, use_gpu);

  dataLayers[0]->getOutputValue()->copyFrom(*inputAnchor);
  dataLayers[1]->getOutputValue()->copyFrom(*inputLoc);
  dataLayers[2]->getOutputValue()->copyFrom(*inputConf);

  // test layer initialize
  bool store_FLAGS_use_gpu = FLAGS_use_gpu;
  FLAGS_use_gpu = use_gpu;
  std::vector<ParameterPtr> parameters;
  LayerPtr proposalLayer;
  initTestLayer(configt, &layerMap, &parameters, &proposalLayer);
  FLAGS_use_gpu = store_FLAGS_use_gpu;
  proposalLayer->forward(PASS_GC);
  checkMatrixEqual(proposalLayer->getOutputValue(), result);
}

TEST(Layer, detectionOutputLayerFwd) {
  bool useGpu = false;
  // CPU case 1.
  MatrixPtr inputLoc;
  MatrixPtr inputConf;
  MatrixPtr inputAnchor;
  MatrixPtr result;
  real nmsThreshold = 0.01;
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
  real inputAnchorData[] = {1, 1, 6, 6, 1, 16, 16, 1, 3, 6, 8, 1, 16, 16,
                            1, 2, 6, 7, 1, 16, 16, 2, 1, 7, 6, 1, 16, 16};
  real resultData[] = {0,
                       1,
                       0.68997448,
                       3.18894059,
                       3.18894059,
                       5.01105940,
                       5.01105940,
                       0,
                       1,
                       0.64565631,
                       3.18894059,
                       5.18894059,
                       5.01105940,
                       7.01105940};
  inputLoc = Matrix::create(1, 16, false, useGpu);
  inputConf = Matrix::create(1, 8, false, useGpu);
  inputAnchor = Matrix::create(1, 28, false, useGpu);
  result = Matrix::create(2, 7, false, useGpu);
  inputLoc->setData(inputLocData);
  inputConf->setData(inputConfData);
  inputAnchor->setData(inputAnchorData);
  result->setData(resultData);
  doOneProposalTest(inputLoc,
                    inputConf,
                    inputAnchor,
                    /* feature_map_width */ 1,
                    /* feature_map_height */ 1,
                    nmsThreshold,
                    useGpu,
                    result);

#ifndef PADDLE_ONLY_CPU
  // GPU case 1.
  useGpu = true;
  inputLoc = Matrix::create(1, 16, false, useGpu);
  inputConf = Matrix::create(1, 8, false, useGpu);
  inputAnchor = Matrix::create(1, 28, false, useGpu);
  inputLoc->copyFrom(inputLocData, 16);
  inputConf->copyFrom(inputConfData, 8);
  inputAnchor->copyFrom(inputAnchorData, 32);

  nmsThreshold = 0.01;
  result = Matrix::create(2, 7, false, useGpu);
  result->copyFrom(resultData, 7);
  doOneProposalTest(inputLoc,
                    inputConf,
                    inputAnchor,
                    /* feature_map_width */ 1,
                    /* feature_map_height */ 1,
                    nmsThreshold,
                    useGpu,
                    result);
#endif
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  initMain(argc, argv);
  return RUN_ALL_TESTS();
}
