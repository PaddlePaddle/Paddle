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

// Do one forward pass of expand layer and check to see if its output
// matches the given result.(Test onlyCPU currently.)
void doOneExpandTest(string trans_type,
                     bool hasSubseq,
                     bool useGpu,
                     Argument& input1,
                     Argument& input2,
                     Argument& result) {
  FLAGS_use_gpu = false;
  // Setting up the expand layer
  TestConfig config;
  config.layerConfig.set_type("expand");

  auto inputType1 =
      trans_type == "non-seq" ? INPUT_DENSE_DIM_DATA : INPUT_SEQUENCE_DATA;
  config.inputDefs.push_back({inputType1, "layer0", 1, 0});
  auto inputType2 =
      hasSubseq ? INPUT_HASSUB_SEQUENCE_DATA : INPUT_SEQUENCE_DATA;

  config.inputDefs.push_back({inputType2, "layer1", 1, 0});
  config.layerConfig.add_inputs();
  config.layerConfig.add_inputs();
  config.layerConfig.set_trans_type(trans_type);

  // data layer initialize
  std::vector<DataLayerPtr> dataLayers;
  LayerMap layerMap;
  vector<Argument> datas;
  initDataLayer(
      config, &dataLayers, &datas, &layerMap, "expand", 1, false, useGpu);
  dataLayers[0]->getOutput() = input1;
  dataLayers[1]->getOutput() = input2;

  // test layer initialize
  std::vector<ParameterPtr> parameters;
  LayerPtr expandLayer;
  initTestLayer(config, &layerMap, &parameters, &expandLayer);
  expandLayer->forward(PASS_GC);
  checkMatrixEqual(expandLayer->getOutputValue(), result.value);
}

TEST(Layer, ExpandLayerFwd) {
  bool useGpu = false;

  // Assume batch_size =3 in all cases.

  // CPU case 1. non-seq expand to seq
  // input1 = 1,2,3
  // input2 = [4,5],[6],[7,8,9]
  // result = [1,1],[2],[3,3,3]
  Argument input1, input2, result;
  input1.value = Matrix::create(3, 1, false, useGpu);
  real input1Data[] = {1, 2, 3};
  input1.value->setData(input1Data);

  input2.value = Matrix::create(6, 1, false, useGpu);
  real input2Data[] = {4, 5, 6, 7, 8, 9};
  input2.value->setData(input2Data);
  input2.sequenceStartPositions = ICpuGpuVector::create(4, useGpu);
  int input2Seq[] = {0, 2, 3, 6};
  input2.sequenceStartPositions->copyFrom(input2Seq, 4, useGpu);

  result.value = Matrix::create(6, 1, false, useGpu);
  real resultData[] = {1, 1, 2, 3, 3, 3};
  result.value->setData(resultData);

  doOneExpandTest("non-seq", false, useGpu, input1, input2, result);

  // CPU case 2. non-seq expand to sub-seq
  // NOTE: input1.batch_size == input2.sequencelength in this case.
  // i.e, input1 expands by input2.sequence
  // input1 = 1,2,3
  // input2 = [[4,5]],[[6]],[[7],[8,9]]
  // result = [[1,1]],[[2]],[[3],[3,3]]
  input2.subSequenceStartPositions = ICpuGpuVector::create(5, useGpu);
  int input2SubSeq[] = {0, 2, 3, 4, 6};
  input2.subSequenceStartPositions->copyFrom(input2SubSeq, 5, useGpu);

  doOneExpandTest("non-seq", true, useGpu, input1, input2, result);

  // CPU case 3. seq expand to sub-seq
  // input1 = [1,2],[3],[4]
  // input2 = [[4,5]],[[6]],[[7],[8,9]]
  // result = [[1,1]],[[2]],[[3],[4,4]]
  Matrix::resizeOrCreate(input1.value, 4, 1, false, useGpu);
  real input1Data_case3[] = {1, 2, 3, 4};
  input1.value->setData(input1Data_case3);

  input1.sequenceStartPositions = ICpuGpuVector::create(4, useGpu);
  int input1Seq[] = {0, 2, 3, 4};
  input1.sequenceStartPositions->copyFrom(input1Seq, 4, useGpu);

  real resultData_case3[] = {1, 1, 2, 3, 4, 4};
  result.value->setData(resultData_case3);

  doOneExpandTest("seq", true, useGpu, input1, input2, result);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  initMain(argc, argv);
  return RUN_ALL_TESTS();
}
