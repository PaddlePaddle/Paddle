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
void doOnePriorBoxTest(size_t feature_map_width,
                       size_t feature_map_height,
                       size_t image_width,
                       size_t image_height,
                       vector<int> min_size,
                       vector<int> max_size,
                       vector<real> aspect_ratio,
                       vector<real> variance,
                       bool use_gpu,
                       MatrixPtr& result) {
  // Setting up the priorbox layer
  TestConfig configt;
  configt.layerConfig.set_type("priorbox");

  configt.inputDefs.push_back({INPUT_DATA, "featureMap", 1, 0});
  LayerInputConfig* input = configt.layerConfig.add_inputs();
  configt.inputDefs.push_back({INPUT_DATA, "image", 1, 0});
  configt.layerConfig.add_inputs();
  PriorBoxConfig* pb = input->mutable_priorbox_conf();
  for (size_t i = 0; i < min_size.size(); i++) pb->add_min_size(min_size[i]);
  for (size_t i = 0; i < max_size.size(); i++) pb->add_max_size(max_size[i]);
  for (size_t i = 0; i < variance.size(); i++) pb->add_variance(variance[i]);
  for (size_t i = 0; i < aspect_ratio.size(); i++)
    pb->add_aspect_ratio(aspect_ratio[i]);

  // data layer initialize
  std::vector<DataLayerPtr> dataLayers;
  LayerMap layerMap;
  vector<Argument> datas;
  initDataLayer(
      configt, &dataLayers, &datas, &layerMap, "priorbox", 1, false, use_gpu);
  dataLayers[0]->getOutput().setFrameHeight(feature_map_height);
  dataLayers[0]->getOutput().setFrameWidth(feature_map_width);
  dataLayers[1]->getOutput().setFrameHeight(image_height);
  dataLayers[1]->getOutput().setFrameWidth(image_width);

  // test layer initialize
  std::vector<ParameterPtr> parameters;
  LayerPtr priorboxLayer;
  initTestLayer(configt, &layerMap, &parameters, &priorboxLayer);
  priorboxLayer->forward(PASS_GC);
  checkMatrixEqual(priorboxLayer->getOutputValue(), result);
}

TEST(Layer, priorBoxLayerFwd) {
  vector<int> minSize;
  vector<int> maxSize;
  vector<real> aspectRatio;
  vector<real> variance;
  bool useGpu = false;

  minSize.push_back(276);
  maxSize.push_back(330);
  variance.push_back(0.1);
  variance.push_back(0.1);
  variance.push_back(0.2);
  variance.push_back(0.2);

  // CPU case 1.
  MatrixPtr result;
  real resultData[] = {0.04,
                       0.04,
                       0.96,
                       0.96,
                       0.1,
                       0.1,
                       0.2,
                       0.2,
                       0,
                       0,
                       1,
                       1,
                       0.1,
                       0.1,
                       0.2,
                       0.2};
  result = Matrix::create(1, 2 * 8, false, useGpu);
  result->setData(resultData);
  doOnePriorBoxTest(/* feature_map_width */ 1,
                    /* feature_map_height */ 1,
                    /* image_width */ 300,
                    /* image_height */ 300,
                    minSize,
                    maxSize,
                    aspectRatio,
                    variance,
                    useGpu,
                    result);
  // CPU case 2.
  variance[1] = 0.2;
  variance[3] = 0.1;
  maxSize.pop_back();
  real resultData2[] = {0,     0,     0.595, 0.595, 0.1, 0.2, 0.2, 0.1,
                        0.405, 0,     1,     0.595, 0.1, 0.2, 0.2, 0.1,
                        0,     0.405, 0.595, 1,     0.1, 0.2, 0.2, 0.1,
                        0.405, 0.405, 1,     1,     0.1, 0.2, 0.2, 0.1};
  Matrix::resizeOrCreate(result, 1, 4 * 8, false, useGpu);
  result->setData(resultData2);
  doOnePriorBoxTest(/* feature_map_width */ 2,
                    /* feature_map_height */ 2,
                    /* image_width */ 400,
                    /* image_height */ 400,
                    minSize,
                    maxSize,
                    aspectRatio,
                    variance,
                    useGpu,
                    result);
  // CPU case 3.
  aspectRatio.push_back(2);
  real resultData3[] = {0.04,     0.04, 0.96, 0.96,       0.1,        0.2,
                        0.2,      0.1,  0,    0.17473088, 1,          0.825269,
                        0.1,      0.2,  0.2,  0.1,        0.17473088, 0,
                        0.825269, 1,    0.1,  0.2,        0.2,        0.1};
  Matrix::resizeOrCreate(result, 1, 3 * 8, false, useGpu);
  result->setData(resultData3);
  doOnePriorBoxTest(/* feature_map_width */ 1,
                    /* feature_map_height */ 1,
                    /* image_width */ 300,
                    /* image_height */ 300,
                    minSize,
                    maxSize,
                    aspectRatio,
                    variance,
                    useGpu,
                    result);

#ifdef PADDLE_WITH_CUDA
  // reset the input parameters
  variance[1] = 0.1;
  variance[3] = 0.2;
  maxSize.push_back(330);
  aspectRatio.pop_back();
  MatrixPtr resultGpu;
  useGpu = true;
  // GPU case 1.
  resultGpu = Matrix::create(1, 2 * 8, false, useGpu);
  resultGpu->copyFrom(resultData, 2 * 8);
  doOnePriorBoxTest(/* feature_map_width */ 1,
                    /* feature_map_height */ 1,
                    /* image_width */ 300,
                    /* image_height */ 300,
                    minSize,
                    maxSize,
                    aspectRatio,
                    variance,
                    useGpu,
                    resultGpu);
  // GPU case 2.
  variance[1] = 0.2;
  variance[3] = 0.1;
  maxSize.pop_back();
  Matrix::resizeOrCreate(resultGpu, 1, 4 * 8, false, useGpu);
  resultGpu->copyFrom(resultData2, 4 * 8);
  doOnePriorBoxTest(/* feature_map_width */ 2,
                    /* feature_map_height */ 2,
                    /* image_width */ 400,
                    /* image_height */ 400,
                    minSize,
                    maxSize,
                    aspectRatio,
                    variance,
                    useGpu,
                    resultGpu);
  // GPU case 3.
  aspectRatio.push_back(2);
  Matrix::resizeOrCreate(resultGpu, 1, 3 * 8, false, useGpu);
  resultGpu->copyFrom(resultData3, 3 * 8);
  doOnePriorBoxTest(/* feature_map_width */ 1,
                    /* feature_map_height */ 1,
                    /* image_width */ 300,
                    /* image_height */ 300,
                    minSize,
                    maxSize,
                    aspectRatio,
                    variance,
                    useGpu,
                    resultGpu);
#endif
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  initMain(argc, argv);
  return RUN_ALL_TESTS();
}
