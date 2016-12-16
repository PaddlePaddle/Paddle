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
#include "./paddle/utils/CommandLineParser.h"
#include "ModelConfig.pb.h"
#include "paddle/gserver/layers/DataLayer.h"
#include "paddle/gserver/layers/ExpandConvTransLayer.h"
#include "paddle/math/MathUtils.h"
#include "paddle/trainer/Trainer.h"
#include "paddle/utils/GlobalConstants.h"

#include "LayerGradUtil.h"
#include "TestUtil.h"

using namespace paddle;  // NOLINT
using namespace std;     // NOLINT

P_DECLARE_bool(use_gpu);
P_DECLARE_int32(gpu_id);
P_DECLARE_double(checkgrad_eps);
P_DECLARE_bool(thread_local_rand_use_global_seed);
P_DECLARE_bool(prev_batch_state);

// Do one forward pass of priorBox layer and check to see if its output
// matches the given result
void doOnePriorBoxTest(size_t featureMapWidth,
                       size_t featureMapHeight,
                       size_t imageWidth,
                       size_t imageHeight,
                       vector<int> minSize,
                       vector<int> maxSize,
                       vector<float> aspectRatio,
                       vector<float> variance,
                       MatrixPtr& result) {
  // Setting up the priorbox layer
  TestConfig configt;
  configt.layerConfig.set_type("priorbox");

  configt.inputDefs.push_back({INPUT_DATA, "featureMap", 1, 0});
  LayerInputConfig* input = configt.layerConfig.add_inputs();
  configt.inputDefs.push_back({INPUT_DATA, "image", 1, 0});
  configt.layerConfig.add_inputs();
  PriorBoxConfig* pb = input->mutable_priorbox_conf();
  for (size_t i = 0; i < minSize.size(); i++) pb->add_min_size(minSize[i]);
  for (size_t i = 0; i < maxSize.size(); i++) pb->add_max_size(maxSize[i]);
  for (size_t i = 0; i < aspectRatio.size(); i++)
    pb->add_aspect_ratio(aspectRatio[i]);
  for (size_t i = 0; i < variance.size(); i++) pb->add_variance(variance[i]);

  // data layer initialize
  std::vector<DataLayerPtr> dataLayers;
  LayerMap layerMap;
  vector<Argument> datas;
  initDataLayer(
      configt, &dataLayers, &datas, &layerMap, "priorbox", 1, false, true);
  dataLayers[0]->getOutput().setFrameHeight(featureMapHeight);
  dataLayers[0]->getOutput().setFrameWidth(featureMapWidth);
  dataLayers[1]->getOutput().setFrameHeight(imageHeight);
  dataLayers[1]->getOutput().setFrameWidth(imageWidth);

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
  vector<float> aspectRatio;
  vector<float> variance;

  minSize.push_back(276);
  maxSize.push_back(330);
  variance.push_back(0.1);
  variance.push_back(0.1);
  variance.push_back(0.2);
  variance.push_back(0.2);

  MatrixPtr result;
  result = Matrix::create(1, 2 * 8, false, false);

  float resultData[] = {0.04,
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
  result->setData(resultData);
  doOnePriorBoxTest(/* featureMapWidth */ 1,
                    /* featureMapHeight */ 1,
                    /* imageWidth */ 300,
                    /* imageHeight */ 300,
                    minSize,
                    maxSize,
                    aspectRatio,
                    variance,
                    result);

  variance[1] = 0.2;
  variance[3] = 0.1;
  maxSize.pop_back();
  Matrix::resizeOrCreate(result, 1, 4 * 8, false, false);
  float resultData2[] = {0,     0,     0.595, 0.595, 0.1, 0.2, 0.2, 0.1,
                         0.405, 0,     1,     0.595, 0.1, 0.2, 0.2, 0.1,
                         0,     0.405, 0.595, 1,     0.1, 0.2, 0.2, 0.1,
                         0.405, 0.405, 1,     1,     0.1, 0.2, 0.2, 0.1};
  result->setData(resultData2);
  doOnePriorBoxTest(/* featureMapWidth */ 2,
                    /* featureMapHeight */ 2,
                    /* imageWidth */ 400,
                    /* imageHeight */ 400,
                    minSize,
                    maxSize,
                    aspectRatio,
                    variance,
                    result);

  aspectRatio.push_back(2);
  Matrix::resizeOrCreate(result, 1, 3 * 8, false, false);
  float resultData3[] = {0.04,     0.04, 0.96, 0.96,       0.1,        0.2,
                         0.2,      0.1,  0,    0.17473088, 1,          0.825269,
                         0.1,      0.2,  0.2,  0.1,        0.17473088, 0,
                         0.825269, 1,    0.1,  0.2,        0.2,        0.1};
  result->setData(resultData3);
  doOnePriorBoxTest(/* featureMapWidth */ 1,
                    /* featureMapHeight */ 1,
                    /* imageWidth */ 300,
                    /* imageHeight */ 300,
                    minSize,
                    maxSize,
                    aspectRatio,
                    variance,
                    result);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  initMain(argc, argv);
  FLAGS_thread_local_rand_use_global_seed = true;
  srand(1);
  return RUN_ALL_TESTS();
}
