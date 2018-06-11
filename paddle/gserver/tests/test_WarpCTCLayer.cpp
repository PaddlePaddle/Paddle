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
#include <paddle/utils/Version.h>
#include "ModelConfig.pb.h"
#include "paddle/gserver/layers/CTCLayer.h"
#include "paddle/gserver/layers/DataLayer.h"
#include "paddle/gserver/layers/Layer.h"
#include "paddle/gserver/layers/WarpCTCLayer.h"

#include "paddle/testing/TestUtil.h"

using namespace paddle;  // NOLINT
using namespace std;     // NOLINT

DECLARE_bool(use_gpu);

const real* getData(const Matrix& matrix) {
  if (matrix.useGpu()) {
    MatrixPtr cpuMatrix = Matrix::create(
        matrix.getHeight(), matrix.getWidth(), matrix.isTransposed(), false);
    cpuMatrix->copyFrom(matrix);
    return cpuMatrix->getData();
  } else {
    return matrix.getData();
  }
}

int checkError(const Matrix& matrix1, const Matrix& matrix2) {
  CHECK_EQ(matrix1.getHeight(), matrix2.getHeight());
  CHECK_EQ(matrix1.getWidth(), matrix2.getWidth());
  CHECK_EQ(matrix1.isTransposed(), matrix2.isTransposed());
#ifndef PADDLE_TYPE_DOUBLE
  real err = 1e-3;
#else
  real err = 1e-10;
#endif

  int height = matrix1.getHeight();
  int width = matrix1.getWidth();

  const real* data1 = getData(matrix1);
  const real* data2 = getData(matrix2);
  int count = 0;
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      if (fabs(data1[i * width + j] - data2[i * width + j]) > err) {
        count++;
      }
    }
  }
  EXPECT_EQ(count, 0) << "There are " << count << " different element.";
  return count;
}

void initArgument(size_t batchSize,
                  int layerSize,
                  bool useGpu,
                  Argument& data) {
  data.value = Matrix::create(batchSize, layerSize, false, useGpu);
  data.grad = Matrix::create(batchSize, layerSize, false, useGpu);
  data.value->randomizeUniform();
  data.value->add(-0.5);
  data.grad->zeroMem();

  generateSequenceStartPositions(batchSize, data.sequenceStartPositions);
}

LayerPtr createDataLayer(
    string name, size_t batchSize, int layerSize, bool useGpu, Argument& data) {
  LayerConfig layerConfig;
  layerConfig.set_name(name);
  layerConfig.set_type("data");
  layerConfig.set_size(layerSize);
  LayerPtr layer = LayerPtr(new DataLayer(layerConfig));

  DataLayerPtr dataLayer = std::dynamic_pointer_cast<DataLayer>(layer);
  dataLayer->setData(data);
  dataLayer->forward(PASS_GC);

  return layer;
}

LayerPtr createLabelLayer(string name,
                          size_t batchSize,
                          size_t numClasses,
                          bool useGpu) {
  LayerConfig layerConfig;
  layerConfig.set_name(name);
  layerConfig.set_type("data");
  layerConfig.set_size(1);
  LayerPtr layer = LayerPtr(new DataLayer(layerConfig));

  Argument data;
  data.ids = IVector::create(batchSize, useGpu);
  data.ids->rand(numClasses - 1);

  generateSequenceStartPositions(batchSize, data.sequenceStartPositions);

  DataLayerPtr labelLayer = std::dynamic_pointer_cast<DataLayer>(layer);
  labelLayer->setData(data);
  labelLayer->forward(PASS_GC);

  return layer;
}

LayerPtr createCTCLayer(string name,
                        size_t numClasses,
                        bool useGpu,
                        bool normByTimes,
                        LayerPtr dataLayer,
                        LayerPtr labelLayer) {
  LayerMap layerMap;
  layerMap[dataLayer->getName()] = dataLayer;
  layerMap[labelLayer->getName()] = labelLayer;

  ParameterMap parameterMap;

  LayerConfig layerConfig;
  layerConfig.set_name(name);
  layerConfig.set_type("ctc");
  layerConfig.set_size(numClasses);
  layerConfig.set_norm_by_times(normByTimes);

  layerConfig.add_inputs();
  LayerInputConfig& input0 = *(layerConfig.mutable_inputs(0));
  input0.set_input_layer_name(dataLayer->getName());

  layerConfig.add_inputs();
  LayerInputConfig& input1 = *(layerConfig.mutable_inputs(1));
  input1.set_input_layer_name(labelLayer->getName());

  LayerPtr layer = LayerPtr(new CTCLayer(layerConfig));
  layerMap[layer->getName()] = layer;
  layer->init(layerMap, parameterMap);

  ActivationFunction* softmaxActivation = ActivationFunction::create("softmax");

  softmaxActivation->forward(dataLayer->getOutput()).check();
  layer->forward(PASS_GC);

  layer->backward();
  softmaxActivation->backward(dataLayer->getOutput()).check();

  return layer;
}

LayerPtr createWarpCTCLayer(string name,
                            size_t numClasses,
                            bool useGpu,
                            bool normByTimes,
                            LayerPtr dataLayer,
                            LayerPtr labelLayer) {
  LayerMap layerMap;
  layerMap[dataLayer->getName()] = dataLayer;
  layerMap[labelLayer->getName()] = labelLayer;

  ParameterMap parameterMap;

  LayerConfig layerConfig;
  layerConfig.set_name(name);
  layerConfig.set_type("warp_ctc");
  layerConfig.set_size(numClasses);
  layerConfig.set_blank(numClasses - 1);
  layerConfig.set_norm_by_times(normByTimes);

  layerConfig.add_inputs();
  LayerInputConfig& input0 = *(layerConfig.mutable_inputs(0));
  input0.set_input_layer_name(dataLayer->getName());

  layerConfig.add_inputs();
  LayerInputConfig& input1 = *(layerConfig.mutable_inputs(1));
  input1.set_input_layer_name(labelLayer->getName());

  LayerPtr layer = LayerPtr(new WarpCTCLayer(layerConfig));
  layerMap[layer->getName()] = layer;
  layer->init(layerMap, parameterMap);

  layer->forward(PASS_GC);
  layer->backward();

  return layer;
}

TEST(Layer, WarpCTCLayer) {
  for (auto layerSize : {10, 64}) {
    for (auto batchSize : {1, 10, 32}) {
      for (auto normByTimes : {false, true}) {
        for (auto useGpu : {false, true}) {
#ifndef PADDLE_WITH_CUDA
          if (useGpu) continue;
#endif
          LOG(INFO) << "layerSize=" << layerSize << " batchSize=" << batchSize
                    << " normByTimes = " << normByTimes << " useGpu=" << useGpu;

          FLAGS_use_gpu = useGpu;

          Argument data0;
          initArgument(batchSize, layerSize, useGpu, data0);

          Argument data1;
          data1.resizeAndCopyFrom(data0);

          LayerPtr dataLayer0 =
              createDataLayer("data", batchSize, layerSize, useGpu, data0);
          LayerPtr dataLayer1 =
              createDataLayer("data", batchSize, layerSize, useGpu, data1);

          LayerPtr labelLayer =
              createLabelLayer("label", batchSize, layerSize, useGpu);

          LayerPtr warpctcLayer = createWarpCTCLayer(
              "cost", layerSize, useGpu, normByTimes, dataLayer0, labelLayer);
          LayerPtr ctcLayer = createCTCLayer(
              "cost", layerSize, useGpu, normByTimes, dataLayer1, labelLayer);

          /// Check cost
          LOG(INFO) << "Check cost: "
                    << checkError(*(warpctcLayer->getOutput().value),
                                  *(ctcLayer->getOutput().value))
                    << " different elements.";

          /// Check gradients
          LOG(INFO) << "Check gradients: "
                    << checkError(*(dataLayer0->getOutput().grad),
                                  *(dataLayer1->getOutput().grad))
                    << " different elements";
        }
      }
    }
  }
}
