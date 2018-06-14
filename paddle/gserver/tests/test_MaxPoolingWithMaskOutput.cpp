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
#include "paddle/math/MathUtils.h"
#include "paddle/testing/TestUtil.h"

using namespace paddle;

void setPoolConfig(TestConfig* config,
                   PoolConfig* pool,
                   const string& poolType) {
  (*config).biasSize = 0;
  (*config).layerConfig.set_type("pool");
  (*config).layerConfig.set_num_filters(1);

  int kw = 3, kh = 3;
  int pw = 0, ph = 0;
  int sw = 2, sh = 2;
  pool->set_pool_type(poolType);
  pool->set_channels(1);
  pool->set_size_x(kw);
  pool->set_size_y(kh);
  pool->set_start(0);
  pool->set_padding(pw);
  pool->set_padding_y(ph);
  pool->set_stride(sw);
  pool->set_stride_y(sh);

  int ow = outputSize(pool->img_size(), kw, pw, sw, /* caffeMode */ false);
  int oh = outputSize(pool->img_size_y(), kh, ph, sh, /* caffeMode */ false);
  pool->set_output_x(ow);
  pool->set_output_y(oh);
}

void doOneMaxPoolingWithMaskOutputTest(MatrixPtr& inputMat,
                                       const string& poolType,
                                       bool use_gpu,
                                       MatrixPtr& maskMat) {
  TestConfig config;
  config.inputDefs.push_back({INPUT_DATA, "layer_0", 25, 0});
  LayerInputConfig* input = config.layerConfig.add_inputs();
  PoolConfig* pool = input->mutable_pool_conf();

  pool->set_img_size(5);
  pool->set_img_size_y(5);
  setPoolConfig(&config, pool, poolType);
  config.layerConfig.set_size(pool->output_x() * pool->output_y() *
                              pool->channels());

  config.layerConfig.set_name("MaxPoolWithMask");

  std::vector<DataLayerPtr> dataLayers;
  LayerMap layerMap;
  vector<Argument> datas;

  initDataLayer(config,
                &dataLayers,
                &datas,
                &layerMap,
                "MaxPoolWithMask",
                1,
                false,
                use_gpu);

  dataLayers[0]->getOutputValue()->copyFrom(*inputMat);

  FLAGS_use_gpu = use_gpu;
  std::vector<ParameterPtr> parameters;
  LayerPtr maxPoolingWithMaskOutputLayer;
  initTestLayer(config, &layerMap, &parameters, &maxPoolingWithMaskOutputLayer);
  maxPoolingWithMaskOutputLayer->forward(PASS_GC);

  checkMatrixEqual(maxPoolingWithMaskOutputLayer->getOutput("mask").value,
                   maskMat);
}

TEST(Layer, maxPoolingWithMaskOutputLayerFwd) {
  bool useGpu = false;
  MatrixPtr inputMat;
  MatrixPtr maskMat;
  real inputData[] = {0.1, 0.1, 0.5, 0.5, 1.1, 0.2, 0.2, 0.6, 0.1,
                      0.1, 0.3, 0.3, 0.7, 0.1, 0.1, 0.4, 0.4, 0.8,
                      0.8, 0.1, 1.0, 2.0, 3.0, 0.0, 9.0};
  real maskData[] = {12, 4, 22, 24};

  inputMat = Matrix::create(1, 25, false, useGpu);
  maskMat = Matrix::create(1, 4, false, useGpu);
  inputMat->setData(inputData);
  maskMat->setData(maskData);
  doOneMaxPoolingWithMaskOutputTest(
      inputMat, "max-pool-with-mask", useGpu, maskMat);
#ifdef PADDLE_WITH_CUDA
  useGpu = true;
  inputMat = Matrix::create(1, 25, false, useGpu);
  maskMat = Matrix::create(1, 4, false, useGpu);
  inputMat->copyFrom(inputData, 25);
  maskMat->copyFrom(maskData, 4);
  doOneMaxPoolingWithMaskOutputTest(
      inputMat, "max-pool-with-mask", useGpu, maskMat);
#endif
}
