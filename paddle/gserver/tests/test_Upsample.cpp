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
#include "paddle/math/MathUtils.h"
#include "paddle/testing/TestUtil.h"

void setPoolConfig(paddle::TestConfig* config,
                   paddle::PoolConfig* pool,
                   const string& poolType) {
  (*config).biasSize = 0;
  (*config).layerConfig.set_type("pool");
  (*config).layerConfig.set_num_filters(1);

  int kw = 2, kh = 2;
  int pw = 0, ph = 0;
  int sw = 2, sh = 2;
  pool->set_pool_type(poolType);
  pool->set_channels(2);
  pool->set_size_x(kw);
  pool->set_size_y(kh);
  pool->set_start(0);
  pool->set_padding(pw);
  pool->set_padding_y(ph);
  pool->set_stride(sw);
  pool->set_stride_y(sh);

  int ow =
      paddle::outputSize(pool->img_size(), kw, pw, sw, /* caffeMode */ false);
  int oh =
      paddle::outputSize(pool->img_size_y(), kh, ph, sh, /* caffeMode */ false);
  pool->set_output_x(ow);
  pool->set_output_y(oh);
}

paddle::LayerPtr doOneUpsampleTest(const paddle::MatrixPtr& inputMat,
                                   const string& poolType,
                                   bool use_gpu,
                                   real* tempGradData) {
  /* prepare maxPoolWithMaskLayer */
  paddle::TestConfig config;
  config.inputDefs.push_back({paddle::INPUT_DATA, "layer_0", 128, 0});
  paddle::LayerInputConfig* input = config.layerConfig.add_inputs();
  paddle::PoolConfig* pool = input->mutable_pool_conf();

  pool->set_img_size(8);
  pool->set_img_size_y(8);
  setPoolConfig(&config, pool, "max-pool-with-mask");
  config.layerConfig.set_size(pool->output_x() * pool->output_y() *
                              pool->channels());

  config.layerConfig.set_name("MaxPoolWithMask");

  std::vector<paddle::DataLayerPtr> dataLayers;
  paddle::LayerMap layerMap;
  vector<paddle::Argument> datas;

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
  std::vector<paddle::ParameterPtr> parameters;
  paddle::LayerPtr maxPoolingWithMaskOutputLayer;
  initTestLayer(config, &layerMap, &parameters, &maxPoolingWithMaskOutputLayer);
  maxPoolingWithMaskOutputLayer->forward(paddle::PASS_GC);

  /* prepare the upsample layer */
  paddle::LayerConfig upsampleLayerConfig;
  upsampleLayerConfig.set_type("upsample");
  paddle::LayerInputConfig* input1 = upsampleLayerConfig.add_inputs();
  upsampleLayerConfig.add_inputs();

  paddle::UpsampleConfig* upsampleConfig = input1->mutable_upsample_conf();
  upsampleConfig->set_scale(2);
  paddle::ImageConfig* imageConfig = upsampleConfig->mutable_image_conf();
  imageConfig->set_channels(2);
  imageConfig->set_img_size(4);
  imageConfig->set_img_size_y(4);
  upsampleLayerConfig.set_size(2 * 8 * 8);
  upsampleLayerConfig.set_name("upsample");

  for (size_t i = 0; i < 2; i++) {
    paddle::LayerInputConfig& inputTemp =
        *(upsampleLayerConfig.mutable_inputs(i));
    inputTemp.set_input_layer_name("MaxPoolWithMask");
  }

  paddle::LayerPtr upsampleLayer;
  paddle::ParameterMap parameterMap;
  upsampleLayer = paddle::Layer::create(upsampleLayerConfig);
  layerMap[upsampleLayerConfig.name()] = upsampleLayer;
  upsampleLayer->init(layerMap, parameterMap);
  upsampleLayer->setNeedGradient(true);
  upsampleLayer->forward(paddle::PASS_GC);
  upsampleLayer->getOutputGrad()->copyFrom(tempGradData, 128);
  upsampleLayer->backward();

  return upsampleLayer;
}

TEST(Layer, maxPoolingWithMaskOutputLayerFwd) {
  bool useGpu = false;
  paddle::MatrixPtr inputMat;
  paddle::MatrixPtr inputGPUMat;
  paddle::MatrixPtr tempGradMat;

  inputMat = paddle::Matrix::create(1, 128, false, useGpu);
  inputMat->randomizeUniform();

  tempGradMat = paddle::Matrix::create(1, 128, false, useGpu);
  tempGradMat->randomizeUniform();
  real* tempGradData = tempGradMat->getData();

  paddle::LayerPtr upsampleLayerCPU =
      doOneUpsampleTest(inputMat, "max-pool-with-mask", useGpu, tempGradData);

#ifdef PADDLE_WITH_CUDA
  useGpu = true;
  real* data = inputMat->getData();
  inputGPUMat = paddle::Matrix::create(1, 128, false, useGpu);
  inputGPUMat->copyFrom(data, 128);
  paddle::LayerPtr upsampleLayerGPU = doOneUpsampleTest(
      inputGPUMat, "max-pool-with-mask", useGpu, tempGradData);
  paddle::checkMatrixEqual(upsampleLayerCPU->getOutput("").value,
                           upsampleLayerGPU->getOutput("").value);

  paddle::checkMatrixEqual(upsampleLayerCPU->getPrev(0)->getOutputGrad(),
                           upsampleLayerGPU->getPrev(0)->getOutputGrad());
#endif
}
