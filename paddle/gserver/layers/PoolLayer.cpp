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

#include "PoolLayer.h"
#include "MaxPoolWithMaskLayer.h"
#include "PoolProjectionLayer.h"
#include "paddle/utils/Logging.h"
#ifdef PADDLE_WITH_CUDA
#include "CudnnPoolLayer.h"
#endif
namespace paddle {

REGISTER_LAYER_CREATE_FUNC(pool, &PoolLayer::create);

bool PoolLayer::init(const LayerMap& layerMap,
                     const ParameterMap& parameterMap) {
  /* Initialize the basic parent class */
  Layer::init(layerMap, parameterMap);

  /* the size of inputs for pool-layer is 1 */
  CHECK_EQ(config_.inputs_size(), 1);

  const PoolConfig& conf = config_.inputs(0).pool_conf();
  poolType_ = conf.pool_type();
  channels_ = conf.channels();
  sizeX_ = conf.size_x();
  stride_ = conf.stride();
  outputX_ = conf.output_x();
  imgSize_ = conf.img_size();
  confPadding_ = conf.padding();

  sizeY_ = conf.has_size_y() ? conf.size_y() : conf.size_x();
  imgSizeY_ = conf.has_img_size_y() ? conf.img_size_y() : conf.img_size();
  strideY_ = conf.has_stride_y() ? conf.stride_y() : conf.stride();
  confPaddingY_ = conf.has_padding_y() ? conf.padding_y() : conf.padding();
  outputY_ = conf.has_output_y() ? conf.output_y() : conf.output_x();

  excludeMode_ = conf.has_exclude_mode() ? conf.exclude_mode() : true;
  return true;
}

Layer* PoolLayer::create(const LayerConfig& config) {
  CHECK_EQ(config.inputs_size(), 1);
  const std::string& pool = config.inputs(0).pool_conf().pool_type();
  if (pool == "max-projection" || pool == "avg-projection") {
    return new PoolProjectionLayer(config);
#ifdef PADDLE_WITH_CUDA
  } else if (CudnnPoolLayer::typeCheck(pool)) {
    return new CudnnPoolLayer(config);
#endif
  } else if (pool == "max-pool-with-mask") {
    return new MaxPoolWithMaskLayer(config);
  } else {
    LOG(FATAL) << "Unknown pool type: " << pool;
    return nullptr;
  }
}

}  // namespace paddle
