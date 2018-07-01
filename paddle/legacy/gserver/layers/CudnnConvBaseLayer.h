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

#pragma once

#include <vector>
#include "ConvBaseLayer.h"
#include "Projection.h"
#include "paddle/math/Matrix.h"

namespace paddle {

/**
 * @brief A 2-dimension conv layer implemented by cuDNN. It only
 *        supports GPU mode. We automatic select CudnnConvLayer for GPU
 *        mode and ExpandConvLayer for CPU mode if you set type of "conv".
 *        User also can specfiy type of "exconv" or "cudnn_conv" for
 *        particular type.
 *
 * The config file api is img_conv_layer.
 */
class CudnnConvBaseLayer : public ConvBaseLayer {
 protected:
  std::vector<std::unique_ptr<ProjectionConfig>> projConf_;
  std::vector<std::unique_ptr<Projection>> projections_;

  hl_tensor_descriptor biasDesc_;
  hl_tensor_descriptor outputDesc_;

 public:
  explicit CudnnConvBaseLayer(const LayerConfig& config)
      : ConvBaseLayer(config) {}

  ~CudnnConvBaseLayer();
  void forward(PassType passType) override;
  void backward(const UpdateCallback& callback) override;

  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override;
};

}  // namespace paddle
