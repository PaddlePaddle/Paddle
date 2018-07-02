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

#include "PoolLayer.h"

namespace paddle {

/**
 * @brief CudnnPoolLayer is subclass of PoolLayer, which is implemented by
 * cudnn api and only supports GPU.
 *
 * The config file api is img_pool_layer.
 */

class CudnnPoolLayer : public PoolLayer {
 protected:
  int windowHeight, windowWidth;
  int heightPadding, widthPadding, strideHeight, strideWidth;
  int imageH_, imageW_, outputH_, outputW_;
  /// mode_ is poolint type, inlcuding "cudnn-max-pool", "cudnn-avg-pool"
  /// "cudnn-avg-excl-pad-pool".
  hl_pooling_mode_t mode_;
  /// cudnn tensor descriptor for input.
  hl_tensor_descriptor inputDesc_;
  /// cudnn tensor descriptor for output.
  hl_tensor_descriptor outputDesc_;
  /// A description of a pooling operation.
  hl_pooling_descriptor poolingDesc_;

 public:
  static bool typeCheck(const std::string& poolType,
                        hl_pooling_mode_t* mode = nullptr);
  explicit CudnnPoolLayer(const LayerConfig& config);
  ~CudnnPoolLayer();
  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override;

  /**
   * Reshape input and output tensor descriptor.
   * The batch size maybe change during training in last batch of each pass.
   * So reshaping is needed.
   */
  void reshape(int batchSize);
  void forward(PassType passType) override;
  void backward(const UpdateCallback& callback = nullptr) override;
};

}  // namespace paddle
