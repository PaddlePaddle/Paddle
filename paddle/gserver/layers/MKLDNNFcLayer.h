/* Copyright (c) 2017 PaddlePaddle Authors. All Rights Reserve.

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

#include "MKLDNNLayer.h"
#include "mkldnn.hpp"

namespace paddle {

/**
 * @brief A subclass of MKLDNNLayer fc layer.
 *
 * The config file api is mkldnn_fc
 */
class MKLDNNFcLayer : public MKLDNNLayer {
protected:
  // input layer size, can not be change after init
  size_t iLayerSize_;  // == ic * ih * iw

  // if has already init the weight
  bool hasInitedWgt_;

  // fc weight and bias
  std::unique_ptr<Weight> weight_;
  std::unique_ptr<Weight> biases_;

public:
  explicit MKLDNNFcLayer(const LayerConfig& config)
      : MKLDNNLayer(config), hasInitedWgt_(false) {}

  ~MKLDNNFcLayer() {}

  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override;

  void convertWeightsFromPaddle() override;

  void convertWeightsToPaddle() override;

  void forward(PassType passType) override;

  void backward(const UpdateCallback& callback) override;

  void updateInputData() override;

protected:
  /**
   * reshape the input image sizes
   * and reset output buffer size
   * and reset mkldnn forward
   */
  void reshape();

  /**
   * reset the forward primitve and memory
   * only would be called when input size changes
   */
  void resetFwd();

  /**
   * reset the backward primitve and memory for mkldnn fc
   * only would be called when needed
   */
  void resetBwd();
};

}  // namespace paddle
