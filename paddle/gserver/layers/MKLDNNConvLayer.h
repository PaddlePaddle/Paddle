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
 * @brief A subclass of MKLDNNLayer conv layer.
 *
 * The config file api is mkldnn_conv
 */
class MKLDNNConvLayer : public MKLDNNLayer {
protected:
  // padding height and width
  int ph_, pw_;
  // stride height and width
  int sh_, sw_;
  // dilation height and width
  int dh_, dw_;
  // filter(kenerl) height and width
  int fh_, fw_;
  // group number
  int gp_;

  MKLDNNMatrixPtr userInVal_;
  MKLDNNMatrixPtr userOutVal_;
  std::shared_ptr<mkldnn::reorder> cvtInVal_;
  std::shared_ptr<mkldnn::reorder> cvtOutVal_;

  // if has already init the weight
  bool hasInitedWgt_;

  // True by default. This impact the calculation of output size.
  // For example:
  // - input(+padding): 0123456789
  // - imageSize(+padding) = 10;
  // - filterSize = 3;
  // - stride = 2;
  // - caffeMode_ is true:
  // - output: (012), (234), (456), (678)
  // - outputSize = 4;
  // - caffeMode_ is false:
  // - output: (012), (234), (456), (678), (9)
  // - outputSize = 5;
  bool caffeMode_;

  // weight and bias
  std::unique_ptr<Weight> weight_;
  std::unique_ptr<Weight> biases_;
  // internal matrix for input value and output value
  MatrixPtr internalInVal_;
  MatrixPtr internalOutVal_;

public:
  explicit MKLDNNConvLayer(const LayerConfig& config)
      : MKLDNNLayer(config), hasInitedWgt_(false), caffeMode_(true) {}

  ~MKLDNNConvLayer() {}

  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override;

  void convertWeightsFromPaddle() override;

  void convertWeightsToPaddle() override;

  void forward(PassType passType) override;

  void backward(const UpdateCallback& callback) override;

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
