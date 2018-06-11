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

#include "Layer.h"
#include "paddle/math/Matrix.h"
#include "paddle/utils/ThreadLocal.h"

namespace paddle {

/**
 * This layer just simply add all input layers together, then activate
 * the sum inputs. Each input of this layer should be the same size,
 * which is also the output size of this layer.
 * \f[
 *   y=f(\sum_{i}x_i + b)
 * \f]
 * where \f$y\f$ is output, \f$x\f$ is input, \f$b\f$ is bias, and \f$f\f$ is
 * activation function.
 *
 * The config file api is addto_layer.
 */
class AddtoLayer : public Layer {
 protected:
  std::unique_ptr<Weight> biases_;

 public:
  explicit AddtoLayer(const LayerConfig& config) : Layer(config) {}

  ~AddtoLayer() {}

  /**
   * Intialization of AddtoLayer.
   */
  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override;

  /**
   * Forward propagation.
   * @note There is no weight matrix for each input,
   *       because it just a simple add operation.
   */
  void forward(PassType passType) override;

  /**
   * Backward propagation.
   */
  void backward(const UpdateCallback& callback = nullptr) override;
};

}  // namespace paddle
