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

#include "ConvBaseOperator.h"
#include "paddle/math/MathUtils.h"
#include "paddle/math/Matrix.h"

namespace paddle {

/**
 * @brief ConvTransOperator takes two inputs to perform the convolution.
 * The first input is the image, and the second input is the convolution kernel.
 * The height of data for two inputs are the same. Each data of the first input
 * is convolved with each data of the second input indepedently.
 *
 * The config file api is conv_operator.
 */

class ConvTransOperator : public ConvBaseOperator {
 public:
  ConvTransOperator(const OperatorConfig &config, bool useGpu)
      : ConvBaseOperator(config, useGpu) {}
  /**
   * Free workspace in device and destroy cudnn tensor descriptor.
   */
  virtual ~ConvTransOperator() {}
  void forward() override;
  void backward() override;
  void reshape(int batchSize) override;
};

}  // namespace paddle
