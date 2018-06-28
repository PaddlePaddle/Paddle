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

#include <cudnn.h>
#include "BatchNormBaseLayer.h"
#include "Layer.h"
#include "paddle/utils/Stat.h"

namespace paddle {

/**
 * @brief Cudnn Batch normalization layer use to cuDNN lib to implentment.
 * @note Cudnn version must >= v4.0, and better to use the latest version
 * (v5.1).
 *
 * The config file api is batch_norm_layer.
 */

class CudnnBatchNormLayer : public BatchNormBaseLayer {
 public:
  explicit CudnnBatchNormLayer(const LayerConfig& config)
      : BatchNormBaseLayer(config) {}

  ~CudnnBatchNormLayer();

  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override;
  /**
   * reshape tensor of ioDesc_.
   */
  void reshape(int batchSize);

  void forward(PassType passType) override;
  void backward(const UpdateCallback& callback = nullptr) override;

 protected:
  /// Epsilon value used in the batch normalization formula.
  /// Same epsilon value should be used in forward and backward functions.
  double eps_;

  /// Input/output tensor descriptor desc
  hl_tensor_descriptor ioDesc_;
  /// Shared tensor descriptor desc for the 6 tenros:
  /// bnScale, bnBias, running mean/var, save_mean/var
  hl_tensor_descriptor bnParamDesc_;

  /**
   * @brief The gradient of weight and bias in cudnn api can not be empty.
   * If set is_static for weight or bias, it will not allocate memory for them,
   * and the gradient is NULL. In this case, will use two matrix.
   */
  MatrixPtr tmpWGrad_, tmpBiasGrad_;
};

}  // namespace paddle
