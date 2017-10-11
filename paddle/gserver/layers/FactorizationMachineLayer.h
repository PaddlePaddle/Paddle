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

#pragma once

#include "Layer.h"
#include "paddle/math/Matrix.h"
#include "paddle/utils/ThreadLocal.h"

namespace paddle {
/**
 * @brief The Factorization Machine models pairwise (order-2) feature
 * interactions as inner product of the learned latent vectors corresponding
 * to each input feature.
 *
 * The Factorization Machine can effectively capture feature interactions
 * especially when the input is sparse. While in principle FM can model higher
 * order feature interaction, in practice usually only order-2 feature
 * interactions are considered. The Factorization Machine Layer here only
 * computes the order-2 interations with the formula:
 *
 * \f[
 *     y = \sum_{i=1}^{n-1}\sum_{j=i+1}^n\langle v_i, v_j \rangle x_i x_j
 * \f]
 *
 * The config file api is factorization_machine.
 */

class FactorizationMachineLayer : public Layer {
protected:
  /// The latent vectors, shape: (size, factorSize_)
  std::unique_ptr<Weight> latentVectors_;
  /// The hyperparameter that defines the dimensionality of the factorization
  size_t factorSize_;

public:
  explicit FactorizationMachineLayer(const LayerConfig& config)
      : Layer(config) {}
  ~FactorizationMachineLayer() {}

  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override;

  void forward(PassType passType) override;
  void backward(const UpdateCallback& callback = nullptr) override;
};

}  // namespace paddle
