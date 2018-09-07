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

#include "GruCompute.h"
#include "Layer.h"
#include "SequenceToBatch.h"
#include "paddle/math/Matrix.h"

namespace paddle {

/**
 * @brief Please refer to "Junyoung Chung, Empirical Evaluation
 * of Gated Recurrent Neural Networks on Sequence Modeling".
 *
 * GatedRecurrentLayer takes 1 input layer with size * 3.
 * Input layer is diveded into 3 equal parts: (xz_t, xr_t, xi_t).
 * parameter and biasParameter is also diveded into 3 equal parts:
 *   - parameter consists of (U_z, U_r, U)
 *   - baisParameter consists of (bias_z, bias_r, bias_o)
 *
 * \f[
 * update \ gate: z_t = actGate(xz_t + U_z * h_{t-1} + bias_z) \\
 * reset \ gate: r_t = actGate(xr_t + U_r * h_{t-1} + bias_r) \\
 * output \ candidate: {h}_t = actNode(xi_t + U * dot(r_t, h_{t-1}) + bias_o) \\
 * hidden \ activation: h_t = dot((1-z_t), h_{t-1}) + dot(z_t, {h}_t) \\
 * \f]
 *
 * @note
 * - dot denotes "element-wise multiplication".
 * - actNode is defined by config active_type
 * - actGate is defined by config actvie_gate_type
 *
 * The config file is grumemory.
 */

class GatedRecurrentLayer : public Layer, public GruCompute {
 public:
  explicit GatedRecurrentLayer(const LayerConfig& config) : Layer(config) {}

  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override;

  void forward(PassType passType) override;

  void backward(const UpdateCallback& callback) override;

  void resetState() override;

  void setState(LayerStatePtr state) override;

  LayerStatePtr getState() override;

 protected:
  void forwardSequence(int batchSize,
                       size_t numSequences,
                       const int* starts,
                       MatrixPtr inputValue);
  void backwardSequence(int batchSize,
                        size_t numSequences,
                        const int* starts,
                        MatrixPtr inputGrad);

  void forwardBatch(int batchSize,
                    size_t numSequences,
                    const int* starts,
                    MatrixPtr inputValue);
  void backwardBatch(int batchSize, MatrixPtr inputGrad);

 protected:
  std::unique_ptr<Weight> weight_;
  std::unique_ptr<Weight> gateWeight_;
  std::unique_ptr<Weight> stateWeight_;
  std::unique_ptr<Weight> bias_;

  Argument gate_;
  Argument resetOutput_;

  bool reversed_;
  bool useBatch_;
  std::unique_ptr<SequenceToBatch> batchValue_;
  std::unique_ptr<SequenceToBatch> batchGrad_;
  std::unique_ptr<ActivationFunction> activationGate_;

  MatrixPtr prevOutput_;
};

}  // namespace paddle
