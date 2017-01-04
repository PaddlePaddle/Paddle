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

namespace paddle {
/**
 * @brief A layer for computing cosine similarity between a vector
 * and each row of a matrix
 * out[i] = cos_scale * cos(in1, in2(i,:));
 * @note used in NEURAL TURING MACHINE
 *
 * Input1: a vector (batchSize * dataDim)
 *
 * Input2: a matrix in vector form (batchSize * (weightDim*dataDim))
 *
 * Output: a vector (batchSize * weightDim)
 */

class CosSimVecMatLayer : public Layer {
public:
  explicit CosSimVecMatLayer(const LayerConfig& config) : Layer(config) {}

  ~CosSimVecMatLayer() {}

  bool init(const LayerMap& layerMap, const ParameterMap& parameterMap);

  void forward(PassType passType);
  void backward(const UpdateCallback& callback = nullptr);

protected:
  MatrixPtr tmpMtx0;
  MatrixPtr tmpMtx1;
  MatrixPtr tmpRow0;
  MatrixPtr tmpRow1;
  MatrixPtr tmpRow2;
  MatrixPtr tmpRow3;
};

}  // namespace paddle
