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
 * @brief The SelectiveFullyConnectedLayer class
 *
 * SelectiveFullyConnectedLayer differs from FullyConnectedLayer by that it
 * requires an additional input to indicate several selected columns, and only
 * compute the multiplications between the input matrices and the selected
 * columns of the parameter matrices of this layer. If the selected columns is
 * not specified, SelectiveFullyConnected layer acts exactly like
 * FullyConnectedLayer.
 *
 * The config file api is selective_fc_layer.
 */
class SelectiveFullyConnectedLayer : public Layer {
 protected:
  WeightList weights_;
  std::unique_ptr<Weight> biases_;

 private:
  /**
   * Get selected columns each forward.
   */
  void getSelectiveCols();

  MatrixPtr mmat_;
  /// cpuSelCols_ is a CpuSparseMatrix, used to save selected columns.
  MatrixPtr cpuSelCols_;
  /// CpuSparseMatrix or GpuSparseMatrix. In CPU mode, selCols_ points
  /// to cpuSelCols_.
  MatrixPtr selCols_;
  size_t inputNum_;

  /// interOutput_ shared same memory with output_.value.
  MatrixPtr interOutput_;

  /// if fullOutput_ is false, interOutGrad_ sparse matrix
  MatrixPtr interOutGrad_;

  /// if true, means output_.value is the same as Fc Layer
  bool fullOutput_;

 public:
  explicit SelectiveFullyConnectedLayer(const LayerConfig& config)
      : Layer(config), selCols_(nullptr) {}

  ~SelectiveFullyConnectedLayer() {}
  void prefetch() override;

  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override;

  Weight& getWeight(int idx) { return *weights_[idx]; }

  /**
   * @brief Resize the output matrix size.
   * And reset value to zero
   */
  void reserveOutput(size_t height, size_t width, size_t nnz);

  /**
   * @brief Fill candidates to select several activations as output.
   * @param candidates specifies several selected columns of the parameter
   * matrices of this layer.
   * Multiplications only between the input matrices and the selected columns
   * are computed.
   * If the candidates is a nullptr, selective fc layer acts exactly like the
   * fully connected layer.
   * @note CURRENTLY, THIS METHOD IS ONLY USED FOR BEAM SEARCH
   */
  void fillSelectiveData(
      const std::shared_ptr<std::vector<std::pair<int*, size_t>>>& candidates);

  void forward(PassType passType) override;
  void backward(const UpdateCallback& callback = nullptr) override;

 private:
  /**
   * @brief Make SelectiveFC act as FullyConnectedLayer
   */
  void fillFullySelectiveData() { fullOutput_ = true; }
};
}  // namespace paddle
