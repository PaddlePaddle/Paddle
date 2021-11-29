/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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

#include <vector>

#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {

// Get row matrix shape from a vector shape. If the rank of x_dim > 1, the
// original x_dim is returned.
paddle::framework::DDim RowMatrixFromVector(
    const paddle::framework::DDim& x_dim);

// Get column matrix shape from a vector shape. If the ran of y_dim > 1, the
// original y_dim is returned.
paddle::framework::DDim ColumnMatrixFromVector(
    const paddle::framework::DDim& y_dim);

// Get result of transposing vector x according to axis.
std::vector<int64_t> TransposeVector(const std::vector<int64_t>& x,
                                     const std::vector<int>& axis);

/**
 * Reshape the x,y,out tensor to 3-D or 2-D tensor by matrix descriptor
 * Out = matmul(x, y)
 *
 * This method will first calculate X,Y matrix sequence, and then calculate
 * the out shape.
 *
 * Assume X = [BatchSize, H1, W1], Y = [BatchSize, H2, W2]
 * The out = [BatchSize, H1, W2]
 *
 * If there is no batch size in `X` and `Y`, the out will be [H1, W2]
 * If any of `X` and `Y` has batch size BatchSize, the out will have the
 * BatchSize.
 */
void ReshapeXYOutIntoMatrixSequence(framework::Tensor* x, framework::Tensor* y,
                                    framework::Tensor* out, bool trans_x,
                                    bool trans_y);

// Reshape a rank-3 tensor from P x M x N to (P * M) x N.
// Identity op if the tensor is not of rank 3.
framework::Tensor FoldInitDims(const framework::Tensor& input);

// Reshape a rank-3 tensor from P x M x N to M x (P * N).
// (Warning: This requires transposing data and writes into new memory.)
// Identity op if the tensor is not of rank 3.
template <typename DeviceContext, typename T>
framework::Tensor FoldHeadAndLastDims(const DeviceContext& context,
                                      const framework::Tensor& input) {
  auto in_dims = input.dims();
  if (in_dims.size() != 3) {
    return input;
  }
  framework::Tensor output;
  output.Resize({in_dims[1], in_dims[0], in_dims[2]});
  output.mutable_data<T>(context.GetPlace());
  std::vector<int> axis = {1, 0, 2};
  math::Transpose<DeviceContext, T, 3> trans;
  trans(context, input, &output, axis);
  output.Resize({in_dims[1], in_dims[0] * in_dims[2]});
  return output;
}

framework::DDim GetDimForInput(const framework::InferShapeContext& ctx,
                               const char input_letter);
};  // namespace operators
};  // namespace paddle
