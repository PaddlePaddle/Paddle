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

#include <string>

#include "paddle/fluid/framework/shape_inference.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/matmul_utils.h"

namespace paddle {
namespace operators {

// Get row matrix shape from a vector shape. If the rank of x_dim > 1, the
// original x_dim is returned.
paddle::framework::DDim RowMatrixFromVector(
    const paddle::framework::DDim& x_dim) {
  return x_dim.size() > 1 ? x_dim : paddle::framework::make_ddim({1, x_dim[0]});
}

// Get column matrix shape from a vector shape. If the ran of y_dim > 1, the
// original y_dim is returned.
paddle::framework::DDim ColumnMatrixFromVector(
    const paddle::framework::DDim& y_dim) {
  return y_dim.size() > 1 ? y_dim : paddle::framework::make_ddim({y_dim[0], 1});
}

std::vector<int64_t> TransposeVector(const std::vector<int64_t>& x,
                                     const std::vector<int>& axis) {
  size_t in_rank = x.size();
  size_t axis_size = axis.size();

  auto axis_set = std::set<int>(axis.begin(), axis.end());
  PADDLE_ENFORCE_EQ(axis_set.size(), axis_size,
                    paddle::platform::errors::InvalidArgument(
                        "In an axis array, elements must be unique."));

  PADDLE_ENFORCE_EQ(in_rank, axis_size,
                    paddle::platform::errors::InvalidArgument(
                        "The input dimension's size "
                        "should be equal to the axis's size. "
                        "But received dimension is %d, "
                        "axis's size is %d",
                        in_rank, axis_size));

  PADDLE_ENFORCE_LT(*std::max_element(axis.begin(), axis.end()), axis_size,
                    paddle::platform::errors::InvalidArgument(
                        "Axis values must be ranging from 0 to (dims - 1)."));

  std::vector<int64_t> new_x(x.size());
  for (size_t i = 0; i < x.size(); i++) {
    new_x[i] = x[axis[i]];
  }
  return new_x;
}
/**
 * Reshape a tensor to 3-D or 2-D tensor by matrix descriptor.
 *
 * The shape would be [BatchSize, H, W] or [H, W].
 * If transposed, `H,W` will be swapped.
 */
void ReshapeTensorIntoMatrixSequence(framework::Tensor* x,
                                     const math::MatDescriptor& descriptor) {
  int64_t h, w;
  h = descriptor.height_;
  w = descriptor.width_;
  if (descriptor.trans_) {
    std::swap(w, h);
  }
  if (descriptor.batch_size_) {
    x->Resize({descriptor.batch_size_, h, w});
  } else {
    x->Resize({h, w});
  }
}

void ReshapeXYOutIntoMatrixSequence(framework::Tensor* x, framework::Tensor* y,
                                    framework::Tensor* out, bool trans_x,
                                    bool trans_y) {
  auto x_dim = RowMatrixFromVector(x->dims());
  auto y_dim = ColumnMatrixFromVector(y->dims());
  auto mat_dim_x = math::CreateMatrixDescriptor(x_dim, 0, trans_x);
  auto mat_dim_y = math::CreateMatrixDescriptor(y_dim, 0, trans_y);
  if (mat_dim_x.batch_size_ == 0 && mat_dim_y.batch_size_ == 0) {
    out->Resize({mat_dim_x.height_, mat_dim_y.width_});
  } else {
    out->Resize({std::max(mat_dim_x.batch_size_, mat_dim_y.batch_size_),
                 mat_dim_x.height_, mat_dim_y.width_});
  }

  ReshapeTensorIntoMatrixSequence(x, mat_dim_x);
  ReshapeTensorIntoMatrixSequence(y, mat_dim_y);
}

framework::Tensor FoldInitDims(const framework::Tensor& input) {
  auto output = input;
  auto in_dims = input.dims();
  if (in_dims.size() == 3) {
    output.Resize({in_dims[0] * in_dims[1], in_dims[2]});
  }
  return output;
}

framework::DDim GetDimForInput(const framework::InferShapeContext& ctx,
                               const char input_letter) {
  PADDLE_ENFORCE((input_letter == 'X' || input_letter == 'Y'),
                 paddle::platform::errors::InvalidArgument(
                     "Input name should be a single character 'X' or 'Y'."));
  std::string input_name{input_letter};
  auto shape = ctx.Attrs().Get<std::vector<int>>("fused_reshape_" + input_name);
  auto axis =
      ctx.Attrs().Get<std::vector<int>>("fused_transpose_" + input_name);
  auto dim = ctx.GetInputDim(input_name);

  PADDLE_ENFORCE_GT(dim.size(), 0,
                    platform::errors::InvalidArgument(
                        "The Input(%s) has not been initialized properly. The "
                        "shape of Input(%s) = [%s].",
                        dim));
  if (!shape.empty() && !axis.empty()) {
    PADDLE_ENFORCE_GE(
        shape.size(), 2,
        platform::errors::InvalidArgument(
            "shape_%s attribute of MatMulOp was implemented for 2, 3 "
            "or 4 dimensions.",
            input_name));
    PADDLE_ENFORCE_LE(
        shape.size(), 4,
        platform::errors::InvalidArgument(
            "shape_%s attribute of MatMulOp was implemented for 2, 3 "
            "or 4 dimensions.",
            input_name));
    PADDLE_ENFORCE_EQ(
        shape.size(), axis.size(),
        platform::errors::InvalidArgument(
            "Ranks of shape_%s and axis_%s attributes of MatMulOp "
            "must be equal.",
            input_name, input_name));

    int num_negative = std::count(shape.begin(), shape.end(), -1);
    PADDLE_ENFORCE_LE(num_negative, 1,
                      platform::errors::InvalidArgument(
                          "The max number of -1 in fused_reshape_%s is 1 "
                          "but received %d.",
                          input_name, num_negative));

    auto it_zero = std::find(shape.begin(), shape.end(), 0);
    if (it_zero != shape.end()) {
      for (uint64_t i = 0; i < shape.size(); i++) {
        if (shape[i] == 0) {
          PADDLE_ENFORCE_LT(i, dim.size(),
                            platform::errors::InvalidArgument(
                                "The index of 0 in fused_reshape_%s ",
                                "should be less than output dim size, ",
                                "but the index is %d and output dim size is %d",
                                input_name, i, dim.size()));
          shape[i] = dim.at(i);
        }
      }
    }

    // if "-1" is present then one of reshape dims must be infered
    auto it_negative = std::find(shape.begin(), shape.end(), -1);
    if (it_negative != shape.end()) {
      int64_t dim_product = 1;
      for (int i = 0; i < dim.size(); i++) {
        dim_product *= dim.at(i);
      }

      int64_t shape_product = std::accumulate(shape.begin(), shape.end(), -1,
                                              std::multiplies<int>());
      int index = std::distance(shape.begin(), it_negative);
      shape[index] = dim_product / shape_product;
    }

    dim = dim.reshape(shape).transpose(axis);
  }
  return dim;
}
}  // namespace operators
}  // namespace paddle
