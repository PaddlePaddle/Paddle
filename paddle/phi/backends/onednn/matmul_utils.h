// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#pragma once

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"

namespace phi {
namespace funcs {

// Reshape a rank-3 tensor from P x M x N to (P * M) x N.
// Identity op if the tensor is not of rank 3.
static DenseTensor FoldInitDims(const DenseTensor& input) {
  auto output = input;
  auto in_dims = input.dims();
  if (in_dims.size() == 3) {
    output.Resize({in_dims[0] * in_dims[1], in_dims[2]});
  }
  return output;
}

// Get row matrix shape from a vector shape. If the rank of x_dim > 1, the
// original x_dim is returned.
static DDim RowMatrixFromVector(const DDim& x_dim) {
  if (x_dim.size() > 1) {
    return x_dim;
  }
  return make_ddim({1, x_dim[0]});
}

// Get column matrix shape from a vector shape. If the ran of y_dim > 1, the
// original y_dim is returned.
static DDim ColumnMatrixFromVector(const DDim& y_dim) {
  if (y_dim.size() > 1) {
    return y_dim;
  }
  return make_ddim({y_dim[0], 1});
}

// Reshape a tensor to 3-D or 2-D tensor by matrix descriptor.
// The shape would be [BatchSize, H, W] or [H, W].
// If transposed, `H,W` will be swapped.
static void ReshapeTensorIntoMatrixSequence(DenseTensor* x,
                                            const MatDescriptor& descriptor) {
  int64_t h;
  int64_t w;
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
static void ReshapeXYOutIntoMatrixSequence(DenseTensor* x,
                                           DenseTensor* y,
                                           DenseTensor* out,
                                           bool trans_x,
                                           bool trans_y) {
  auto x_dim = RowMatrixFromVector(x->dims());
  auto y_dim = ColumnMatrixFromVector(y->dims());
  auto mat_dim_x = CreateMatrixDescriptor(x_dim, 0, trans_x);
  auto mat_dim_y = CreateMatrixDescriptor(y_dim, 0, trans_y);
  if (mat_dim_x.batch_size_ == 0 && mat_dim_y.batch_size_ == 0) {
    out->Resize({mat_dim_x.height_, mat_dim_y.width_});
  } else {
    out->Resize({(std::max)(mat_dim_x.batch_size_, mat_dim_y.batch_size_),
                 mat_dim_x.height_,
                 mat_dim_y.width_});
  }

  ReshapeTensorIntoMatrixSequence(x, mat_dim_x);
  ReshapeTensorIntoMatrixSequence(y, mat_dim_y);
}

}  // namespace funcs
}  // namespace phi
