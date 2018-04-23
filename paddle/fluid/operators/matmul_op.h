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
#include <algorithm>
#include <functional>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/math/matmul.h"

namespace paddle {
namespace operators {
namespace matmul_detail {

using Tensor = framework::Tensor;
using DDim = framework::DDim;
using framework::make_ddim;
using framework::vectorize;

template <typename DeviceContext, typename T>
class MatMulKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const Tensor& x = *context.Input<Tensor>("X");
    const Tensor& y = *context.Input<Tensor>("Y");
    Tensor* out = context.Output<Tensor>("Out");
    out->mutable_data<T>(context.GetPlace());
    bool transpose_x = context.Attr<bool>("transpose_X");
    bool transpose_y = context.Attr<bool>("transpose_Y");

    math::MatMulFunctor<DeviceContext, T>()(
        context.template device_context<DeviceContext>(), x, transpose_x, y,
        transpose_y, T(1), out, T(0));
  }
};

template <typename T>
inline Tensor Reshape(const Tensor& input, const DDim& dims) {
  Tensor output;
  output.ShareDataWith(input);
  output.Resize(dims);
  return output;
}

// Reshape a rank-3 tensor from P x M x N to (P * M) x N.
// Identity op if the tensor is not of rank 3.
template <typename T>
Tensor CombineBatchAndM(const Tensor& input) {
  Tensor output;
  output.ShareDataWith(input);
  auto in_dims = input.dims();
  if (in_dims.size() == 3) {
    std::vector<int64_t> out_dims = {in_dims[0] * in_dims[1], in_dims[2]};
    output.Resize(make_ddim(out_dims));
  }
  return output;
}

// Reshape a rank-3 tensor from P x M x N to M x (P * N).
// (Warning: This requires transposing data and writes into new memory.)
// Identity op if the tensor is not of rank 3.
template <typename DeviceContext, typename T>
Tensor CombineBatchAndN(const DeviceContext& context, const Tensor& input) {
  Tensor output;
  auto in_dims = input.dims();
  if (in_dims.size() == 3) {
    output.Resize({in_dims[1], in_dims[0], in_dims[2]});
    output.mutable_data<T>(context.GetPlace());
    std::vector<int> axis = {1, 0, 2};
    math::Transpose<DeviceContext, T, 3> trans;
    trans(context, input, &output, axis);
    std::vector<int64_t> out_dims = {in_dims[1], in_dims[0] * in_dims[2]};
    output.Resize({in_dims[1], in_dims[0] * in_dims[2]});
  } else {
    output.ShareDataWith(input);
  }
  return output;
}

// Using dimensional constraints on matrix multiplication, it is
// straight-forward to check the following table for when X and Y
// are both matrices.
//
// transpose_X | False    | True     | False    | True
// transpose_Y | False    | False    | True     | True
// -----------+----------+----------+----------+-----------
//        dX = | dOut Y^T | Y dOut^T | dOut Y   | Y^T dOut^T
//        dY = | X^T dOut | X dOut   | dOut^T X | dOut^T X^T
//
// When X is a vector of size K, we treat it instead as a matrix of shape
// (1, K). Similarly, when Y is a vector of size K, we treat it instead as
// a matrix of shape (K, 1).
//
// When X and Y are both 3-dimensional tensors, then the first dimension
// the batch dimension can be ignored and the exact same formulas apply
// as for two matrices.
//
// Finally, when, e.g., X is a 3-dimensional tensor but Y is a matrix, we end
// up with formulas like
//
//   dY_{ij} = \sum_{p, m} X_{pmi} dOut_{pmj}
//
// To handle this sort of scenario, we reshape X : P x M x K, dOut: P x M x N
// to X: (P * M) x K, dOut: (P * M) x N.
template <typename DeviceContext, typename T>
class MatMulGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const Tensor& x = *context.Input<Tensor>("X");
    const Tensor& y = *context.Input<Tensor>("Y");
    const Tensor& dout = *context.Input<Tensor>(framework::GradVarName("Out"));
    Tensor* dx = context.Output<Tensor>(framework::GradVarName("X"));
    Tensor* dy = context.Output<Tensor>(framework::GradVarName("Y"));
    bool transpose_x = context.Attr<bool>("transpose_X");
    bool transpose_y = context.Attr<bool>("transpose_Y");

    std::vector<int64_t> x_dims = vectorize(x.dims());
    std::vector<int64_t> y_dims = vectorize(y.dims());

    // If X is a vector, reshape it to a matrix.
    if (x_dims.size() == 1) {
      x_dims.insert(x_dims.begin(), 1);
    }

    // If Y is a vector, reshape it to a matrix.
    if (y_dims.size() == 1) {
      y_dims.push_back(1);
    }

    int batch_count = 0;
    // The first rank-2 dimensions are accumulated on the batch_count, and the
    // last two dimensions are used for matrix multiplication.
    if (x_dims.size() > 3) {
      batch_count = accumulate(x_dims.begin(), x_dims.end() - 2, 1,
                               std::multiplies<int>());
    }
    // Fix the dOut dimensions.
    int M = 0, N = 0, batchCountX = 0, batchCountY = 0;

    switch (x_dims.size()) {
      case 2:
        M = transpose_x ? x_dims[1] : x_dims[0];
        break;
      case 3:
        batchCountX = x_dims[0];
        M = transpose_x ? x_dims[2] : x_dims[1];
        break;
      default:
        batchCountX = batch_count;
        size_t mat_s = x_dims.size() - 2;
        M = transpose_x ? x_dims[mat_s + 1] : x_dims[mat_s];
    }

    switch (y_dims.size()) {
      case 2:
        N = transpose_y ? y_dims[0] : y_dims[1];
        break;
      case 3:
        batchCountY = y_dims[0];
        N = transpose_y ? y_dims[1] : y_dims[2];
        break;
      default:
        batchCountY = batch_count;
        size_t mat_s = y_dims.size() - 2;
        N = transpose_y ? y_dims[mat_s] : y_dims[mat_s + 1];
    }
    if (batchCountX && batchCountY) {
      PADDLE_ENFORCE_EQ(
          batchCountX, batchCountY,
          "When Input(X) and Input(Y) are both three dimensional, they "
          "must have the same batch dimension.");
    }
    int batchCount = std::max(batchCountX, batchCountY);
    std::vector<int64_t> dout_dims = {M, N};
    if (batchCount) {
      if (x_dims.size() > 3) {
        dout_dims.insert(dout_dims.begin(), x_dims.begin(), x_dims.end() - 2);
      } else {
        dout_dims.insert(dout_dims.begin(), batchCount);
      }
    }
    Tensor X = Reshape<T>(x, make_ddim(x_dims));
    Tensor Y = Reshape<T>(y, make_ddim(y_dims));
    Tensor dOut = Reshape<T>(dout, make_ddim(dout_dims));

    auto& dev_ctx = context.template device_context<DeviceContext>();
    if (dx) {
      dx->mutable_data<T>(context.GetPlace());
      const Tensor& dOut_for_dX =
          (x_dims.size() == 2 && y_dims.size() == 3)
              ? CombineBatchAndN<DeviceContext, T>(dev_ctx, dOut)
              : dOut;
      if (x_dims.size() == 2 && y_dims.size() == 3) {
        Y = transpose_y ? CombineBatchAndM<T>(Y)
                        : CombineBatchAndN<DeviceContext, T>(dev_ctx, Y);
      }
      if (transpose_x) {
        math::MatMulFunctor<DeviceContext, T>()(
            dev_ctx, Y, transpose_y, dOut_for_dX, transpose_x, T(1), dx, T(0));
      } else {
        math::MatMulFunctor<DeviceContext, T>()(
            dev_ctx, dOut_for_dX, transpose_x, Y, !transpose_y, T(1), dx, T(0));
      }
    }

    if (dy) {
      dy->mutable_data<T>(context.GetPlace());
      const Tensor& dOut_for_dY = (y_dims.size() == 2 && x_dims.size() == 3)
                                      ? CombineBatchAndM<T>(dOut)
                                      : dOut;
      if (y_dims.size() == 2 && x_dims.size() == 3) {
        X = transpose_x ? CombineBatchAndN<DeviceContext, T>(dev_ctx, X)
                        : CombineBatchAndM<T>(X);
        dOut = CombineBatchAndM<T>(dOut);
      }
      if (transpose_y) {
        math::MatMulFunctor<DeviceContext, T>()(
            dev_ctx, dOut_for_dY, transpose_y, X, transpose_x, T(1), dy, T(0));
      } else {
        math::MatMulFunctor<DeviceContext, T>()(
            dev_ctx, X, !transpose_x, dOut_for_dY, transpose_y, T(1), dy, T(0));
      }
    }
  }
};
}  // namespace matmul_detail

using matmul_detail::MatMulKernel;
using matmul_detail::MatMulGradKernel;

}  // namespace operators
}  // namespace paddle
