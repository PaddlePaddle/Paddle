/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

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
#include <utility>
#include <vector>

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/reduce_ops/reduce_sum_op.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/complex_functors.h"

// only can include the headers in paddle/phi/api dirs
#include "paddle/phi/api/lib/utils/tensor_utils.h"
#include "paddle/phi/kernels/matmul_grad_kernel.h"
#include "paddle/phi/kernels/matmul_kernel.h"

#if defined(__NVCC__) || defined(__HIPCC__)
#include "paddle/fluid/operators/reduce_ops/reduce_op.cu.h"
#endif

namespace paddle {
namespace operators {

static framework::DDim GetDimForInput(const framework::InferShapeContext& ctx,
                                      const std::string input_name) {
  auto shape = ctx.Attrs().Get<std::vector<int>>("fused_reshape_" + input_name);
  auto axis =
      ctx.Attrs().Get<std::vector<int>>("fused_transpose_" + input_name);
  auto dim = ctx.GetInputDim(input_name);

  PADDLE_ENFORCE_GT(dim.size(),
                    0,
                    platform::errors::InvalidArgument(
                        "The Input(%s) has not been initialized properly. The "
                        "shape of Input(%s) = [%s].",
                        dim));

  if (!shape.empty() && !axis.empty()) {
    dim = dim.reshape(shape).transpose(axis);
  }
  return dim;
}

class MatMulV2Op : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "matmul_v2");
    OP_INOUT_CHECK(ctx->HasInput("Y"), "Input", "Y", "matmul_v2");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "matmul_v2");
    bool trans_x = ctx->Attrs().Get<bool>("trans_x");
    bool trans_y = ctx->Attrs().Get<bool>("trans_y");

    std::vector<int64_t> dims_x = phi::vectorize(GetDimForInput(*ctx, "X"));
    std::vector<int64_t> dims_y = phi::vectorize(GetDimForInput(*ctx, "Y"));
    auto ndims_x = dims_x.size();
    auto ndims_y = dims_y.size();
    PADDLE_ENFORCE_GT(ndims_x,
                      0,
                      platform::errors::InvalidArgument(
                          "The Input(X) dims size must be greater than 0,"
                          " but received dims size is 0. "));
    PADDLE_ENFORCE_GT(ndims_y,
                      0,
                      platform::errors::InvalidArgument(
                          "The Input(Y) dims size must be greater than 0,"
                          " but received dims size is 0. "));

    bool x_broadcasted = false, y_broadcasted = false;
    if (ndims_x == 1) {
      dims_x.insert(dims_x.begin(), 1);
      ndims_x = 2;
      x_broadcasted = true;
    }

    if (ndims_y == 1) {
      dims_y.push_back(1);
      ndims_y = 2;
      y_broadcasted = true;
    }

    size_t M, N;
    if (trans_x) {
      M = dims_x[ndims_x - 1];
    } else {
      M = dims_x[ndims_x - 2];
    }
    if (trans_y) {
      N = dims_y[ndims_y - 2];
    } else {
      N = dims_y[ndims_y - 1];
    }

    std::vector<int64_t> new_dims;
    if (ndims_x > ndims_y) {
      new_dims.assign(dims_x.begin(), dims_x.end() - 2);
    } else if (ndims_x < ndims_y) {
      new_dims.assign(dims_y.begin(), dims_y.end() - 2);
    } else {
      new_dims.reserve(ndims_x);
      for (size_t i = 0; i < ndims_x - 2; ++i) {
        new_dims.push_back(std::max(dims_x[i], dims_y[i]));
      }
    }
    if (!x_broadcasted) {
      new_dims.push_back(M);
    }
    if (!y_broadcasted) {
      new_dims.push_back(N);
    }
    if (x_broadcasted && y_broadcasted) {
      new_dims.push_back(1);
    }

    auto ddim_out = phi::make_ddim(new_dims);

#ifdef PADDLE_WITH_MKLDNN
    auto shape = ctx->Attrs().Get<std::vector<int>>("fused_reshape_Out");
    auto axis = ctx->Attrs().Get<std::vector<int>>("fused_transpose_Out");

    if (!shape.empty() && !axis.empty()) {
      ddim_out = ddim_out.transpose(axis).reshape(shape);
    }
#endif

    ctx->SetOutputDim("Out", ddim_out);
    ctx->ShareLoD("X", "Out");
  }

 protected:
  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto input_data_type =
        OperatorWithKernel::IndicateOrPromoteVarDataTypes(ctx, "X", "Y");
    return phi::KernelKey(input_data_type, ctx.GetPlace());
  }

  phi::KernelKey GetKernelTypeForVar(
      const std::string& var_name,
      const phi::DenseTensor& tensor,
      const phi::KernelKey& expected_kernel_type) const override {
    if (framework::IsComplexType(expected_kernel_type.dtype())) {
      // only promote inputs’s types when contains complex input
      return phi::KernelKey(tensor.place(), tensor.layout(), tensor.dtype());
    } else {
#ifdef PADDLE_WITH_MKLDNN
      // When matmul_v2 is first oneDNN op in a chain (there was some non oneDNN
      // op previously) then we also need to rotate shape NHWC -> NCWH
      if ((expected_kernel_type.layout() == phi::DataLayout::ONEDNN) &&
          (tensor.layout() != phi::DataLayout::ONEDNN) &&
          phi::OneDNNContext::tls().get_cur_paddle_data_layout() ==
              phi::DataLayout::kNHWC) {
        return phi::KernelKey(tensor.place(),
                              phi::DataLayout::kNHWC,
                              expected_kernel_type.dtype());
      }
#endif
      return phi::KernelKey(
          tensor.place(), tensor.layout(), expected_kernel_type.dtype());
    }
  }
};

class MatMulV2OpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() final;

 protected:
  virtual void Apply() {}
};

// Reshape a rank-3 tensor from P x M x N to (P * M) x N.
// Identity op if the tensor is not of rank 3.
static phi::DenseTensor FoldInitDims(const phi::DenseTensor& input) {
  auto output = input;
  auto in_dims = input.dims();
  if (in_dims.size() == 3) {
    output.Resize({in_dims[0] * in_dims[1], in_dims[2]});
  }
  return output;
}

/**
 * Get row matrix shape from a vector shape. If the rank of x_dim > 1, the
 * original x_dim is returned.
 */
static framework::DDim RowMatrixFromVector(const framework::DDim& x_dim) {
  if (x_dim.size() > 1) {
    return x_dim;
  }
  return phi::make_ddim({1, x_dim[0]});
}

/**
 * Get column matrix shape from a vector shape. If the ran of y_dim > 1, the
 * original y_dim is returned.
 */
static framework::DDim ColumnMatrixFromVector(const framework::DDim& y_dim) {
  if (y_dim.size() > 1) {
    return y_dim;
  }
  return phi::make_ddim({y_dim[0], 1});
}

/**
 * Reshape a tensor to 3-D or 2-D tensor by matrix descriptor.
 *
 * The shape would be [BatchSize, H, W] or [H, W].
 * If transposed, `H,W` will be swapped.
 */
static void ReshapeTensorIntoMatrixSequence(
    phi::DenseTensor* x, const phi::funcs::MatDescriptor& descriptor) {
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

static void ReshapeXYOutIntoMatrixSequence(phi::DenseTensor* x,
                                           phi::DenseTensor* y,
                                           phi::DenseTensor* out,
                                           bool trans_x,
                                           bool trans_y) {
  auto x_dim = RowMatrixFromVector(x->dims());
  auto y_dim = ColumnMatrixFromVector(y->dims());
  auto mat_dim_x = phi::funcs::CreateMatrixDescriptor(x_dim, 0, trans_x);
  auto mat_dim_y = phi::funcs::CreateMatrixDescriptor(y_dim, 0, trans_y);
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

}  // namespace operators
}  // namespace paddle
