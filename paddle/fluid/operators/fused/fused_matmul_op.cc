//   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/matmul_v2_op.h"

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/phi/core/infermeta_utils.h"

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

class FusedMatmulOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "fused_matmul");
    OP_INOUT_CHECK(ctx->HasInput("Y"), "Input", "Y", "fused_matmul");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "fused_matmul");
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
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto input_data_type =
        OperatorWithKernel::IndicateOrPromoteVarDataTypes(ctx, "X", "Y");
    return framework::OpKernelType(input_data_type, ctx.GetPlace());
  }

  framework::OpKernelType GetKernelTypeForVar(
      const std::string& var_name,
      const phi::DenseTensor& tensor,
      const framework::OpKernelType& expected_kernel_type) const override {
    if (framework::IsComplexType(expected_kernel_type.data_type_)) {
      // only promote inputs’s types when contains complex input
      return framework::OpKernelType(
          framework::TransToProtoVarType(tensor.dtype()),
          tensor.place(),
          tensor.layout());
    } else {
#ifdef PADDLE_WITH_MKLDNN
      // When matmul_v2 is first oneDNN op in a chain (there was some non oneDNN
      // op previously) then we also need to rotate shape NHWC -> NCWH
      if ((expected_kernel_type.data_layout_ == phi::DataLayout::ONEDNN) &&
          (tensor.layout() != phi::DataLayout::ONEDNN) &&
          phi::OneDNNContext::tls().get_cur_paddle_data_layout() ==
              phi::DataLayout::kNHWC) {
        return framework::OpKernelType(expected_kernel_type.data_type_,
                                       tensor.place(),
                                       phi::DataLayout::kNHWC);
      }
#endif
      return framework::OpKernelType(
          expected_kernel_type.data_type_, tensor.place(), tensor.layout());
    }
  }
};

class FusedMatmulOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "tensor of shape (d0, d1 ... M, K)");
    AddInput("Y", "tensor of shape (d0, d1 ... K, N)");
    AddOutput("Out", "tensor of shape (d0, d1 ... M, N)");
    AddAttr<bool>("trans_x",
                  "Set true to transpose the last two dimensions of X before "
                  "doing multiplication")
        .SetDefault(false);
    AddAttr<bool>("trans_y",
                  "Set true to transpose the last two dimensions of Y before "
                  "doing multiplication")
        .SetDefault(false);
    AddAttr<std::string>("fuse_activation",
                         "Activation type from matmul_activation fuse pass")
        .SetDefault("");
    AddAttr<std::vector<int>>(
        "fused_reshape_X",
        "Reshape's shape attribute from reshape_transpose_matmul fuse pass")
        .SetDefault({});
    AddAttr<std::vector<int>>(
        "fused_transpose_X",
        "Transpose's axis attribute from reshape_transpose_matmul fuse pass")
        .SetDefault({});
    AddAttr<std::vector<int>>(
        "fused_reshape_Y",
        "Reshape's shape attribute from reshape_transpose_matmul fuse pass")
        .SetDefault({});
    AddAttr<std::vector<int>>(
        "fused_transpose_Y",
        "Transpose's axis attribute from reshape_transpose_matmul fuse pass")
        .SetDefault({});
    AddAttr<std::vector<int>>(
        "fused_reshape_Out",
        "Reshape's shape attribute from matmul_transpose_reshape fuse pass")
        .SetDefault({});
    AddAttr<std::vector<int>>(
        "fused_transpose_Out",
        "Transpose's axis attribute from matmul_transpose_reshape fuse pass")
        .SetDefault({});
    AddComment(
        R"DOC(Matrix multiplication Out = X * Y. A has shape (d0, d1 ... M, K),
        B has shape (d0, d1 ... K, N), Out has shape ((d0, d1 ... M, N)).
        In addition, it also follows the broadcast rule which is similar as
        numpy.matmul.
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    fused_matmul,
    ops::FusedMatmulOp,
    ops::FusedMatmulOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);
