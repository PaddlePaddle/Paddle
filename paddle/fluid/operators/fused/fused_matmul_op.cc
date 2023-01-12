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

namespace paddle {
namespace operators {

static std::vector<int64_t> GetInputShape(phi::DDim dim,
                                          std::vector<int> shape,
                                          std::vector<int> axis) {
  PADDLE_ENFORCE_GT(dim.size(),
                    0,
                    platform::errors::InvalidArgument(
                        "The Input(%s) has not been initialized properly. The "
                        "shape of Input(%s) = [%s].",
                        dim));

  if (!shape.empty() && !axis.empty()) {
    dim = dim.reshape(shape).transpose(axis);
  }
  return phi::vectorize(dim);
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

    std::vector<int64_t> dims_x =
        GetInputShape(ctx->GetInputDim("X"),
                      ctx->Attrs().Get<std::vector<int>>("fused_reshape_X"),
                      ctx->Attrs().Get<std::vector<int>>("fused_transpose_X"));
    std::vector<int64_t> dims_y =
        GetInputShape(ctx->GetInputDim("Y"),
                      ctx->Attrs().Get<std::vector<int>>("fused_reshape_Y"),
                      ctx->Attrs().Get<std::vector<int>>("fused_transpose_Y"));

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

    auto shape = ctx->Attrs().Get<std::vector<int>>("fused_reshape_Out");
    auto axis = ctx->Attrs().Get<std::vector<int>>("fused_transpose_Out");

    if (!shape.empty() && !axis.empty()) {
      ddim_out = ddim_out.transpose(axis).reshape(shape);
    }

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

class FusedMatmulOpMaker : public MatMulV2OpMaker {
 protected:
  void Apply() override {
    AddInput("ResidualData",
             "Extra input from matmul_elementwise_add_mkldnn_fuse_pass")
        .AsDispensable()
        .AsExtra();
    AddAttr<float>("matmul_alpha", "Output scale used in matmul_v1")
        .SetDefault(1.0f);
    AddAttr<std::string>(
        "fuse_activation",
        "Activation type from matmul_activation_mkldnn_fuse_pass")
        .SetDefault("");
    AddAttr<float>("fuse_alpha",
                   "Activation alpha from matmul_activation_mkldnn_fuse_pass")
        .SetDefault(0.0f);
    AddAttr<float>("fuse_beta",
                   "Activation beta from matmul_activation_mkldnn_fuse_pass")
        .SetDefault(0.0f);
    AddAttr<float>("fused_output_scale",
                   "Output scale from operator_scale_onednn_fuse_pass")
        .SetDefault(1.0f);
    AddAttr<std::vector<int>>("fused_reshape_X",
                              "Reshape's shape attribute from "
                              "reshape_transpose_matmul_mkldnn_fuse_pass")
        .SetDefault({});
    AddAttr<std::vector<int>>("fused_transpose_X",
                              "Transpose's axis attribute from "
                              "reshape_transpose_matmul_mkldnn_fuse_pass")
        .SetDefault({});
    AddAttr<std::vector<int>>("fused_reshape_Y",
                              "Reshape's shape attribute from "
                              "reshape_transpose_matmul_mkldnn_fuse_pass")
        .SetDefault({});
    AddAttr<std::vector<int>>("fused_transpose_Y",
                              "Transpose's axis attribute from "
                              "reshape_transpose_matmul_mkldnn_fuse_pass")
        .SetDefault({});
    AddAttr<std::vector<int>>("fused_reshape_Out",
                              "Reshape's shape attribute from "
                              "matmul_transpose_reshape_mkldnn_fuse_pass")
        .SetDefault({});
    AddAttr<std::vector<int>>("fused_transpose_Out",
                              "Transpose's axis attribute from "
                              "matmul_transpose_reshape_mkldnn_fuse_pass")
        .SetDefault({});
    AddAttr<std::string>("mkldnn_data_type", "oneDNN operator data type")
        .SetDefault("float32")
        .InEnum({"float32", "int8", "bfloat16"});
    AddAttr<float>("Scale_x", "Matmul X input quantization scale")
        .SetDefault(1.0f);
    AddAttr<float>("Scale_y", "Matmul Y input quantization scale")
        .SetDefault(1.0f);
    AddAttr<float>("Scale_in_eltwise", "Matmul ResidualData quantization scale")
        .SetDefault(0.0f);
    AddAttr<float>("Scale_out", "Matmul output quantization scale")
        .SetDefault(1.0f);
    AddAttr<bool>("force_fp32_output",
                  "Flag determining if output should be converted to FP32")
        .SetDefault(false);
    AddComment(
        R"DOC(Matrix multiplication extended with oneDNN-specific fusion logic.)DOC");
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
