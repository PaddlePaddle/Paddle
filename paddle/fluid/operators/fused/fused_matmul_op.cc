//   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include <string>
#include <vector>

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/binary.h"

namespace paddle {
namespace operators {

static std::vector<int64_t> GetInputShape(phi::DDim dim,
                                          std::vector<int> shape,
                                          std::vector<int> axis) {
  PADDLE_ENFORCE_GT(dim.size(),
                    0,
                    common::errors::InvalidArgument(
                        "The Input(%s) has not been initialized properly. The "
                        "shape of Input(%s) = [%s].",
                        dim));

  auto is_input_fused = (!shape.empty() && !axis.empty());
  if (is_input_fused) {
    dim = dim.reshape(shape).transpose(axis);
  }
  return common::vectorize(dim);
}

class FusedMatmulOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "matmul_v2");
    OP_INOUT_CHECK(ctx->HasInput("Y"), "Input", "Y", "matmul_v2");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "matmul_v2");
    bool trans_x = ctx->Attrs().Get<bool>("trans_x");
    bool trans_y = ctx->Attrs().Get<bool>("trans_y");

    std::vector<int64_t> dims_x = common::vectorize(ctx->GetInputDim("X"));
    std::vector<int64_t> dims_y = common::vectorize(ctx->GetInputDim("Y"));
    auto ndims_x = dims_x.size();
    auto ndims_y = dims_y.size();
    PADDLE_ENFORCE_GT(
        ndims_x,
        0,
        common::errors::InvalidArgument(
            "The first input tensor X's dimension size must be greater than 0,"
            " but received the first input tensor X's dimension size is 0. "));
    PADDLE_ENFORCE_GT(
        ndims_y,
        0,
        common::errors::InvalidArgument(
            "The second input tensor Y's dimension size must be greater than 0,"
            " but received the second input tensor Y's dimension size is 0. "));

    bool x_broadcasted = false;
    bool y_broadcasted = false;

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

    size_t M = 0, N = 0;
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
      new_dims.push_back(M);  // NOLINT
    }
    if (!y_broadcasted) {
      new_dims.push_back(N);  // NOLINT
    }

    ctx->SetOutputDim("Out", common::make_ddim(new_dims));
    ctx->ShareLoD("X", "Out");
  };

 protected:
  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto input_data_type =
        OperatorWithKernel::IndicateOrPromoteVarDataTypes(ctx, "X", "Y");
    return phi::KernelKey(input_data_type, ctx.GetPlace());
  };

  phi::KernelKey GetKernelTypeForVar(
      const std::string& var_name,
      const phi::DenseTensor& tensor,
      const phi::KernelKey& expected_kernel_type) const override {
    if (framework::IsComplexType(expected_kernel_type.dtype())) {
      // only promote inputs’s types when contains complex input
      return phi::KernelKey(tensor.place(), tensor.layout(), tensor.dtype());
    } else {
#ifdef PADDLE_WITH_DNNL
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
  };
};

class FusedMatmulOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() final {
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
    AddComment(
        R"DOC(Matrix multiplication Out = X * Y. A has shape (d0, d1 ... M, K),
        B has shape (d0, d1 ... K, N), Out has shape ((d0, d1 ... M, N)).
        In addition, it also follows the broadcast rule which is similar as
        numpy.matmul.
)DOC");
    Apply();
  };

 protected:
  void Apply() {
    AddInput("ResidualData",
             "Extra input from matmul_elementwise_add_onednn_fuse_pass")
        .AsDispensable()
        .AsExtra();
    AddAttr<float>("matmul_alpha", "Output scale used in matmul_v1")
        .SetDefault(1.0f);
    AddAttr<std::string>(
        "fuse_activation",
        "Activation type from matmul_activation_onednn_fuse_pass")
        .SetDefault("");
    AddAttr<float>("fuse_alpha",
                   "Activation alpha from matmul_activation_onednn_fuse_pass")
        .SetDefault(0.0f);
    AddAttr<float>("fuse_beta",
                   "Activation beta from matmul_activation_onednn_fuse_pass")
        .SetDefault(0.0f);
    AddAttr<float>("fused_output_scale",
                   "Output scale from operator_scale_onednn_fuse_pass")
        .SetDefault(1.0f);
    AddAttr<std::vector<int>>("fused_reshape_X",
                              "Reshape's shape attribute from "
                              "reshape_transpose_matmul_onednn_fuse_pass")
        .SetDefault({});
    AddAttr<std::vector<int>>("fused_transpose_X",
                              "Transpose's axis attribute from "
                              "reshape_transpose_matmul_onednn_fuse_pass")
        .SetDefault({});
    AddAttr<std::vector<int>>("fused_reshape_Y",
                              "Reshape's shape attribute from "
                              "reshape_transpose_matmul_onednn_fuse_pass")
        .SetDefault({});
    AddAttr<std::vector<int>>("fused_transpose_Y",
                              "Transpose's axis attribute from "
                              "reshape_transpose_matmul_onednn_fuse_pass")
        .SetDefault({});
    AddAttr<std::vector<int>>("fused_reshape_Out",
                              "Reshape's shape attribute from "
                              "matmul_transpose_reshape_onednn_fuse_pass")
        .SetDefault({});
    AddAttr<std::vector<int>>("fused_transpose_Out",
                              "Transpose's axis attribute from "
                              "matmul_transpose_reshape_onednn_fuse_pass")
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
DECLARE_INFER_SHAPE_FUNCTOR(fused_matmul,
                            FusedMatmulInferShapeFunctor,
                            PD_INFER_META(phi::FusedMatmulInferMeta));
REGISTER_OPERATOR(
    fused_matmul,
    ops::FusedMatmulOp,
    ops::FusedMatmulOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>,
    FusedMatmulInferShapeFunctor);
