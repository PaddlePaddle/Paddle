/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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
#include <vector>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/device/gpu/gpu_dnn.h"

namespace paddle {
namespace operators {

class FusedDgradDreluBnBwdWeightOpMaker
    : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    // dgrad inputs
    AddInput("dY", "Grad input of dgrad");
    AddInput("W", "Filter input of dgrad");
    // fuse_add inputs (optional)
    AddInput("dX_branch", "Gradient of the optional branch.").AsDispensable();
    // drelu inputs (optional)
    AddInput("Relu_X", "Residual input to relu").AsDispensable();
    // dBN1 inputs
    AddInput("BN1_mean", "");
    AddInput("BN1_inv_std", "");
    AddInput("BN1_scale", "");
    AddInput("BN1_bias", "");
    AddInput("BN1_X", "The input to BN1");
    // dBN2 inputs (optional)
    AddInput("BN2_mean", "").AsDispensable();
    AddInput("BN2_inv_std", "").AsDispensable();
    AddInput("BN2_scale", "").AsDispensable();
    AddInput("BN2_bias", "").AsDispensable();
    AddInput("BN2_X", "The input to BN2").AsDispensable();
    // drelu outputs
    AddOutput("dX", "Output of fused dgrad drelu");
    // dBN1 outputs
    AddOutput("BN1_dGamma", "");
    AddOutput("BN1_dBeta", "");
    AddOutput("BN1_eqscale_dy", "");
    AddOutput("BN1_eqscale_x", "");
    AddOutput("BN1_eqbias", "");
    // dBN2 outputs (optional)
    AddOutput("BN2_dGamma", "").AsDispensable();
    AddOutput("BN2_dBeta", "").AsDispensable();
    AddOutput("BN2_eqscale_dy", "").AsDispensable();
    AddOutput("BN2_eqscale_x", "").AsDispensable();
    AddOutput("BN2_eqbias", "").AsDispensable();
    // conv params
    AddAttr<std::vector<int>>("strides",
                              "(vector<int> default:{1, 1}), the "
                              "strides(h_stride, w_stride) of "
                              "convolution operator.")
        .SetDefault({1, 1});
    AddAttr<std::vector<int>>("paddings",
                              "(vector<int> default:{0, 0}), the "
                              "paddings(pad_height_top, pad_height_bottom, "
                              "pad_width_left, pad_wifth_right)  of "
                              "convolution operator.")
        .SetDefault({0, 0});
    AddAttr<std::string>(
        "padding_algorithm",
        "(string, default \"EXPLICIT\") An optional string from: \"EXPLICIT\","
        "\"SAME\",\"VALID\". Set to \"EXPLICIT\" for explicit padding. "
        "Set to \"SAME\" or \"VALID\" for algorithm of padding. ")
        .SetDefault("EXPLICIT");
    AddAttr<std::vector<int>>("dilations",
                              "(vector<int> default:{1, 1}), the "
                              "dilations(h_dilation, w_dilation) of "
                              "convolution operator.")
        .SetDefault({1, 1});
    // fusion options
    AddAttr<bool>(
        "fuse_shortcut",
        R"DOC((bool, default false). Whether a residual is added before relu.)DOC")
        .SetDefault(false);
    AddAttr<bool>(
        "fuse_dual",
        R"DOC((bool, default false). Whether two resblocks are added before relu.)DOC")
        .SetDefault(false);
    AddAttr<bool>(
        "fuse_add",
        R"DOC((bool, default false). Whether to add an additional gradient after dgrad.)DOC")
        .SetDefault(false);
    AddAttr<int>(
        "groups",
        "(int default:1), the groups number of the convolution operator. ")
        .SetDefault(1);
    AddComment(R"DOC(
FusedDgradDreluBnBwdWeight Operator
It fuses the backward of the following patterns:
(1)    BN -> ReLU -> Conv

                       |---> (optional branch)
(2)    BN1 -> Add -> ReLU -> Conv
       BN2 ----^

                      |---> (optional branch)
(3)    BN -> Add -> ReLU -> Conv
  (shortcut)--^

The meaning of three attributes are:
- fuse_shortcut: Whether a shortcut is added in the forward pattern, as in (2).
- fuse_dual: Whether two BN outputs are added in the forward pattern, as in (3).
- fuse_add: Whether ReLU output is used in a forward node other than Conv,
  marked in (2)(3) as (optional branch). In this case, the gradient of the branch
  should be added to the output dgrad.

Requirements:
- All tensors should have layout NHWC, except that W is NCHW.
- BN_dGamma, BN_dBeta, BN_eqscale_dy, BN_eqscale_x, BN_eqbias
  BN_mean, BN_inv_std, BN_scale, BN_bias should have shape [C]
  and dtype FP32.
- BN_X, dX, Relu_X should have input shape of Conv and dtype FP16.
)DOC");
  }
};

class FusedDgradDreluBnBwdWeightOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    // check basic IOs
    OP_INOUT_CHECK(
        ctx->HasInput("dY"), "Input", "dY", "FusedDgradDreluBnBwdWeight");
    OP_INOUT_CHECK(
        ctx->HasInput("W"), "Input", "W", "FusedDgradDreluBnBwdWeight");
    OP_INOUT_CHECK(ctx->HasInput("BN1_mean"),
                   "Input",
                   "BN1_mean",
                   "FusedDgradDreluBnBwdWeight");
    OP_INOUT_CHECK(ctx->HasInput("BN1_inv_std"),
                   "Input",
                   "BN1_inv_std",
                   "FusedDgradDreluBnBwdWeight");
    OP_INOUT_CHECK(ctx->HasInput("BN1_scale"),
                   "Input",
                   "BN1_scale",
                   "FusedDgradDreluBnBwdWeight");
    OP_INOUT_CHECK(ctx->HasInput("BN1_bias"),
                   "Input",
                   "BN1_bias",
                   "FusedDgradDreluBnBwdWeight");
    OP_INOUT_CHECK(
        ctx->HasInput("BN1_X"), "Input", "BN1_X", "FusedDgradDreluBnBwdWeight");

    OP_INOUT_CHECK(
        ctx->HasOutput("dX"), "Output", "dX", "FusedDgradDreluBnBwdWeight");
    OP_INOUT_CHECK(ctx->HasOutput("BN1_dGamma"),
                   "Output",
                   "BN1_dGamma",
                   "FusedDgradDreluBnBwdWeight");
    OP_INOUT_CHECK(ctx->HasOutput("BN1_dBeta"),
                   "Output",
                   "BN1_dBeta",
                   "FusedDgradDreluBnBwdWeight");
    OP_INOUT_CHECK(ctx->HasOutput("BN1_eqscale_dy"),
                   "Output",
                   "BN1_eqscale_dy",
                   "FusedDgradDreluBnBwdWeight");
    OP_INOUT_CHECK(ctx->HasOutput("BN1_eqscale_x"),
                   "Output",
                   "BN1_eqscale_x",
                   "FusedDgradDreluBnBwdWeight");
    OP_INOUT_CHECK(ctx->HasOutput("BN1_eqbias"),
                   "Output",
                   "BN1_eqbias",
                   "FusedDgradDreluBnBwdWeight");

    bool fuse_shortcut = ctx->Attrs().Get<bool>("fuse_shortcut");
    bool fuse_dual = ctx->Attrs().Get<bool>("fuse_dual");
    bool fuse_add = ctx->Attrs().Get<bool>("fuse_add");

    PADDLE_ENFORCE_EQ(
        fuse_shortcut && fuse_dual,
        0,
        platform::errors::InvalidArgument(
            "fuse_shortcut and fuse_dual should not be set at the same time."
            "Got fuse_shortcut=%d, fuse_dual=%d.",
            fuse_shortcut,
            fuse_dual));

    if (fuse_shortcut) {
      OP_INOUT_CHECK(ctx->HasInput("Relu_X"),
                     "Input",
                     "Relu_X",
                     "FusedDgradDreluBnBwdWeight");
    }

    if (fuse_dual) {
      OP_INOUT_CHECK(ctx->HasInput("BN2_mean"),
                     "Input",
                     "BN2_mean",
                     "FusedDgradDreluBnBwdWeight");
      OP_INOUT_CHECK(ctx->HasInput("BN2_inv_std"),
                     "Input",
                     "BN2_inv_std",
                     "FusedDgradDreluBnBwdWeight");
      OP_INOUT_CHECK(ctx->HasInput("BN2_scale"),
                     "Input",
                     "BN2_scale",
                     "FusedDgradDreluBnBwdWeight");
      OP_INOUT_CHECK(ctx->HasInput("BN2_bias"),
                     "Input",
                     "BN2_bias",
                     "FusedDgradDreluBnBwdWeight");
      OP_INOUT_CHECK(ctx->HasInput("BN2_X"),
                     "Input",
                     "BN2_X",
                     "FusedDgradDreluBnBwdWeight");

      OP_INOUT_CHECK(ctx->HasOutput("BN2_dGamma"),
                     "Output",
                     "BN2_dGamma",
                     "FusedDgradDreluBnBwdWeight");
      OP_INOUT_CHECK(ctx->HasOutput("BN2_dBeta"),
                     "Output",
                     "BN2_dBeta",
                     "FusedDgradDreluBnBwdWeight");
      OP_INOUT_CHECK(ctx->HasOutput("BN2_eqscale_dy"),
                     "Output",
                     "BN2_eqscale_dy",
                     "FusedDgradDreluBnBwdWeight");
      OP_INOUT_CHECK(ctx->HasOutput("BN2_eqscale_x"),
                     "Output",
                     "BN2_eqscale_x",
                     "FusedDgradDreluBnBwdWeight");
      OP_INOUT_CHECK(ctx->HasOutput("BN2_eqbias"),
                     "Output",
                     "BN2_eqbias",
                     "FusedDgradDreluBnBwdWeight");
    }

    if (fuse_add) {
      OP_INOUT_CHECK(ctx->HasInput("dX_branch"),
                     "Input",
                     "dX_branch",
                     "FusedDgradDreluBnBwdWeight");
    }

    int groups = ctx->Attrs().Get<int>("groups");
    PADDLE_ENFORCE_EQ(groups,
                      1,
                      platform::errors::InvalidArgument(
                          "Expect group to be 1, got %d.", groups));

    auto x_dims = ctx->GetInputDim("BN1_X");  // [N, H, W, C]
    auto y_dims = ctx->GetInputDim("dY");
    auto w_dims = ctx->GetInputDim("W");         // [K, C, R, S]
    auto c_dims = ctx->GetInputDim("BN1_mean");  // [C]

    // sanity checks
    PADDLE_ENFORCE_EQ(y_dims[y_dims.size() - 1],
                      w_dims[0],
                      platform::errors::InvalidArgument(
                          "dY should be of shape [N, P, Q, K]."
                          "W should be of shape [K, C, R, S]."
                          "The last dim of dY should match the first of W."
                          "Got dY shape=[%s], W shape=[%s]",
                          y_dims,
                          w_dims));
    PADDLE_ENFORCE_EQ(x_dims[x_dims.size() - 1],
                      w_dims[1],
                      platform::errors::InvalidArgument(
                          "X should be of shape [N, H, W, C]."
                          "W should be of shape [K, C, R, S]."
                          "The last dim of dY should match the second of W."
                          "Got dY shape=[%s], W shape=[%s]",
                          x_dims,
                          w_dims));
    PADDLE_ENFORCE_EQ(
        (c_dims.size() == 1) && c_dims[0] == w_dims[1],
        1,
        platform::errors::InvalidArgument("BN1_mean should be of shape [%d]."
                                          "Got: [%s]",
                                          w_dims[1],
                                          c_dims));

    // set output shapes
    ctx->SetOutputDim("dX", x_dims);
    ctx->SetOutputDim("BN1_dGamma", c_dims);
    ctx->SetOutputDim("BN1_dBeta", c_dims);
    ctx->SetOutputDim("BN1_eqscale_dy", c_dims);
    ctx->SetOutputDim("BN1_eqscale_x", c_dims);
    ctx->SetOutputDim("BN1_eqbias", c_dims);

    if (fuse_dual) {
      ctx->SetOutputDim("BN2_dGamma", c_dims);
      ctx->SetOutputDim("BN2_dBeta", c_dims);
      ctx->SetOutputDim("BN2_eqscale_dy", c_dims);
      ctx->SetOutputDim("BN2_eqscale_x", c_dims);
      ctx->SetOutputDim("BN2_eqbias", c_dims);
    }
  }

  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const {
    auto data_type = OperatorWithKernel::IndicateVarDataType(ctx, "dY");
    return phi::KernelKey(data_type, ctx.GetPlace());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(fused_dgrad_drelu_bnbwdweight,
                             ops::FusedDgradDreluBnBwdWeightOp,
                             ops::FusedDgradDreluBnBwdWeightOpMaker);
