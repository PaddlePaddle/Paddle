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
#include "paddle/fluid/operators/conv_op.h"
#include "paddle/fluid/platform/device/gpu/gpu_dnn.h"

namespace paddle {
namespace operators {

class FusedConvBnWgradOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    // inputs
    AddInput("BN_X", "Input to BN");
    AddInput("W", "Only to infer shape, not involved in computation.");
    AddInput("dY", "");
    AddInput("Scale", "");
    AddInput("Bias", "");
    // outputs
    AddOutput("dW", "");
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
    AddAttr<int>(
        "groups",
        "(int default:1), the groups number of the convolution operator. ")
        .SetDefault(1);
    AddComment(R"DOC(
FusedConvBnWgrad Operator
Ref: https://docs.nvidia.com/deeplearning/cudnn/developer-guide/index.html#convbnwgrad
It fuses the following pattern:

X = ReLU(BN_X * Scale + Bias)
dW = Wgrad(dY, X)

Requirements:
- All tensors should be FP16 and have NHWC layout
- Scale, Bias should have shape [C], where C is the channel
  dimension of BN_X.
)DOC");
  }
};

class FusedConvBnWgradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    // check basic IOs
    OP_INOUT_CHECK(ctx->HasInput("BN_X"), "Input", "BN_X", "FusedConvBnWgrad");
    OP_INOUT_CHECK(ctx->HasInput("dY"), "Input", "dY", "FusedConvBnWgrad");
    OP_INOUT_CHECK(ctx->HasInput("W"), "Input", "W", "FusedConvBnWgrad");
    OP_INOUT_CHECK(
        ctx->HasInput("Scale"), "Input", "Scale", "FusedConvBnWgrad");
    OP_INOUT_CHECK(ctx->HasInput("Bias"), "Input", "Bias", "FusedConvBnWgrad");
    OP_INOUT_CHECK(ctx->HasOutput("dW"), "Output", "dW", "FusedConvBnWgrad");

    auto w_dims = ctx->GetInputDim("W");      // [N, H, W, C]
    auto c_dims = ctx->GetInputDim("Scale");  // [C]

    int groups = ctx->Attrs().Get<int>("groups");
    PADDLE_ENFORCE_EQ(groups,
                      1,
                      platform::errors::InvalidArgument(
                          "Expect group to be 1, got %d.", groups));

    // sanity checks
    PADDLE_ENFORCE_EQ(
        (c_dims.size() == 1) && c_dims[0] == w_dims[1],
        1,
        platform::errors::InvalidArgument("Scale should be of shape [%d]."
                                          "Got: [%s]",
                                          w_dims[1],
                                          c_dims));
    // set output shapes
    ctx->SetOutputDim("dW", w_dims);
  }

  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const {
    auto data_type = OperatorWithKernel::IndicateVarDataType(ctx, "BN_X");
    return phi::KernelKey(data_type, ctx.GetPlace());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(fused_conv_bn_wgrad,
                             ops::FusedConvBnWgradOp,
                             ops::FusedConvBnWgradOpMaker);
