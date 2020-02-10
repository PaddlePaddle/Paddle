/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/fc_op.h"
#include <vector>

namespace paddle {
namespace operators {

class FCOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInput("Input"), true,
                      "X(Input) of Fully Connected should not be null.");
    PADDLE_ENFORCE_EQ(ctx->HasOutput("Out"), true,
                      "Out(Output) of Fully Connected should not be null.");
    PADDLE_ENFORCE_EQ(ctx->HasInput("W"), true,
                      "W(Input) of Fully Connected should not be null.");

    auto in_dims = ctx->GetInputDim("Input");
    auto w_dims = ctx->GetInputDim("W");
    bool padding_weights = ctx->Attrs().Get<bool>("padding_weights");

    if (ctx->HasInput("Bias")) {
      auto bias_dims = ctx->GetInputDim("Bias");
      auto w_dims1 = padding_weights ? w_dims[1] - 4 : w_dims[1];
      if (bias_dims.size() == 2) {
        PADDLE_ENFORCE_EQ(bias_dims[0], 1,
                          platform::errors::InvalidArgument(
                              "The shape of Bias is invalid."
                              "The height of Bias should be 1."
                              "But received height of Bias is %d.",
                              bias_dims[0]));
        PADDLE_ENFORCE_EQ(
            bias_dims[1], w_dims1,
            platform::errors::InvalidArgument(
                "The shape of Bias is invalid."
                "The width of Bias should be equal to width of Weight."
                "But received width of Bias is %d and width of Weight is %d.",
                bias_dims[1], w_dims1));
      } else if (bias_dims.size() == 1) {
        PADDLE_ENFORCE_EQ(
            bias_dims[0], w_dims1,
            platform::errors::InvalidArgument(
                "The shape of Bias is invalid."
                "The height of Bias should be equal to the width of weight."
                "But received height of Bias is %d and width of Weight is %d.",
                bias_dims[0], w_dims1));
      }
    }

    auto& activation_type = ctx->Attrs().Get<std::string>("activation_type");
    if (!activation_type.empty()) {
      PADDLE_ENFORCE_EQ(activation_type, "relu",
                        "Activation %s is not supportetd in fc now.",
                        activation_type.c_str());
    }
    if (ctx->Attrs().Get<bool>("use_mkldnn")) {
      PADDLE_ENFORCE_EQ(
          in_dims.size() >= 2 && in_dims.size() <= 4, true,
          platform::errors::Unimplemented(
              "Fully Connected input should be 2D, 3D or 4D tensor."));
    }
    PADDLE_ENFORCE_EQ(w_dims.size(), 2,
                      "Fully Connected weights should be 2-D tensor.");
    int in_num_col_dims = ctx->Attrs().Get<int>("in_num_col_dims");
    PADDLE_ENFORCE_GT(
        in_dims.size(), in_num_col_dims,
        "The input tensor Input's rank of FCOp should be larger than "
        "in_num_col_dims.");

    std::vector<int64_t> output_dims;
    FCOutputSize(in_dims, w_dims, output_dims, in_num_col_dims,
                 padding_weights);

    ctx->SetOutputDim("Out", framework::make_ddim(output_dims));
    ctx->ShareLoD("Input", "Out");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    framework::LibraryType library = framework::LibraryType::kPlain;
    framework::DataLayout layout = framework::DataLayout::kAnyLayout;
    int customized_type_value =
        framework::OpKernelType::kDefaultCustomizedTypeValue;
    auto input_data_type =
        OperatorWithKernel::IndicateVarDataType(ctx, "Input");
    if (ctx.Attr<bool>("use_mkldnn")) {
      library = framework::LibraryType::kMKLDNN;
      layout = framework::DataLayout::kMKLDNN;
      using framework::proto::VarType;
      customized_type_value = (input_data_type == VarType::INT8 ||
                               input_data_type == VarType::UINT8)
                                  ? kFCMKLDNNINT8
                                  : kFCMKLDNNFP32;
    }
    return framework::OpKernelType(input_data_type, ctx.GetPlace(), layout,
                                   library, customized_type_value);
  }
};

class FCOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Input",
             "(Tensor), The input tensor of fully connected operator.");
    AddInput("W", "(Tensor), The weight fc op with shape (I, O).");
    AddInput("Bias", "(Tensor, optional) Bias vector with shape (1 x O")
        .AsDispensable();
    AddOutput("Out",
              "(Tensor) The output tensor of fully connected operator. ");
    AddAttr<int>("in_num_col_dims",
                 "(int, default 1), The fc op can take tensors with more than "
                 "two dimensions as its inputs.")
        .SetDefault(1)
        .EqualGreaterThan(1);
    AddAttr<std::string>("activation_type",
                         "Activation type used in fully connected operator.")
        .SetDefault("");
    AddAttr<bool>("use_mkldnn",
                  "(bool, default false) Only used in mkldnn kernel")
        .SetDefault(false);
    AddAttr<bool>(
        "padding_weights",
        "(bool, default false) When padding weights in the fc fuse pass, "
        "the 'padding_weights' attribute is set as true.")
        .SetDefault(false);
    AddAttr<bool>(framework::kAllKernelsMustComputeRuntimeShape,
                  "Skip calling InferShape() function in the runtime.")
        .SetDefault(true);
    /* int8 parameters */
    AddAttr<bool>("use_quantizer",
                  "(bool, default false) "
                  "Set to true for operators that should be quantized and use "
                  "int8 kernel. "
                  "Only used on CPU.")
        .SetDefault(false);
    AddAttr<float>("Scale_in",
                   "(float, default 1.0f), The quantize scale of input data")
        .SetDefault(1.0f);
    AddAttr<std::vector<float>>("Scale_weights",
                                "(std::vector<float>, default {1.0f}), The "
                                "quantize scale of weights data")
        .SetDefault({1.0f});
    AddAttr<float>("Scale_out",
                   "(float, default 1.0f), The quantize scale of output data")
        .SetDefault(1.0f);
    AddAttr<bool>("force_fp32_output",
                  "(bool, default false) Force INT8 kernel output FP32, only "
                  "used in MKL-DNN INT8")
        .SetDefault(false);
    AddComment(R"DOC(
Fully Connected Operator.

The fully connected operation calculates the output based on the input, weights and bias.
The size of each dimension of the parameters checked in the infer-shape.
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(
    fc, ops::FCOp, ops::FCOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OP_CPU_KERNEL(
    fc, ops::FCOpKernel<paddle::platform::CPUDeviceContext, float>,
    ops::FCOpKernel<paddle::platform::CPUDeviceContext, double>);
