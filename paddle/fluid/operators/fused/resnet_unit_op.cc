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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

// Shape of bitmask
static framework::DDim GetBitmaskDims(std::vector<int> out_shape) {
  int c = out_shape.back();
  int64_t nhw = std::accumulate(out_shape.begin(), out_shape.end(), 1,
                                std::multiplies<int>()) /
                c;
  int32_t c_int32_elems = ((c + 63) & ~63) / 32;
  int32_t nhw_int32_elems = ((nhw + 31) & ~31);
  std::vector<int> bitmask_shape = {nhw_int32_elems, c_int32_elems, 1};
  return framework::make_ddim(bitmask_shape);
}

class ResNetUnitOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const {
    // Check input
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "ResNetUnitOp");
    OP_INOUT_CHECK(ctx->HasInput("FilterX"), "Input", "FilterX",
                   "ResNetUnitOp");
    OP_INOUT_CHECK(ctx->HasInput("ScaleX"), "Input", "ScaleX", "ResNetUnitOp");
    OP_INOUT_CHECK(ctx->HasInput("BiasX"), "Input", "BiasX", "ResNetUnitOp");
    OP_INOUT_CHECK(ctx->HasInput("MeanX"), "Input", "MeanX", "ResNetUnitOp");
    OP_INOUT_CHECK(ctx->HasInput("VarX"), "Input", "VarX", "ResNetUnitOp");

    bool fuse_add = ctx->Attrs().Get<bool>("fuse_add");
    bool has_shortcut = ctx->Attrs().Get<bool>("has_shortcut");
    if (fuse_add || has_shortcut) {
      OP_INOUT_CHECK(ctx->HasInput("Z"), "Input", "Z", "ResNetUnitOp");
    }
    if (has_shortcut) {
      OP_INOUT_CHECK(ctx->HasInput("FilterZ"), "Input", "FilterZ",
                     "ResNetUnitOp");
      OP_INOUT_CHECK(ctx->HasInput("ScaleZ"), "Input", "ScaleZ",
                     "ResNetUnitOp");
      OP_INOUT_CHECK(ctx->HasInput("BiasZ"), "Input", "BiasZ", "ResNetUnitOp");
      OP_INOUT_CHECK(ctx->HasInput("MeanZ"), "Input", "MeanZ", "ResNetUnitOp");
      OP_INOUT_CHECK(ctx->HasInput("VarZ"), "Input", "VarZ", "ResNetUnitOp");
    }

    // Check output
    OP_INOUT_CHECK(ctx->HasOutput("Y"), "Output", "Y", "ResNetUnitOp");
    OP_INOUT_CHECK(ctx->HasOutput("BitMask"), "Output", "BitMask",
                   "ResNetUnitOp");
    OP_INOUT_CHECK(ctx->HasOutput("ConvX"), "Output", "ConvX", "ResNetUnitOp");
    OP_INOUT_CHECK(ctx->HasOutput("SavedMeanX"), "Output", "SavedMeanX",
                   "ResNetUnitOp");
    OP_INOUT_CHECK(ctx->HasOutput("SavedInvstdX"), "Output", "SavedInvstdX",
                   "ResNetUnitOp");
    OP_INOUT_CHECK(ctx->HasOutput("RunningMeanX"), "Output", "RunningMeanX",
                   "ResNetUnitOp");
    OP_INOUT_CHECK(ctx->HasOutput("RunningVarX"), "Output", "RunningVarX",
                   "ResNetUnitOp");
    if (has_shortcut) {
      OP_INOUT_CHECK(ctx->HasOutput("ConvZ"), "Output", "ConvZ",
                     "ResNetUnitOp");
      OP_INOUT_CHECK(ctx->HasOutput("SavedMeanZ"), "Output", "SavedMeanZ",
                     "ResNetUnitOp");
      OP_INOUT_CHECK(ctx->HasOutput("SavedInvstdZ"), "Output", "SavedInvstdZ",
                     "ResNetUnitOp");
      OP_INOUT_CHECK(ctx->HasOutput("RunningMeanZ"), "Output", "RunningMeanZ",
                     "ResNetUnitOp");
      OP_INOUT_CHECK(ctx->HasOutput("RunningVarZ"), "Output", "RunningVarZ",
                     "ResNetUnitOp");
    }

    // make sure Mean/RunningMean and Var/RunningVar share memory
    PADDLE_ENFORCE_EQ(
        ctx->Inputs("MeanX")[0], ctx->Outputs("RunningMeanX")[0],
        platform::errors::InvalidArgument(
            "MeanX and RunningMeanX should share the same memory"));
    PADDLE_ENFORCE_EQ(ctx->Inputs("VarX")[0], ctx->Outputs("RunningVarX")[0],
                      platform::errors::InvalidArgument(
                          "VarX and RunningVarX should share the same memory"));
    if (has_shortcut) {
      PADDLE_ENFORCE_EQ(
          ctx->Inputs("MeanZ")[0], ctx->Outputs("RunningMeanZ")[0],
          platform::errors::InvalidArgument(
              "MeanZ and RunningMeanZ should share the same memory"));
      PADDLE_ENFORCE_EQ(
          ctx->Inputs("VarZ")[0], ctx->Outputs("RunningVarZ")[0],
          platform::errors::InvalidArgument(
              "VarZ and RunningVarZ should share the same memory"));
    }

    // Check dims of inputs
    const auto x_dims = ctx->GetInputDim("X");
    const auto w_dims = ctx->GetInputDim("FilterX");
    std::vector<int64_t> bn_param_shape =
        framework::vectorize(ctx->GetInputDim("ScaleX"));
    if (1 == bn_param_shape.size()) {
      bn_param_shape = {1, 1, 1, bn_param_shape[0]};
    }
    framework::DDim bn_param_dims = framework::make_ddim(bn_param_shape);
    PADDLE_ENFORCE_EQ(x_dims.size(), 4, platform::errors::InvalidArgument(
                                            "The dimensions of input "
                                            "must equal to 4."
                                            "But received: the shape of input "
                                            "= [%s], the dimension of input = "
                                            "[%d]",
                                            x_dims, x_dims.size()));
    PADDLE_ENFORCE_EQ(w_dims.size(), 4,
                      platform::errors::InvalidArgument(
                          "The dimensions of filter "
                          "must equal to 4."
                          "But received: the shape of filter "
                          "= [%s], the dimension of filter = [%d] ",
                          w_dims, w_dims.size()));
    PADDLE_ENFORCE_EQ(bn_param_dims.size(), 4,
                      platform::errors::InvalidArgument(
                          "The dimensions of bn param "
                          "must equal to 4."
                          "But received: the shape of bn param "
                          "= [%s], the dimension of bn param = [%d] ",
                          bn_param_dims, bn_param_dims.size()));
    auto data_format = ctx->Attrs().Get<std::string>("data_format");
    PADDLE_ENFORCE_EQ(
        data_format, "NHWC",
        platform::errors::InvalidArgument("The data format must equal to NHWC. "
                                          "But received: the data format "
                                          "= [%s]",
                                          data_format));
    // Calculate the dims of outputs
    int batch = x_dims[0];
    int output_channel = w_dims[0];
    int filter_size = w_dims[2];
    int stride = ctx->Attrs().Get<int>("stride");
    int padding = ctx->Attrs().Get<int>("padding");
    int out_h = (x_dims[1] + padding * 2 - filter_size) / stride + 1;
    int out_w = (x_dims[2] + padding * 2 - filter_size) / stride + 1;
    std::vector<int> out_shape = {batch, out_h, out_w, output_channel};

    auto y_dims = framework::make_ddim(out_shape);
    auto bitmask_dims = GetBitmaskDims(out_shape);
    // Set dims of outputs
    ctx->SetOutputDim("Y", y_dims);
    ctx->SetOutputDim("BitMask", bitmask_dims);
    ctx->SetOutputDim("ConvX", y_dims);
    ctx->SetOutputDim("SavedMeanX", bn_param_dims);
    ctx->SetOutputDim("SavedInvstdX", bn_param_dims);
    ctx->SetOutputDim("RunningMeanX", bn_param_dims);
    ctx->SetOutputDim("RunningVarX", bn_param_dims);
    if (has_shortcut) {
      ctx->SetOutputDim("ConvZ", y_dims);
      ctx->SetOutputDim("SavedMeanZ", bn_param_dims);
      ctx->SetOutputDim("SavedInvstdZ", bn_param_dims);
      ctx->SetOutputDim("RunningMeanZ", bn_param_dims);
      ctx->SetOutputDim("RunningVarZ", bn_param_dims);
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const {
    auto input_data_type = OperatorWithKernel::IndicateVarDataType(ctx, "X");
    // By default, the type of the scale, bias, mean,
    // and var tensors should be float when input tensor's dtype is float16.
    auto bn_param_type = framework::proto::VarType::FP32;

    PADDLE_ENFORCE_EQ(bn_param_type, ctx.Input<Tensor>("ScaleX")->type(),
                      platform::errors::InvalidArgument(
                          "Scale input should be of float type"));
    PADDLE_ENFORCE_EQ(bn_param_type, ctx.Input<Tensor>("BiasX")->type(),
                      platform::errors::InvalidArgument(
                          "Bias input should be of float type"));
    framework::LibraryType library = framework::LibraryType::kPlain;
    framework::DataLayout layout = framework::DataLayout::kAnyLayout;
    return framework::OpKernelType(input_data_type, ctx.GetPlace(), layout,
                                   library);
  }
};

class ResNetUnitOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddInput("X", "The input 1 tensor");
    AddInput("FilterX", "Filter tensor of input 1");
    AddInput("ScaleX", "Scale tensor of input 1 used in batchnorm");
    AddInput("BiasX", "Bias tensor of input 1 used in batchnorm");
    AddInput("MeanX", "Mean tensor of input 1 used in batchnorm");
    AddInput("VarX", "Variance tensor of input 1 used in batchnorm");
    AddInput("Z", "The input 2 tensor").AsDispensable();
    AddInput("FilterZ", "Filter tensor of input 2").AsDispensable();
    AddInput("ScaleZ", "Scale tensor of input 2").AsDispensable();
    AddInput("BiasZ", "Bias tensor of input 2").AsDispensable();
    AddInput("MeanZ", "Mean tensor of input 2").AsDispensable();
    AddInput("VarZ", "Variance tensor of input 2").AsDispensable();
    AddOutput("Y", "The result of the resnet unit");
    AddOutput("BitMask", "The bitmask generated after relu");
    AddOutput("ConvX", "The output of input 1 after conv");
    AddOutput("SavedMeanX", "Mean of input 1 in the current batch");
    AddOutput("SavedInvstdX", "Invstd of input 1 in the current batch");
    AddOutput("RunningMeanX", "Shared memory with MeanX");
    AddOutput("RunningVarX", "Shared memory with VarX");
    AddOutput("ConvZ", "The output of input 2 after conv").AsDispensable();
    AddOutput("SavedMeanZ", "Mean of input 1 in the current batch")
        .AsDispensable();
    AddOutput("SavedInvstdZ", "Invstd of input 1 in the current batch")
        .AsDispensable();
    AddOutput("RunningMeanZ", "Shared memory with MeanZ").AsDispensable();
    AddOutput("RunningVarZ", "Shared memory with VarZ").AsDispensable();
    AddAttr<int>("stride", "").SetDefault(1);
    AddAttr<int>("stride_z", "").SetDefault(1);
    AddAttr<int>("padding", "").SetDefault(0);
    AddAttr<int>("dilation", "").SetDefault(1);
    AddAttr<int>("group", "").SetDefault(1);
    AddAttr<float>("momentum", "").SetDefault(0.9);
    AddAttr<float>("epsilon", "").SetDefault(1e-5);
    AddAttr<std::string>("data_format", "").SetDefault("NHWC");
    AddAttr<bool>("fuse_add", "").SetDefault(false);
    AddAttr<bool>("has_shortcut", "").SetDefault(false);
    AddAttr<bool>("use_global_stats", "").SetDefault(false);
    AddAttr<bool>("is_test",
                  "(bool, default false) Set to true for inference only, false "
                  "for training. Some layers may run faster when this is true.")
        .SetDefault(false);
    AddAttr<bool>("use_addto", "").SetDefault(false);
    AddAttr<std::string>("act_type", "The activation type to be fused.")
        .SetDefault("relu");
    AddComment(R"DOC(
Fusion op of the basic unit of resnet block. 

The implementation is based on the latest fusion op interface in cuDNN v8.0.
For more details: 
https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnFusedOps_t

)DOC");
  }
};

class ResNetUnitGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const {
    // check input
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "ResNetUnitGradOp");
    OP_INOUT_CHECK(ctx->HasInput("FilterX"), "Input", "FilterX",
                   "ResNetUnitGradOp");
    OP_INOUT_CHECK(ctx->HasInput("ConvX"), "Input", "ConvX",
                   "ResNetUnitGradOp");
    OP_INOUT_CHECK(ctx->HasInput("ScaleX"), "Input", "ScaleX",
                   "ResNetUnitGradOp");
    OP_INOUT_CHECK(ctx->HasInput("BiasX"), "Input", "BiasX",
                   "ResNetUnitGradOp");
    OP_INOUT_CHECK(ctx->HasInput("SavedMeanX"), "Input", "SavedMeanX",
                   "ResNetUnitGradOp");
    OP_INOUT_CHECK(ctx->HasInput("SavedInvstdX"), "Input", "SavedInvstdX",
                   "ResNetUnitGradOp");

    bool fuse_add = ctx->Attrs().Get<bool>("fuse_add");
    bool has_shortcut = ctx->Attrs().Get<bool>("has_shortcut");
    if (fuse_add || has_shortcut) {
      OP_INOUT_CHECK(ctx->HasInput("Z"), "Input", "Z", "ResNetUnitGradOp");
    }
    if (has_shortcut) {
      OP_INOUT_CHECK(ctx->HasInput("FilterZ"), "Input", "FilterZ",
                     "ResNetUnitGradOp");
      OP_INOUT_CHECK(ctx->HasInput("ConvZ"), "Input", "ConvZ",
                     "ResNetUnitGradOp");
      OP_INOUT_CHECK(ctx->HasInput("ScaleZ"), "Input", "ScaleZ",
                     "ResNetUnitGradOp");
      OP_INOUT_CHECK(ctx->HasInput("BiasZ"), "Input", "BiasZ",
                     "ResNetUnitGradOp");
      OP_INOUT_CHECK(ctx->HasInput("SavedMeanZ"), "Input", "SavedMeanZ",
                     "ResNetUnitGradOp");
      OP_INOUT_CHECK(ctx->HasInput("SavedInvstdZ"), "Input", "SavedInvstdZ",
                     "ResNetUnitGradOp");
    }
    OP_INOUT_CHECK(ctx->HasInput("Y"), "Input", "Y", "ResNetUnitGradOp");
    OP_INOUT_CHECK(ctx->HasInput("BitMask"), "Input", "BitMask",
                   "ResNetUnitGradOp");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Y")), "Input",
                   framework::GradVarName("Y"), "ResNetUnitGradOp");

    // check output
    OP_INOUT_CHECK(ctx->HasOutput(framework::GradVarName("X")), "Output",
                   framework::GradVarName("X"), "ResNetUnitGradOp");
    OP_INOUT_CHECK(ctx->HasOutput(framework::GradVarName("FilterX")), "Output",
                   framework::GradVarName("FilterX"), "ResNetUnitGradOp");
    OP_INOUT_CHECK(ctx->HasOutput(framework::GradVarName("ScaleX")), "Output",
                   framework::GradVarName("ScaleX"), "ResNetUnitGradOp");
    OP_INOUT_CHECK(ctx->HasOutput(framework::GradVarName("BiasX")), "Output",
                   framework::GradVarName("BiasX"), "ResNetUnitGradOp");
    if (fuse_add) {
      OP_INOUT_CHECK(ctx->HasOutput(framework::GradVarName("Z")), "Output",
                     framework::GradVarName("Z"), "ResNetUnitGradOp");
    }
    if (has_shortcut) {
      OP_INOUT_CHECK(ctx->HasOutput(framework::GradVarName("FilterZ")),
                     "Output", framework::GradVarName("FilterZ"),
                     "ResNetUnitGradOp");
      OP_INOUT_CHECK(ctx->HasOutput(framework::GradVarName("ScaleZ")), "Output",
                     framework::GradVarName("ScaleZ"), "ResNetUnitGradOp");
      OP_INOUT_CHECK(ctx->HasOutput(framework::GradVarName("BiasZ")), "Output",
                     framework::GradVarName("BiasZ"), "ResNetUnitGradOp");
    }
    const auto x_dims = ctx->GetInputDim("X");
    const auto filter_x_dims = ctx->GetInputDim("FilterX");
    const auto param_dims = ctx->GetInputDim("ScaleX");
    ctx->SetOutputDim(framework::GradVarName("X"), x_dims);
    ctx->SetOutputDim(framework::GradVarName("FilterX"), filter_x_dims);
    ctx->SetOutputDim(framework::GradVarName("ScaleX"), param_dims);
    ctx->SetOutputDim(framework::GradVarName("BiasX"), param_dims);
    if (fuse_add || has_shortcut) {
      const auto z_dims = ctx->GetInputDim("Z");
      ctx->SetOutputDim(framework::GradVarName("Z"), z_dims);
    }
    if (has_shortcut) {
      const auto filter_z_dims = ctx->GetInputDim("FilterZ");
      ctx->SetOutputDim(framework::GradVarName("FilterZ"), filter_z_dims);
      ctx->SetOutputDim(framework::GradVarName("ScaleZ"), param_dims);
      ctx->SetOutputDim(framework::GradVarName("BiasZ"), param_dims);
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const {
    PADDLE_ENFORCE_NOT_NULL(
        ctx.InputVar(framework::GradVarName("Y")),
        platform::errors::NotFound(
            "Can not find Y@GRAD in the execution context."));

    framework::LibraryType library = framework::LibraryType::kPlain;
    framework::DataLayout layout = framework::DataLayout::kAnyLayout;

    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"), ctx.GetPlace(),
        layout, library);
  }
};

template <typename T>
class ResNetUnitGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("resnet_unit_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput("FilterX", this->Input("FilterX"));
    op->SetInput("ConvX", this->Output("ConvX"));
    op->SetInput("ScaleX", this->Input("ScaleX"));
    op->SetInput("BiasX", this->Input("BiasX"));
    op->SetInput("SavedMeanX", this->Output("SavedMeanX"));
    op->SetInput("SavedInvstdX", this->Output("SavedInvstdX"));
    op->SetInput("Z", this->Input("Z"));
    op->SetInput("FilterZ", this->Input("FilterZ"));
    op->SetInput("ConvZ", this->Output("ConvZ"));
    op->SetInput("ScaleZ", this->Input("ScaleZ"));
    op->SetInput("BiasZ", this->Input("BiasZ"));
    op->SetInput("SavedMeanZ", this->Output("SavedMeanZ"));
    op->SetInput("SavedInvstdZ", this->Output("SavedInvstdZ"));
    op->SetInput("Y", this->Output("Y"));
    op->SetInput("BitMask", this->Output("BitMask"));
    op->SetInput(framework::GradVarName("Y"), this->OutputGrad("Y"));

    op->SetAttrMap(this->Attrs());

    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetOutput(framework::GradVarName("FilterX"),
                  this->InputGrad("FilterX"));
    op->SetOutput(framework::GradVarName("ScaleX"), this->InputGrad("ScaleX"));
    op->SetOutput(framework::GradVarName("BiasX"), this->InputGrad("BiasX"));
    op->SetOutput(framework::GradVarName("Z"), this->InputGrad("Z"));
    op->SetOutput(framework::GradVarName("FilterZ"),
                  this->InputGrad("FilterZ"));
    op->SetOutput(framework::GradVarName("ScaleZ"), this->InputGrad("ScaleZ"));
    op->SetOutput(framework::GradVarName("BiasZ"), this->InputGrad("BiasZ"));
  }
};

class ResNetUnitOpInferVarType
    : public framework::PassInDtypeAndVarTypeToOutput {
 protected:
  std::unordered_map<std::string, std::string>& GetInputOutputWithSameType()
      const override {
    static std::unordered_map<std::string, std::string> m{{"X", /*->*/ "Y"}};
    return m;
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(resnet_unit, ops::ResNetUnitOp, ops::ResNetUnitOpMaker,
                  ops::ResNetUnitOpInferVarType,
                  ops::ResNetUnitGradOpMaker<paddle::framework::OpDesc>,
                  ops::ResNetUnitGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(resnet_unit_grad, ops::ResNetUnitGradOp);
