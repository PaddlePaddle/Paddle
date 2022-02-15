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

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

inline std::vector<int64_t> CorrelationOutputSize(int batch, int input_height,
                                                  int input_width, int stride1,
                                                  int stride2, int kernel_size,
                                                  int pad_size,
                                                  int max_displacement) {
  std::vector<int64_t> output_shape({batch});
  int kernel_radius = (kernel_size - 1) / 2;
  int border_radius = kernel_radius + max_displacement;
  int padded_input_height = input_height + 2 * pad_size;
  int padded_input_width = input_width + 2 * pad_size;
  int output_channel = ((max_displacement / stride2) * 2 + 1) *
                       ((max_displacement / stride2) * 2 + 1);
  output_shape.push_back(output_channel);
  int output_height =
      std::ceil(static_cast<float>(padded_input_height - 2 * border_radius) /
                static_cast<float>(stride1));
  int output_width =
      std::ceil(static_cast<float>(padded_input_width - 2 * border_radius) /
                static_cast<float>(stride1));
  output_shape.push_back(output_height);
  output_shape.push_back(output_width);
  return output_shape;
}

class CorrelationOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Input1", "Input is a 4-D Tensor with shape [N, C, H, W]");
    AddInput("Input2", "Input is a 4-D Tensor with shape [N, C, H, W]");
    AddOutput("Output",
              "(Tensor) The output tensor of correlation operator. "
              "It has same data fromat and data type as the Input.");
    AddAttr<int>("pad_size", "pad size for input1 and input2");
    AddAttr<int>("kernel_size", "kernel size of input1 and input2");
    AddAttr<int>("max_displacement", "max displacement of input1 and input2");
    AddAttr<int>("stride1", "Input1 stride");
    AddAttr<int>("stride2", "Input2 stride");
    AddAttr<int>("corr_type_multiply", "correlation coefficient").SetDefault(1);
    AddComment(
        R"DOC(Correlation of two feature map. Only support NCHW data format.)DOC");
  }
};

class CorrelationOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("Input1"), "Input", "X", "CorrelationOp");
    OP_INOUT_CHECK(ctx->HasInput("Input2"), "Input", "Y", "CorrelationOp");
    int stride1 = ctx->Attrs().Get<int>("stride1");
    int stride2 = ctx->Attrs().Get<int>("stride2");
    int max_displacement = ctx->Attrs().Get<int>("max_displacement");
    int pad_size = ctx->Attrs().Get<int>("pad_size");
    int kernel_size = ctx->Attrs().Get<int>("kernel_size");

    auto in_dims = ctx->GetInputDim("Input1");
    auto in2_dims = ctx->GetInputDim("Input2");

    PADDLE_ENFORCE_EQ(in_dims.size() == 4, true,
                      platform::errors::InvalidArgument(
                          "Input(X) of CorrelationOp must be 4 dims."
                          "But received dims is %d.",
                          in_dims.size()));

    PADDLE_ENFORCE_EQ(in2_dims.size() == 4, true,
                      platform::errors::InvalidArgument(
                          "Input(Y) of CorrelationOp must be 4 dims."
                          "But received dims is %d.",
                          in2_dims.size()));
    std::vector<int64_t> output_shape =
        CorrelationOutputSize(in_dims[0], in_dims[2], in_dims[3], stride1,
                              stride2, kernel_size, pad_size, max_displacement);
    ctx->SetOutputDim("Output", framework::make_ddim(output_shape));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto input_data_type =
        OperatorWithKernel::IndicateVarDataType(ctx, "Input1");
    PADDLE_ENFORCE_EQ(
        input_data_type,
        framework::TransToProtoVarType(ctx.Input<Tensor>("Input2")->dtype()),
        platform::errors::InvalidArgument(
            "X and Y shoule have the same datatype"));
    return framework::OpKernelType(input_data_type, ctx.GetPlace());
  }

  framework::OpKernelType GetKernelTypeForVar(
      const std::string& var_name, const Tensor& tensor,
      const framework::OpKernelType& expected_kernel_type) const override {
    return framework::OpKernelType(expected_kernel_type.data_type_,
                                   tensor.place(), tensor.layout());
  }
};

template <typename T>
class CorrelationOpGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("correlation_grad");
    op->SetInput("Input1", this->Input("Input1"));
    op->SetInput("Input2", this->Input("Input2"));
    op->SetInput(framework::GradVarName("Output"), this->OutputGrad("Output"));
    op->SetOutput(framework::GradVarName("Input1"), this->InputGrad("Input1"));
    op->SetOutput(framework::GradVarName("Input2"), this->InputGrad("Input2"));
    op->SetAttrMap(this->Attrs());
  }
};

class CorrelationOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("Input1"), "Input", "X", "CorrelationOp");
    OP_INOUT_CHECK(ctx->HasInput("Input2"), "Input", "Y", "CorrelationOp");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Output")), "Input",
                   "Output@GRAD", "CorrelationGradOp");

    auto in1_dims = ctx->GetInputDim("Input1");
    auto in2_dims = ctx->GetInputDim("Input2");
    ctx->SetOutputDim(framework::GradVarName("Input1"), in1_dims);
    ctx->SetOutputDim(framework::GradVarName("Input2"), in2_dims);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "Input1"), ctx.GetPlace());
  }
};

template <typename T>
class CorrelationKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE_EQ(
        platform::is_gpu_place(ctx.GetPlace()), true,
        platform::errors::Unimplemented("Correlation only supports GPU now."));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(correlation, ops::CorrelationOp, ops::CorrelationOpMaker,
                  ops::CorrelationOpGradMaker<paddle::framework::OpDesc>,
                  ops::CorrelationOpGradMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(correlation_grad, ops::CorrelationOpGrad);
REGISTER_OP_CPU_KERNEL(correlation, ops::CorrelationKernel<float>,
                       ops::CorrelationKernel<double>);
