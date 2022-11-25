/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
Indicesou may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

<<<<<<< HEAD
=======
#include "paddle/fluid/operators/unpool_op.h"

>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e
#include <memory>
#include <string>
#include <vector>

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/infermeta/backward.h"
#include "paddle/phi/infermeta/binary.h"

namespace paddle {
namespace operators {

class Unpool2dOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput(
        "X",
        "(Tensor) The input tensor of unpool operator. "
        "The format of input tensor is NCHW. Where N is batch size, C is the "
        "number of channels, H and W is the height and width of feature.");
    AddInput(
        "Indices",
        "(Tensor) The input tensor of the indices given out by MaxPool2d. "
        "The format of input tensor is NCHW. Where N is batch size, C is the "
        "number of channels, H and W is the height and width of feature.");
    AddOutput("Out",
              "(Tensor) The output tensor of unpool operator."
              "The format of output tensor is also NCHW."
              "Where N is batch size, C is "
              "the number of channels, H and W is the height and "
              "width of feature.");
    AddAttr<std::vector<int>>(
        "ksize",
        "(vector), the unpooling window size(height, width) "
        "of unpooling operator.");
    AddAttr<std::vector<int>>("strides",
                              "(vector, default:{1, 1}), "
                              "strides (height, width) of unpooling operator.")
        .SetDefault({1, 1});
    AddAttr<std::vector<int>>("paddings",
                              "(vector default:{0,0}), "
                              "paddings (height, width) of unpooling operator.")
        .SetDefault({0, 0});
    AddAttr<std::string>(
        "unpooling_type",
        "(string), unpooling type, can be \"max\" for max-unpooling ")
        .InEnum({"max"});
    AddAttr<std::vector<int>>("output_size",
                              "(vector, optional). The shape of output.")
        .SetDefault({0, 0})
        .SupportTensor();
    AddAttr<std::string>(
        "data_format",
        "(string, default NCHW) Only used in "
        "An optional string from: \"NHWC\", \"NCHW\". "
        "Defaults to \"NHWC\". Specify the data format of the output data, "
        "the input will be transformed automatically. ")
        .SetDefault("NCHW");
    AddComment(R"DOC(
Input shape is: $(N, C_{in}, H_{in}, W_{in})$, Output shape is:
$(N, C_{out}, H_{out}, W_{out})$, where
$$
H_{out} = (H_{in}-1) * strides[0] - 2 * paddings[0] + ksize[0] \\
W_{out} = (W_{in}-1) * strides[1] - 2 * paddings[1] + ksize[1]
$$
Paper: http://www.matthewzeiler.com/wp-content/uploads/2017/07/iccv2011.pdf
)DOC");
  }
};

class Unpool3dOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput(
        "X",
        "(Tensor) The input tensor of unpool operator. "
        "The format of input tensor is NCDHW. Where N is batch size, C is the "
        "number of channels, D, H and W is the depth, height and width of "
        "feature.");
    AddInput(
        "Indices",
        "(Tensor) The input tensor of the indices given out by MaxPool3d. "
        "The format of input tensor is NCDHW. Where N is batch size, C is the "
        "number of channels, D, H and W is the depth, height and width of "
        "feature.");
    AddOutput("Out",
              "(Tensor) The output tensor of unpool operator."
              "The format of output tensor is also NCDHW."
              "Where N is batch size, C is "
              "the number of channels, D, H and W is the depth, height and "
              "width of feature.");
    AddAttr<std::vector<int>>(
        "ksize",
        "(vector), the unpooling window size(depth, height, width) "
        "of unpooling operator.");
    AddAttr<std::vector<int>>(
        "strides",
        "(vector, default:{1, 1, 1}), "
        "strides (depth, height, width) of unpooling operator.")
        .SetDefault({1, 1, 1});
    AddAttr<std::vector<int>>(
        "paddings",
        "(vector default:{0, 0,0}), "
        "paddings (depth, height, width) of unpooling operator.")
        .SetDefault({0, 0, 0});
    AddAttr<std::string>(
        "unpooling_type",
        "(string), unpooling type, can be \"max\" for max-unpooling ")
        .InEnum({"max"});
    AddAttr<std::vector<int>>("output_size",
                              "(vector, optional). The shape of output.")
        .SetDefault({0, 0, 0});
    AddAttr<std::string>(
        "data_format",
        "(string, default NCDHW)"
        "Defaults to \"NCDHW\". Specify the data format of the output data, ")
        .SetDefault("NCDHW");
    AddComment(R"DOC(
Input shape is: $(N, C_{in}, D_{in}, H_{in}, W_{in})$, Output shape is:
$(N, C_{out}, D_{out}, H_{out}, W_{out})$, where
$$
D_{out} = (D_{in}-1) * strides[0] - 2 * paddings[0] + ksize[0] \\
H_{out} = (H_{in}-1) * strides[1] - 2 * paddings[1] + ksize[1] \\
W_{out} = (W_{in}-1) * strides[2] - 2 * paddings[2] + ksize[2]
$$
)DOC");
  }
};

int UnpoolOutputSize(int input_size, int ksize, int padding, int stride) {
  int output_size = (input_size - 1) * stride - 2 * padding + ksize;
  return output_size;
}

class UnpoolOp : public framework::OperatorWithKernel {
 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"),
        ctx.device_context());
  }

 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
<<<<<<< HEAD
=======
  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "Unpool");
    OP_INOUT_CHECK(ctx->HasInput("Indices"), "Input", "Indices", "Unpool");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "Unpool");
    auto in_x_dims = ctx->GetInputDim("X");
    auto in_y_dims = ctx->GetInputDim("Indices");
    std::string unpooling_type =
        ctx->Attrs().Get<std::string>("unpooling_type");
    std::vector<int> ksize = ctx->Attrs().Get<std::vector<int>>("ksize");
    std::vector<int> strides = ctx->Attrs().Get<std::vector<int>>("strides");
    std::vector<int> paddings = ctx->Attrs().Get<std::vector<int>>("paddings");
    std::vector<int> output_size =
        ctx->Attrs().Get<std::vector<int>>("output_size");
    PADDLE_ENFORCE_EQ(in_x_dims.size() == 4,
                      true,
                      platform::errors::InvalidArgument(
                          "Unpool Intput(X) must be of 4-dimensional, but "
                          "received Input(X)'s dimensions is %d.",
                          in_x_dims.size()));
    PADDLE_ENFORCE_EQ(in_x_dims,
                      in_y_dims,
                      platform::errors::InvalidArgument(
                          "The dimensions of Input(X) must equal to be"
                          "the dimensions of Input(Indices), but received"
                          "dimensions of Input(X) is [%d], received dimensions"
                          "of Input(Indices) is [%d]",
                          in_x_dims,
                          in_y_dims));

    std::vector<int64_t> output_shape({in_x_dims[0], in_x_dims[1]});
    for (size_t i = 0; i < ksize.size(); ++i) {
      if (!ctx->IsRuntime() && in_x_dims[i + 2] <= 0) {
        output_shape.push_back(-1);
      } else {
        output_shape.push_back(output_size[i]);
      }
    }
    ctx->SetOutputDim("Out", phi::make_ddim(output_shape));
  }
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e
};

class Unpool3dOp : public framework::OperatorWithKernel {
 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"),
        ctx.device_context());
  }

 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
<<<<<<< HEAD
=======
  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "Unpool3d");
    OP_INOUT_CHECK(ctx->HasInput("Indices"), "Input", "Indices", "Unpool3d");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "Unpool3d");
    auto in_x_dims = ctx->GetInputDim("X");
    auto in_y_dims = ctx->GetInputDim("Indices");
    std::string unpooling_type =
        ctx->Attrs().Get<std::string>("unpooling_type");
    std::vector<int> ksize = ctx->Attrs().Get<std::vector<int>>("ksize");
    std::vector<int> strides = ctx->Attrs().Get<std::vector<int>>("strides");
    std::vector<int> paddings = ctx->Attrs().Get<std::vector<int>>("paddings");
    std::vector<int> output_size =
        ctx->Attrs().Get<std::vector<int>>("output_size");
    PADDLE_ENFORCE_EQ(in_x_dims.size() == 5,
                      true,
                      platform::errors::InvalidArgument(
                          "Unpool Intput(X) must be of 5-dimensional, but "
                          "received Input(X)'s dimensions is %d.",
                          in_x_dims.size()));
    PADDLE_ENFORCE_EQ(in_x_dims,
                      in_y_dims,
                      platform::errors::InvalidArgument(
                          "The dimensions of Input(X) must equal to be"
                          "the dimensions of Input(Indices), but received"
                          "dimensions of Input(X) is [%d], received dimensions"
                          "of Input(Indices) is [%d]",
                          in_x_dims,
                          in_y_dims));

    std::vector<int64_t> output_shape({in_x_dims[0], in_x_dims[1]});
    for (size_t i = 0; i < ksize.size(); ++i) {
      if (!ctx->IsRuntime() && in_x_dims[i + 2] <= 0) {
        output_shape.push_back(-1);
      } else {
        output_shape.push_back(output_size[i]);
      }
    }
    ctx->SetOutputDim("Out", phi::make_ddim(output_shape));
  }
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e
};

template <typename T>
class UnpoolOpGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;
  void Apply(GradOpPtr<T> op) const override {
    op->SetType(this->ForwardOpType() + "_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput("Indices", this->Input("Indices"));
    op->SetInput("Out", this->Output("Out"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetAttrMap(this->Attrs());
  }
};

template <typename T>
class Unpool3dOpGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;
  void Apply(GradOpPtr<T> op) const override {
    op->SetType(this->ForwardOpType() + "_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput("Indices", this->Input("Indices"));
    op->SetInput("Out", this->Output("Out"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetAttrMap(this->Attrs());
  }
};

class UnpoolOpGrad : public framework::OperatorWithKernel {
 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"),
        ctx.device_context());
  }

 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
<<<<<<< HEAD
=======
  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "UnpoolGrad");
    OP_INOUT_CHECK(ctx->HasOutput(framework::GradVarName("X")),
                   "Output",
                   framework::GradVarName("X"),
                   "UnpoolGrad");
    ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("X"));
  }
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e
};

class Unpool3dOpGrad : public framework::OperatorWithKernel {
 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"),
        ctx.device_context());
  }

 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
<<<<<<< HEAD
=======
  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "Unpool3dGrad");
    OP_INOUT_CHECK(ctx->HasOutput(framework::GradVarName("X")),
                   "Output",
                   framework::GradVarName("X"),
                   "Unpool3dGrad");
    ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("X"));
  }
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
<<<<<<< HEAD
DECLARE_INFER_SHAPE_FUNCTOR(unpool,
                            UnpoolInferShapeFunctor,
                            PD_INFER_META(phi::UnpoolInferMeta));
=======
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e
REGISTER_OPERATOR(unpool,
                  ops::UnpoolOp,
                  ops::Unpool2dOpMaker,
                  ops::UnpoolOpGradMaker<paddle::framework::OpDesc>,
                  ops::UnpoolOpGradMaker<paddle::imperative::OpBase>,
                  UnpoolInferShapeFunctor);

<<<<<<< HEAD
DECLARE_INFER_SHAPE_FUNCTOR(unpool_grad,
                            UnpoolGradInferShapeFunctor,
                            PD_INFER_META(phi::UnchangedInferMeta));

REGISTER_OPERATOR(unpool_grad, ops::UnpoolOpGrad, UnpoolGradInferShapeFunctor);

DECLARE_INFER_SHAPE_FUNCTOR(unpool,
                            Unpool3dInferShapeFunctor,
                            PD_INFER_META(phi::Unpool3dInferMeta));

=======
REGISTER_OPERATOR(unpool_grad, ops::UnpoolOpGrad);
REGISTER_OP_CPU_KERNEL(unpool,
                       ops::UnpoolKernel<phi::CPUContext, float>,
                       ops::UnpoolKernel<phi::CPUContext, double>);
REGISTER_OP_CPU_KERNEL(unpool_grad,
                       ops::UnpoolGradKernel<phi::CPUContext, float>,
                       ops::UnpoolGradKernel<phi::CPUContext, double>);

>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e
REGISTER_OPERATOR(unpool3d,
                  ops::Unpool3dOp,
                  ops::Unpool3dOpMaker,
                  ops::Unpool3dOpGradMaker<paddle::framework::OpDesc>,
                  ops::Unpool3dOpGradMaker<paddle::imperative::OpBase>,
                  Unpool3dInferShapeFunctor);

DECLARE_INFER_SHAPE_FUNCTOR(unpool3d_grad,
                            Unpool3dGradInferShapeFunctor,
                            PD_INFER_META(phi::UnchangedInferMeta));

<<<<<<< HEAD
REGISTER_OPERATOR(unpool3d_grad,
                  ops::Unpool3dOpGrad,
                  Unpool3dGradInferShapeFunctor);
=======
REGISTER_OPERATOR(unpool3d_grad, ops::Unpool3dOpGrad);
REGISTER_OP_CPU_KERNEL(unpool3d,
                       ops::Unpool3dKernel<phi::CPUContext, float>,
                       ops::Unpool3dKernel<phi::CPUContext, double>);
REGISTER_OP_CPU_KERNEL(unpool3d_grad,
                       ops::Unpool3dGradKernel<phi::CPUContext, float>,
                       ops::Unpool3dGradKernel<phi::CPUContext, double>);
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e
