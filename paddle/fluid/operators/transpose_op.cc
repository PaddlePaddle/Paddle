/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

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
#include <vector>

#include "paddle/fluid/operators/transpose_op.h"

namespace paddle {
namespace operators {

phi::KernelKey TransposeOp::GetExpectedKernelType(
    const framework::ExecutionContext &ctx) const {
  auto data_type = OperatorWithKernel::IndicateVarDataType(ctx, "X");
  auto &data_format = ctx.Attr<std::string>("data_format");
  phi::DataLayout layout_ = common::StringToDataLayout(data_format);
  return phi::KernelKey(
      ctx.GetPlace(), layout_, phi::TransToPhiDataType(data_type));
}

class TransposeOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput(
        "X",
        "(Tensor) The input tensor, tensors with rank up to 6 are supported.");
    AddOutput("Out", "(Tensor)The output tensor.");
    AddAttr<std::vector<int>>(
        "axis",
        "(vector<int>) A list of values, and the size of the list should be "
        "the same with the input tensor rank. This operator permutes the input "
        "tensor's axes according to the values given.");
    AddAttr<bool>("use_mkldnn",
                  "(bool, default false) Only used in mkldnn kernel")
        .SetDefault(false)
        .AsExtra();
    AddAttr<std::string>(
        "data_format",
        "(string, default NCHW) Only used in "
        "An optional string from: \"NHWC\", \"NCHW\". "
        "Defaults to \"NHWC\". Specify the data format of the output data, "
        "the input will be transformed automatically. ")
        .SetDefault("AnyLayout")
        .AsExtra();
    AddAttr<std::string>(
        "mkldnn_data_type",
        "(string, default \"float32\"). Data type of mkldnn kernel")
        .SetDefault("float32")
        .InEnum({"float32", "int8", "bfloat16"})
        .AsExtra();
    AddComment(R"DOC(
Transpose Operator.

The input tensor will be permuted according to the axes given.
The behavior of this operator is similar to how `numpy.transpose` works.

- suppose the input `X` is a 2-D tensor:
    $$
    X = \begin{pmatrix}
    0 &1 &2 \\
    3 &4 &5
    \end{pmatrix}$$

    the given `axes` is: $[1, 0]$, and $Y$ = transpose($X$, axis)

    then the output $Y$ is:

    $$
    Y = \begin{pmatrix}
         0 &3 \\
         1 &4  \\
         2 &5
    \end{pmatrix}$$

- Given a input tensor with shape $(N, C, H, W)$ and the `axes` is
$[0, 2, 3, 1]$, then shape of the output tensor will be: $(N, H, W, C)$.

)DOC");
  }
};

class TransposeOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto data_type = OperatorWithKernel::IndicateVarDataType(
        ctx, framework::GradVarName("Out"));
    std::string data_format = ctx.Attr<std::string>("data_format");
    phi::DataLayout layout_ = common::StringToDataLayout(data_format);
    return phi::KernelKey(
        ctx.GetPlace(), layout_, phi::TransToPhiDataType(data_type));
  }
};

void Transpose2Op::InferShape(framework::InferShapeContext *ctx) const {
  using CompatMetaTensor = framework::CompatMetaTensor;
  CompatMetaTensor x(ctx->GetInputVarPtrs("X")[0], ctx->IsRuntime());
  CompatMetaTensor out(ctx->GetOutputVarPtrs("Out")[0], ctx->IsRuntime());
  std::vector<int> axis = ctx->Attrs().Get<std::vector<int>>("axis");
  phi::TransposeInferMeta(x, axis, &out);

  if (!ctx->HasOutput("XShape")) return;
  const auto &in_dims = ctx->GetInputDim("X");
  std::vector<int64_t> x_shape_dim(in_dims.size() + 1);
  x_shape_dim[0] = 0;
  for (int i = 0; i < in_dims.size(); ++i) {
    x_shape_dim[i + 1] = in_dims[i];
  }
  ctx->SetOutputDim("XShape", common::make_ddim(x_shape_dim));
  ctx->ShareLoD("X", /*->*/ "XShape");
}

phi::KernelKey Transpose2Op::GetExpectedKernelType(
    const framework::ExecutionContext &ctx) const {
  auto data_type = OperatorWithKernel::IndicateVarDataType(ctx, "X");
  auto &data_format = ctx.Attr<std::string>("data_format");
  phi::DataLayout layout_ = common::StringToDataLayout(data_format);
  return phi::KernelKey(
      ctx.GetPlace(), layout_, phi::TransToPhiDataType(data_type));
}

void Transpose2OpMaker::Make() {
  AddInput(
      "X",
      "(Tensor) The input tensor, tensors with rank up to 6 are supported.");
  AddOutput("Out", "(Tensor)The output tensor.");
  AddAttr<std::vector<int>>(
      "axis",
      "(vector<int>) A list of values, and the size of the list should be "
      "the same with the input tensor rank. This operator permutes the input "
      "tensor's axes according to the values given.");
  AddOutput("XShape", "(Tensor)The output tensor.").AsIntermediate().AsExtra();
  AddComment(R"DOC(
Transpose Operator.

The input tensor will be permuted according to the axes given.
The behavior of this operator is similar to how `numpy.transpose` works.

- suppose the input `X` is a 2-D tensor:
    $$
    X = \begin{pmatrix}
    0 &1 &2 \\
    3 &4 &5
    \end{pmatrix}$$

    the given `axes` is: $[1, 0]$, and $Y$ = transpose($X$, axis)

    then the output $Y$ is:

    $$
    Y = \begin{pmatrix}
         0 &3 \\
         1 &4  \\
         2 &5
    \end{pmatrix}$$

- Given a input tensor with shape $(N, C, H, W)$ and the `axes` is
$[0, 2, 3, 1]$, then shape of the output tensor will be: $(N, H, W, C)$.

)DOC");
  Apply();
}

template <typename T>
class Transpose2GradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

  void Apply(GradOpPtr<T> grad_op) const override {
    grad_op->SetType("transpose2_grad");
    grad_op->SetInput("XShape", this->Output("XShape"));
    grad_op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    grad_op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    grad_op->SetAttrMap(this->Attrs());
  }
};

class Transpose2CompositeGradOpMaker : public prim::CompositeGradOpMakerBase {
  using prim::CompositeGradOpMakerBase::CompositeGradOpMakerBase;

 public:
  void Apply() override {
    paddle::Tensor xshape = this->GetSingleForwardOutput("XShape");
    paddle::Tensor out_grad = this->GetSingleOutputGrad("Out");
    paddle::Tensor dx = this->GetSingleInputGrad("X");
    auto *dx_ptr = this->GetOutputPtr(&dx);
    std::string dx_name = this->GetOutputName(dx);
    std::vector<int> axis =
        static_cast<std::vector<int>>(this->Attr<std::vector<int>>("axis"));
    VLOG(6) << "Running transpose2_grad composite func";
    prim::transpose_grad<prim::DescTensor>(out_grad, axis, dx_ptr);
    this->RecoverOutputName(dx, dx_name);
  }
};

template <typename T>
class Transpose2DoubleGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

  void Apply(GradOpPtr<T> grad_op) const override {
    grad_op->SetType("transpose2");
    grad_op->SetInput("X", this->OutputGrad(framework::GradVarName("X")));
    grad_op->SetOutput("Out", this->InputGrad(framework::GradVarName("Out")));
    grad_op->SetOutput("XShape", this->Input("XShape"));
    grad_op->SetAttrMap(this->Attrs());
  }
};

class Transpose2OpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    framework::proto::VarType::Type data_type =
        OperatorWithKernel::IndicateVarDataType(ctx,
                                                framework::GradVarName("Out"));
    std::string data_format = ctx.Attr<std::string>("data_format");
    phi::DataLayout layout_ = common::StringToDataLayout(data_format);
    return phi::KernelKey(
        ctx.GetPlace(), layout_, phi::TransToPhiDataType(data_type));
  }
};

class TransposeGradInferVarType : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext *ctx) const override {
    ctx->SyncTypeAndDataType(framework::GradVarName("Out"),
                             framework::GradVarName("X"));
  }
};

}  // namespace operators
}  // namespace paddle
DECLARE_INFER_SHAPE_FUNCTOR(transpose,
                            TransposeInferShapeFunctor,
                            PD_INFER_META(phi::TransposeInferMeta));

DECLARE_INFER_SHAPE_FUNCTOR(transpose_grad,
                            TransposeGradInferShapeFunctor,
                            PD_INFER_META(phi::TransposeGradInferMeta));

DECLARE_INFER_SHAPE_FUNCTOR(transpose2_grad,
                            Transpose2GradInferShapeFunctor,
                            PD_INFER_META(phi::TransposeGradInferMeta));

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    transpose,
    ops::TransposeOp,
    ops::TransposeOpMaker,
    paddle::framework::DefaultGradOpMaker<paddle::framework::OpDesc, true>,
    paddle::framework::DefaultGradOpMaker<paddle::imperative::OpBase, true>,
    TransposeInferShapeFunctor);

REGISTER_OPERATOR(transpose_grad,
                  ops::TransposeOpGrad,
                  ops::TransposeGradInferVarType,
                  TransposeGradInferShapeFunctor);

REGISTER_OPERATOR(transpose2,
                  ops::Transpose2Op,
                  ops::Transpose2OpMaker,
                  ops::Transpose2GradMaker<paddle::framework::OpDesc>,
                  ops::Transpose2GradMaker<paddle::imperative::OpBase>,
                  ops::Transpose2CompositeGradOpMaker);
REGISTER_OPERATOR(transpose2_grad,
                  ops::Transpose2OpGrad,
                  ops::TransposeGradInferVarType,
                  ops::Transpose2DoubleGradMaker<paddle::framework::OpDesc>,
                  ops::Transpose2DoubleGradMaker<paddle::imperative::OpBase>,
                  Transpose2GradInferShapeFunctor);
