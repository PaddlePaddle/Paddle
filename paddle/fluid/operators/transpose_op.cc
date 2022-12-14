/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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

#ifdef PADDLE_WITH_MKLDNN
#include "paddle/fluid/platform/mkldnn_helper.h"
#endif
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

class TransposeOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "Transpose");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "Transpose");
    auto x_dims = ctx->GetInputDim("X");
    std::vector<int> axis = ctx->Attrs().Get<std::vector<int>>("axis");

    size_t x_rank = x_dims.size();
    size_t axis_size = axis.size();

    // Note: x_rank > axis_size when fuse squeeze2 + transpose2, else x_rank ==
    // axis_size
    PADDLE_ENFORCE_GE(x_rank,
                      axis_size,
                      platform::errors::InvalidArgument(
                          "The input tensor's dimension "
                          "should be equal to or greater than the axis's size. "
                          "But received input tensor's dimension is %d, "
                          "axis's size is %d",
                          x_rank,
                          axis_size));

    std::vector<int> count(axis_size, 0);
    for (size_t i = 0; i < axis_size; i++) {
      PADDLE_ENFORCE_GE(axis[i],
                        0,
                        platform::errors::InvalidArgument(
                            "The axis should be greater than or equal to 0."
                            "But received %d of axis[%d]",
                            axis[i],
                            i));

      PADDLE_ENFORCE_EQ(
          axis[i] < static_cast<int>(axis_size) && ++count[axis[i]] == 1,
          true,
          platform::errors::InvalidArgument(
              "Each element of Attribute axis should "
              "be a unique value range from 0 to (dims - 1), "
              "where the dims is the axis's size, "
              "unique value means this axis value can appear only once. "
              "But received axis[%d] is %d, axis_size is %d, "
              "count[axis[%d]] is %d",
              i,
              axis[i],
              axis_size,
              i,
              count[axis[i]]));
    }

    framework::DDim out_dims(x_dims);
#ifdef PADDLE_WITH_MKLDNN
    // Here we need to match dims to paddle layout
    // as we are producing non-oneDNN result
    if (ctx->IsRunMKLDNNKernel() && (x_dims.size() >= 3) &&
        (phi::OneDNNContext::tls().get_cur_paddle_data_layout() ==
         phi::DataLayout::kNHWC)) {
      auto dims = phi::vectorize<int>(x_dims);
      std::rotate(dims.begin() + 1, dims.begin() + 2, dims.end());
      x_dims = x_dims.reshape(dims);
      VLOG(3)
          << "Rotating Shape in Transpose from: kMKLDNN to: kNHWC output_shape";
    }
#endif
    for (size_t i = 0; i < axis_size; i++) {
      out_dims[i] = x_dims[axis[i]];
    }
    ctx->SetOutputDim("Out", out_dims);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto data_type = OperatorWithKernel::IndicateVarDataType(ctx, "X");
    auto &data_format = ctx.Attr<std::string>("data_format");
    phi::DataLayout layout_ = phi::StringToDataLayout(data_format);
    return framework::OpKernelType(data_type, ctx.GetPlace(), layout_);
  }
};

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
    AddAttr<bool>(
        "use_quantizer",
        "(bool, default false) "
        "This parameter is no longer used. Use 'mkldnn_data_type' instead.")
        .SetDefault(false)
        .AsExtra();
    AddAttr<std::string>(
        "mkldnn_data_type",
        "(string, default \"float32\"). Data type of mkldnn kernel")
        .SetDefault("float32")
        .InEnum({"float32", "int8", "bfloat16"})
        .AsExtra();
    /* int8 parameters */
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

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "TransposeOpGrad");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")),
                   "Input",
                   framework::GradVarName("Out"),
                   "TransposeOpGrad");
    auto x_dims = ctx->GetInputDim("X");
    ctx->SetOutputDim(framework::GradVarName("X"), x_dims);
    if (ctx->HasOutput(framework::GradVarName("X"))) {
      ctx->SetOutputDim(framework::GradVarName("X"), x_dims);
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto data_type = OperatorWithKernel::IndicateVarDataType(
        ctx, framework::GradVarName("Out"));
    std::string data_format = ctx.Attr<std::string>("data_format");
    phi::DataLayout layout_ = phi::StringToDataLayout(data_format);
    return framework::OpKernelType(data_type, ctx.GetPlace(), layout_);
  }
};

// FIXME(zcd): transpose2 adds an intermediate output(XShape) based on
// transpose, the XShape is used to carry the shape and lod of X which
// will be used in transpose_grad, in this way, the framework can reuse
// the memory of X immediately the transpose2_op is finished.
// Considering compatibility issues, we could not fix transpose2_op
class Transpose2Op : public TransposeOp {
 public:
  Transpose2Op(const std::string &type,
               const framework::VariableNameMap &inputs,
               const framework::VariableNameMap &outputs,
               const framework::AttributeMap &attrs)
      : TransposeOp(type, inputs, outputs, attrs) {}

  void InferShape(framework::InferShapeContext *ctx) const override {
    TransposeOp::InferShape(ctx);
    if (!ctx->HasOutput("XShape")) return;
    const auto &in_dims = ctx->GetInputDim("X");
    std::vector<int64_t> x_shape_dim(in_dims.size() + 1);
    x_shape_dim[0] = 0;
    for (int i = 0; i < in_dims.size(); ++i) {
      x_shape_dim[i + 1] = in_dims[i];
    }
    ctx->SetOutputDim("XShape", phi::make_ddim(x_shape_dim));
    ctx->ShareLoD("X", /*->*/ "XShape");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    framework::proto::VarType::Type data_type =
        OperatorWithKernel::IndicateVarDataType(ctx, "X");
    std::string data_format = ctx.Attr<std::string>("data_format");
    phi::DataLayout layout_ = phi::StringToDataLayout(data_format);
    return framework::OpKernelType(data_type, ctx.GetPlace(), layout_);
  }
};

class Transpose2OpMaker : public framework::OpProtoAndCheckerMaker {
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
    AddOutput("XShape", "(Tensor)The output tensor.")
        .AsIntermediate()
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

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(
        ctx->HasInput("XShape"), "Input", "XShape", "Transpose2OpGrad");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")),
                   "Input",
                   framework::GradVarName("Out"),
                   "Transpose2OpGrad");
    if (ctx->HasOutput(framework::GradVarName("X"))) {
      auto xshape_dim = ctx->GetInputDim("XShape");
      auto x_shape_dim = phi::slice_ddim(xshape_dim, 1, xshape_dim.size());
      ctx->SetOutputDim(framework::GradVarName("X"), x_shape_dim);
      ctx->ShareLoD("XShape", framework::GradVarName("X"));
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    framework::proto::VarType::Type data_type =
        OperatorWithKernel::IndicateVarDataType(ctx,
                                                framework::GradVarName("Out"));
    std::string data_format = ctx.Attr<std::string>("data_format");
    phi::DataLayout layout_ = phi::StringToDataLayout(data_format);
    return framework::OpKernelType(data_type, ctx.GetPlace(), layout_);
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

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    transpose,
    ops::TransposeOp,
    ops::TransposeOpMaker,
    paddle::framework::DefaultGradOpMaker<paddle::framework::OpDesc, true>,
    paddle::framework::DefaultGradOpMaker<paddle::imperative::OpBase, true>);
REGISTER_OPERATOR(transpose_grad,
                  ops::TransposeOpGrad,
                  ops::TransposeGradInferVarType);

REGISTER_OPERATOR(transpose2,
                  ops::Transpose2Op,
                  ops::Transpose2OpMaker,
                  ops::Transpose2GradMaker<paddle::framework::OpDesc>,
                  ops::Transpose2GradMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(transpose2_grad,
                  ops::Transpose2OpGrad,
                  ops::TransposeGradInferVarType,
                  ops::Transpose2DoubleGradMaker<paddle::framework::OpDesc>,
                  ops::Transpose2DoubleGradMaker<paddle::imperative::OpBase>);
