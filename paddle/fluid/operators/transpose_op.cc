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

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/unary.h"

#ifdef PADDLE_WITH_MKLDNN
#include "paddle/fluid/platform/mkldnn_helper.h"
#endif
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/prim/api/composite_backward/composite_backward_api.h"
#include "paddle/fluid/prim/utils/static/composite_grad_desc_maker.h"

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

    int x_rank = x_dims.size();
    int axis_size = axis.size();

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

    std::vector<int> formated_axis = axis;
    std::vector<int> count(axis_size, 0);
    for (int i = 0; i < axis_size; i++) {
      PADDLE_ENFORCE_LT(axis[i],
                        axis_size,
                        platform::errors::InvalidArgument(
                            "The reduce dim index %d should be in the "
                            "range [ -dimension(X), dimension(X) ) "
                            "which dimesion = %d. But received dim index = %d.",
                            i,
                            axis_size,
                            axis[i]));
      PADDLE_ENFORCE_GE(axis[i],
                        -axis_size,
                        platform::errors::InvalidArgument(
                            "The reduce dim index %d should be in the "
                            "range [ -dimension(X), dimension(X) )  "
                            "which dimesion = %d. But received dim index = %d.",
                            i,
                            axis_size,
                            axis[i]));

      if (axis[i] < 0) {
        formated_axis[i] = axis[i] + axis_size;
      }
      PADDLE_ENFORCE_EQ(++count[formated_axis[i]],
                        1,
                        platform::errors::InvalidArgument(
                            "Each element of axis should be unique. but "
                            "axis[%d] is %d appear not only once",
                            i,
                            axis[i]));
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
    for (int i = 0; i < axis_size; i++) {
      out_dims[i] = x_dims[formated_axis[i]];
    }
    ctx->SetOutputDim("Out", out_dims);
  }

 protected:
  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto data_type = OperatorWithKernel::IndicateVarDataType(ctx, "X");
    auto &data_format = ctx.Attr<std::string>("data_format");
    phi::DataLayout layout_ = phi::StringToDataLayout(data_format);
    return phi::KernelKey(
        ctx.GetPlace(), layout_, phi::TransToPhiDataType(data_type));
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

 protected:
  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto data_type = OperatorWithKernel::IndicateVarDataType(
        ctx, framework::GradVarName("Out"));
    std::string data_format = ctx.Attr<std::string>("data_format");
    phi::DataLayout layout_ = phi::StringToDataLayout(data_format);
    return phi::KernelKey(
        ctx.GetPlace(), layout_, phi::TransToPhiDataType(data_type));
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
  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    framework::proto::VarType::Type data_type =
        OperatorWithKernel::IndicateVarDataType(ctx, "X");
    std::string data_format = ctx.Attr<std::string>("data_format");
    phi::DataLayout layout_ = phi::StringToDataLayout(data_format);
    return phi::KernelKey(
        ctx.GetPlace(), layout_, phi::TransToPhiDataType(data_type));
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
    VLOG(6) << "Runing transpose2_grad composite func";
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
    phi::DataLayout layout_ = phi::StringToDataLayout(data_format);
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
    paddle::framework::DefaultGradOpMaker<paddle::imperative::OpBase, true>);
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
