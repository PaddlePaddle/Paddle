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

#include "paddle/fluid/operators/fill_diagonal_op.h"

namespace paddle {
namespace operators {

int64_t CalStride(framework::DDim dim) {
  int rank = dim.size();
  int64_t dimsum = 1;
  int64_t strides = 0;
  for (int i = rank - 1; i >= 0; i--) {
    strides += dimsum;
    dimsum *= dim[i];
  }
  return strides;
}

class FillIDiagonalOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddComment(R"DOC(Fill replace operator
                Fill the diagonal of an tensor with 'value'.
                )DOC");
    AddInput("X", "(Tensor) The input tensor.");
    AddOutput("Out",
              "Tensor, the output tensor, with the same shape and data type "
              "as input(x)");
    AddAttr<float>(
        "value",
        "The float values of tensor, whose dim is one, and no need of grad")
        .SetDefault(0);
    AddAttr<bool>("wrap",
                  "the diagonal 'wrapped' after N columns for tall matrices")
        .SetDefault(false);
    AddAttr<int>("offset",
                 "offset of diagonal, zero means no offset, positive means "
                 "offset to up-right corner; negtive means offset to "
                 "bottom-left corner")
        .SetDefault(0);
  }
};

class FillIDiagonalOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *context) const override {
    OP_INOUT_CHECK(context->HasInput("X"), "Input", "X", "FillIDiagonal");
    OP_INOUT_CHECK(context->HasOutput("Out"), "Output", "Out", "FillIDiagonal");
    auto x_dims = context->GetInputDim("X");
    context->SetOutputDim("Out", x_dims);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"), ctx.GetPlace());
  }
};

class FillIDiagonalOpVarTypeInference : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext *ctx) const override {
    auto var_type = ctx->GetInputType("X", 0);
    auto data_type = ctx->GetInputDataType("X", 0);
    ctx->SetOutputType("Out", var_type, framework::ALL_ELEMENTS);
    ctx->SetOutputDataType("Out", data_type, framework::ALL_ELEMENTS);
  }
};

template <typename T>
class FillIDiagonalKernel : public framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext &ctx) const override {
    auto fill_val = ctx.template Attr<float>("value");
    auto *out = ctx.Output<framework::Tensor>("Out");
    auto offset = ctx.Attr<int>("offset");
    auto wrap = ctx.Attr<bool>("wrap");

    auto *xin = ctx.Input<framework::Tensor>("X");

    T temp_var = static_cast<T>(fill_val);

    T *out_data = out->mutable_data<T>(ctx.GetPlace());
    framework::TensorCopy(*xin, ctx.GetPlace(), out);

    auto out_dims = out->dims();
    auto strides = CalStride(out_dims);
    auto size = out->numel();

    // The wrap mode supported only the dims equels to 2; In wrap mode, the
    // value will be filled in cycles
    if (!wrap) {
      size = std::min(size, out_dims[1] * out_dims[1]);
    }

    for (int64_t i = offset; i < size; i += strides) {
      out_data[i] = temp_var;
    }
  }
};

class FillIDiagonalGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")), "Input",
                   "Out@GRAD", "mul");
    auto x_dims = ctx->GetInputDim(framework::GradVarName("Out"));
    auto x_grad_name = framework::GradVarName("X");
    if (ctx->HasOutput(x_grad_name)) {
      ctx->SetOutputDim(x_grad_name, x_dims);
    }
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    // Note: don't get data type from ctx.Input<framework::Tensor>("Input");
    auto dtype =
        ctx.Input<framework::Tensor>(framework::GradVarName("Out"))->type();
    return framework::OpKernelType(dtype, ctx.GetPlace());
  }
};

template <typename T>
class FillIDiagonalGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> retv) const override {
    retv->SetType("fill_diagonal_grad");
    retv->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    retv->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    retv->SetAttrMap(this->Attrs());
  }
};

template <typename T>
class FillIDiagonalGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext &ctx) const override {
    auto *dx = ctx.Output<framework::Tensor>(framework::GradVarName("X"));
    auto *dout = ctx.Input<framework::Tensor>(framework::GradVarName("Out"));

    auto offset = ctx.Attr<int>("offset");
    auto wrap = ctx.Attr<bool>("wrap");

    if (dx) {
      auto *data = dx->mutable_data<T>(ctx.GetPlace());
      framework::TensorCopy(*dout, ctx.GetPlace(), dx);

      auto dx_dims = dx->dims();
      auto strides = CalStride(dx_dims);
      auto size = dx->numel();
      auto wrapsize = std::min(size, dx_dims[1] * dx_dims[1]);

      // The wrap mode supported only the dims equels to 2; In wrap mode, the
      // value will be filled in cycles
      if (wrap) {
        wrapsize = size;
      }

      for (int64_t i = offset; i < wrapsize; i += strides) {
        data[i] = T(0);
      }
    }
  }
};

DECLARE_INPLACE_OP_INFERER(FillIDiagonalOpInplaceInferer, {"X", "Out"});
DECLARE_INPLACE_OP_INFERER(FillIDiagonalGradOpInplaceInferer,
                           {framework::GradVarName("Out"),
                            framework::GradVarName("X")});

}  // namespace operators
}  // namespace paddle
namespace ops = paddle::operators;

REGISTER_OPERATOR(fill_diagonal, ops::FillIDiagonalOp,
                  ops::FillIDiagonalOpMaker,
                  ops::FillIDiagonalOpVarTypeInference,
                  ops::FillIDiagonalGradOpMaker<paddle::framework::OpDesc>,
                  ops::FillIDiagonalGradOpMaker<paddle::imperative::OpBase>,
                  ops::FillIDiagonalOpInplaceInferer);

REGISTER_OPERATOR(fill_diagonal_grad, ops::FillIDiagonalGradOp,
                  ops::FillIDiagonalGradOpInplaceInferer);

REGISTER_OP_CPU_KERNEL(fill_diagonal, ops::FillIDiagonalKernel<float>,
                       ops::FillIDiagonalKernel<double>,
                       ops::FillIDiagonalKernel<int64_t>,
                       ops::FillIDiagonalKernel<int>,
                       ops::FillIDiagonalKernel<paddle::platform::float16>,
                       ops::FillIDiagonalKernel<bool>);

REGISTER_OP_CPU_KERNEL(fill_diagonal_grad, ops::FillIDiagonalGradKernel<float>,
                       ops::FillIDiagonalGradKernel<double>,
                       ops::FillIDiagonalGradKernel<int64_t>,
                       ops::FillIDiagonalGradKernel<int>,
                       ops::FillIDiagonalGradKernel<paddle::platform::float16>,
                       ops::FillIDiagonalGradKernel<bool>);
