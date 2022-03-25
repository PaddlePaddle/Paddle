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

#include "paddle/fluid/operators/fill_diagonal_tensor_op.h"

namespace paddle {
namespace operators {

// calculate the offset\new_dims\(strides of dim1/dim2)\matoffset
void CalMatDims(framework::DDim out_dims, int dim1, int dim2, int64_t *offset,
                int64_t *new_dims, int64_t *strides, int64_t *matoffset) {
  int64_t dimprod = 1, batchdim = 1;
  int rank = out_dims.size();
  int matoffidx = 0;
  for (int i = rank - 1; i >= 0; i--) {
    if (i == dim2) {
      strides[0] = dimprod;
    } else if (i == dim1) {
      strides[1] = dimprod;
    } else {
      batchdim *= out_dims[i];
      // matoffset calculate the offset position of the diagonal defined by dim1
      // and dim2
      // the first circle calculate the final free dimension
      // and then calculate the front free dim one by one
      if (matoffidx == 0) {
        for (int64_t j = 0; j < out_dims[i]; j++) {
          matoffset[matoffidx] = dimprod * j;
          matoffidx++;
        }
      } else {
        auto size = matoffidx;
        for (int64_t j = 1; j < out_dims[i]; j++) {
          for (int64_t k = 0; k < size; k++) {
            matoffset[matoffidx] = matoffset[k] + dimprod * j;
            matoffidx++;
          }
        }
      }
    }
    dimprod *= out_dims[i];
  }

  auto diagdim = dim1;
  if (*offset >= 0) {
    diagdim = std::min(out_dims[dim1], out_dims[dim2] - *offset);
    *offset *= strides[0];
  } else {
    diagdim = std::min(out_dims[dim1] + *offset, out_dims[dim2]);
    *offset *= -strides[1];
  }
  new_dims[0] = batchdim;
  new_dims[1] = diagdim;
  return;
}

class FillDiagonalTensorOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddComment(R"DOC(Fill replace operator
                Fill the diagonal of an tensor with `Y` Tensor.
                )DOC");
    AddInput("X", "(Tensor) The input tensor.");
    AddInput("Y", "(Tensor) The input tensor to fill in.");
    AddOutput("Out",
              "Tensor, the output tensor, with the same shape and data type "
              "as input(x)");
    AddAttr<int>("dim1", "the first dim to figure out the diagonal")
        .SetDefault(0);
    AddAttr<int>("dim2", "the second dim to figure out the diagonal")
        .SetDefault(1);
    AddAttr<int64_t>("offset",
                     "offset of diagonal, zero means no offset, positive means "
                     "offset to up-right corner; negtive means offset to "
                     "bottom-left corner")
        .SetDefault(0);
  }
};

class FillDiagonalTensorOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *context) const override {
    OP_INOUT_CHECK(context->HasInput("X"), "Input", "X", "FillDiagonalTensor");
    OP_INOUT_CHECK(context->HasOutput("Out"), "Output", "Out",
                   "FillDiagonalTensor");
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

class FillDiagonalTensorOpVarTypeInference
    : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext *ctx) const override {
    auto var_type = ctx->GetInputType("X", 0);
    auto data_type = ctx->GetInputDataType("X", 0);
    ctx->SetOutputType("Out", var_type, framework::ALL_ELEMENTS);
    ctx->SetOutputDataType("Out", data_type, framework::ALL_ELEMENTS);
  }
};

template <typename T>
class FillDiagonalTensorKernel : public framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext &ctx) const override {
    auto *out = ctx.Output<framework::Tensor>("Out");
    auto *srctensor = ctx.Input<framework::Tensor>("Y");
    auto dim1 = ctx.Attr<int>("dim1");
    auto dim2 = ctx.Attr<int>("dim2");
    auto offset = ctx.Attr<int64_t>("offset");
    auto *xin = ctx.Input<framework::Tensor>("X");

    T *out_data = out->mutable_data<T>(ctx.GetPlace());
    const T *fill_data = srctensor->data<T>();

    framework::TensorCopy(*xin, ctx.GetPlace(), out);
    auto out_dims = out->dims();
    auto matdims = srctensor->dims();
    auto fill_dims = phi::flatten_to_2d(matdims, matdims.size() - 1);

    int64_t new_dims[2], strides[2];
    std::vector<int64_t> matdim;
    matdim.resize(fill_dims[0]);
    CalMatDims(out_dims, dim1, dim2, &offset, new_dims, strides, matdim.data());
    PADDLE_ENFORCE_EQ(
        new_dims[0], fill_dims[0],
        platform::errors::InvalidArgument("The dims should be %d x %d, but get "
                                          "%d x %d in fill tensor Y",
                                          new_dims[0], new_dims[1],
                                          fill_dims[0], fill_dims[1]));
    PADDLE_ENFORCE_EQ(
        new_dims[1], fill_dims[1],
        platform::errors::InvalidArgument("The dims should be %d x %d, but get "
                                          "%d x %d in fill tensor Y",
                                          new_dims[0], new_dims[1],
                                          fill_dims[0], fill_dims[1]));

    auto size = out->numel();
    for (int64_t i = 0; i < fill_dims[0]; i += 1) {
      auto sumoff = matdim[i] + offset;
      for (int64_t j = 0; j < fill_dims[1]; j += 1) {
        auto fill_index = j * (strides[1] + strides[0]) + sumoff;
        if (fill_index < size) {
          out_data[fill_index] = fill_data[i * fill_dims[1] + j];
        }
      }
    }
  }
};

class FillDiagonalTensorGradOp : public framework::OperatorWithKernel {
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
    return framework::OpKernelType(framework::TransToProtoVarType(dtype),
                                   ctx.GetPlace());
  }
};

template <typename T>
class FillDiagonalTensorGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> retv) const override {
    retv->SetType("fill_diagonal_tensor_grad");
    retv->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    retv->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    retv->SetAttrMap(this->Attrs());
  }
};

template <typename T>
class FillDiagonalTensorGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext &ctx) const override {
    auto *dx = ctx.Output<framework::Tensor>(framework::GradVarName("X"));
    auto *dout = ctx.Input<framework::Tensor>(framework::GradVarName("Out"));

    auto dim1 = ctx.Attr<int>("dim1");
    auto dim2 = ctx.Attr<int>("dim2");
    auto offset = ctx.Attr<int64_t>("offset");
    auto matrows = 1;

    if (dx) {
      auto *data = dx->mutable_data<T>(ctx.GetPlace());

      auto dx_dims = dx->dims();
      for (int i = 0; i < dx_dims.size(); i++) {
        if (i != dim1 && i != dim2) {
          matrows *= dx_dims[i];
        }
      }

      int64_t new_dims[2], strides[2];
      std::vector<int64_t> matdim;
      matdim.resize(matrows);
      CalMatDims(dx_dims, dim1, dim2, &offset, new_dims, strides,
                 matdim.data());

      auto size = dx->numel();
      framework::TensorCopy(*dout, ctx.GetPlace(), dx);

      for (int64_t i = 0; i < new_dims[0]; i += 1) {
        auto sumoff = matdim[i] + offset;
        for (int64_t j = 0; j < new_dims[1]; j += 1) {
          auto fill_index = j * (strides[1] + strides[0]) + sumoff;
          if (fill_index < size) {
            data[fill_index] = 0;
          }
        }
      }
    }
  }
};

DECLARE_INPLACE_OP_INFERER(FillDiagonalTensorOpInplaceInferer, {"X", "Out"});
DECLARE_INPLACE_OP_INFERER(FillDiagonalTensorGradOpInplaceInferer,
                           {framework::GradVarName("Out"),
                            framework::GradVarName("X")});

}  // namespace operators
}  // namespace paddle
namespace ops = paddle::operators;

REGISTER_OPERATOR(
    fill_diagonal_tensor, ops::FillDiagonalTensorOp,
    ops::FillDiagonalTensorOpMaker, ops::FillDiagonalTensorOpVarTypeInference,
    ops::FillDiagonalTensorGradOpMaker<paddle::framework::OpDesc>,
    ops::FillDiagonalTensorGradOpMaker<paddle::imperative::OpBase>,
    ops::FillDiagonalTensorOpInplaceInferer);

REGISTER_OPERATOR(fill_diagonal_tensor_grad, ops::FillDiagonalTensorGradOp,
                  ops::FillDiagonalTensorGradOpInplaceInferer);

REGISTER_OP_CPU_KERNEL(
    fill_diagonal_tensor, ops::FillDiagonalTensorKernel<float>,
    ops::FillDiagonalTensorKernel<double>,
    ops::FillDiagonalTensorKernel<int64_t>, ops::FillDiagonalTensorKernel<int>,
    ops::FillDiagonalTensorKernel<int8_t>,
    ops::FillDiagonalTensorKernel<uint8_t>,
    ops::FillDiagonalTensorKernel<paddle::platform::float16>,
    ops::FillDiagonalTensorKernel<paddle::platform::complex<float>>,
    ops::FillDiagonalTensorKernel<paddle::platform::complex<double>>,
    ops::FillDiagonalTensorKernel<bool>);

REGISTER_OP_CPU_KERNEL(
    fill_diagonal_tensor_grad, ops::FillDiagonalTensorGradKernel<float>,
    ops::FillDiagonalTensorGradKernel<double>,
    ops::FillDiagonalTensorGradKernel<int64_t>,
    ops::FillDiagonalTensorGradKernel<int>,
    ops::FillDiagonalTensorGradKernel<int8_t>,
    ops::FillDiagonalTensorGradKernel<uint8_t>,
    ops::FillDiagonalTensorGradKernel<paddle::platform::float16>,
    ops::FillDiagonalTensorGradKernel<paddle::platform::complex<float>>,
    ops::FillDiagonalTensorGradKernel<paddle::platform::complex<double>>,
    ops::FillDiagonalTensorGradKernel<bool>);
