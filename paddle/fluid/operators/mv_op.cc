/* Copyright (c) 2017 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <algorithm>
#include <utility>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/blas.h"
#ifdef PADDLE_WITH_MKLDNN
#include "paddle/fluid/platform/mkldnn_helper.h"
#endif

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class MVKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto &x = GET_DATA_SAFELY(context.Input<framework::Tensor>("X"), "Input",
                              "X", "mv");
    auto &y = GET_DATA_SAFELY(context.Input<framework::Tensor>("Y"), "Input",
                              "Y", "mv");
    auto *out = context.Output<framework::Tensor>("Out");
    out->mutable_data<T>(context.GetPlace());

    auto blas = math::GetBlas<DeviceContext, T>(context);

    // y.Resize(framework::make_ddim({y.dims()[0], 1}));

    // out->Resize(framework::make_ddim({x.dims()[0], 1}));
    blas.MatMul(x, y, out);
    // out->Resize(framework::make_ddim({x.dims()[0]}));
  }
};

class MVOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The first input of mv op");
    AddInput("Y", "The second input of mv op");
    AddOutput("Out", "The output of mv op");
    AddComment(R"DOC(
MV Operator.

This operator is used to perform matrix vector multiplication
of the input tensors `X` and `Y`.
)DOC");
  }
};

class MVOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext *context) const override {
    OP_INOUT_CHECK(context->HasInput("X"), "Input", "X", "mv");
    OP_INOUT_CHECK(context->HasInput("Y"), "Input", "Y", "mv");
    OP_INOUT_CHECK(context->HasOutput("Out"), "Output", "Out", "mv");

    auto dim_x = context->GetInputDim("X");
    auto dim_y = context->GetInputDim("Y");
    PADDLE_ENFORCE_EQ(
        dim_x.size(), 2,
        platform::errors::InvalidArgument(
            "The rank of input X should be 2, but is %d", dim_x.size()));
    PADDLE_ENFORCE_EQ(
        dim_y.size(), 2,
        platform::errors::InvalidArgument(
            "The rank of input Y should be 2, but is %d", dim_y.size()));
    // PADDLE_ENFORCE_EQ(
    //     dim_y[1] == 1, true,
    //     platform::errors::InvalidArgument(
    //         "The length of input Y' second dim should be 1, but is %d",
    //         dim_y[1]));
    PADDLE_ENFORCE_EQ(dim_x[1] == dim_y[0], true,
                      platform::errors::InvalidArgument(
                          "The length of input X' second dim should equal the "
                          "length of input Y,"
                          " but X[%d, %d], Y[%d, %d]",
                          dim_x[0], dim_x[1], dim_y[0], dim_y[1]));

    framework::DDim dim_out = framework::make_ddim({dim_x[0], dim_y[1]});

    context->SetOutputDim("Out", dim_out);
    context->ShareLoD("X", /*->*/ "Out");
  }
};

// Using dimensional constraints on matrix multiplication, it is
// straight-forward to check the following table for when X and Y
// are both matrices.
//
// dX = | dOut Y^T
// dY = | X^T dOut
template <typename DeviceContext, typename T>
class MVGradKernel : public framework::OpKernel<T> {
 public:
  void MatMul(const framework::ExecutionContext &context,
              const framework::Tensor &a, bool trans_a,
              const framework::Tensor &b, bool trans_b,
              framework::Tensor *out) const {
    out->mutable_data<T>(context.GetPlace());
    auto blas = math::GetBlas<DeviceContext, T>(context);
    // auto mat_dim_a = math::CreateMatrixDescriptor(a.dims(), 0, trans_a);
    // auto mat_dim_b = math::CreateMatrixDescriptor(b.dims(), 0, trans_b);

    // blas.MatMul(a, mat_dim_a, b, mat_dim_b, T(0), out, T(0));

    blas.MatMul(a, trans_a, b, trans_b, out);
  }

  void CalcInputGrad(const framework::ExecutionContext &context,
                     const framework::Tensor &a, bool trans_a,
                     const framework::Tensor &b, bool trans_b,
                     framework::Tensor *out) const {
    if (out == nullptr) return;

    MatMul(context, a, trans_a, b, trans_b, out);
  }

  void Compute(const framework::ExecutionContext &context) const override {
    auto x = *context.Input<framework::Tensor>("X");
    auto y = *context.Input<framework::Tensor>("Y");
    auto dout =
        *context.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto *dx = context.Output<framework::Tensor>(framework::GradVarName("X"));
    auto *dy = context.Output<framework::Tensor>(framework::GradVarName("Y"));

    framework::DDim dx_dims;
    if (dx) {
      dx_dims = dx->dims();
      if (dx_dims != x.dims()) {
        dx->Resize(x.dims());
      }
    }

    framework::DDim dy_dims;
    if (dy) {
      dy_dims = dy->dims();
      if (dy_dims != y.dims()) {
        dy->Resize(y.dims());
      }
    }

    CalcInputGrad(context, dout, false, y, true, dx);
    CalcInputGrad(context, x, true, dout, false, dy);

    if (dx) {
      if (dx_dims != x.dims()) {
        dx->Resize(dx_dims);
      }
    }
    if (dy) {
      if (dy_dims != y.dims()) {
        dy->Resize(dy_dims);
      }
    }
  }
};

template <typename T>
class MVOpGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> retv) const override {
    retv->SetType("mv_grad");
    retv->SetInput("X", this->Input("X"));
    retv->SetInput("Y", this->Input("Y"));
    retv->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    retv->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    retv->SetOutput(framework::GradVarName("Y"), this->InputGrad("Y"));
  }
};

class MVOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext *context) const override {
    OP_INOUT_CHECK(context->HasInput("X"), "Input", "X", "matmul");
    OP_INOUT_CHECK(context->HasInput("Y"), "Input", "Y", "matmul");
    OP_INOUT_CHECK(context->HasInput(framework::GradVarName("Out")), "Input",
                   "Out@GRAD", "matmul");
    auto x_dims = context->GetInputDim("X");
    auto y_dims = context->GetInputDim("Y");

    auto x_grad_name = framework::GradVarName("X");
    auto y_grad_name = framework::GradVarName("Y");

    if (context->HasOutput(x_grad_name)) {
      context->SetOutputDim(x_grad_name, x_dims);
    }
    if (context->HasOutput(y_grad_name)) {
      context->SetOutputDim(y_grad_name, y_dims);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

// REGISTER_OPERATOR(
//     mv, ops::MVOp, ops::MVOpMaker,
//     paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
//     paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(mv, ops::MVOp, ops::MVOpMaker,
                  ops::MVOpGradMaker<paddle::framework::OpDesc>,
                  ops::MVOpGradMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(mv_grad, ops::MVOpGrad);

REGISTER_OP_CPU_KERNEL(
    mv, ops::MVKernel<paddle::platform::CPUDeviceContext, float>,
    ops::MVKernel<paddle::platform::CPUDeviceContext, double>);
REGISTER_OP_CPU_KERNEL(
    mv_grad, ops::MVGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::MVGradKernel<paddle::platform::CPUDeviceContext, double>);

#ifdef PADDLE_WITH_CUDA
REGISTER_OP_CUDA_KERNEL(
    mv, ops::MVKernel<paddle::platform::CUDADeviceContext, float>,
    ops::MVKernel<paddle::platform::CUDADeviceContext, double>,
    ops::MVKernel<paddle::platform::CUDADeviceContext,
                  paddle::platform::float16>);
REGISTER_OP_CUDA_KERNEL(
    mv_grad, ops::MVGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::MVGradKernel<paddle::platform::CUDADeviceContext, double>,
    ops::MVGradKernel<paddle::platform::CUDADeviceContext,
                      paddle::platform::float16>);
#endif
