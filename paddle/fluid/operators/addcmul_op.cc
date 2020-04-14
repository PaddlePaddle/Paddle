// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/operators/addcmul_op.h"
#include <algorithm>
#include <memory>
#include <vector>
#include "paddle/fluid/operators/elementwise/elementwise_op.h"

namespace paddle {
namespace operators {

template <typename T>
struct SameDimsElemwiseAdd<
    platform::CPUDeviceContext, T,
    typename std::enable_if<std::is_floating_point<T>::value>::type> {
  void operator()(const framework::ExecutionContext &ctx,
                  const framework::Tensor *x, const framework::Tensor *y,
                  framework::Tensor *z) {
    auto blas = math::GetBlas<platform::CPUDeviceContext, T>(ctx);
    blas.VADD(x->numel(), x->data<T>(), y->data<T>(), z->data<T>());
  }
};

template <typename T>
struct SameDimsElemwiseAdd<
    platform::CPUDeviceContext, T,
    typename std::enable_if<!std::is_floating_point<T>::value>::type> {
  void operator()(const framework::ExecutionContext &ctx,
                  const framework::Tensor *x, const framework::Tensor *y,
                  framework::Tensor *z) {
    auto eigen_x = framework::EigenVector<T>::Flatten(*x);
    auto eigen_y = framework::EigenVector<T>::Flatten(*y);
    auto eigen_z = framework::EigenVector<T>::Flatten(*z);
    auto &place = *ctx.template device_context<platform::CPUDeviceContext>()
                       .eigen_device();
    eigen_z.device(place) = eigen_x + eigen_y;
  }
};

template <typename T>
struct SameDimsElemwiseMul<
    platform::CPUDeviceContext, T,
    typename std::enable_if<std::is_floating_point<T>::value>::type> {
  void operator()(const framework::ExecutionContext &ctx,
                  const framework::Tensor *x, const framework::Tensor *y,
                  framework::Tensor *z) {
    auto blas = math::GetBlas<platform::CPUDeviceContext, T>(ctx);
    blas.VMUL(x->numel(), x->data<T>(), y->data<T>(), z->data<T>());
  }
};

template <typename T>
struct SameDimsElemwiseMul<
    platform::CPUDeviceContext, T,
    typename std::enable_if<!std::is_floating_point<T>::value>::type> {
  void operator()(const framework::ExecutionContext &ctx,
                  const framework::Tensor *x, const framework::Tensor *y,
                  framework::Tensor *z) {
    auto eigen_x = framework::EigenVector<T>::Flatten(*x);
    auto eigen_y = framework::EigenVector<T>::Flatten(*y);
    auto eigen_z = framework::EigenVector<T>::Flatten(*z);
    auto &place = *ctx.template device_context<platform::CPUDeviceContext>()
                       .eigen_device();
    eigen_z.device(place) = eigen_x * eigen_y;
  }
};

class AddcmulOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Input", "(Tensor) The input tensor to be added.");
    AddInput("Tensor1", "(Tensor) The tensor to be multiplied.");
    AddInput("Tensor2", "(Tensor) The tensor to be multiplied. ");
    AddOutput("Out", "(Tensor) The output tensor.");
    AddAttr<float>("value",
                   "(double default:1.0), The multiplier for Tensor1 * Tensor2")
        .SetDefault(1.0);
    AddAttr<int>("axis", "(int default:-1), The axis for broadcast")
        .SetDefault(-1);
    AddComment(R"DOC(
**Addcmul Operator**

Calculate the element-wise multiplication of tensor1 and tensor2,
then multiply the result by value, and add it to input

$$
out_i = input_i + value * tensor1_i * tensor2_i
$$

The shape of input, tensor1 and tensor2 must be broadcastable.

)DOC");
  }
};

class AddcmulOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInput("Input"), true,
                      platform::errors::NotFound(
                          "Input(Input) of AddcmulOp should not be null"));
    PADDLE_ENFORCE_EQ(ctx->HasInput("Tensor1"), true,
                      platform::errors::NotFound(
                          "Input(Tensor1) of AddcmulOp should not be null"));
    PADDLE_ENFORCE_EQ(ctx->HasInput("Tensor2"), true,
                      platform::errors::NotFound(
                          "Input(Tensor2) of AddcmulOp should not be null"));
    PADDLE_ENFORCE_EQ(ctx->HasOutput("Out"), true,
                      platform::errors::NotFound(
                          "Output(Out) of AddcmulOp should not be null."));

    auto tensor1_dims = ctx->GetInputDim("Tensor1");
    auto tensor2_dims = ctx->GetInputDim("Tensor2");
    int axis = ctx->Attrs().Get<int>("axis");
    axis = (axis == -1 ? std::abs(tensor1_dims.size() - tensor2_dims.size())
                       : axis);
    int max_dim = std::max(tensor1_dims.size(), tensor2_dims.size());
    std::vector<int> tensor1_dims_array(max_dim);
    std::vector<int> tensor2_dims_array(max_dim);
    std::vector<int> muled_dims_array(max_dim);
    GetBroadcastDimsArrays(tensor1_dims, tensor2_dims,
                           tensor1_dims_array.data(), tensor2_dims_array.data(),
                           muled_dims_array.data(), max_dim, axis);
    auto muled_dims = framework::make_ddim(muled_dims_array);

    auto input_dims = ctx->GetInputDim("Input");
    max_dim = std::max(input_dims.size(), muled_dims.size());
    std::vector<int> input_dims_array(max_dim);
    std::vector<int> out_dims_array(max_dim);
    axis = ctx->Attrs().Get<int>("axis");

    axis =
        (axis == -1 ? std::abs(input_dims.size() - muled_dims.size()) : axis);
    GetBroadcastDimsArrays(input_dims, muled_dims, input_dims_array.data(),
                           muled_dims_array.data(), out_dims_array.data(),
                           max_dim, axis);
    ctx->SetOutputDim("Out", framework::make_ddim(out_dims_array));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto input_data_type =
        OperatorWithKernel::IndicateVarDataType(ctx, "Input");
    return framework::OpKernelType(input_data_type, ctx.device_context());
  }
};

template <typename T>
class AddcmulGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("addcmul_grad");
    op->SetInput("Input", this->Input("Input"));
    op->SetInput("Tensor1", this->Input("Tensor1"));
    op->SetInput("Tensor2", this->Input("Tensor2"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));

    op->SetOutput(framework::GradVarName("Input"), this->InputGrad("Input"));
    op->SetOutput(framework::GradVarName("Tensor1"),
                  this->InputGrad("Tensor1"));
    op->SetOutput(framework::GradVarName("Tensor2"),
                  this->InputGrad("Tensor2"));

    op->SetAttrMap(this->Attrs());
  }
};

class AddcmulGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    auto out_grad_name = framework::GradVarName("Out");

    PADDLE_ENFORCE_EQ(ctx->HasInput(framework::GradVarName("Out")), true,
                      platform::errors::NotFound(
                          "the gradient of output(Out) must not be null"));
    auto input_grad_name = framework::GradVarName("Input");
    auto tensor1_grad_name = framework::GradVarName("Tensor1");
    auto tensor2_grad_name = framework::GradVarName("Tensor2");
    if (ctx->HasOutput(input_grad_name)) {
      ctx->ShareDim("Input", /*->*/ input_grad_name);
    }
    if (ctx->HasOutput(tensor1_grad_name)) {
      ctx->ShareDim("Tensor1", /*->*/ tensor1_grad_name);
    }
    if (ctx->HasOutput(tensor2_grad_name)) {
      ctx->ShareDim("Tensor2", /*->*/ tensor2_grad_name);
    }
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto input_data_type = OperatorWithKernel::IndicateVarDataType(
        ctx, framework::GradVarName("Out"));
    return framework::OpKernelType(input_data_type, ctx.device_context());
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(addcmul, ops::AddcmulOp, ops::AddcmulOpMaker,
                  ops::AddcmulGradOpMaker<paddle::framework::OpDesc>,
                  ops::AddcmulGradOpMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(addcmul_grad, ops::AddcmulGradOp);

REGISTER_OP_CPU_KERNEL(
    addcmul, ops::AddcmulKernel<paddle::platform::CPUDeviceContext, float>,
    ops::AddcmulKernel<paddle::platform::CPUDeviceContext, double>);
REGISTER_OP_CPU_KERNEL(
    addcmul_grad,
    ops::AddcmulGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::AddcmulGradKernel<paddle::platform::CPUDeviceContext, double>);
