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

#include "paddle/fluid/operators/kthvalue_op.h"
#include <memory>
#include "paddle/fluid/framework/generator.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"

namespace paddle {
namespace operators {

class KthvalueOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "kthvalue");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "kthvalue");
    OP_INOUT_CHECK(ctx->HasOutput("Indices"), "Output", "Indices", "kthvalue");
    auto input_dims = ctx->GetInputDim("X");
    const int& dim_size = input_dims.size();
    int axis = static_cast<int>(ctx->Attrs().Get<int>("axis"));
    PADDLE_ENFORCE_LT(axis, dim_size,
                      paddle::platform::errors::InvalidArgument(
                          "the axis must be [-%d, %d), but received %d .",
                          dim_size, dim_size, axis));
    PADDLE_ENFORCE_GE(axis, -dim_size,
                      paddle::platform::errors::InvalidArgument(
                          "the axis must be [-%d, %d), but received %d .",
                          dim_size, dim_size, axis));
    if (axis < 0) axis += dim_size;
    int k = static_cast<int>(ctx->Attrs().Get<int>("k"));
    PADDLE_ENFORCE_GE(
        k, 1, paddle::platform::errors::InvalidArgument(
                  "the k in the kthvalue must >= 1, but received %d .", k));
    PADDLE_ENFORCE_GE(input_dims.size(), 1,
                      paddle::platform::errors::InvalidArgument(
                          "input of kthvalue must have >= 1d shape"));
    if (ctx->IsRuntime()) {
      PADDLE_ENFORCE_GE(
          input_dims[axis], k,
          paddle::platform::errors::InvalidArgument(
              "input of kthvalue must have >= %d columns in axis of %d", k,
              axis));
    }
    bool keepdim = ctx->Attrs().Get<bool>("keepdim");
    std::vector<int64_t> dimvec;
    for (int64_t i = 0; i < axis; i++) {
      dimvec.emplace_back(input_dims[i]);
    }
    if (keepdim) {
      dimvec.emplace_back(static_cast<int64_t>(1));
    }
    for (int64_t i = axis + 1; i < dim_size; i++) {
      dimvec.emplace_back(input_dims[i]);
    }
    framework::DDim dims = framework::make_ddim(dimvec);
    ctx->SetOutputDim("Out", dims);
    ctx->SetOutputDim("Indices", dims);
    ctx->ShareLoD("X", "Out");
    ctx->ShareLoD("X", "Indices");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"),
        ctx.device_context());
  }
};

class KthvalueOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddComment(R"DOC(
    This operator find the k-th smallest elements in the specific axis of a Tensor.
    It will return the values and corresponding indices.
    )DOC");
    AddInput("X", "(Tensor) The input of Kthvalue op");
    AddOutput("Out", "(Tensor) The values of k-th smallest elements of input");
    AddOutput("Indices",
              "(Tensor) The indices of k-th smallest elements of input");
    AddAttr<int>(
        "k",
        "(int, default 1) k for k-th smallest elements to look for along "
        "the tensor).")
        .SetDefault(1);
    AddAttr<int>("axis",
                 "the axis to sort and get the k indices, value."
                 "if not set, will get k-th value in last axis.")
        .SetDefault(-1);
    AddAttr<bool>("keepdim", "Keep the dim that to reduce.").SetDefault(false);
  }
};

class KthvalueOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("X"), true,
        platform::errors::InvalidArgument("Input(X) should be not null"));
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("Indices"), true,
        platform::errors::InvalidArgument("Input(Indices) should be not null"));
    PADDLE_ENFORCE_EQ(ctx->HasInput(framework::GradVarName("Out")), true,
                      platform::errors::InvalidArgument(
                          "Grad Input(Out) should be not null"));
    PADDLE_ENFORCE_EQ(
        ctx->HasOutput(framework::GradVarName("X")), true,
        platform::errors::InvalidArgument("Grad Output(X) should be not null"));

    auto x_dims = ctx->GetInputDim("X");
    ctx->SetOutputDim(framework::GradVarName("X"), x_dims);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto data_type = OperatorWithKernel::IndicateVarDataType(
        ctx, framework::GradVarName("Out"));
    return framework::OpKernelType(data_type, ctx.device_context());
  }
};

template <typename T>
class KthvalueGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("kthvalue_grad");
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetInput("X", this->Input("X"));
    op->SetInput("Indices", this->Output("Indices"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(kthvalue, ops::KthvalueOp, ops::KthvalueOpMaker,
                  ops::KthvalueGradOpMaker<paddle::framework::OpDesc>,
                  ops::KthvalueGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OP_CPU_KERNEL(
    kthvalue, ops::KthvalueCPUKernel<paddle::platform::CPUPlace, float>,
    ops::KthvalueCPUKernel<paddle::platform::CPUPlace, double>,
    ops::KthvalueCPUKernel<paddle::platform::CPUPlace, int32_t>,
    ops::KthvalueCPUKernel<paddle::platform::CPUPlace, int64_t>);

REGISTER_OPERATOR(kthvalue_grad, ops::KthvalueOpGrad);
REGISTER_OP_CPU_KERNEL(
    kthvalue_grad,
    ops::KthvalueGradCPUKernel<paddle::platform::CPUPlace, float>,
    ops::KthvalueGradCPUKernel<paddle::platform::CPUPlace, double>,
    ops::KthvalueGradCPUKernel<paddle::platform::CPUPlace, int32_t>,
    ops::KthvalueGradCPUKernel<paddle::platform::CPUPlace, int64_t>);
