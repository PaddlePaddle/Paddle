// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/stack_op.h"
#include <memory>
#include <vector>

namespace plat = paddle::platform;
namespace ops = paddle::operators;

namespace paddle {
namespace operators {

class StackOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE_GT(ctx->Inputs("X").size(), 0,
                      platform::errors::InvalidArgument(
                          "Number of Inputs(X) must be larger than 0, but"
                          " received value is:%d.",
                          ctx->Inputs("X").size()));
    PADDLE_ENFORCE_EQ(ctx->HasOutput("Y"), true,
                      platform::errors::InvalidArgument(
                          "Output(Y) of stack_op should not be null."));

    auto input_dims = ctx->GetInputsDim("X");
    for (size_t i = 1; i < input_dims.size(); ++i) {
      PADDLE_ENFORCE_EQ(input_dims[i], input_dims[0],
                        platform::errors::InvalidArgument(
                            "Dims of all Inputs(X) must be the same, but"
                            " received input %d dim is:%d not equal to input 0"
                            " dim:%d.",
                            i, input_dims[i], input_dims[0]));
    }

    // Only lod of X[0] would be shared with Y
    ctx->ShareLoD("X", /*->*/ "Y");

    int axis = ctx->Attrs().Get<int>("axis");
    int rank = input_dims[0].size();
    PADDLE_ENFORCE_GE(
        axis, -(rank + 1),
        platform::errors::InvalidArgument(
            "Attr(axis) must be inside [-(rank+1), rank+1), where rank = %d, "
            "but received axis is:%d.",
            rank, axis));

    PADDLE_ENFORCE_LT(
        axis, rank + 1,
        platform::errors::InvalidArgument(
            "Attr(axis) must be inside [-(rank+1), rank+1), where rank = %d, "
            "but received axis is:%d",
            rank, axis));

    if (axis < 0) axis += (rank + 1);

    auto vec = phi::vectorize<int>(input_dims[0]);
    vec.insert(vec.begin() + axis, input_dims.size());
    ctx->SetOutputDim("Y", phi::make_ddim(vec));
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto input_data_type =
        framework::OperatorWithKernel::IndicateVarDataType(ctx, "X");

#ifdef PADDLE_WITH_MKLDNN
    if (this->CanMKLDNNBeUsed(ctx, input_data_type)) {
      return framework::OpKernelType(input_data_type, ctx.GetPlace(),
                                     framework::DataLayout::kMKLDNN,
                                     framework::LibraryType::kMKLDNN);
    }
#endif
    return framework::OpKernelType(input_data_type, ctx.GetPlace());
  }
};

class StackOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The input of stack op.").AsDuplicable();
    AddOutput("Y", "The output of stack op.");
    AddAttr<int>("axis",
                 "The axis along which all of the Inputs(X) should be stacked.")
        .SetDefault(0);
    AddAttr<bool>(
        "use_mkldnn",
        "(bool, default false) Indicates if MKL-DNN kernel will be used")
        .SetDefault(false)
        .AsExtra();
    AddComment(R"DOC(
Stack Operator.
Stack all of the Inputs(X) into one tensor along Attr(axis). The dims of all Inputs(X) must be the same.
)DOC");
  }
};

class StackOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE_EQ(
        ctx->HasInput(framework::GradVarName("Y")), true,
        platform::errors::InvalidArgument("Input(Y@Grad) not exist."));

    int axis = ctx->Attrs().Get<int>("axis");
    auto dy_dim = ctx->GetInputDim(framework::GradVarName("Y"));
    int rank = dy_dim.size();
    PADDLE_ENFORCE_GE(
        axis, -rank,
        platform::errors::InvalidArgument(
            "Attr(axis) must be inside [-rank, rank), where rank = %d, "
            "but received axis is:%d.",
            rank, axis));
    PADDLE_ENFORCE_LT(
        axis, rank,
        platform::errors::InvalidArgument(
            "Attr(axis) must be inside [-rank, rank), where rank = %d, "
            "but received axis is:%d.",
            rank, axis));

    if (axis < 0) axis += rank;
    PADDLE_ENFORCE_EQ(
        ctx->Outputs(framework::GradVarName("X")).size(),
        static_cast<size_t>(dy_dim[axis]),
        platform::errors::InvalidArgument(
            "Number of Outputs(X@Grad) is equal to dy dim at axis, but"
            " received outputs size is:%d, dy dims is:%d.",
            ctx->Outputs(framework::GradVarName("X")).size(),
            static_cast<size_t>(dy_dim[axis])));

    auto vec = phi::vectorize<int>(dy_dim);
    vec.erase(vec.begin() + axis);
    ctx->SetOutputsDim(
        framework::GradVarName("X"),
        std::vector<framework::DDim>(dy_dim[axis], phi::make_ddim(vec)));
  }
};

template <typename T>
class StackGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("stack_grad");
    op->SetInput(framework::GradVarName("Y"), this->OutputGrad("Y"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X", false));
    op->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OPERATOR(stack, ops::StackOp, ops::StackOpMaker,
                  ops::StackGradOpMaker<paddle::framework::OpDesc>,
                  ops::StackGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(stack_grad, ops::StackOpGrad);

REGISTER_OP_CPU_KERNEL(
    stack, ops::StackKernel<plat::CPUDeviceContext, float>,
    ops::StackKernel<plat::CPUDeviceContext, double>,
    ops::StackKernel<plat::CPUDeviceContext, int>,
    ops::StackKernel<plat::CPUDeviceContext, int64_t>,
    ops::StackKernel<plat::CPUDeviceContext, paddle::platform::bfloat16>);

REGISTER_OP_CPU_KERNEL(
    stack_grad, ops::StackGradKernel<plat::CPUDeviceContext, float>,
    ops::StackGradKernel<plat::CPUDeviceContext, double>,
    ops::StackGradKernel<plat::CPUDeviceContext, int>,
    ops::StackGradKernel<plat::CPUDeviceContext, int64_t>,
    ops::StackGradKernel<plat::CPUDeviceContext, paddle::platform::bfloat16>);
