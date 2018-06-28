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

#include "paddle/fluid/operators/compare_op.h"
#include <string>
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {
template <typename OpComment>
class CompareOpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    OpComment comment;
    AddInput("X", string::Sprintf("the left hand operand of %s operator",
                                  comment.type));
    AddInput("Y", string::Sprintf("the right hand operand of %s operator",
                                  comment.type));
    AddAttr<bool>("force_cpu",
                  "Force fill output variable to cpu "
                  "memory. Otherwise, fill output variable to the running "
                  "device [default true].")
        .SetDefault(true);
    AddOutput("Out", string::Sprintf("n-dim bool tensor. Each element is %s",
                                     comment.equation));
    AddComment(string::Sprintf(R"DOC(
It operates element-wise on X and Y, and returns the Out. Each of them is a
N-dim tensor. X and Y could be any type.  The each element of the Out tensor is
calculated by $%s$
)DOC",
                               comment.equation));
    AddAttr<int>(
        "axis",
        "The start dimension index for broadcasting Y onto X. [default -1]")
        .SetDefault(-1)
        .EqualGreaterThan(-1);
  }
};

template <typename OpComment>
class CompareOpInferShape : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *context) const override {
    OpComment comment;
    PADDLE_ENFORCE(context->HasInput("X"), "%s operator must has input X",
                   comment.type);
    PADDLE_ENFORCE(context->HasInput("Y"), "%s operator must has input Y",
                   comment.type);
    auto dim_x = context->GetInputDim("X");
    auto dim_y = context->GetInputDim("Y");
    PADDLE_ENFORCE_GE(dim_x.size(), dim_y.size(),
                      "The size of dim_y should not be greater than dim_x's.");

    context->SetOutputDim("Out", context->GetInputDim("X"));
    context->ShareLoD("X", "Out");
  }
};

class CompareOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    framework::OpKernelType kt = OperatorWithKernel::GetExpectedKernelType(ctx);
    // CompareOp kernel's device type is decided by input tensor place
    bool force_cpu = ctx.Attr<bool>("force_cpu");
    kt.place_ = force_cpu ? platform::CPUPlace()
                          : ctx.Input<framework::LoDTensor>("X")->place();
    return kt;
  }
};

}  // namespace operators
}  // namespace paddle

#define REGISTER_COMPARE_OP(op_type, _equation)                      \
  struct _##op_type##Comment {                                       \
    static char type[];                                              \
    static char equation[];                                          \
  };                                                                 \
  char _##op_type##Comment::type[]{#op_type};                        \
  char _##op_type##Comment::equation[]{_equation};                   \
  REGISTER_OPERATOR(                                                 \
      op_type, ::paddle::operators::CompareOp,                       \
      ::paddle::operators::CompareOpProtoMaker<_##op_type##Comment>, \
      ::paddle::operators::CompareOpInferShape<_##op_type##Comment>, \
      ::paddle::framework::EmptyGradOpMaker);

REGISTER_COMPARE_OP(less_than, "Out = X < Y");
REGISTER_COMPARE_KERNEL(less_than, CPU, paddle::operators::LessThanFunctor);
REGISTER_COMPARE_OP(less_equal, "Out = X <= Y");
REGISTER_COMPARE_KERNEL(less_equal, CPU, paddle::operators::LessEqualFunctor);
REGISTER_COMPARE_OP(greater_than, "Out = X > Y");
REGISTER_COMPARE_KERNEL(greater_than, CPU,
                        paddle::operators::GreaterThanFunctor);
REGISTER_COMPARE_OP(greater_equal, "Out = X >= Y");
REGISTER_COMPARE_KERNEL(greater_equal, CPU,
                        paddle::operators::GreaterEqualFunctor);
REGISTER_COMPARE_OP(equal, "Out = X == Y");
REGISTER_COMPARE_KERNEL(equal, CPU, paddle::operators::EqualFunctor);
REGISTER_COMPARE_OP(not_equal, "Out = X != Y");
REGISTER_COMPARE_KERNEL(not_equal, CPU, paddle::operators::NotEqualFunctor);
