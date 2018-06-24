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

#include "paddle/fluid/operators/logical_op.h"
#include <string>
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {
template <typename OpComment>
class BinaryLogicalOpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    OpComment comment;
    AddInput("X",
             string::Sprintf("(LoDTensor) Left hand operand of %s operator",
                             comment.type));
    AddInput("Y",
             string::Sprintf("(LoDTensor) Right hand operand of %s operator",
                             comment.type));
    AddOutput("Out", string::Sprintf(
                         "(LoDTensor) n-dim bool tensor. Each element is %s",
                         comment.equation));
    AddComment(string::Sprintf(R"DOC(%s Operator

It operates element-wise on X and Y, and returns the Out. X, Y and Out are N-dim boolean tensors.
Each element of Out is calculated by %s
)DOC",
                               comment.type, comment.equation));
  }
};

template <typename OpComment>
class UnaryLogicalOpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    OpComment comment;
    AddInput("X", string::Sprintf("(LoDTensor) Operand of %s operator",
                                  comment.type));
    AddOutput("Out", string::Sprintf(
                         "(LoDTensor) n-dim bool tensor. Each element is %s",
                         comment.equation));
    AddComment(string::Sprintf(R"DOC(%s Operator

It operates element-wise on X, and returns the Out. X and Out are N-dim boolean tensors.
Each element of Out is calculated by %s
)DOC",
                               comment.type, comment.equation));
  }
};

template <typename OpComment>
class BinaryLogicalOpInferShape : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *context) const override {
    OpComment comment;
    PADDLE_ENFORCE(context->HasInput("X"),
                   "Input(X) of %s operator must not be null", comment.type);
    PADDLE_ENFORCE(context->HasInput("Y"),
                   "Input(Y) of %s operator must not be null", comment.type);
    auto dim_x = context->GetInputDim("X");
    auto dim_y = context->GetInputDim("Y");
    PADDLE_ENFORCE_EQ(framework::product(dim_x), framework::product(dim_y),
                      "The number of elements in X and Y should be same");

    context->SetOutputDim("Out", context->GetInputDim("X"));
    context->ShareLoD("X", "Out");
  }
};

template <typename OpComment>
class UnaryLogicalOpInferShape : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *context) const override {
    OpComment comment;
    PADDLE_ENFORCE(context->HasInput("X"),
                   "Input(X) of %s operator must not be null", comment.type);
    auto dim_x = context->GetInputDim("X");

    context->SetOutputDim("Out", context->GetInputDim("X"));
    context->ShareLoD("X", "Out");
  }
};

class LogicalOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    framework::OpKernelType kt = OperatorWithKernel::GetExpectedKernelType(ctx);
    // LogicalOp kernel's device type is decided by input tensor place
    kt.place_ = ctx.Input<framework::LoDTensor>("X")->place();
    return kt;
  }
};

}  // namespace operators
}  // namespace paddle

#define REGISTER_BINARY_LOGICAL_OP(op_type, _equation)                     \
  struct _##op_type##Comment {                                             \
    static char type[];                                                    \
    static char equation[];                                                \
  };                                                                       \
  char _##op_type##Comment::type[]{#op_type};                              \
  char _##op_type##Comment::equation[]{_equation};                         \
  REGISTER_OPERATOR(                                                       \
      op_type, ::paddle::operators::LogicalOp,                             \
      ::paddle::operators::BinaryLogicalOpProtoMaker<_##op_type##Comment>, \
      ::paddle::operators::BinaryLogicalOpInferShape<_##op_type##Comment>, \
      ::paddle::framework::EmptyGradOpMaker);

#define REGISTER_UNARY_LOGICAL_OP(op_type, _equation)                     \
  struct _##op_type##Comment {                                            \
    static char type[];                                                   \
    static char equation[];                                               \
  };                                                                      \
  char _##op_type##Comment::type[]{#op_type};                             \
  char _##op_type##Comment::equation[]{_equation};                        \
  REGISTER_OPERATOR(                                                      \
      op_type, ::paddle::operators::LogicalOp,                            \
      ::paddle::operators::UnaryLogicalOpProtoMaker<_##op_type##Comment>, \
      ::paddle::operators::UnaryLogicalOpInferShape<_##op_type##Comment>, \
      ::paddle::framework::EmptyGradOpMaker);

REGISTER_BINARY_LOGICAL_OP(logical_and, "$$Out = X \\&\\& Y$$");
REGISTER_BINARY_LOGICAL_KERNEL(logical_and, CPU,
                               paddle::operators::LogicalAndFunctor);
REGISTER_BINARY_LOGICAL_OP(logical_or, "$$Out = X || Y$$");
REGISTER_BINARY_LOGICAL_KERNEL(logical_or, CPU,
                               paddle::operators::LogicalOrFunctor);
REGISTER_UNARY_LOGICAL_OP(logical_not, "$$Out = !X$$");
REGISTER_UNARY_LOGICAL_KERNEL(logical_not, CPU,
                              paddle::operators::LogicalNotFunctor);
REGISTER_BINARY_LOGICAL_OP(logical_xor,
                           "$$Out = (X || Y) \\&\\& !(X \\&\\& Y)$$");
REGISTER_BINARY_LOGICAL_KERNEL(logical_xor, CPU,
                               paddle::operators::LogicalXorFunctor);
