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

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/binary.h"

namespace paddle {
namespace operators {

template <typename OpComment>
class CompareReduceOpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    OpComment comment;
    AddInput(
        "X",
        string::Sprintf("the left hand operand of %s operator", comment.type));
    AddInput(
        "Y",
        string::Sprintf("the right hand operand of %s operator", comment.type));
    AddOutput(
        "Out",
        string::Sprintf("tensor with a bool element. If all "
                        "element %s, the Out tensor is [True], else [False]",
                        comment.equation));
    AddComment(string::Sprintf(R"DOC(
It operates element-wise on X and Y, and returns the Out. X, Y is a
N-dim tensor, which could be any type. If all element $%s$, the Out tensor
is [True], else [False]
)DOC",
                               comment.equation));
  }
};

template <typename OpComment>
class CompareReduceOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
};

}  // namespace operators
}  // namespace paddle

#define REGISTER_COMPARE_REDUCE_OP(op_type, _equation)                     \
  struct _##op_type##Comment {                                             \
    static char type[];                                                    \
    static char equation[];                                                \
  };                                                                       \
  char _##op_type##Comment::type[]{#op_type};                              \
  char _##op_type##Comment::equation[]{_equation};                         \
  DECLARE_INFER_SHAPE_FUNCTOR(op_type,                                     \
                              op_type##_InferShapeFunctor,                 \
                              PD_INFER_META(phi::CompareAllInferMeta));    \
  REGISTER_OPERATOR(                                                       \
      op_type,                                                             \
      ::paddle::operators::CompareReduceOp<_##op_type##Comment>,           \
      ::paddle::operators::CompareReduceOpProtoMaker<_##op_type##Comment>, \
      ::paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,    \
      ::paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>,   \
      op_type##_InferShapeFunctor);

REGISTER_COMPARE_REDUCE_OP(equal_all, "X == Y");
