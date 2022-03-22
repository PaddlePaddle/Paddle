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

#include <memory>
#include <vector>

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/unary.h"

namespace paddle {
namespace operators {

class ReverseOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
};

class ReverseOpVarTypeInference : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext* ctx) const override {
    ctx->SetOutputType("Out", ctx->GetInputType("X"));
    ctx->SetOutputDataType("Out", ctx->GetInputDataType("X"));
  }
};

class ReverseOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The LoDTensor to be flipped.");
    AddOutput("Out", "The LoDTensor after flipping.");
    AddAttr<std::vector<int>>(
        "axis", "The axises that along which order of elements is reversed.");
    AddComment(R"DOC(
      Reverse Operator.

      Reverse the order of elements in the input LoDTensor along given axises.

      Case 1:
        Given
            X = [[1, 2, 3, 4, 5]
                 [6, 7, 8, 9, 10]
                 [11, 12, 13, 14, 15]],
        and
            axis = [0],
        we get:
            Out = [[11, 12, 13, 14, 15]
                   [6, 7, 8, 9, 10]
                   [1, 2, 3, 4, 5]].
        
      Case 2:
        Given
            X = [[[1, 2, 3, 4]
                  [5, 6, 7, 8]]
                 [[9, 10, 11, 12]
                  [13, 14, 15, 16]]],
        and
            axis = [0, 2],
        we get:
            Out = [[[12, 11, 10, 9]
                    [16, 15, 14, 13]]
                   [[4, 3, 2, 1]
                    [8, 7, 6, 5]]],
    )DOC");
  }
};

template <typename T>
class ReverseGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

  void Apply(GradOpPtr<T> grad_op) const override {
    grad_op->SetType("reverse");
    grad_op->SetInput("X", this->OutputGrad("Out"));
    grad_op->SetOutput("Out", this->InputGrad("X"));
    grad_op->SetAttr("axis", this->GetAttr("axis"));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
DECLARE_INFER_SHAPE_FUNCTOR(reverse, ReverseInferShapeFunctor,
                            PD_INFER_META(phi::ReverseInferMeta));
REGISTER_OPERATOR(reverse, ops::ReverseOp, ops::ReverseOpMaker,
                  ops::ReverseGradMaker<paddle::framework::OpDesc>,
                  ops::ReverseGradMaker<paddle::imperative::OpBase>,
                  ops::ReverseOpVarTypeInference, ReverseInferShapeFunctor);
REGISTER_OPERATOR(reverse_grad, ops::ReverseOp, ops::ReverseOpVarTypeInference);
