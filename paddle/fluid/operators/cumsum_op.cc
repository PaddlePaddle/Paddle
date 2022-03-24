/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/unary.h"

namespace paddle {
namespace operators {

class CumOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
};

class CumsumOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "Input of cumsum operator");
    AddOutput("Out", "Output of cumsum operator");
    AddAttr<int>("axis",
                 "The dimension to accumulate along. -1 means the last "
                 "dimension [default -1].")
        .SetDefault(-1);
    AddAttr<bool>("flatten",
                  "Whether to compute the cumsum over the flattened array. "
                  "[default false].")
        .SetDefault(false);
    AddAttr<bool>("exclusive",
                  "Whether to perform exclusive cumsum. [default false].")
        .SetDefault(false);
    AddAttr<bool>("reverse",
                  "If true, the cumsum is performed in the reversed direction. "
                  "[default false].")
        .SetDefault(false);
    AddComment(R"DOC(
The cumulative sum of the elements along a given axis.
By default, the first element of the result is the same of the first element of
the input. If exlusive is true, the first element of the result is 0.
)DOC");
  }
};

template <typename T>
class CumsumGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> grad_op) const override {
    grad_op->SetType("cumsum");
    grad_op->SetInput("X", this->OutputGrad("Out"));
    grad_op->SetOutput("Out", this->InputGrad("X"));
    grad_op->SetAttr("axis", BOOST_GET_CONST(int, this->GetAttr("axis")));
    grad_op->SetAttr("flatten",
                     BOOST_GET_CONST(bool, this->GetAttr("flatten")));
    grad_op->SetAttr("reverse",
                     !BOOST_GET_CONST(bool, this->GetAttr("reverse")));
    grad_op->SetAttr("exclusive",
                     BOOST_GET_CONST(bool, this->GetAttr("exclusive")));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
using CPU = paddle::platform::CPUDeviceContext;
DECLARE_INFER_SHAPE_FUNCTOR(cumsum, CumsumInferShapeFunctor,
                            PD_INFER_META(phi::CumsumInferMeta));
REGISTER_OPERATOR(cumsum, ops::CumOp, ops::CumsumOpMaker,
                  ops::CumsumGradMaker<paddle::framework::OpDesc>,
                  ops::CumsumGradMaker<paddle::imperative::OpBase>,
                  CumsumInferShapeFunctor);

REGISTER_OP_VERSION(cumsum)
    .AddCheckpoint(
        R"ROC(
      Upgrade cumsum add a new attribute [flatten].
    )ROC",
        paddle::framework::compatible::OpVersionDesc().NewAttr(
            "flatten",
            "In order to compute the cumsum over the flattened array when the "
            "argument `axis` in python API is None.",
            false));
