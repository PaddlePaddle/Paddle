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
#include "paddle/phi/infermeta/ternary.h"

namespace paddle {
namespace operators {

class AccuracyOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "Out"), ctx.GetPlace());
  }
};

class AccuracyOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    // TODO(typhoonzero): support both inference value and indices.
    AddInput("Out", "The network output of topk (inferences)");
    AddInput("Indices", "The network output of topk (indices)");
    AddInput("Label", "Label of the training data");
    // TODO(typhoonzero): AddInput("Weight", ...
    AddOutput("Accuracy", "The accuracy of current batch");
    AddOutput("Correct", "The correct samples count of current batch");
    AddOutput("Total", "The samples count of current batch");

    AddComment(R"DOC(
Accuracy Operator.

It will print accuracy rate for classification.
The accuracy is calculated as follows:

$$accuracy = \frac{NumOfCorrectPredicts}{NumOfAllSamples}$$

Both the input Out and Label can carry the LoD (Level of Details)
information, or not. But the output only shares the LoD information
with the input Out(Inference).

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

// FIXME(typhoonzero): types of T is for infernece data.
// label data is always int.
DECLARE_INFER_SHAPE_FUNCTOR(accuracy,
                            AccuracyInferShapeFunctor,
                            PD_INFER_META(phi::AccuracyInferMeta));
namespace ops = paddle::operators;
REGISTER_OPERATOR(
    accuracy,
    ops::AccuracyOp,
    ops::AccuracyOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>,
    AccuracyInferShapeFunctor);
