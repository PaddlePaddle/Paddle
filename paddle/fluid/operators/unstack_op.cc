/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <memory>
#include <string>
#include <vector>
#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/for_range.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/unary.h"

namespace paddle {
namespace operators {

class UnStackOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
};

class UnStackOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The input of unstack op.");
    AddOutput("Y", "The output of unstack op.").AsDuplicable();
    AddAttr<int>("axis", "The axis along which Input(X) should be unstacked.")
        .SetDefault(0);
    AddAttr<int>("num", "The number of outputs(Y).").GreaterThan(0);
    AddComment(R"DOC(
      UnStack Operator.

      UnStack Input(X) into several tensors along Attr(axis).
    )DOC");
  }
};

template <typename T>
class UnStackGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("unstack_grad");
    op->SetInput(framework::GradVarName("Y"), this->OutputGrad("Y"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetAttrMap(this->Attrs());
  }
};

class UnStackGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE_GT(ctx->Inputs(framework::GradVarName("Y")).size(), 0,
                      platform::errors::InvalidArgument(
                          "The Inputs(Y@Grad) of unstack operator are empty."));
    OP_INOUT_CHECK(ctx->HasOutput(framework::GradVarName("X")), "Output", "X",
                   "UnStackGrad");
    auto input_dims = ctx->GetInputsDim(framework::GradVarName("Y"));
    for (size_t i = 1; i < input_dims.size(); ++i) {
      PADDLE_ENFORCE_EQ(
          input_dims[i], input_dims[0],
          platform::errors::InvalidArgument(
              "The dimensions of all Inputs(Y@Grad) must be the same,"
              "but received Inputs(Y@Grad)'s %d-th dimension is %d, "
              "Inputs(Y@Grad)'s 0-th to %d-th dimension is %d.",
              i, input_dims[i], i - 1, input_dims[0]));
    }

    int axis = ctx->Attrs().Get<int>("axis");
    int rank = input_dims[0].size();
    PADDLE_ENFORCE_GE(axis, -(rank + 1),
                      platform::errors::InvalidArgument(
                          "The attribute axis is out of range, it must be "
                          "inside [-(rank+1), rank+1), where rank = %d",
                          rank));
    PADDLE_ENFORCE_LT(axis, rank + 1,
                      platform::errors::InvalidArgument(
                          "The attribute axis is out of range, it must be "
                          "inside [-(rank+1), rank+1), where rank = %d",
                          rank));
    if (axis < 0) axis += (rank + 1);

    auto vec = phi::vectorize<int>(input_dims[0]);
    vec.insert(vec.begin() + axis, input_dims.size());
    ctx->SetOutputDim(framework::GradVarName("X"), phi::make_ddim(vec));
  }
};

}  // namespace operators
}  // namespace paddle

namespace plat = paddle::platform;
namespace ops = paddle::operators;

DECLARE_INFER_SHAPE_FUNCTOR(unstack, UnStackInferMetaFunctor,
                            PD_INFER_META(phi::UnStackInferMeta));

REGISTER_OPERATOR(unstack, ops::UnStackOp, ops::UnStackOpMaker,
                  ops::UnStackGradOpMaker<paddle::framework::OpDesc>,
                  ops::UnStackGradOpMaker<paddle::imperative::OpBase>,
                  UnStackInferMetaFunctor);

REGISTER_OPERATOR(unstack_grad, ops::UnStackGradOp);
