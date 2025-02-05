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

#include "paddle/fluid/operators/collective/c_split_op.h"

#include "paddle/common/enforce.h"

namespace paddle {
namespace operators {

class CSplitOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInput("X"),
                      true,
                      common::errors::InvalidArgument(
                          "The input 'X' for c_split must be provided."));
    PADDLE_ENFORCE_EQ(ctx->HasOutput("Out"),
                      true,
                      common::errors::InvalidArgument(
                          "The output 'Out' for c_split must be provided."));
    int nranks = ctx->Attrs().Get<int>("nranks");
    int rank = ctx->Attrs().Get<int>("rank");
    int ring_id = ctx->Attrs().Get<int>("ring_id");
    PADDLE_ENFORCE_GE(
        nranks,
        2,
        common::errors::InvalidArgument("The number of ranks (%d) for c_split "
                                        "must be greater than 1.",
                                        nranks));
    PADDLE_ENFORCE_GE(
        ring_id,
        0,
        common::errors::InvalidArgument(
            "The ring_id (%d) for c_split must be non-negative.", ring_id));
    PADDLE_ENFORCE_GE(
        rank,
        0,
        common::errors::InvalidArgument(
            "The rank (%d) for c_split must be non-negative.", rank));
    PADDLE_ENFORCE_LT(rank,
                      nranks,
                      common::errors::InvalidArgument(
                          "The value of rank (%d) for c_split must "
                          "be less than that of nranks.",
                          rank,
                          nranks));

    phi::DDim dim = ctx->GetInputDim("X");
    PADDLE_ENFORCE_EQ(
        dim[dim.size() - 1] % nranks,
        0,
        common::errors::InvalidArgument("The last dimension (%d) of the X "
                                        "should be divisible by nranks (%d)",
                                        dim[dim.size() - 1],
                                        nranks));

    dim[dim.size() - 1] = dim[dim.size() - 1] / nranks;
    if (dim[0] < 0) dim[0] = -1;
    ctx->SetOutputDim("Out", dim);
  }

 protected:
  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return phi::KernelKey(OperatorWithKernel::IndicateVarDataType(ctx, "X"),
                          ctx.GetPlace());
  }
};

template <typename T>
class CSplitOpGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> retv) const override {
    retv->SetType("c_allgather");
    retv->SetInput("X", this->OutputGrad("Out"));
    retv->SetOutput("Out", this->InputGrad("X"));
    retv->SetAttrMap(this->Attrs());
  }
};

class CSplitOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor) tensor to be split.");
    AddOutput("Out", "(Tensor) the result of split.");
    AddAttr<int>("rank", "(int default 0) rank id.").SetDefault(0);
    AddAttr<int>("nranks", "(int default 1) number of ranks.").SetDefault(1);
    AddAttr<int>("ring_id", "(int default 0) ring id.").SetDefault(0);
    AddAttr<bool>("use_model_parallel",
                  "(bool default false) use this op with model parallel.")
        .SetDefault(true);
    AddComment(R"DOC(
CSplit Operator
Split the tensor evenly according to its rank.
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(c_split,
                  ops::CSplitOp,
                  ops::CSplitOpGradMaker<paddle::framework::OpDesc>,
                  ops::CSplitOpGradMaker<paddle::imperative::OpBase>,
                  ops::CSplitOpMaker);
