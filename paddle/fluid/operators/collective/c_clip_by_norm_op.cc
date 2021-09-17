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
#include "paddle/fluid/operators/collective/c_clip_by_norm_op.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

class CollectiveClipByNormOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(
        ctx->HasInputs("X"), true,
        platform::errors::InvalidArgument(
            "Input(X) of CollectiveClipByNormOp should not be null. "
            "Please check if it is created correctly."));
    PADDLE_ENFORCE_EQ(
        ctx->HasOutputs("Out"), true,
        platform::errors::InvalidArgument(
            "Output(Out) of CollectiveClipByNormOp should not be null. "
            "Please check if it is created correctly."));
    auto max_norm = ctx->Attrs().Get<float>("max_norm");
    PADDLE_ENFORCE_GT(max_norm, 0, platform::errors::InvalidArgument(
                                       "max_norm should be greater than 0. "
                                       "Received max_norm is %f.",
                                       max_norm));

    PADDLE_ENFORCE_GT(ctx->Inputs("X").size(), static_cast<size_t>(0),
                      platform::errors::InvalidArgument(
                          "The CollectiveClipByNormOp operator has no input."));
    PADDLE_ENFORCE_EQ(
        ctx->Inputs("X").size(), ctx->Outputs("Out").size(),
        platform::errors::InvalidArgument(
            "The input(X) and output(Out) should have same size in "
            "Operator(c_clip_by_norm), size of input(X) is %d "
            "and size of output(Out) is %d.",
            ctx->Inputs("X").size(), ctx->Outputs("Out").size()));
    auto x_dims = ctx->GetInputsDim("X");
    ctx->SetOutputsDim("Out", x_dims);
    ctx->ShareLoD("X", "Out");
  }
};

class CollectiveClipByNormOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput(
        "X",
        "(Tensors) The inputs of c_clip_by_norm op and data type is float32."
        "The number of dimensions must be between [1, 9].")
        .AsDuplicable();
    AddOutput("Out",
              "(Tensors) The output of c_clip_by_norm op with shape as input(X)"
              "The data type is float32.")
        .AsDuplicable();
    AddAttr<float>("max_norm", "(float) The maximum norm value.");
    AddAttr<int>("ring_id", "(int) The ring id for allreduce.");
    AddAttr<bool>("use_calc_stream",
                  "(bool) whether use calc stream for "
                  "allreduce")
        .SetDefault(false);
    AddComment(R"DOC(
CollectiveClipByNorm Operator.

This operator is used under model-parallel circumstance.
The math equation for this op is same with the equation
of ClipByNorm.
During the MP, some vars will be split into each rank.
When calc the norm of the split vars, all values from
every parts should be counted.
In this op, we first calc the square_sum value for values
in current mp rank. Then do a c_allreduce_sum operation to
get the final square_sum value for all values from each
parts. Then do the sqrt operations.
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_WITHOUT_GRADIENT(c_clip_by_norm, ops::CollectiveClipByNormOp,
                             ops::CollectiveClipByNormOpMaker);

REGISTER_OP_CPU_KERNEL(
    c_clip_by_norm,
    ops::CollectiveClipByNormCPUKernel<plat::CPUDeviceContext, float>,
    ops::CollectiveClipByNormCPUKernel<plat::CPUDeviceContext, double>);
