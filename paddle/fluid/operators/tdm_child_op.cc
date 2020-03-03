/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/tdm_child_op.h"
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/sampler.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace operators {
class TDMChildOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddInput("Input",
             "Input(Tensor), dtype support int64 Input variable is the node id "
             "of TDM-Tree");
    AddInput("Tree_embedding",
             "Tree_embedding(Tensor), dtype support int64 stores the tree "
             "structure by node");
    AddAttr<int>("Ancestor_nums", "Ancestor_nums(int)",
                 "The nums of input (batch_size, ancestor_nums)");
    AddAttr<int>("Child_nums", "Child_nums(int)",
                 "The output nums of child, if the node hasn't enough child, "
                 "it will padding 0 until child nums = Child_nums");
    AddOutput("Child",
              "Return thr child node id of input, "
              "if input don't have child,return nothing");
    AddOutput("Item_mask",
              "Item_mask has the same shape with Child"
              "If child is leaf node, mask = 1");
    AddComment(R"DOC("TDM Child")DOC");
  }
};

class TDMChildOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInput("Input"), true,
                      "Inputs(Input) of TdmChild should not be null.");
    PADDLE_ENFORCE_EQ(ctx->HasInput("Tree_embedding"), true,
                      "Inputs(Tree_embedding) of TdmChild should not be null.");
    int ancestor_nums = ctx->Attrs().Get<int>("Ancestor_nums");
    int child_nums = ctx->Attrs().Get<int>("Child_nums");

    auto ddim = framework::make_ddim({-1, ancestor_nums, child_nums});
    if (ctx->IsRuntime()) {
      // do something in Runtime
    } else {
      ctx->SetOutputDim("Child", ddim);
      ctx->SetOutputDim("Item_mask", ddim);
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto data_type = OperatorWithKernel::IndicateVarDataType(ctx, "Input");
    return framework::OpKernelType(data_type, ctx.device_context());
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(
    tdm_child, ops::TDMChildOp, ops::TDMChildOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OP_CPU_KERNEL(
    tdm_child, ops::TDMChildKernel<paddle::platform::CPUPlace, float>,
    ops::TDMChildKernel<paddle::platform::CPUPlace, double>,
    ops::TDMChildKernel<paddle::platform::CPUPlace, int>,
    ops::TDMChildKernel<paddle::platform::CPUPlace, int64_t>);
