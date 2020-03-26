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
    AddInput("X",
             "X(Tensor), dtype support int32/int64, X variable is the "
             "node id of TDM-Tree");
    AddInput(
        "Tree_info",
        "Tree_info(Tensor), dtype support int32/int64, it stores the node "
        "information in the following format: item_id(shape=1), "
        "layer_id(shape=1), parent_id(shape=1), child_id(shape=Child_nums)");
    AddAttr<int>("Child_nums", "Child_nums(int)",
                 "The child nums of one node, if the node hasn't enough child, "
                 "it should padding 0 until child nums equal to Child_nums");
    AddOutput("Child",
              "Return the children's node_id of input node, "
              "if input don't have child, return 0");
    AddOutput("Leaf_mask",
              "Leaf_mask has the same shape with Child"
              "If child is leaf node, leaf_mask value = 1, else = 0");
    AddComment(R"DOC("TDM Child")DOC");
  }
};

class TDMChildOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInput("X"), true,
                      platform::errors::InvalidArgument(
                          "Inputs(X) of TdmChild should not be null."));
    PADDLE_ENFORCE_EQ(ctx->HasInput("Tree_info"), true,
                      platform::errors::InvalidArgument(
                          "Inputs(Tree_info) of TdmChild should not be null."));

    int child_nums = ctx->Attrs().Get<int>("Child_nums");
    PADDLE_ENFORCE_GT(
        child_nums, 0,
        platform::errors::InvalidArgument(
            "ValueError: The value of the 'Child_nums' must greater than 0. "
            "But received Child_nums value = %d, ",
            child_nums));

    auto info_dims = ctx->GetInputDim("Tree_info");
    auto input_dims = ctx->GetInputDim("X");

    PADDLE_ENFORCE_EQ(
        info_dims.size(), 2,
        platform::errors::InvalidArgument(
            "ShapeError: The dimensions of the 'tree info' must be 2. "
            "But received tree info's dimensions = %d, "
            "tree info's shape = [%s].",
            info_dims.size(), info_dims));

    auto output_dims = framework::vectorize(input_dims);
    output_dims.push_back(child_nums);
    ctx->SetOutputDim("Child", framework::make_ddim(output_dims));
    ctx->SetOutputDim("Leaf_mask", framework::make_ddim(output_dims));

    if (ctx->GetOutputsVarType("Child")[0] ==
        framework::proto::VarType::LOD_TENSOR) {
      ctx->ShareLoD("X", /*->*/ "Child");
      ctx->ShareLoD("X", /*->*/ "Leaf_mask");
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto data_type = OperatorWithKernel::IndicateVarDataType(ctx, "X");
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
