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

#include "paddle/fluid/operators/detection/distribute_fpn_proposals_op.h"

namespace paddle {
namespace operators {

class DistributeFpnProposalsOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("FpnRois"),
                   "Input(FpnRois) shouldn't be null");
    PADDLE_ENFORCE_GE(
        ctx->Outputs("MultiFpnRois").size(), 1UL,
        "Outputs(MultiFpnRois) of DistributeOp should not be empty");
    size_t min_level = static_cast<size_t>(ctx->Attrs().Get<int>("min_level"));
    size_t max_level = static_cast<size_t>(ctx->Attrs().Get<int>("max_level"));
    PADDLE_ENFORCE_GE(max_level, min_level,
                      "max_level must not lower than min_level");
    // Set the output shape
    size_t num_out_rois = max_level - min_level + 1;
    std::vector<framework::DDim> outs_dims;
    outs_dims.reserve(num_out_rois);
    for (size_t i = 0; i < num_out_rois; ++i) {
      framework::DDim out_dim = {-1, 4};
      outs_dims.push_back(out_dim);
    }
    ctx->SetOutputsDim("MultiFpnRois", outs_dims);
    ctx->SetOutputDim("RestoreIndex", {-1, 1});
    if (!ctx->IsRuntime()) {
      for (size_t i = 0; i < num_out_rois; ++i) {
        ctx->SetLoDLevel("MultiFpnRois", ctx->GetLoDLevel("FpnRois"), i);
      }
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto data_type = OperatorWithKernel::IndicateVarDataType(ctx, "FpnRois");
    return framework::OpKernelType(data_type, ctx.device_context());
  }
};

class DistributeFpnProposalsOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("FpnRois", "(LoDTensor) The rois at all levels in shape (-1, 4)");
    AddOutput("MultiFpnRois", "(LoDTensor) Output with distribute operator")
        .AsDuplicable();
    AddOutput("RestoreIndex",
              "(Tensor) An array of positive number which is "
              "used to restore the order of FpnRois");
    AddAttr<int>("min_level",
                 "The lowest level of FPN layer where the"
                 " proposals come from");
    AddAttr<int>("max_level",
                 "The highest level of FPN layer where the"
                 " proposals come from");
    AddAttr<int>("refer_level",
                 "The referring level of FPN layer with"
                 " specified scale");
    AddAttr<int>("refer_scale",
                 "The referring scale of FPN layer with"
                 " specified level");
    AddComment(R"DOC(
This operator distribute all proposals into different fpn level,
 with respect to scale of the proposals, the referring scale and
 the referring level. Besides, to restore the order of proposals,
we return an array which indicate the original index of rois in
 current proposals.
)DOC");
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    distribute_fpn_proposals, ops::DistributeFpnProposalsOp,
    ops::DistributeFpnProposalsOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OP_CPU_KERNEL(distribute_fpn_proposals,
                       ops::DistributeFpnProposalsOpKernel<float>,
                       ops::DistributeFpnProposalsOpKernel<double>);
