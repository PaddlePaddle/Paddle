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

#include "paddle/fluid/operators/collective/select_scatter_op.h"

namespace paddle {
namespace operators {

class SelectScatterOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("local_input_buf"), "Input", "local_input_buf",
                   "SelectScatter");
    OP_INOUT_CHECK(ctx->HasInput("local_expert_count"), "Input",
                   "local_expert_count", "SelectScatter");
    OP_INOUT_CHECK(ctx->HasInput("global_expert_count"), "Input",
                   "global_expert_count", "SelectScatter");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "SelectScatter");
    int ring_id = ctx->Attrs().Get<int>("ring_id");
    PADDLE_ENFORCE_GE(ring_id, 0, platform::errors::InvalidArgument(
                                      "The ring_id (%d) for select "
                                      "scatter/gather op must be non-negative.",
                                      ring_id));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "local_input_buf"),
        ctx.GetPlace());
  }
};

class SelectScatterOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddInput("local_input_buf", "(Tensor) tensor send.");
    AddInput("local_expert_count", "(Tensor) tensor send.");
    AddInput("global_expert_count", "(Tensor) tensor send.");
    // AddAttr<std::vector<int>>("local_expert_count", "The shape of the
    // output");
    // AddAttr<std::vector<int>>("global_expert_count", "The shape of the
    // output");
    AddAttr<int>("in_feat", "(int default 0) nccl communication ring id.")
        .SetDefault(0);
    AddAttr<int>("n_expert", "(int default 0) nccl communication ring id.")
        .SetDefault(0);
    AddAttr<int>("world_size", "(int default 0) nccl communication ring id.")
        .SetDefault(0);
    AddOutput("Out", "(Tensor) the result of alltoall.");

    AddAttr<int>("ring_id", "(int default 0) nccl communication ring id.")
        .SetDefault(0);
    AddAttr<bool>(
        "use_calc_stream",
        "(bool default false) eject CUDA operations to calculation stream.")
        .SetDefault(false);
    AddComment(R"DOC(
AllToAll Operator
Scatter tensors from all participators to all participators.
)DOC");
  }
};

template <typename T>
class SelectScatterOpGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> retv) const override {
    retv->SetType("select_gather");
    retv->SetInput("output_buf", this->OutputGrad("Out"));
    retv->SetInput("local_expert_count", this->Input("local_expert_count"));
    retv->SetInput("global_expert_count", this->Input("global_expert_count"));
    retv->SetOutput("Out", this->InputGrad("local_input_buf"));
    retv->SetAttrMap(this->Attrs());
    retv->SetAttr("out_feat", retv->GetAttr("in_feat"));
    // auto local_expert_count = retv->GetAttr("local_expert_count");
    // auto global_expert_count = retv->GetAttr("global_expert_count");
    // retv->SetAttr("local_expert_count", global_expert_count);
    // retv->SetAttr("global_expert_count", local_expert_count);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OPERATOR(select_scatter, ops::SelectScatterOp,
                  ops::SelectScatterOpMaker,
                  ops::SelectScatterOpGradMaker<paddle::framework::OpDesc>,
                  ops::SelectScatterOpGradMaker<paddle::imperative::OpBase>)

REGISTER_OP_CPU_KERNEL(select_scatter, ops::SelectScatterOpCPUKernel<float>,
                       ops::SelectScatterOpCPUKernel<double>,
                       ops::SelectScatterOpCPUKernel<int>,
                       ops::SelectScatterOpCPUKernel<int64_t>,
                       ops::SelectScatterOpCPUKernel<plat::float16>);
