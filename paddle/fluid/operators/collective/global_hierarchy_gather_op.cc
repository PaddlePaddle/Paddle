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

#include "paddle/fluid/operators/collective/global_hierarchy_gather_op.h"

namespace paddle {
namespace operators {

class GlobalHierarchyGatherOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "GlobalHierarchyGather");
    OP_INOUT_CHECK(ctx->HasInput("local_count"), "Input", "local_count",
                   "GlobalHierarchyGather");

    OP_INOUT_CHECK(ctx->HasInput("mp_global_count"), "Input", "mp_global_count",
                   "GlobalHierarchyGather");

    OP_INOUT_CHECK(ctx->HasInput("dp_global_count"), "Input", "dp_global_count",
                   "GlobalHierarchyGather");

    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out",
                   "GlobalHierarchyGather");
    int inside_ring_id = ctx->Attrs().Get<int>("inside_ring_id");
    PADDLE_ENFORCE_GE(inside_ring_id, 0,
                      platform::errors::InvalidArgument(
                          "The inside_ring_id (%d) for global hierarchy gather "
                          "op must be non-negative.",
                          inside_ring_id));

    int outside_ring_id = ctx->Attrs().Get<int>("outside_ring_id");
    PADDLE_ENFORCE_GE(outside_ring_id, 0,
                      platform::errors::InvalidArgument(
                          "The outside_ring_id (%d) for global hierarchy "
                          "gather op must be non-negative.",
                          outside_ring_id));

    auto input_dims = ctx->GetInputDim("X");
    auto ndim_input = input_dims.size();
    // dim check
    PADDLE_ENFORCE_EQ(ndim_input, 2,
                      platform::errors::InvalidArgument(
                          "The input tensor's dimension must be 2. "
                          "But received input's dimension = %d.",
                          ndim_input));
    framework::DDim out_dims = framework::make_ddim({-1, -1});
    ctx->SetOutputDim("Out", out_dims);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"), ctx.GetPlace());
  }
};

class GlobalHierarchyGatherOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddInput("X", "(Tensor) tensor send.");
    AddInput("local_count",
             "(Tensor) Tensor which has n_expert * world_size elements that "
             "indicates"
             "how many data needed to be received from each expert.");

    AddInput("mp_global_count",
             "(Tensor) Tensor which has n_expert * world_size elements that "
             "indicates"
             "how many data needed to be received from each expert.");

    AddInput("dp_global_count",
             "(Tensor) Tensor which has n_expert * world_size elements that "
             "indicates"
             "how many data needed to be received from each expert.");

    AddOutput("Out", "(Tensor) the result of global_hierarchy_gather.");
    AddAttr<int>("inside_ring_id",
                 "(int default 0) nccl communication ring id.")
        .SetDefault(0);

    AddAttr<int>("outside_ring_id",
                 "(int default 0) nccl communication ring id.")
        .SetDefault(0);
    AddAttr<bool>(
        "use_calc_stream",
        "(bool default false) eject CUDA operations to calculation stream.")
        .SetDefault(false);
    AddComment(R"DOC(
Global Hierarchy Gather Operator
refer to Global Hierarchy Scatter.
)DOC");
  }
};

template <typename T>
class GlobalHierarchyGatherOpGradMaker
    : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> retv) const override {
    retv->SetType("global_hierarchy_scatter");
    retv->SetInput("X", this->OutputGrad("Out"));
    retv->SetInput("local_count", this->Input("local_count"));
    retv->SetInput("mp_global_count", this->Input("mp_global_count"));
    retv->SetInput("dp_global_count", this->Input("dp_global_count"));
    retv->SetInput("outside_ring_id", this->Input("outside_ring_id"));
    retv->SetOutput("Out", this->InputGrad("X"));
    retv->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OPERATOR(
    global_hierarchy_gather, ops::GlobalHierarchyGatherOp,
    ops::GlobalHierarchyGatherOpMaker,
    ops::GlobalHierarchyGatherOpGradMaker<paddle::framework::OpDesc>,
    ops::GlobalHierarchyGatherOpGradMaker<paddle::imperative::OpBase>)

REGISTER_OP_CPU_KERNEL(global_hierarchy_gather,
                       ops::GlobalHierarchyGatherOpCPUKernel<float>,
                       ops::GlobalHierarchyGatherOpCPUKernel<double>,
                       ops::GlobalHierarchyGatherOpCPUKernel<int>,
                       ops::GlobalHierarchyGatherOpCPUKernel<int64_t>,
                       ops::GlobalHierarchyGatherOpCPUKernel<plat::float16>);
