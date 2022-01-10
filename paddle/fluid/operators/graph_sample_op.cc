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

#include "paddle/fluid/operators/graph_sample_op.h"

namespace paddle {
namespace operators {

class GraphSampleOP : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("Src"), "Input", "Src", "GraphSampling");
    OP_INOUT_CHECK(ctx->HasInput("Dst_Count"), "Input", "Dst_Count",
                   "GraphSampling");
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "GraphSampling");
    OP_INOUT_CHECK(ctx->HasOutput("Out_Src"), "Output", "Out_Src",
                   "GraphSampling");
    OP_INOUT_CHECK(ctx->HasOutput("Out_Dst"), "Output", "Out_Dst",
                   "GraphSampling");
    OP_INOUT_CHECK(ctx->HasOutput("Sample_index"), "Output", "Sample_index",
                   "GraphSampling");
    // 是否限制所有输入输出均为1维向量，或者2维向量第二维为1.

    const std::vector<int>& sample_sizes =
        ctx->Attrs().Get<std::vector<int>>("sample_sizes");
    PADDLE_ENFORCE_EQ(
        !sample_sizes.empty(), true,
        platform::errors::InvalidArgument(
            "The parameter 'sample_sizes' in GraphSampleOp must be set. "
            "But received 'sample_sizes' is empty."));

    ctx->SetOutputDim("Out_Src", {-1});
    ctx->SetOutputDim("Out_Dst", {-1});
    ctx->SetOutputDim("Sample_index", {-1});
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"),
        ctx.device_context());
  }
};

class GraphSampleOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    // AddInput("Unique_Dst", "The unique dst index tensor.");
    AddInput("Src", "The src index tensor after sorted by dst.");
    AddInput("Dst_Count",
             "The indegree cumsum of dst intex, starts from 0, end with number "
             "of edges");
    AddInput("X", "The input center nodes index tensor.");
    AddOutput("Out_Src",
              "The output src edges tensor after sampling and reindex.");
    AddOutput("Out_Dst",
              "The output dst edges tensor after sampling and reindex.");
    AddOutput("Sample_index",
              "The original index of the sampling nodes and center nodes.");
    AddAttr<std::vector<int>>("sample_sizes",
                              "The sample sizes of graphsage sampling method.")
        .SetDefault({});
    AddComment(R"DOC(
Graph Learning Sampling operator, mainly for graphsage sampling method currently.

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
using CPU = paddle::platform::CPUDeviceContext;

REGISTER_OPERATOR(graph_sample, ops::GraphSampleOP, ops::GraphSampleOpMaker);
REGISTER_OP_CPU_KERNEL(graph_sample, ops::GraphSampleOpKernel<CPU, int>,
                       ops::GraphSampleOpKernel<CPU, int64_t>);
