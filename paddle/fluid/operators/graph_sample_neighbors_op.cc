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

#include "paddle/fluid/operators/graph_sample_neighbors_op.h"

namespace paddle {
namespace operators {

class GraphSampleNeighborsOP : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("Src"), "Input", "Src",
                   "GraphSampleNeighbors");
    OP_INOUT_CHECK(ctx->HasInput("Dst_Count"), "Input", "Dst_Count",
                   "GraphSampleNeighbors");
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "GraphSampleNeighbors");
    OP_INOUT_CHECK(ctx->HasOutput("Out_Src"), "Output", "Out_Src",
                   "GraphSampleNeighbors");
    OP_INOUT_CHECK(ctx->HasOutput("Out_Dst"), "Output", "Out_Dst",
                   "GraphSampleNeighbors");
    OP_INOUT_CHECK(ctx->HasOutput("Sample_index"), "Output", "Sample_index",
                   "GraphSampleNeighbors");
    OP_INOUT_CHECK(ctx->HasOutput("Reindex_X"), "Output", "Reindex_X",
                   "GraphSampleNeighbors");
    // 是否限制所有输入输出均为1维向量，或者2维向量第二维为1.

    const std::vector<int>& sample_sizes =
        ctx->Attrs().Get<std::vector<int>>("sample_sizes");
    PADDLE_ENFORCE_EQ(
        !sample_sizes.empty(), true,
        platform::errors::InvalidArgument(
            "The parameter 'sample_sizes' in GraphSampleOp must be set. "
            "But received 'sample_sizes' is empty."));
    const bool& return_eids = ctx->Attrs().Get<bool>("return_eids");
    if (return_eids) {
      OP_INOUT_CHECK(ctx->HasInput("Src_Eids"), "Input", "Src_Eids",
                     "GraphSampleNeighbors");
      OP_INOUT_CHECK(ctx->HasOutput("Out_Eids"), "Output", "Out_Eids",
                     "GraphSampleNeighbors");
      ctx->SetOutputDim("Out_Eids", {-1});
    }

    ctx->SetOutputDim("Out_Src", {-1, 1});
    ctx->SetOutputDim("Out_Dst", {-1, 1});
    ctx->SetOutputDim("Sample_index", {-1});

    auto dims = ctx->GetInputDim("X");
    ctx->SetOutputDim("Reindex_X", dims);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "Src"),
        ctx.device_context());
  }
};

class GraphSampleNeighborsOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Src", "The src index tensor after sorted by dst.");
    AddInput("Src_Eids", "The eids of the input graph edges.").AsDispensable();
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
    AddOutput("Reindex_X", "The reindex node id of the input nodes.");
    AddOutput("Out_Eids", "The eids of the sample edges.").AsIntermediate();
    AddAttr<std::vector<int>>(
        "sample_sizes", "The sample sizes of graph sample neighbors method.")
        .SetDefault({});
    AddAttr<bool>("return_eids",
                  "Whether to return the eids of the sample edges.")
        .SetDefault(false);
    AddComment(R"DOC(
Graph Learning Sampling Neighbors operator, for graphsage sampling method.

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
using CPU = paddle::platform::CPUDeviceContext;

REGISTER_OPERATOR(graph_sample_neighbors, ops::GraphSampleNeighborsOP,
                  ops::GraphSampleNeighborsOpMaker);
REGISTER_OP_CPU_KERNEL(graph_sample_neighbors,
                       ops::GraphSampleNeighborsOpKernel<CPU, int>,
                       ops::GraphSampleNeighborsOpKernel<CPU, int64_t>);
