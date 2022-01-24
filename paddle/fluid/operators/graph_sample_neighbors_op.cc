/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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

void InputShapeCheck(const framework::DDim& dims, std::string tensor_name) {
  if (dims.size() == 2) {
    PADDLE_ENFORCE_EQ(dims[1], 1, platform::errors::InvalidArgument(
                                      "The last dim of %s should be 1 when it "
                                      "is 2D, but we get %d",
                                      tensor_name, dims[1]));
  } else {
    PADDLE_ENFORCE_EQ(
        dims.size(), 1,
        platform::errors::InvalidArgument(
            "The %s should be 1D, when it is not 2D, but we get %d",
            tensor_name, dims.size()));
  }
}

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
    OP_INOUT_CHECK(ctx->HasOutput("Sample_Index"), "Output", "Sample_Index",
                   "GraphSampleNeighbors");
    OP_INOUT_CHECK(ctx->HasOutput("Reindex_X"), "Output", "Reindex_X",
                   "GraphSampleNeighbors");

    // Restrict all the inputs as 1-dim tensor, or 2-dim tensor with the second
    // dim as 1.
    InputShapeCheck(ctx->GetInputDim("Src"), "Src");
    InputShapeCheck(ctx->GetInputDim("Dst_Count"), "Dst_Count");
    InputShapeCheck(ctx->GetInputDim("X"), "X");

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
      InputShapeCheck(ctx->GetInputDim("Src_Eids"), "Src_Eids");
      OP_INOUT_CHECK(ctx->HasOutput("Out_Eids"), "Output", "Out_Eids",
                     "GraphSampleNeighbors");
      ctx->SetOutputDim("Out_Eids", {-1});
    }

    ctx->SetOutputDim("Out_Src", {-1, 1});
    ctx->SetOutputDim("Out_Dst", {-1, 1});
    ctx->SetOutputDim("Sample_Index", {-1});

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
    AddInput("Src", "The src index tensor of graph edges after sorted by dst.");
    AddInput("Src_Eids", "The eids of the input graph edges.").AsDispensable();
    AddInput("Dst_Count",
             "The cumulative sum of the number of src neighbors of dst index, "
             "starts from 0, end with number of edges");
    AddInput("X", "The input center nodes index tensor.");
    AddOutput("Out_Src",
              "The output src edges tensor after sampling and reindex.");
    AddOutput("Out_Dst",
              "The output dst edges tensor after sampling and reindex.");
    AddOutput("Sample_Index",
              "The original index of the center nodes and sampling nodes");
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
                       ops::GraphSampleNeighborsOpKernel<CPU, int32_t>,
                       ops::GraphSampleNeighborsOpKernel<CPU, int64_t>);
