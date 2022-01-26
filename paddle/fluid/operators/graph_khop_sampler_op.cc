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

#include "paddle/fluid/operators/graph_khop_sampler_op.h"

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

class GraphKhopSamplerOP : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("Row"), "Input", "Row", "GraphKhopSampler");
    OP_INOUT_CHECK(ctx->HasInput("Col_Ptr"), "Input", "Col_Ptr",
                   "GraphKhopSampler");
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "GraphKhopSampler");
    OP_INOUT_CHECK(ctx->HasOutput("Out_Src"), "Output", "Out_Src",
                   "GraphKhopSampler");
    OP_INOUT_CHECK(ctx->HasOutput("Out_Dst"), "Output", "Out_Dst",
                   "GraphKhopSampler");
    OP_INOUT_CHECK(ctx->HasOutput("Sample_Index"), "Output", "Sample_Index",
                   "GraphKhopSampler");
    OP_INOUT_CHECK(ctx->HasOutput("Reindex_X"), "Output", "Reindex_X",
                   "GraphKhopSampler");

    // Restrict all the inputs as 1-dim tensor, or 2-dim tensor with the second
    // dim as 1.
    InputShapeCheck(ctx->GetInputDim("Row"), "Row");
    InputShapeCheck(ctx->GetInputDim("Col_Ptr"), "Col_Ptr");
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
      OP_INOUT_CHECK(ctx->HasInput("Eids"), "Input", "Eids",
                     "GraphKhopSampler");
      InputShapeCheck(ctx->GetInputDim("Eids"), "Eids");
      OP_INOUT_CHECK(ctx->HasOutput("Out_Eids"), "Output", "Out_Eids",
                     "GraphKhopSampler");
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
        OperatorWithKernel::IndicateVarDataType(ctx, "Row"),
        ctx.device_context());
  }
};

class GraphKhopSamplerOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Row", "The src index tensor of graph edges after sorted by dst.");
    AddInput("Eids", "The eids of the input graph edges.").AsDispensable();
    AddInput("Col_Ptr",
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

REGISTER_OPERATOR(graph_khop_sampler, ops::GraphKhopSamplerOP,
                  ops::GraphKhopSamplerOpMaker);
REGISTER_OP_CPU_KERNEL(graph_khop_sampler,
                       ops::GraphKhopSamplerOpKernel<CPU, int32_t>,
                       ops::GraphKhopSamplerOpKernel<CPU, int64_t>);
