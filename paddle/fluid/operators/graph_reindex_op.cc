// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/multiary.h"

namespace paddle {
namespace operators {

class GraphReindexOP : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"),
        ctx.device_context());
  }
};

class GraphReindexOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The destination nodes of the input graph.");
    AddInput("Neighbors", "The neighbor nodes of the destination nodes `X`.");
    AddInput("Count", "The number of neighbor nodes of each destination node.");
    // Note(daisiming): If using buffer hashtable, we must ensure the number of
    // nodes of the input graph should be no larger than maximum(int32).
    AddInput("HashTable_Value",
             "One of the buffer tensor of hashtable for reindex")
        .AsDispensable();
    AddInput("HashTable_Index",
             "One of the buffer tensor of hashtable for reindex")
        .AsDispensable();
    AddAttr<bool>("flag_buffer_hashtable",
                  "Define whether using the buffer hashtable.")
        .SetDefault(false);
    AddOutput("Reindex_Src",
              "The source node index of graph edges after reindex.");
    AddOutput("Reindex_Dst",
              "The destination node index of graph edges after reindex.");
    AddOutput("Out_Nodes", "The original index of graph nodes before reindex");

    AddComment(R"DOC(
Graph Reindex operator.
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

DECLARE_INFER_SHAPE_FUNCTOR(graph_reindex, GraphReindexInferShapeFunctor,
                            PD_INFER_META(phi::GraphReindexInferMeta));

REGISTER_OPERATOR(
    graph_reindex, ops::GraphReindexOP, ops::GraphReindexOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>,
    GraphReindexInferShapeFunctor);
