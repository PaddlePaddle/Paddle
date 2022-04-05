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

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/multiary.h"

namespace paddle {
namespace operators {

class GraphSampleNeighborsOP : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "Row"),
        ctx.device_context());
  }
};

class GraphSampleNeighborsOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Row",
             "One of the components of the CSC format of the input graph.");
    AddInput("Col_Ptr",
             "One of the components of the CSC format of the input graph.");
    AddInput("X", "The input center nodes index tensor.");
    AddInput("Eids", "The edge ids of the input graph.").AsDispensable();
    AddInput("Perm_Buffer", "Permutation buffer for fisher-yates sampling.")
        .AsDispensable();
    AddOutput("Out", "The neighbors of input nodes X after sampling.");
    AddOutput("Out_Count",
              "The number of sample neighbors of input nodes respectively.");
    AddOutput("Out_Eids", "The eids of the sample edges");
    AddAttr<int>(
        "sample_size", "The sample size of graph sample neighbors method. ",
        "Set default value as -1, means return all neighbors of nodes.")
        .SetDefault(-1);
    AddAttr<bool>("return_eids",
                  "Whether to return the eid of the sample edges.")
        .SetDefault(false);
    AddAttr<bool>("flag_perm_buffer",
                  "Using the permutation for fisher-yates sampling in GPU"
                  "Set default value as false, means not using it.")
        .SetDefault(false);
    AddComment(R"DOC(
Graph Learning Sampling Neighbors operator, for graphsage sampling method.
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

DECLARE_INFER_SHAPE_FUNCTOR(graph_sample_neighbors,
                            GraphSampleNeighborsInferShapeFunctor,
                            PD_INFER_META(phi::GraphSampleNeighborsInferMeta));

REGISTER_OPERATOR(
    graph_sample_neighbors, ops::GraphSampleNeighborsOP,
    ops::GraphSampleNeighborsOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>,
    GraphSampleNeighborsInferShapeFunctor);
