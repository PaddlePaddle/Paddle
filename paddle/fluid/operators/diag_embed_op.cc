// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/phi/infermeta/unary.h"

namespace paddle {
namespace operators {

class DiagEmbedOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
};

class DiagEmbedOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Input", "The input tensor. Must be at least 1-dimensional.");
    AddOutput("Out", "A matrix whose certain 2D planes is diagonal matrix.");

    AddAttr<int>(
        "offset",
        R"DOC((int, default 0), which diagonal to consider. Default: 0 (main diagonal).
        )DOC")
        .SetDefault(0);
    AddAttr<int>(
        "dim1",
        R"DOC((int, default -2), first dimension with respect to which to take diagonal. Default: -2.
        )DOC")
        .SetDefault(-2);
    AddAttr<int>(
        "dim2",
        R"DOC((int, default -1), second dimension with respect to which to take diagonal. Default: -1.
        )DOC")
        .SetDefault(-1);

    AddComment(R"DOC(Creates a tensor whose diagonals of certain 2D planes
              (specified by dim1 and dim2) are filled by input.
              To facilitate creating batched diagonal matrices,
              the 2D planes formed by the last two dimensions of the returned tensor
              are chosen by default.
              )DOC");
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
DECLARE_INFER_SHAPE_FUNCTOR(diag_embed,
                            DiagEmbedInferShapeFunctor,
                            PD_INFER_META(phi::DiagEmbedInferMeta));

REGISTER_OPERATOR(
    diag_embed,
    ops::DiagEmbedOp,
    ops::DiagEmbedOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>,
    DiagEmbedInferShapeFunctor);
