/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <memory>

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/nullary.h"

namespace paddle {
namespace operators {

class TriuIndicesOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        framework::proto::VarType::Type(ctx.Attr<int>("dtype")),
        ctx.GetPlace());
  }
};

class TriuIndicesOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddOutput("out",
              "Tensor, the output tensor, with the shape (2,x), x bounded by "
              "[0,row*col])");
    AddAttr<int>("row",
                 "int number, the input of triu_indices op"
                 "which describes the number of row of the matrix")
        .SetDefault(0);
    AddAttr<int>("col",
                 "int number, the input of triu_indices op"
                 "which describes the number of col of the matrix")
        .SetDefault(0);
    AddAttr<int>(
        "offset",
        "int number, the input of triu_indices op bounded by [1-rows,cols-1"
        "which describes the dignalline index of the upper triangular part of "
        "the matrix")
        .SetDefault(0);
    AddAttr<int>("dtype", "data type ,the input of triu_indices op")
        .SetDefault(framework::proto::VarType::INT64);

    AddComment(R"DOC(
  TriuIndices Operator.
  The triu_indices operator returns the indices of the upper triangular part of the matrix
  whose rows and cols is known. It is a 2-by-x tensor, where the first row contains row coordinates
  of all indices and the second row contains column coordinates. Indices are ordered based on
  rows and then columns. The upper triangular part of the matrix is defined as the elements on
  and below the diagonal.
  The argument offset controls which diagonal to consider, default value is 0.
  A positive value includes just as fewer diagonals above the main diagonal,
  and similarly a negative value excludes just as fewer diagonals below the main diagonal
  )DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
DECLARE_INFER_SHAPE_FUNCTOR(triu_indices,
                            TriuIndicesInferShapeFunctor,
                            PD_INFER_META(phi::TriuIndicesInferMeta));

REGISTER_OPERATOR(
    triu_indices,
    ops::TriuIndicesOp,
    ops::TriuIndicesOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>,
    TriuIndicesInferShapeFunctor);
