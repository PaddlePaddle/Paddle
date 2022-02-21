/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/unbind_op.h"
#include <string>
#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/unary.h"

namespace paddle {
namespace operators {
using framework::Tensor;

class UnbindOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
};

class UnbindOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor) Input tensor of the split operator.");
    AddOutput("Out", "(Tensor) Output tensors of the unbind operator.")
        .AsDuplicable();
    AddComment(R"DOC(
Unbind operator

Remove a tensor dimension.

Example:
  Input = [[1,2],
           [3,4],
           [5,6]]
  axis = 0
  Output[0] = [1,2]
  Output[1] = [3,4]
  Output[2] = [5,6]

    )DOC");
    AddAttr<int>("axis",
                 "(int, default 0) "
                 "dimension to remove.")
        .SetDefault(0);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

DELCARE_INFER_SHAPE_FUNCTOR(unbind, UnbindInferShapeFunctor,
                            PT_INFER_META(phi::UnbindInferMeta));

REGISTER_OPERATOR(unbind, ops::UnbindOp, ops::UnbindOpMaker,
                  ops::UnbindGradMaker<paddle::framework::OpDesc>,
                  ops::UnbindGradMaker<paddle::imperative::OpBase>,
                  UnbindInferShapeFunctor);
