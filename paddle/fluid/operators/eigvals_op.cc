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

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/unary.h"

namespace paddle {
namespace operators {
class EigvalsOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(Tensor), A complex- or real-valued tensor with shape (*, n, n)"
             "where * is zero or more batch dimensions");
    AddOutput("Out",
              "(Tensor) The output tensor with shape (*,n) cointaining the "
              "eigenvalues of X.");
    AddComment(R"DOC(eigvals operator
        Return the eigenvalues of one or more square matrices. The eigenvalues are complex even when the input matrices are real.
        )DOC");
  }
};

class EigvalsOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
};

}  // namespace operators
}  // namespace paddle
namespace ops = paddle::operators;

DECLARE_INFER_SHAPE_FUNCTOR(eigvals,
                            EigvalsInferShapeFunctor,
                            PD_INFER_META(phi::EigvalsInferMeta));

REGISTER_OPERATOR(eigvals,
                  ops::EigvalsOp,
                  ops::EigvalsOpMaker,
                  EigvalsInferShapeFunctor);
