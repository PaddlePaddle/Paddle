// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/framework/operator.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/unary.h"

namespace paddle {
namespace operators {
class DirichletOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Alpha", "(Tensor), The dirichlet Alpha parameter");
    AddOutput("Out", "(Tensor), The output tensor of sample");
    AddComment(R"DOC(Sample random data from dirichlet distribution.)DOC");
  }
};

class DirichletOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
};

}  // namespace operators
}  // namespace paddle

DECLARE_INFER_SHAPE_FUNCTOR(dirichlet,
                            DirichletInferShapeFunctor,
                            PD_INFER_META(phi::DirichletInferMeta));

REGISTER_OP_WITHOUT_GRADIENT(dirichlet,
                             paddle::operators::DirichletOp,
                             paddle::operators::DirichletOpMaker,
                             DirichletInferShapeFunctor);
