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

// filename: diag_block_op.cc
#include <string>
#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/binary.h"
#include "paddle/phi/infermeta/multiary.h"
#include "paddle/phi/infermeta/nullary.h"
#include "paddle/phi/infermeta/ternary.h"
#include "paddle/phi/infermeta/unary.h"

namespace paddle {
namespace operators {

class DiagBlockOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor[]), input 0 of diag_block op.").AsDuplicable();
    AddAttr<double>("ss", "dummy attr");
    AddOutput("Out", "(Tensor), output 0 of diag_block op.");
    AddComment(R"DOC(
TODO: Documentation of diag_block op.
)DOC");
  }
};

class DiagBlockOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
};

DECLARE_INFER_SHAPE_FUNCTOR(diag_block, DiagBlockInferShapeFunctor,
                            PD_INFER_META(phi::DiagBlockInferMeta));

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(diag_block, ops::DiagBlockOp, ops::DiagBlockOpMaker,
                  ops::DiagBlockInferShapeFunctor);
