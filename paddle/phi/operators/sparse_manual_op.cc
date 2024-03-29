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

#include <memory>
#include <string>

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/binary.h"
#include "paddle/phi/infermeta/multiary.h"
#include "paddle/phi/infermeta/sparse/binary.h"
#include "paddle/phi/infermeta/sparse/unary.h"
#include "paddle/phi/infermeta/unary.h"

namespace paddle {
namespace operators {

class SparseIndicesOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("x", "(Tensor), input 0 of sparse_indices op.");
    AddOutput("out", "(Tensor), output 0 of sparse_indices op.");
    AddComment(R"DOC(
TODO: Documentation of sparse_indices op.
)DOC");
  }
};

class SparseIndicesOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
};

DECLARE_INFER_SHAPE_FUNCTOR(sparse_indices,
                            SparseIndicesInferShapeFunctor,
                            PD_INFER_META(phi::sparse::IndicesInferMeta));

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(sparse_indices,
                  ops::SparseIndicesOp,
                  ops::SparseIndicesOpMaker,
                  ops::SparseIndicesInferShapeFunctor);
