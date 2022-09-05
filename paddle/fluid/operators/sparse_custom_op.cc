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
#include "paddle/phi/infermeta/sparse/unary.h"

namespace paddle {
namespace operators {

class SparseCooTensorOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Values", "(Tensor), input 0 of sparse_coo_tensor op.");
    AddInput("Indices", "(Tensor), input 1 of sparse_coo_tensor op.");
    AddOutput("Out", "(Tensor), output 0 of sparse_coo_tensor op.");
    AddAttr<std::vector<int>>(
        "dense_shape", "(vector<int>), attribute 0 for sparse_coo_tensor op.");
    AddComment(R"DOC(
TODO: Documentation of sparse_coo_tensor op.
)DOC");
  }
};

class SparseCooTensorOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
};

DECLARE_INFER_SHAPE_FUNCTOR(sparse_coo_tensor,
                            SparseCooTensorInferShapeFunctor,
                            PD_INFER_META(phi::sparse::UnchangedInferMeta));

class ValuesCooOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor), input 0 of values_coo op.");
    AddOutput("Out", "(Tensor), output 0 of values_coo op.");
    AddComment(R"DOC(
TODO: Documentation of values_coo op.
)DOC");
  }
};

class ValuesCooOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
};

DECLARE_INFER_SHAPE_FUNCTOR(values_coo,
                            ValuesCooInferShapeFunctor,
                            PD_INFER_META(phi::sparse::UnchangedInferMeta));

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(sparse_coo_tensor,
                  ops::SparseCooTensorOp,
                  ops::SparseCooTensorOpMaker,
                  // ops::TraceGradOpMaker<paddle::framework::OpDesc>,
                  // ops::TraceGradOpMaker<paddle::imperative::OpBase>,
                  ops::SparseCooTensorInferShapeFunctor);

REGISTER_OPERATOR(values_coo,
                  ops::ValuesCooOp,
                  ops::ValuesCooOpMaker,
                  // ops::TraceGradOpMaker<paddle::framework::OpDesc>,
                  // ops::TraceGradOpMaker<paddle::imperative::OpBase>,
                  ops::ValuesCooInferShapeFunctor);
