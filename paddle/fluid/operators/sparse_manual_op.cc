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
#include "paddle/phi/infermeta/sparse/binary.h"
#include "paddle/phi/infermeta/sparse/unary.h"
#include "paddle/phi/infermeta/unary.h"

namespace paddle {
namespace operators {

class SparseSparseCooTensorOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("values", "(Tensor), input 0 of sparse_coo_tensor op.");
    AddInput("indices", "(Tensor), input 1 of sparse_coo_tensor op.");
    AddOutput("out", "(Tensor), output 0 of sparse_coo_tensor op.");
    AddAttr<std::vector<int>>(
        "dense_shape", "(vector<int>), attribute 0 for sparse_coo_tensor op.");
    AddComment(R"DOC(
TODO: Documentation of sparse_coo_tensor op.
)DOC");
  }
};

class SparseSparseCooTensorOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
};

DECLARE_INFER_SHAPE_FUNCTOR(
    sparse_sparse_coo_tensor,
    SparseSparseCooTensorInferShapeFunctor,
    PD_INFER_META(phi::sparse::SparseCooTensorInferMeta));

class SparseValuesOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("x", "(Tensor), input 0 of sparse_values op.");
    AddOutput("out", "(Tensor), output 0 of sparse_values op.");
    AddComment(R"DOC(
TODO: Documentation of sparse_values op.
)DOC");
  }
};

class SparseValuesOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
};

DECLARE_INFER_SHAPE_FUNCTOR(sparse_values,
                            SparseValuesInferShapeFunctor,
                            PD_INFER_META(phi::sparse::ValuesInferMeta));

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

class SparseToDenseOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("x", "(Tensor), input 0 of sparse_to_dense op.");
    AddOutput("out", "(Tensor), output 0 of sparse_to_dense op.");
    AddComment(R"DOC(
TODO: Documentation of sparse_to_dense op.
)DOC");
  }
};

class SparseToDenseOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
};

DECLARE_INFER_SHAPE_FUNCTOR(sparse_to_dense,
                            SparseToDenseInferShapeFunctor,
                            PD_INFER_META(phi::UnchangedInferMeta));

class SparseReluOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("x", "(Tensor), input 0 of sparse_relu op.");
    AddOutput("out", "(Tensor), output 0 of sparse_relu op.");
    AddComment(R"DOC(
TODO: Documentation of sparse_relu op.
)DOC");
  }
};

class SparseReluOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
};

DECLARE_INFER_SHAPE_FUNCTOR(sparse_relu,
                            SparseReluInferShapeFunctor,
                            PD_INFER_META(phi::UnchangedInferMeta));

class SparseConv3dOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("x", "(Tensor), input 0 of sparse_conv3d op.");
    AddInput("kernel", "(Tensor), input 1 of sparse_conv3d op.");
    AddOutput("out", "(Tensor), output 0 of sparse_conv3d op.");
    AddOutput("rulebook", "(Tensor), output 1 of sparse_conv3d op.");
    AddOutput("counter", "(Tensor), output 2 of sparse_conv3d op.");
    AddAttr<std::vector<int>>(
        "paddings", "(vector<int>), attribute 0 for sparse_conv3d op.");
    AddAttr<std::vector<int>>(
        "dilations", "(vector<int>), attribute 1 for sparse_conv3d op.");
    AddAttr<std::vector<int>>(
        "strides", "(vector<int>), attribute 2 for sparse_conv3d op.");
    AddAttr<int>("groups", "(int), attribute 3 for sparse_conv3d op.");
    AddAttr<bool>("subm", "(bool), attribute 4 for conv3d_coo op.");
    AddAttr<std::string>("key", "(string), attribute 5 for sparse_conv3d op.")
        .SetDefault("");
    AddComment(R"DOC(
TODO: Documentation of sparse_conv3d op.
)DOC");
  }
};

class SparseConv3dOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
};

DECLARE_INFER_SHAPE_FUNCTOR(sparse_conv3d,
                            SparseConv3dInferShapeFunctor,
                            PD_INFER_META(phi::sparse::Conv3dInferMeta));

class SparseAddOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("x", "(Tensor), input 0 of sparse_add op.");
    AddInput("y", "(Tensor), input 1 of sparse_add op.");
    AddOutput("out", "(Tensor), output 0 of sparse_add op.");
    AddComment(R"DOC(
TODO: Documentation of sparse_add op.
)DOC");
  }
};

class SparseAddOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
};

DECLARE_INFER_SHAPE_FUNCTOR(sparse_add,
                            SparseAddInferShapeFunctor,
                            PD_INFER_META(phi::UnchangedInferMeta));

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(sparse_sparse_coo_tensor,
                  ops::SparseSparseCooTensorOp,
                  ops::SparseSparseCooTensorOpMaker,
                  ops::SparseSparseCooTensorInferShapeFunctor);

REGISTER_OPERATOR(sparse_values,
                  ops::SparseValuesOp,
                  ops::SparseValuesOpMaker,
                  ops::SparseValuesInferShapeFunctor);

REGISTER_OPERATOR(sparse_indices,
                  ops::SparseIndicesOp,
                  ops::SparseIndicesOpMaker,
                  ops::SparseIndicesInferShapeFunctor);

REGISTER_OPERATOR(sparse_to_dense,
                  ops::SparseToDenseOp,
                  ops::SparseToDenseOpMaker,
                  ops::SparseToDenseInferShapeFunctor);

REGISTER_OPERATOR(sparse_relu,
                  ops::SparseReluOp,
                  ops::SparseReluOpMaker,
                  ops::SparseReluInferShapeFunctor);

REGISTER_OPERATOR(sparse_conv3d,
                  ops::SparseConv3dOp,
                  ops::SparseConv3dOpMaker,
                  ops::SparseConv3dInferShapeFunctor);

REGISTER_OPERATOR(sparse_add,
                  ops::SparseAddOp,
                  ops::SparseAddOpMaker,
                  ops::SparseAddInferShapeFunctor);
