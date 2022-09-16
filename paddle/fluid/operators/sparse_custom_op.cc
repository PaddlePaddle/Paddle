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

DECLARE_INFER_SHAPE_FUNCTOR(
    sparse_coo_tensor,
    SparseCooTensorInferShapeFunctor,
    PD_INFER_META(phi::sparse::SparseCooTensorInferMeta));

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
                            PD_INFER_META(phi::sparse::ValuesInferMeta));

class IndicesCooOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor), input 0 of indices_coo op.");
    AddOutput("Out", "(Tensor), output 0 of indices_coo op.");
    AddComment(R"DOC(
TODO: Documentation of indices_coo op.
)DOC");
  }
};

class IndicesCooOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
};

DECLARE_INFER_SHAPE_FUNCTOR(indices_coo,
                            IndicesCooInferShapeFunctor,
                            PD_INFER_META(phi::UnchangedInferMeta));

class CooToDenseOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor), input 0 of values_coo op.");
    AddOutput("Out", "(Tensor), output 0 of values_coo op.");
    AddComment(R"DOC(
TODO: Documentation of values_coo op.
)DOC");
  }
};

class CooToDenseOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
};

DECLARE_INFER_SHAPE_FUNCTOR(coo_to_dense,
                            CooToDenseInferShapeFunctor,
                            PD_INFER_META(phi::UnchangedInferMeta));

class ReluCooOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor), input 0 of relu_coo op.");
    AddOutput("Out", "(Tensor), output 0 of relu_coo op.");
    AddComment(R"DOC(
TODO: Documentation of relu_coo op.
)DOC");
  }
};

class ReluCooOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
};

DECLARE_INFER_SHAPE_FUNCTOR(relu_coo,
                            ReluCooInferShapeFunctor,
                            PD_INFER_META(phi::UnchangedInferMeta));

class ShapeCooOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor), input 0 of relu_coo op.");
    AddOutput("Out", "(Tensor), output 0 of relu_coo op.");
    AddComment(R"DOC(
TODO: Documentation of relu_coo op.
)DOC");
  }
};

class ShapeCooOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
};

DECLARE_INFER_SHAPE_FUNCTOR(shape_coo,
                            ShapeCooInferShapeFunctor,
                            PD_INFER_META(phi::UnchangedInferMeta));

class Conv3dCooOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor), input 0 of conv3d_coo op.");
    AddInput("Kernel", "(Tensor), input 1 of conv3d_coo op.");
    AddOutput("Out", "(Tensor), output 0 of conv3d_coo op.");
    AddOutput("Rulebook", "(Tensor), output 1 of conv3d_coo op.");
    AddOutput("Counter", "(Tensor), output 2 of conv3d_coo op.");
    AddAttr<std::vector<int>>("paddings",
                              "(vector<int>), attribute 0 for conv3d_coo op.");
    AddAttr<std::vector<int>>("dilations",
                              "(vector<int>), attribute 1 for conv3d_coo op.");
    AddAttr<std::vector<int>>("strides",
                              "(vector<int>), attribute 2 for conv3d_coo op.");
    AddAttr<int>("groups", "(int), attribute 3 for conv3d_coo op.");
    AddAttr<bool>("subm", "(bool), attribute 4 for conv3d_coo op.");
    AddAttr<std::string>("key", "(string), attribute 5 for conv3d_coo op.")
        .SetDefault("");
    AddComment(R"DOC(
TODO: Documentation of conv3d_coo op.
)DOC");
  }
};

class Conv3dCooOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
};

DECLARE_INFER_SHAPE_FUNCTOR(conv3d_coo,
                            Conv3dCooInferShapeFunctor,
                            PD_INFER_META(phi::sparse::Conv3dInferMeta));

class ValuesAddCooCooOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor), input 0 of values_add_coo_coo op.");
    AddInput("Y", "(Tensor), input 1 of values_add_coo_coo op.");
    AddOutput("Out", "(Tensor), output 0 of relu_coo op.");
    AddComment(R"DOC(
TODO: Documentation of values_add_coo_coo op.
)DOC");
  }
};

class ValuesAddCooCooOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
};

DECLARE_INFER_SHAPE_FUNCTOR(values_add_coo_coo,
                            ValuesAddCooCooInferShapeFunctor,
                            PD_INFER_META(phi::UnchangedInferMeta));

class ValuesAddCooDenseOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor), input 0 of values_add_coo_dense op.");
    AddInput("Y", "(Tensor), input 1 of values_add_coo_dense op.");
    AddOutput("Out", "(Tensor), output 0 of relu_coo op.");
    AddComment(R"DOC(
TODO: Documentation of values_add_coo_dense op.
)DOC");
  }
};

class ValuesAddCooDenseOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
};

DECLARE_INFER_SHAPE_FUNCTOR(values_add_coo_dense,
                            ValuesAddCooDenseInferShapeFunctor,
                            PD_INFER_META(phi::UnchangedInferMeta));

class CastCooOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor), input 0 of cast_coo op.");
    AddOutput("Out", "(Tensor), output 0 of cast_coo op.");
    AddAttr<int>("index_dtype", "(int), attribute 0 for cholesky op.");
    AddAttr<int>("value_dtype", "(int), attribute 0 for cholesky op.");
    AddComment(R"DOC(
TODO: Documentation of cast_coo op.
)DOC");
  }
};

class CastCooOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
};

DECLARE_INFER_SHAPE_FUNCTOR(cast_coo,
                            CastCooInferShapeFunctor,
                            PD_INFER_META(phi::UnchangedInferMeta));

class AddCooCooOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor), input 0 of add_coo_coo op.");
    AddInput("Y", "(Tensor), input 1 of add_coo_coo op.");
    AddOutput("Out", "(Tensor), output 0 of add_coo_coo op.");
    AddComment(R"DOC(
TODO: Documentation of add_coo_coo op.
)DOC");
  }
};

class AddCooCooOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
};

DECLARE_INFER_SHAPE_FUNCTOR(add_coo_coo,
                            AddCooCooInferShapeFunctor,
                            PD_INFER_META(phi::UnchangedInferMeta));

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

REGISTER_OPERATOR(indices_coo,
                  ops::IndicesCooOp,
                  ops::IndicesCooOpMaker,
                  ops::IndicesCooInferShapeFunctor);

REGISTER_OPERATOR(coo_to_dense,
                  ops::CooToDenseOp,
                  ops::CooToDenseOpMaker,
                  // ops::TraceGradOpMaker<paddle::framework::OpDesc>,
                  // ops::TraceGradOpMaker<paddle::imperative::OpBase>,
                  ops::CooToDenseInferShapeFunctor);

REGISTER_OPERATOR(relu_coo,
                  ops::ReluCooOp,
                  ops::ReluCooOpMaker,
                  // ops::TraceGradOpMaker<paddle::framework::OpDesc>,
                  // ops::TraceGradOpMaker<paddle::imperative::OpBase>,
                  ops::ReluCooInferShapeFunctor);

REGISTER_OPERATOR(shape_coo,
                  ops::ShapeCooOp,
                  ops::ShapeCooOpMaker,
                  ops::ShapeCooInferShapeFunctor);

REGISTER_OPERATOR(conv3d_coo,
                  ops::Conv3dCooOp,
                  ops::Conv3dCooOpMaker,
                  // ops::TraceGradOpMaker<paddle::framework::OpDesc>,
                  // ops::TraceGradOpMaker<paddle::imperative::OpBase>,
                  ops::Conv3dCooInferShapeFunctor);

REGISTER_OPERATOR(values_add_coo_coo,
                  ops::ValuesAddCooCooOp,
                  ops::ValuesAddCooCooOpMaker,
                  ops::ValuesAddCooCooInferShapeFunctor);

REGISTER_OPERATOR(values_add_coo_dense,
                  ops::ValuesAddCooDenseOp,
                  ops::ValuesAddCooDenseOpMaker,
                  ops::ValuesAddCooDenseInferShapeFunctor);

REGISTER_OPERATOR(cast_coo,
                  ops::CastCooOp,
                  ops::CastCooOpMaker,
                  ops::CastCooInferShapeFunctor);

REGISTER_OPERATOR(add_coo_coo,
                  ops::AddCooCooOp,
                  ops::AddCooCooOpMaker,
                  ops::AddCooCooInferShapeFunctor);
