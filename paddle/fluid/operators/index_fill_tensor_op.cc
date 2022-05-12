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
#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/binary.h"

namespace paddle {
namespace operators {

class IndexFillTensorOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"), ctx.GetPlace());
  }
};

class IndexFillTensorOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(Tensor), "
             "the input feature data of IndexFillTensorOp, dtype should be"
             "bool, int32, int64, float16, float32, float64.");
    AddInput("FillValue",
             "(Tensor>), "
             "the input tensor of IndexFillTensorOp, dtype should be"
             "the same as input tensor X");
    AddInput("IndexTensor",
             "(Tensor, optional) If provided, index fill will use this."
             "It has the highest priority of IndexTensor and attr(index).")
        .AsDispensable();
    AddInput("AxisTensor",
             "(Tensor) If provided, use this as "
             "axis, this has a higher priority than "
             "attr(axis), the shape of this tensor MUST BE 1.")
        .AsDispensable();
    AddAttr<std::vector<int>>(
        "index",
        "(list<int>) Starting indices of corresponding axis in `axes`");
    AddAttr<int>("axis", "(int), the dimension in which we index.");
    AddAttr<float>("fill_value", "(float), The value to be filled.");
    AddOutput(
        "Out",
        "(Tensor, default Tensor<float>),"
        " the output of  IndexFillTensorOp, whose dtype is the same as X.");
    AddComment(R"DOC(
                IndexFillTensor operator
                Fills the elements of the input tensor with value
                by selecting the indices in the order given in index.

                This operator also supports inplace modification.
        )DOC");
  }
};

template <typename T>
class IndexFillTensorGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

  void Apply(GradOpPtr<T> op) const override {
    op->SetType("index_fill_tensor_grad");
    if (this->HasInput("AxisTensor")) {
      op->SetInput("AxisTensor", this->Input("AxisTensor"));
    }
    if (this->HasInput("IndexTensor")) {
      op->SetInput("IndexTensor", this->Input("IndexTensor"));
    }
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetOutput(framework::GradVarName("FillValue"),
                  this->InputGrad("FillValue"));
    op->SetAttrMap(this->Attrs());
  }
};

class IndexFillTensorGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Out")),
                                   ctx.GetPlace());
  }
};

DECLARE_INPLACE_OP_INFERER(IndexFillTensorInplaceInferer, {"X", "Out"});
DECLARE_INPLACE_OP_INFERER(IndexFillTensorGradInplaceInferer,
                           {framework::GradVarName("Out"),
                            framework::GradVarName("X")});

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
DECLARE_INFER_SHAPE_FUNCTOR(index_fill_tensor, IndexFillTensorInferShapeFunctor,
                            PD_INFER_META(phi::IndexFillTensorInferMeta));

REGISTER_OPERATOR(index_fill_tensor, ops::IndexFillTensorOp,
                  ops::IndexFillTensorOpMaker,
                  ops::IndexFillTensorGradMaker<paddle::framework::OpDesc>,
                  ops::IndexFillTensorGradMaker<paddle::imperative::OpBase>,
                  ops::IndexFillTensorInplaceInferer,
                  IndexFillTensorInferShapeFunctor);

DECLARE_INFER_SHAPE_FUNCTOR(index_fill_tensor_grad,
                            IndexFillTensorGradInferShapeFunctor,
                            PD_INFER_META(phi::IndexFillTensorGradInferMeta));

REGISTER_OPERATOR(index_fill_tensor_grad, ops::IndexFillTensorGradOp,
                  ops::IndexFillTensorGradInplaceInferer,
                  IndexFillTensorGradInferShapeFunctor);
