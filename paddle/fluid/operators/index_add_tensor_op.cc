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

class IndexAddTensorOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"), ctx.GetPlace());
  }
};

class IndexAddTensorOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(Tensor), "
             "the input feature data of IndexAddTensorOp, dtype should be"
             "bool, int32, int64, float16, float32, float64.");
    AddInput("AddValue",
             "(Tensor>), "
             "the input tensor of IndexAddTensorOp, dtype should be"
             "the same as input tensor X");
    AddInput("IndexTensor",
             "(Tensor, optional) If provided, index_add will use this."
             "It has higher priority than attr(index).")
        .AsDispensable();
    AddInput("AxisTensor",
             "(Tensor) If provided, use this as "
             "axis, this has a higher priority than "
             "attr(axis), the shape of this tensor MUST BE (1,).")
        .AsDispensable();
    AddAttr<std::vector<int64_t>>(
        "index",
        "(list<int>) indices of corresponding axis in `axis`");
    AddAttr<int>("axis", "(int), the dimension in which we index.");
    AddOutput(
        "Out",
        "(Tensor, default Tensor<float>),"
        " the output of  IndexAddTensorOp, whose dtype is the same as X.");
    AddComment(R"DOC(
                IndexAddTensor operator
                Add the elements of the input tensor with value
                by selecting the indices in the order given in index.
                This operator also supports inplace modification.
        )DOC");
  }
};

template <typename T>
class IndexAddTensorGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

  void Apply(GradOpPtr<T> op) const override {
    op->SetType("index_add_tensor_grad");
    if (this->HasInput("AxisTensor")) {
      op->SetInput("AxisTensor", this->Input("AxisTensor"));
    }
    if (this->HasInput("IndexTensor")) {
      op->SetInput("IndexTensor", this->Input("IndexTensor"));
    }
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetOutput(framework::GradVarName("AddValue"),
                  this->InputGrad("AddValue"));
    op->SetAttrMap(this->Attrs());
  }
};

class IndexAddTensorGradOp : public framework::OperatorWithKernel {
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

DECLARE_INPLACE_OP_INFERER(IndexAddTensorInplaceInferer, {"X", "Out"});
DECLARE_INPLACE_OP_INFERER(IndexAddTensorGradInplaceInferer,
                           {framework::GradVarName("Out"),
                            framework::GradVarName("X")});

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
DECLARE_INFER_SHAPE_FUNCTOR(index_add_tensor, IndexAddTensorInferShapeFunctor,
                            PD_INFER_META(phi::IndexAddTensorInferMeta));

REGISTER_OPERATOR(index_add_tensor, ops::IndexAddTensorOp,
                  ops::IndexAddTensorOpMaker,
                  ops::IndexAddTensorGradMaker<paddle::framework::OpDesc>,
                  ops::IndexAddTensorGradMaker<paddle::imperative::OpBase>,
                  ops::IndexAddTensorInplaceInferer,
                  IndexAddTensorInferShapeFunctor);

DECLARE_INFER_SHAPE_FUNCTOR(index_add_tensor_grad,
                            IndexAddTensorGradInferShapeFunctor,
                            PD_INFER_META(phi::IndexAddTensorGradInferMeta));

REGISTER_OPERATOR(index_add_tensor_grad, ops::IndexAddTensorGradOp,
                  ops::IndexAddTensorGradInplaceInferer,
                  IndexAddTensorGradInferShapeFunctor);