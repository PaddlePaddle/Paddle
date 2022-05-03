/*Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

class IndexFillOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"), ctx.GetPlace());
  }
};

class IndexFillOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(Tensor, default input Tensor<float>), "
             "the input feature data of IndexFillOp, dtype should be"
             "int32, int64, float16, float32, float64.");
    AddInput("Index",
             "(Tensor, default 1-d Tensor<int>), "
             "the 1-D tensor containing the indices to index, "
             "dtype should be int32, int64");
    AddAttr<int>("axis",
                 "(int, default 0), "
                 "the dimension in which we index.")
        .SetDefault(0);
    AddAttr<float>("fill_value",
                   "(float, default 0.0f) The value to be filled.")
        .SetDefault(0.0f);
    AddOutput("Out",
              "(Tensor, default Tensor<float>),"
              " the output of  IndexFillOp, whose dtype is the same as X.");
    AddComment(R"DOC(
                IndexFill operator
                Fills the elements of the input tensor with value
                by selecting the indices in the order given in index.

                This operator also supports inplace modification.
        )DOC");
  }
};

template <typename T>
class IndexFillGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

  void Apply(GradOpPtr<T> op) const override {
    op->SetType("index_fill_grad");
    op->SetInput("Index", this->Input("Index"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetAttrMap(this->Attrs());
  }
};

class IndexFillGradOp : public framework::OperatorWithKernel {
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

DECLARE_INPLACE_OP_INFERER(IndexFillInplaceInferer, {"X", "Out"});
DECLARE_INPLACE_OP_INFERER(IndexFillGradInplaceInferer,
                           {framework::GradVarName("Out"),
                            framework::GradVarName("X")});
DECLARE_NO_NEED_BUFFER_VARS_INFERER(IndexFillGradNoNeedBufferVarsInferer, "X");

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
DECLARE_INFER_SHAPE_FUNCTOR(index_fill, IndexFillInferShapeFunctor,
                            PD_INFER_META(phi::IndexFillInferMeta));

REGISTER_OPERATOR(index_fill, ops::IndexFillOp, ops::IndexFillOpMaker,
                  ops::IndexFillGradMaker<paddle::framework::OpDesc>,
                  ops::IndexFillGradMaker<paddle::imperative::OpBase>,
                  ops::IndexFillInplaceInferer, IndexFillInferShapeFunctor);

DECLARE_INFER_SHAPE_FUNCTOR(index_fill_grad, IndexFillGradInferShapeFunctor,
                            PD_INFER_META(phi::IndexFillGradInferMeta));

REGISTER_OPERATOR(index_fill_grad, ops::IndexFillGradOp,
                  ops::IndexFillGradInplaceInferer,
                  ops::IndexFillGradNoNeedBufferVarsInferer,
                  IndexFillGradInferShapeFunctor);
