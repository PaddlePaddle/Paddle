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
#include "paddle/phi/infermeta/backward.h"
#include "paddle/phi/infermeta/unary.h"

namespace paddle {
namespace operators {

class NanmedianOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"), ctx.GetPlace());
  }
};

class NanmedianOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(Tensor), "
             "the input feature data of NanmedianOp, dtype should be"
             "int32, int64, float16, float32 or float64.");
    AddOutput(
        "MedianIndex",
        "Store the index position of median values, The calculation differs "
        "in the odd or even valid elements numbers."
        "Along the axis, two elements contributed to the median value in "
        "each row."
        "If the amount of valid elements were even, both were the same.")
        .AsIntermediate()
        .AsExtra();
    AddOutput("Out",
              "(Tensor),"
              " the output of  NanmedianOp, whose dtype is the same as X");
    AddAttr<bool>("keepdim",
                  "(bool, default true) "
                  "If true, retain the reduced axis with length 1.")
        .SetDefault(true);
    AddAttr<std::vector<int>>("axis",
                              "(std::vector<int>). List of integers,"
                              " indicating the dimensions to calculate medians")
        .SetDefault({});
    AddComment(R"DOC(
                Nanmedian operator

                This operator is considered as an extention of median operation,
                which supports specifically the case of NaN values in the input.

                If all the elements in input are NaN it will also return NaN.
                If no elements in input are Nan, this op is identical to thie median op.

                If the valid count of elements is a even number, the average value of
                the elements in the middle is calculated as the median.

                This operator can also supports multiple axis.
        )DOC");
  }
};

template <typename T>
class NanmedianGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

  void Apply(GradOpPtr<T> op) const override {
    op->SetType("nanmedian_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput("MedianIndex", this->Output("MedianIndex"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetAttrMap(this->Attrs());
  }
};

class NanmedianGradOp : public framework::OperatorWithKernel {
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

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
DECLARE_INFER_SHAPE_FUNCTOR(nanmedian,
                            NanmedianInferShapeFunctor,
                            PD_INFER_META(phi::NanmedianInferMeta));

REGISTER_OPERATOR(nanmedian,
                  ops::NanmedianOp,
                  ops::NanmedianOpMaker,
                  ops::NanmedianGradMaker<paddle::framework::OpDesc>,
                  ops::NanmedianGradMaker<paddle::imperative::OpBase>,
                  NanmedianInferShapeFunctor);

DECLARE_INFER_SHAPE_FUNCTOR(nanmedian_grad,
                            NanmedianGradInferShapeFunctor,
                            PD_INFER_META(phi::NanmedianGradInferMeta));

REGISTER_OPERATOR(nanmedian_grad,
                  ops::NanmedianGradOp,
                  NanmedianGradInferShapeFunctor);
