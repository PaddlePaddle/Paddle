// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <memory>
#include <vector>
#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/backward.h"
#include "paddle/phi/infermeta/multiary.h"

namespace plat = paddle::platform;
namespace ops = paddle::operators;

namespace paddle {
namespace operators {

class StackOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto input_data_type =
        framework::OperatorWithKernel::IndicateVarDataType(ctx, "X");

#ifdef PADDLE_WITH_MKLDNN
    if (this->CanMKLDNNBeUsed(ctx, input_data_type)) {
      return framework::OpKernelType(input_data_type, ctx.GetPlace(),
                                     framework::DataLayout::kMKLDNN,
                                     framework::LibraryType::kMKLDNN);
    }
#endif
    return framework::OpKernelType(input_data_type, ctx.GetPlace());
  }
};

class StackOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The input of stack op.").AsDuplicable();
    AddOutput("Y", "The output of stack op.");
    AddAttr<int>("axis",
                 "The axis along which all of the Inputs(X) should be stacked.")
        .SetDefault(0);
    AddAttr<bool>(
        "use_mkldnn",
        "(bool, default false) Indicates if MKL-DNN kernel will be used")
        .SetDefault(false)
        .AsExtra();
    AddComment(R"DOC(
Stack Operator.
Stack all of the Inputs(X) into one tensor along Attr(axis). The dims of all Inputs(X) must be the same.
)DOC");
  }
};

class StackOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
};

template <typename T>
class StackGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("stack_grad");
    op->SetInput(framework::GradVarName("Y"), this->OutputGrad("Y"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X", false));
    op->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

DECLARE_INFER_SHAPE_FUNCTOR(stack, StackInferMetaFunctor,
                            PD_INFER_META(phi::StackInferMeta));
DECLARE_INFER_SHAPE_FUNCTOR(stack_grad, StackGradInferMetaFunctor,
                            PD_INFER_META(phi::StackGradInferMeta));
REGISTER_OPERATOR(stack, ops::StackOp, ops::StackOpMaker,
                  ops::StackGradOpMaker<paddle::framework::OpDesc>,
                  ops::StackGradOpMaker<paddle::imperative::OpBase>,
                  StackInferMetaFunctor);
REGISTER_OPERATOR(stack_grad, ops::StackOpGrad, StackGradInferMetaFunctor);
