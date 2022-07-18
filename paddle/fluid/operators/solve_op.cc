/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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
#include <unordered_map>
#include <vector>

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/phi/core/ddim.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/backward.h"
#include "paddle/phi/infermeta/binary.h"

namespace paddle {
namespace operators {

using framework::OpKernelType;
using framework::Tensor;

class SolveOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const {
    framework::LibraryType library = framework::LibraryType::kPlain;
    framework::DataLayout layout = framework::DataLayout::kAnyLayout;
    int customized_type_value =
        framework::OpKernelType::kDefaultCustomizedTypeValue;
    auto input_data_type = OperatorWithKernel::IndicateVarDataType(ctx, "X");
    return framework::OpKernelType(input_data_type,
                                   ctx.GetPlace(),
                                   layout,
                                   library,
                                   customized_type_value);
  }
};

class SolveOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor), The first input tensor of solve op.");
    AddInput("Y", "(Tensor), The second input tensor of solve op.");
    AddOutput("Out", "(Tensor), The output tensor of solve op.");
    AddComment(R"DOC(
          Solve Operator.
          This operator is used to computes the solution of a square system of 
          linear equations with a unique solution for input $X$ and $Y$.

          The equation is:
          $$Out = X^-1 * Y$$
)DOC");
  }
};

class SolveOpInferVarType : public framework::PassInDtypeAndVarTypeToOutput {
 protected:
  std::unordered_map<std::string, std::string>& GetInputOutputWithSameType()
      const override {
    static std::unordered_map<std::string, std::string> m{{"X", /*->*/ "Out"}};
    return m;
  }
};

class SolveGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
};

template <typename T>
class SolveOpGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> retv) const override {
    retv->SetType("solve_grad");
    retv->SetInput("X", this->Input("X"));
    retv->SetInput("Y", this->Input("Y"));
    // reuse the linalg.solve forward output
    retv->SetInput("Out", this->Output("Out"));
    retv->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    retv->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    retv->SetOutput(framework::GradVarName("Y"), this->InputGrad("Y"));
    retv->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

DECLARE_INFER_SHAPE_FUNCTOR(solve,
                            SolveInferShapeFunctor,
                            PD_INFER_META(phi::SolveInferMeta));

REGISTER_OPERATOR(solve,
                  ops::SolveOp,
                  ops::SolveOpMaker,
                  ops::SolveOpInferVarType,
                  ops::SolveOpGradMaker<paddle::framework::OpDesc>,
                  ops::SolveOpGradMaker<paddle::imperative::OpBase>,
                  SolveInferShapeFunctor);

DECLARE_INFER_SHAPE_FUNCTOR(solve_grad,
                            SolveGradInferShapeFunctor,
                            PD_INFER_META(phi::GeneralBinaryGradInferMeta));

REGISTER_OPERATOR(solve_grad, ops::SolveGradOp, SolveGradInferShapeFunctor);
