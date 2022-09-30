/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/unary.h"

namespace paddle {
namespace operators {

using framework::OpKernelType;

class FlipOp : public framework::OperatorWithKernel {
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

class FlipOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor), The input tensor of flip op.");
    AddOutput("Out", "(Tensor), The output tensor of flip op.");
    AddAttr<std::vector<int>>("axis", "The axes to flip on.");
    AddComment(R"DOC(
          Flip Operator.
          Reverse the order of a n-D tensor along given axis in axes.
      )DOC");
  }
};

class FlipOpInferVarType : public framework::PassInDtypeAndVarTypeToOutput {
 protected:
  std::unordered_map<std::string, std::string>& GetInputOutputWithSameType()
      const override {
    static std::unordered_map<std::string, std::string> m{{"X", /*->*/ "Out"}};
    return m;
  }
};

template <typename T>
class FlipOpGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> retv) const override {
    retv->SetType("flip");
    retv->SetInput("X", this->OutputGrad("Out"));
    retv->SetOutput("Out", this->InputGrad("X"));
    retv->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
DECLARE_INFER_SHAPE_FUNCTOR(flip,
                            FlipInferShapeFunctor,
                            PD_INFER_META(phi::FlipInferMeta));
REGISTER_OPERATOR(flip,
                  ops::FlipOp,
                  ops::FlipOpMaker,
                  ops::FlipOpInferVarType,
                  ops::FlipOpGradMaker<paddle::framework::OpDesc>,
                  ops::FlipOpGradMaker<paddle::imperative::OpBase>,
                  FlipInferShapeFunctor);

/* ==========================  register checkpoint ===========================*/
REGISTER_OP_VERSION(flip).AddCheckpoint(
    R"ROC(Upgrade flip, add new attr [axis] and delete attr [dims].)ROC",
    paddle::framework::compatible::OpVersionDesc()
        .NewAttr("axis",
                 "The added attr 'axis' doesn't set default value.",
                 paddle::none)
        .DeleteAttr("dims", "The attr 'dims' is deleted."));
