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

#include <algorithm>
#include <string>
#include <vector>

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/unary.h"

namespace paddle {
namespace operators {

class MultinomialOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "A tensor contains probabilities of categories");
    AddOutput("Out", "The output tensor of multinomial op");
    AddAttr<int>("num_samples", "number of the generated samples")
        .SetDefault(1)
        .SupportTensor();
    AddAttr<bool>("replacement", "can a category be sampled more than once")
        .SetDefault(false);
    AddComment(R"DOC(
This OP returns a Tensor filled with the sampled categoris according to Multinomial probabilities.

      Out ~ Multinomial(X)

)DOC");
  }
};

class MultinomialOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto input_data_type =
        framework::OperatorWithKernel::IndicateVarDataType(ctx, "X");
    return framework::OpKernelType(input_data_type, ctx.GetPlace());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
DECLARE_INFER_SHAPE_FUNCTOR(multinomial,
                            MultinomialInferShapeFunctor,
                            PD_INFER_META(phi::MultinomialInferMeta));
REGISTER_OPERATOR(
    multinomial,
    ops::MultinomialOp,
    ops::MultinomialOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>,
    MultinomialInferShapeFunctor);
