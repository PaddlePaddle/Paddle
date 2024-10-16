/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/infermeta/multiary.h"

namespace paddle {
namespace operators {

class LarsMomentumOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto input_data_type =
        OperatorWithKernel::IndicateVarDataType(ctx, "Param");
    return phi::KernelKey(input_data_type, ctx.GetPlace());
  }
};

class LarsMomentumOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Param",
             "(phi::DenseTensor, default phi::DenseTensor<float>) "
             "Input parameter that has to be updated")
        .AsDuplicable();
    AddInput("Grad",
             "(phi::DenseTensor, default phi::DenseTensor<float>) "
             "Input gradient of the parameter")
        .AsDuplicable();
    AddInput("Velocity",
             "(phi::DenseTensor, default phi::DenseTensor<float>) "
             "Input velocity (corresponding to the parameter) "
             "that has to be updated")
        .AsDuplicable();
    AddInput("LearningRate",
             "(phi::DenseTensor, default phi::DenseTensor<float>) "
             "Input learning rate")
        .AsDuplicable();
    AddInput("MasterParam", "FP32 master weight for AMP.")
        .AsDuplicable()
        .AsDispensable();
    AddOutput("ParamOut",
              "(phi::DenseTensor) This output is updated parameter. "
              "It shared memory with Input(Param).")
        .AsDuplicable();
    AddOutput("VelocityOut",
              "(phi::DenseTensor) This output is updated velocity. "
              "It shared memory with Input(Velocity).")
        .AsDuplicable();
    AddOutput("MasterParamOut",
              "The updated FP32 master weight for AMP. "
              "It shared memory with Input(MasterParam).")
        .AsDuplicable()
        .AsDispensable();
    AddAttr<float>("mu", "(float) Momentum coefficient");
    AddAttr<float>("lars_coeff", "(float, default 0.001) LARS coefficient.")
        .SetDefault(0.001f);
    AddAttr<std::vector<float>>(
        "lars_weight_decay",
        "(std::vector<float>, default 0.0005) LARS weight decay params")
        .SetDefault({0.0005f});
    AddAttr<float>("epsilon",
                   "(float, default 0.0) epsilon to avoid Division by Zero.")
        .SetDefault(0.0f);
    AddAttr<bool>("multi_precision",
                  "(bool, default false) "
                  "Whether to use multi-precision during weight updating.")
        .SetDefault(false);
    AddAttr<float>(
        "rescale_grad",
        "(float, default 1.0) Multiply the gradient with `rescale_grad`"
        "before updating. Often choose to be `1.0/batch_size`.")
        .SetDefault(1.0f);

    AddComment(R"DOC(
Lars Momentum Optimizer.

This optimizer use LARS (https://arxiv.org/abs/1708.03888) to optimize each
weight using a local learning rate:

$$
local\_lr = \eta  *
    \frac{\left \| param \right \|}{\left \| grad \right \| + \beta *\left \| param \right \|} \\
velocity = mu * velocity +
    local\_lr * (grad + \beta * param) \\
param = param - velocity. \\
$$

Note that we use lars_weight_decay here to decay weights, you may need not to
use L2 regularizers in case of using LARS.

)DOC");
  }
};

class LarsMomentumOpVarTypeInference : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext* ctx) const override {}
};
}  // namespace operators
}  // namespace paddle

DECLARE_INFER_SHAPE_FUNCTOR(lars_momentum,
                            LarsMomentumInferShapeFunctor,
                            PD_INFER_META(phi::LarsMomentumInferMeta));

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    lars_momentum,
    ops::LarsMomentumOp,
    ops::LarsMomentumOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>,
    ops::LarsMomentumOpVarTypeInference,
    LarsMomentumInferShapeFunctor);
