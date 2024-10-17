// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/multiary.h"

namespace paddle {
namespace operators {

using Tensor = phi::DenseTensor;

class FusedAdamOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto param_dtype =
        framework::OperatorWithKernel::IndicateVarDataType(ctx, "Params");
    return phi::KernelKey(param_dtype, ctx.GetPlace());
  }

  phi::KernelKey GetKernelTypeForVar(
      const std::string &var_name,
      const phi::DenseTensor &tensor,
      const phi::KernelKey &expected_kernel_type) const override {
    if (var_name == "Beta1Pows" || var_name == "Beta2Pows" ||
        var_name == "SkipUpdate") {
      return phi::KernelKey(phi::Backend::ALL_BACKEND,
                            expected_kernel_type.layout(),
                            expected_kernel_type.dtype());
    } else {
      return phi::KernelKey(
          tensor.place(), tensor.layout(), expected_kernel_type.dtype());
    }
  }
};

class FusedAdamOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Params", "(Tensor) Input parameters").AsDuplicable();
    AddInput("Grads", "(Tensor) Input gradients").AsDuplicable();
    AddInput("LearningRate", "(Tensor, default Tensor<float>) Learning rate");
    AddInput("Moments1", "(Tensor) Input first moments").AsDuplicable();
    AddInput("Moments2", "(Tensor) Input second moments").AsDuplicable();
    AddInput("Moments2Max", "(Tensor) Input second moments max for amsgrad")
        .AsDispensable()
        .AsDuplicable();
    AddInput("Beta1Pows",
             "(Tensor, default Tensor<float>) Input beta1 power accumulator")
        .AsDuplicable();
    AddInput("Beta2Pows",
             "(Tensor, default Tensor<float>) Input beta2 power accumulator")
        .AsDuplicable();
    AddInput("MasterParams", "FP32 master weight for AMP.")
        .AsDispensable()
        .AsDuplicable();
    AddInput("SkipUpdate", "(Tensor<bool>, optional), Skip the update or not.")
        .AsDispensable();

    AddOutput("ParamsOut", "(Tensor) Output parameters").AsDuplicable();
    AddOutput("Moments1Out", "(Tensor) Output first moments").AsDuplicable();
    AddOutput("Moments2Out", "(Tensor) Output second moments").AsDuplicable();
    AddOutput("Moments2MaxOut",
              "(Tensor) Output second moments max for amsgrad")
        .AsDispensable()
        .AsDuplicable();
    AddOutput("Beta1PowsOut", "(Tensor) Output beta1 power accumulator")
        .AsDuplicable();
    AddOutput("Beta2PowsOut", "(Tensor) Output beta2 power accumulator")
        .AsDuplicable();
    AddOutput("MasterParamsOut",
              "The updated FP32 master weight for AMP. "
              "It shared memory with Input(MasterParams).")
        .AsDispensable()
        .AsDuplicable();

    AddAttr<float>("beta1",
                   "(float, default 0.9) "
                   "Exponential decay rate for the "
                   "first moment estimates.")
        .SetDefault(0.9f)
        .SupportTensor();
    AddAttr<float>("beta2",
                   "(float, default 0.999) "
                   "exponential decay rate for the "
                   "second moment estimates.")
        .SetDefault(0.999f)
        .SupportTensor();
    AddAttr<float>("epsilon",
                   "(float, default 1.0e-8) "
                   "Constant for numerical stability")
        .SetDefault(1.0e-8f)
        .SupportTensor();

    AddAttr<int>("chunk_size", "ChunkSize for blocks computing");

    AddAttr<float>("weight_decay",
                   "(float, default 0) "
                   "weight decay (L2 penalty)")
        .SetDefault(0);
    AddAttr<bool>("use_adamw",
                  "(bool, default False) "
                  "Whether to use AdamW"
                  "True for decoupled weight decay")
        .SetDefault(false);
    AddAttr<bool>("multi_precision",
                  "(bool, default false) "
                  "Whether to use multi-precision during weight updating.")
        .SetDefault(false);
    // TODO(zhiqiu): We could set Beta1PowOut and Beta2PowOut
    // as dispensable since they are not used when use_global_beta_pow is true.
    AddAttr<bool>("use_global_beta_pow",
                  "(bool, default false) "
                  "Whether to use global beta_pow for whole model instead of "
                  "creating beta_pow for each parameter.")
        .SetDefault(false);
    AddAttr<bool>("amsgrad",
                  "(bool, default false) "
                  "Whether to use the AMSGrad of this algorithm.")
        .SetDefault(false);

    AddComment(R"DOC(
Adam Optimizer.

This implements the Adam optimizer from Section 2 of the Adam
paper : https://arxiv.org/abs/1412.6980.
Adam is a first-order gradient-based optimization method based on
adaptive estimates of lower-order moments.

Adam updates:

$$
moment\_1\_out = \beta_1 * moment\_1 + (1 - \beta_1) * grad \\
moment\_2_\out = \beta_2 * moment\_2 + (1 - \beta_2) * grad * grad \\
learning\_rate = learning\_rate *
                  \frac{\sqrt{1 - \beta_{2\_pow}}}{1 - \beta_{1\_pow}} \\
param\_out = param - learning\_rate * \frac{moment\_1}{\sqrt{moment\_2} + \epsilon}
$$

AdamW updates:

$$
moment\_1\_out = \beta_1 * moment\_1 + (1 - \beta_1) * grad \\
moment\_2_\out = \beta_2 * moment\_2 + (1 - \beta_2) * grad * grad \\
learning\_rate = learning\_rate *
                  \frac{\sqrt{1 - \beta_{2\_pow}}}{1 - \beta_{1\_pow}} \\
param\_out & = param - learning\_rate * (\frac{moment\_1}{\sqrt{moment\_2} + \epsilon} + \lambda * param)
$$

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
DECLARE_INFER_SHAPE_FUNCTOR(fused_adam,
                            FusedAdamInferShapeFunctor,
                            PD_INFER_META(phi::FusedAdamInferMeta));
REGISTER_OPERATOR(
    fused_adam,
    ops::FusedAdamOp,
    ops::FusedAdamOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>,
    FusedAdamInferShapeFunctor);
