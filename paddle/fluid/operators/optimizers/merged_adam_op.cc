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

#include "paddle/fluid/framework/op_registry.h"

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/multiary.h"

namespace paddle {
namespace operators {

class MergedAdamOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto param_dtype =
        framework::OperatorWithKernel::IndicateVarDataType(ctx, "Param");
    return framework::OpKernelType(param_dtype, ctx.GetPlace());
  }

  framework::OpKernelType GetKernelTypeForVar(
      const std::string& var_name,
      const phi::DenseTensor& tensor,
      const framework::OpKernelType& expected_kernel_type) const override {
    if (var_name == "Beta1Pow" || var_name == "Beta2Pow" ||
        var_name == "SkipUpdate") {
      return expected_kernel_type;
    } else {
      return framework::OpKernelType(
          expected_kernel_type.data_type_, tensor.place(), tensor.layout());
    }
  }
};

class MergedAdamOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Param", "(Tensor, default Tensor<float>) Input parameter")
        .AsDuplicable();
    AddInput("Grad", "(Tensor, default Tensor<float>) Input gradient")
        .AsDuplicable();
    AddInput("LearningRate", "(Tensor, default Tensor<float>) Learning rate")
        .AsDuplicable();
    AddInput("Moment1", "(Tensor, default Tensor<float>) Input first moment")
        .AsDuplicable();
    AddInput("Moment2", "(Tensor, default Tensor<float>) Input second moment")
        .AsDuplicable();
    AddInput("Beta1Pow",
             "(Tensor, default Tensor<float>) Input beta1 power accumulator")
        .AsDuplicable();
    AddInput("Beta2Pow",
             "(Tensor, default Tensor<float>) Input beta2 power accumulator")
        .AsDuplicable();
    AddInput("MasterParam", "FP32 master weight for AMP.")
        .AsDispensable()
        .AsDuplicable();

    AddOutput("ParamOut", "(Tensor) Output parameter").AsDuplicable();
    AddOutput("Moment1Out", "(Tensor) Output first moment").AsDuplicable();
    AddOutput("Moment2Out", "(Tensor) Output second moment").AsDuplicable();
    AddOutput("Beta1PowOut", "(Tensor) Output beta1 power accumulator")
        .AsDuplicable();
    AddOutput("Beta2PowOut", "(Tensor) Output beta2 power accumulator")
        .AsDuplicable();
    AddOutput("MasterParamOut",
              "The updated FP32 master weight for AMP. "
              "It shared memory with Input(MasterParam).")
        .AsDispensable()
        .AsDuplicable();

    AddAttr<float>("beta1",
                   "(float, default 0.9) "
                   "Exponential decay rate for the "
                   "first moment estimates.")
        .SetDefault(0.9f);
    AddAttr<float>("beta2",
                   "(float, default 0.999) "
                   "exponential decay rate for the "
                   "second moment estimates.")
        .SetDefault(0.999f);
    AddAttr<float>("epsilon",
                   "(float, default 1.0e-8) "
                   "Constant for numerical stability")
        .SetDefault(1.0e-8f);
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
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

DECLARE_INFER_SHAPE_FUNCTOR(merged_adam,
                            MergedAdamInferMetaFunctor,
                            PD_INFER_META(phi::MergedAdamInferMeta));

REGISTER_OPERATOR(
    merged_adam,
    ops::MergedAdamOp,
    ops::MergedAdamOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>,
    MergedAdamInferMetaFunctor);
