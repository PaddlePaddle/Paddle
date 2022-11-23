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

#include <string>
#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/infermeta/multiary.h"
#include "paddle/phi/kernels/lamb_kernel.h"

namespace paddle {
namespace operators {

class LambOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const {
    auto input_data_type =
        OperatorWithKernel::IndicateVarDataType(ctx, "Param");
    return framework::OpKernelType(input_data_type, ctx.GetPlace());
  }
  framework::OpKernelType GetKernelTypeForVar(
      const std::string &var_name,
      const phi::DenseTensor &tensor,
      const framework::OpKernelType &expected_kernel_type) const {
    if (var_name == "Beta1Pow" || var_name == "Beta2Pow") {
      return expected_kernel_type;
    } else {
      return framework::OpKernelType(
          expected_kernel_type.data_type_, tensor.place(), tensor.layout());
    }
  }
};

class LambOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Param",
             "(LoDTensor, default LoDTensor<float>) "
             "Input parameter that has to be updated.");
    AddInput("Grad",
             "(LoDTensor, default LoDTensor<float>) "
             "Input gradient of the parameter.");
    AddInput("LearningRate", "(Tensor) Learning rate.");
    AddInput("Moment1", "(Tensor) Input first moment.");
    AddInput("Moment2", "(Tensor) Input second moment.");
    AddInput("Beta1Pow", "(Tensor) Input beta1 power accumulator.");
    AddInput("Beta2Pow", "(Tensor) Input beta2 power accumulator.");
    AddInput("MasterParam",
             "(LoDTensor, default LoDTensor<float>) "
             "Input master parameter that has to be updated.")
        .AsDispensable();
    AddInput(
        "SkipUpdate",
        "(Tensor) Input tensor to determine whether to update the parameter.")
        .AsDispensable();

    AddOutput("ParamOut", "(Tensor) Output parameter.");
    AddOutput("Moment1Out", "(Tensor) Output first moment.");
    AddOutput("Moment2Out", "(Tensor) Output second moment.");
    AddOutput("Beta1PowOut", "(Tensor) Output beta1 power accumulator")
        .AsDispensable();
    AddOutput("Beta2PowOut", "(Tensor) Output beta2 power accumulator")
        .AsDispensable();
    AddOutput("MasterParamOut", "(Tensor) Output master parameter.")
        .AsDispensable();
    AddAttr<float>("weight_decay", "(float) Weight decay rate.");
    AddAttr<float>("beta1",
                   "(float, default 0.9) The exponential decay rate for the "
                   "1st moment estimates.")
        .SetDefault(0.9);
    AddAttr<float>("beta2",
                   "(float, default 0.999) The exponential decay rate for the "
                   "2nd moment estimates.")
        .SetDefault(0.999);
    AddAttr<float>("epsilon",
                   "(float, default 1.0e-6) "
                   "Constant for numerical stability.")
        .SetDefault(1.0e-6f);
    AddAttr<bool>(
        "multi_precision",
        "(bool, default false) Whether to enable multi-precision mode.")
        .SetDefault(false);

    AddComment(R"DOC(
LAMB (Layer-wise Adaptive Moments optimizer for Batching training) Optimizer.

LAMB Optimizer is designed to scale up the batch size of training without losing
accuracy, which supports adaptive element-wise updating and accurate layer-wise
correction. For more information, please refer to https://arxiv.org/abs/1904.00962.

The updating of parameters follows:

$$
m_t &= \beta_1 m_{t - 1}+ (1 - \beta_1)g_t \\

v_t &= \beta_2 v_{t - 1}  + (1 - \beta_2)g_t^2 \\

m_t &= \frac{m_t}{\beta_1^t} \\

v_t &= \frac{v_t}{\beta_2^t} \\

r_t &= \frac{m_t}{\sqrt{v_t}+\epsilon} \\

w_t &= w_{t-1} -\eta_t \frac{\left \| w_{t-1}\right \|}{\left \| r_t + \lambda w_{t-1}\right \|} (r_t + \lambda w_{t-1})
$$

where $m$ is the 1st moment, and $v$ the 2nd moment, $\eta$ the
learning rate, $\lambda$ the weight decay rate.
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
DECLARE_INFER_SHAPE_FUNCTOR(lamb,
                            LambInferMetaFunctor,
                            PD_INFER_META(phi::LambInferMeta));
REGISTER_OPERATOR(
    lamb,
    ops::LambOp,
    ops::LambOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>,
    LambInferMetaFunctor);

/* ==========================  register checkpoint ===========================*/
REGISTER_OP_VERSION(lamb).AddCheckpoint(
    R"ROC(Upgrade lamb, add two new outputs [Beta1PowOut] and [Beta2PowOut].)ROC",
    paddle::framework::compatible::OpVersionDesc()
        .NewInput("Beta1PowOut",
                  "The Output beta1 power accumulator. 'Beta1PowOut' is "
                  "dispensable.")
        .NewInput("Beta2PowOut",
                  "The Output beta2 power accumulator. 'Beta2PowOut' is "
                  "dispensable."));
