/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/optimizers/lamb_op.h"
#include "paddle/fluid/operators/optimizers/adam_op.h"

namespace paddle {
namespace operators {

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

    AddOutput("ParamOut", "(Tensor) Output parameter.");
    AddOutput("Moment1Out", "(Tensor) Output first moment.");
    AddOutput("Moment2Out", "(Tensor) Output second moment.");
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

    AddComment(R"DOC(
LAMB (Layer-wise Adaptive Moments optimizer for Batching training) Optimizer.

LAMB Optimizer is designed to scale up the batch size of training without losing 
accuracy, which supports adaptive element-wise updating and accurate layer-wise 
correction. For more information, please refer to https://arxiv.org/abs/1904.00962.

The updating of parameters follows:

$$
m_t &= \beta_1 m_{t - 1}+ (1 - \beta_1)g_t \\

v_t &= \beta_2 v_{t - 1}  + (1 - \beta_2)g_t^2 \\

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
REGISTER_OP_WITHOUT_GRADIENT(lamb, ops::AdamOp, ops::LambOpMaker);
REGISTER_OP_CPU_KERNEL(
    lamb, ops::LambOpKernel<paddle::platform::CPUDeviceContext, float>,
    ops::LambOpKernel<paddle::platform::CPUDeviceContext, double>);
