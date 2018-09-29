/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/lars_momentum_op.h"
#include "paddle/fluid/operators/momentum_op.h"

namespace paddle {
namespace operators {

class LarsMomentumOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Param",
             "(Tensor, default Tensor<float>) "
             "Input parameter that has to be updated");
    AddInput("Grad",
             "(Tensor, default Tensor<float>) "
             "Input gradient of the parameter");
    AddInput("Velocity",
             "(Tensor, default Tensor<float>) "
             "Input velocity (corresponding to the parameter) "
             "that has to be updated");
    AddInput("LearningRate",
             "(Tensor, default Tensor<float>) "
             "Input learning rate");

    AddOutput("ParamOut",
              "(Tensor) This output is updated parameter. "
              "It shared memory with Input(Param).");
    AddOutput("VelocityOut",
              "(Tensor) This output is updated velocity. "
              "It shared memory with Input(Velocity).");

    AddAttr<float>("mu", "(float) Momentum coefficient");
    AddAttr<float>("lars_coeff", "(float, default 0.001) LARS coefficient.")
        .SetDefault(0.001);
    AddAttr<float>("lars_weight_decay",
                   "(float, default 0.0005) LARS weight decay")
        .SetDefault(0.0005);

    AddComment(R"DOC(
Lars Momentum Optimizer.

This optimizer use LARS (https://arxiv.org/abs/1708.03888) to optimize each
weight using a local learning rate:

$$
learning\_rate *= lars_coeff * sqrt(sumsq(param)) 
    / (sqrt(sumsq(gradient))+ lars\_weight\_decay * sqrt(sumsq(param))) \\
velocity = mu * velocity + 
    (gradient + lars\_weight\_decay * param) \\
param = param - learning\_rate * velocity. \\
$$

Note that we use lars_weight_decay here to decay weights, you may need not to
use L2 regularizers in case of using LARS.

)DOC");
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(lars_momentum, ops::MomentumOp,
                             ops::LarsMomentumOpMaker);
REGISTER_OP_CPU_KERNEL(lars_momentum, ops::LarsMomentumOpKernel<float>,
                       ops::LarsMomentumOpKernel<double>);
