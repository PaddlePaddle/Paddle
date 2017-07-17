/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <paddle/framework/op_registry.h>
#include <paddle/operators/momentum_sgd_op.h>

namespace paddle {
namespace operators {

class MomentumSGDOp : public framework::OperatorWithKernel {
protected:
  void InferShape(
      const std::vector<const framework::Tensor *> &inputs,
      const std::vector<framework::Tensor *> &outputs) const override {}
};

class MomentumSGDOpMaker : public framework::OpProtoAndCheckerMaker {
public:
  MomentumSGDOpMaker(framework::OpProto *proto,
                     framework::OpAttrChecker *op_checker)
      : framework::OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("param", "Parameters to be updated");
    AddInput("grad", "Gradient computed");
    AddInput("moment", "Momentum variable");
    AddInput("lr", "learning rate");
    AddOutput("output_param", "Updated parameters");
    AddOutput("output_grad", "Updated first moment");
    AddOutput("output_moment", "Updated second moment");
    AddAttr<float>("momentum", "")
        .SetDefault(0.9)
        .LargerThan(0.0)
        .LessThan(1.0);
    AddAttr<int>("nesterov", "").SetDefault(0);
    AddComment(R"DOC(

Performs a momentum SGD update for an input gradient and momentum
parameters. Concretely, given inputs (param, grad, m, lr) and attribute
(momentum, nesterov), computes:

    if not nesterov:
        adjusted_gradient = lr * grad + momentum * m
        param = param - adjusted_gradient
        return (adjusted_gradient, adjusted_gradient, param)
    else:
        m_new = momentum * m + lr * grad
        param = param - ((1 + momentum) * m_new - momentum * m),
        return ((1 + momentum) * m_new - momentum * m, m_new, param)

Output is (parameter, grad, momentum).

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OP(momentum_sgd_op,
            paddle::operators::MomentumSGDOp,
            paddle::operators::MomentumSGDOpMaker);
REGISTER_OP_CPU_KERNEL(
    momentum_sgd_op,
    ::paddle::operators::MomentumSGDOpKernel<::paddle::platform::CPUPlace>);
