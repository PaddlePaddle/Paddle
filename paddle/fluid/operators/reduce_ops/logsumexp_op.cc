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

#include "paddle/fluid/operators/reduce_ops/logsumexp_op.h"
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace paddle {
namespace operators {

class LogsumexpOpMaker : public ops::ReduceOpMaker {
 protected:
  virtual std::string GetName() const { return "logsumexp"; }
  virtual std::string GetOpType() const { return "Reduce logsumexp"; }
};

template <typename T>
class LogsumexpGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("logsumexp_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput("Out", this->Output("Out"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetAttrMap(this->Attrs());
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OPERATOR(logsumexp, ops::ReduceOp, ops::LogsumexpOpMaker,
                  ops::LogsumexpGradOpMaker<paddle::framework::OpDesc>,
                  ops::LogsumexpGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(logsumexp_grad, ops::ReduceGradOp);

REGISTER_OP_CPU_KERNEL(logsumexp,
                       ops::ReduceKernel<paddle::platform::CPUDeviceContext,
                                         float, ops::LogsumexpFunctor>,
                       ops::ReduceKernel<paddle::platform::CPUDeviceContext,
                                         double, ops::LogsumexpFunctor>);
REGISTER_OP_CPU_KERNEL(
    logsumexp_grad, ops::ReduceGradKernel<paddle::platform::CPUDeviceContext,
                                          float, ops::LogsumexpGradFunctor>,
    ops::ReduceGradKernel<paddle::platform::CPUDeviceContext, double,
                          ops::LogsumexpGradFunctor>);
