// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/reduce_ops/reduce_sum_op.h"
#include <memory>
#include <string>

namespace paddle {
namespace operators {

// NOTE: Input(Out) is unnecessary in reduce_sum_grad, and Input(X) needs no
// buffer
class ReduceSumOpGradDescMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  std::unique_ptr<framework::OpDesc> Apply() const override {
    std::unique_ptr<framework::OpDesc> op(new framework::OpDesc());
    op->SetType("reduce_sum_grad");
    op->SetInput("X", Input("X"));
    op->SetInput(framework::GradVarName("Out"), OutputGrad("Out"));
    op->SetAttrMap(Attrs());
    op->SetOutput(framework::GradVarName("X"), InputGrad("X"));
    return op;
  }
};

DECLARE_NO_NEED_BUFFER_VARS_INFERENCE(ReduceSumGradNoNeedBufferVarInference,
                                      "X");

}  // namespace operators
}  // namespace paddle

class ReduceSumOpMaker : public ops::ReduceOpMaker {
 protected:
  virtual std::string GetName() const { return "reduce_sum"; }
  virtual std::string GetOpType() const { return "Reduce reduce_sum"; }
};

REGISTER_OPERATOR(reduce_sum, ops::ReduceOp, ReduceSumOpMaker,
                  ops::ReduceSumOpGradDescMaker);
REGISTER_OPERATOR(reduce_sum_grad, ops::ReduceGradOp,
                  ops::ReduceSumGradNoNeedBufferVarInference);

REGISTER_OP_CPU_KERNEL(
    reduce_sum, ops::ReduceKernel<paddle::platform::CPUDeviceContext, float,
                                  ops::SumFunctor>,
    ops::ReduceKernel<paddle::platform::CPUDeviceContext, double,
                      ops::SumFunctor>,
    ops::ReduceKernel<paddle::platform::CPUDeviceContext, int, ops::SumFunctor>,
    ops::ReduceKernel<paddle::platform::CPUDeviceContext, int64_t,
                      ops::SumFunctor>);

template <typename T>
using CPUReduceSumGradKernel =
    ops::ReduceSumGradKernel<paddle::platform::CPUDeviceContext, T,
                             ops::SumGradFunctor, true>;

REGISTER_OP_CPU_KERNEL(reduce_sum_grad, CPUReduceSumGradKernel<float>,
                       CPUReduceSumGradKernel<double>,
                       CPUReduceSumGradKernel<int>,
                       CPUReduceSumGradKernel<int64_t>);
