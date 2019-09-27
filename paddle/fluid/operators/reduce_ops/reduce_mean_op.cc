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

#include "paddle/fluid/operators/reduce_ops/reduce_mean_op.h"
#include <memory>
#include <string>
#include <vector>

namespace paddle {
namespace operators {

// NOTE(dengkaipeng): Input(Out) is unnecessary in reduce_mean_grad
// calcualtion, but will incur a reduce_mean_grad op after
// reduce_mean_grad_grad, delete Input(Out) here.
// This change has no effect on reduce_mean_grad calculations.
class ReduceMeanOpGradDescMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  std::unique_ptr<framework::OpDesc> Apply() const override {
    std::unique_ptr<framework::OpDesc> op(new framework::OpDesc());
    op->SetType("reduce_mean_grad");
    op->SetInput("X", Input("X"));
    op->SetInput(framework::GradVarName("Out"), OutputGrad("Out"));
    op->SetAttrMap(Attrs());
    op->SetOutput(framework::GradVarName("X"), InputGrad("X"));
    return op;
  }
};

class ReduceMeanDoubleGradMaker : public framework::GradOpDescMakerBase {
 public:
  using framework::GradOpDescMakerBase::GradOpDescMakerBase;

  std::vector<std::unique_ptr<framework::OpDesc>> operator()() const override {
    std::vector<std::unique_ptr<framework::OpDesc>> ops;
    auto x_gg = OutputGrad(framework::GradVarName("X"));  // input ddx
    auto out_grads = InputGrad(framework::GradVarName("Out"));
    if (!out_grads.empty()) {
      auto* out_grad_op = new framework::OpDesc();
      out_grad_op->SetType("reduce_mean");
      out_grad_op->SetInput("X", x_gg);
      out_grad_op->SetAttrMap(Attrs());
      out_grad_op->SetOutput("Out", out_grads);
      ops.emplace_back(out_grad_op);
    }

    return ops;
  }
};

DECLARE_NO_NEED_BUFFER_VARS_INFERENCE(ReduceMeanGradNoNeedBufferVarInference,
                                      "X");
}  // namespace operators
}  // namespace paddle

class __reduce_meanMaker__ : public ops::ReduceOpMaker {
 protected:
  virtual std::string GetName() const { return "reduce_mean"; }
  virtual std::string GetOpType() const { return "Reduce reduce_mean"; }
};

REGISTER_OPERATOR(reduce_mean, ops::ReduceOp, __reduce_meanMaker__,
                  ops::ReduceMeanOpGradDescMaker);
REGISTER_OPERATOR(reduce_mean_grad, ops::ReduceGradOp,
                  ops::ReduceMeanDoubleGradMaker,
                  ops::ReduceMeanGradNoNeedBufferVarInference);
REGISTER_OP_CPU_KERNEL(reduce_mean,
                       ops::ReduceKernel<paddle::platform::CPUDeviceContext,
                                         float, ops::MeanFunctor>,
                       ops::ReduceKernel<paddle::platform::CPUDeviceContext,
                                         double, ops::MeanFunctor>,
                       ops::ReduceKernel<paddle::platform::CPUDeviceContext,
                                         int, ops::MeanFunctor>,
                       ops::ReduceKernel<paddle::platform::CPUDeviceContext,
                                         int64_t, ops::MeanFunctor>);

template <typename T>
using CPUReduceMeanGradKernel =
    ops::ReduceGradKernel<paddle::platform::CPUDeviceContext, T,
                          ops::MeanGradFunctor, true>;

REGISTER_OP_CPU_KERNEL(reduce_mean_grad, CPUReduceMeanGradKernel<float>,
                       CPUReduceMeanGradKernel<double>,
                       CPUReduceMeanGradKernel<int>,
                       CPUReduceMeanGradKernel<int64_t>);
