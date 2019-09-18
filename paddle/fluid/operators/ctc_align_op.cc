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

#include "paddle/fluid/operators/ctc_align_op.h"

namespace paddle {
namespace operators {

class CTCAlignOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Input"),
                   "Input of CTCAlignOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Output"),
                   "Output of CTCAlignOp should not be null.");

    auto input_dims = ctx->GetInputDim("Input");

    // TODO(wanghaoshuang): it is tricky to set the wrong dimension here.
    ctx->SetOutputDim("Output", input_dims);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(ctx.Input<Tensor>("Input")->type(),
                                   ctx.device_context());
  }
};

class CTCAlignOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Input",
             "(LodTensor, default: LoDTensor<int>), Its shape is "
             "[Lp, 1], where Lp is the sum of all input sequences' length.");
    AddOutput("Output", "(Tensor, default: Tensor<int>), The align result.");
    AddAttr<int>("blank",
                 "(int, default: 0), the blank label setted in Connectionist "
                 "Temporal Classification (CTC) op.")
        .SetDefault(0);
    AddAttr<bool>("merge_repeated",
                  "(bool, default: true), whether to "
                  "merge repeated elements between two blanks. ")
        .SetDefault(true);
    AddComment(R"DOC(
CTCAlign op is used to merge repeated elements between two blanks
and then delete all blanks in sequence.

Given:
    Input.data = [0, 1, 2, 2, 0, 4, 0, 4, 5, 0, 6,
                  6, 0, 0, 7, 7, 7, 0]
    Input.dims = {18, 1}
    Input.LoD = [[0, 11, 18]]

And:
    blank = 0
    merge_repeated = True

Then:
    Output.data = [1, 2, 4, 4, 5, 6,
                   6, 7]
    Output.dims = {8, 1}
    Output.LoD = [[0, 6, 8]]

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(ctc_align, ops::CTCAlignOp, ops::CTCAlignOpMaker,
                  paddle::framework::EmptyGradOpMaker);
REGISTER_OP_CPU_KERNEL(
    ctc_align, ops::CTCAlignKernel<paddle::platform::CPUDeviceContext, int>,
    ops::CTCAlignKernel<paddle::platform::CPUDeviceContext, int64_t>);
