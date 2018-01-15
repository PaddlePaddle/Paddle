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

#include "paddle/operators/ctc_greedy_decode_op.h"

namespace paddle {
namespace operators {

class CTCGreedyDecodeOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Input"),
                   "Input of CTCGreedyDecodeOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Output"),
                   "Output of CTCGreedyDecodeOp should not be null.");

    auto input_dims = ctx->GetInputDim("Input");

    int sequence_width =
        static_cast<int>(framework::product(input_dims) / input_dims[0]);
    int blank = ctx->Attrs().Get<int>("blank");
    PADDLE_ENFORCE((blank >= 0) && (blank < sequence_width),
                   "The value of Attr(blank) should be in interval [0, %d).",
                   sequence_width);
    // TODO(wanghaoshuang): it is tricky to set the wrong dimension here.
    ctx->SetOutputDim("Output", {input_dims[0], 1});
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        framework::ToDataType(ctx.Input<Tensor>("Input")->type()),
        ctx.device_context());
  }
};

class CTCGreedyDecodeOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  CTCGreedyDecodeOpMaker(OpProto* proto, OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("Input",
             "(LodTensor, default: LoDTensor<float>), the unscaled "
             "probabilities of variable-length sequences, which is a 2-D "
             "Tensor with LoD information. It's shape is "
             "[Lp, num_classes + 1], where Lp is the sum of all input "
             "sequences' length and num_classes is the true number of classes "
             "(not including the blank label).");
    AddOutput("Output", "(Tensor, default: Tensor<int>), the decode result ");
    AddAttr<int>("blank",
                 "(int, default: 0), the blank label setted in Connectionist "
                 "Temporal Classification (CTC) op, and it is in the "
                 "half-opened interval [0, num_classes + 1).")
        .SetDefault(0);
    AddAttr<bool>("merge_repeated",
                  "(bool, default: true), whether to "
                  "merge repeated elements between two blanks. ")
        .SetDefault(true);
    AddComment(R"DOC(
CTCGreedyDecoder is an implementation of the simple best path decoding
algorithm, selecting at each timestep the most likely class at each timestep.
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(ctc_greedy_decode, ops::CTCGreedyDecodeOp,
                  ops::CTCGreedyDecodeOpMaker,
                  paddle::framework::EmptyGradOpMaker);
REGISTER_OP_CPU_KERNEL(
    ctc_greedy_decode,
    ops::CTCGreedyDecodeKernel<paddle::platform::CPUDeviceContext, float>);
