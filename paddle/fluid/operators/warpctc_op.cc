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

#include "paddle/fluid/operators/warpctc_op.h"

namespace paddle {
namespace operators {

class WarpCTCOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Logits"),
                   "Input(Logits) of WarpCTCOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Label"),
                   "Input(Label) of WarpCTCOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("WarpCTCGrad"),
                   "Output(WarpCTCGrad) of WarpCTCOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Loss"),
                   "Output(Loss) of WarpCTCOp should not be null.");

    auto logits_dims = ctx->GetInputDim("Logits");
    int sequence_width =
        static_cast<int>(framework::product(logits_dims) / logits_dims[0]);
    int blank = ctx->Attrs().Get<int>("blank");
    PADDLE_ENFORCE((blank >= 0) && (blank < sequence_width),
                   "The value of Attr(blank) should be in interval [0, %d).",
                   sequence_width);
    // TODO(liuyiqun): it is tricky to set the wrong dimension here.
    ctx->SetOutputDim("Loss", {logits_dims[0], 1});
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        framework::ToDataType(ctx.Input<Tensor>("Logits")->type()),
        ctx.device_context());
  }
};

class WarpCTCOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Logits",
             "(LodTensor, default: LoDTensor<float>), the unscaled "
             "probabilities of variable-length sequences, which is a 2-D "
             "Tensor with LoD information. It's shape is "
             "[Lp, num_classes + 1], where Lp is the sum of all input "
             "sequences' length and num_classes is the true number of classes "
             "(not including the blank label).");
    AddInput("Label",
             "(LodTensor, default: LoDTensor<int>), the ground truth "
             "of variable-length sequence, which is a 2-D Tensor with LoD "
             "information. It is of the shape [Lg, 1], where Lg is th sum of "
             "all labels' length.");
    AddOutput("WarpCTCGrad",
              "(Tensor, default: Tensor<float>), a temporary "
              "output Tensor to store the gradients of warp-ctc, which is "
              "computed with loss together in one call. It is a 3-D Tensor of "
              "the shape [max_sequence_length, batch_size, num_classes + 1].")
        .AsIntermediate();
    AddOutput("Loss",
              "(Tensor, default: Tensor<float>), the Connectionist "
              "Temporal Classification (CTC) loss, which is a 2-D Tensor of "
              "the shape [batch_size, 1]");
    AddAttr<int>("blank",
                 "(int, default: 0), the blank label of Connectionist "
                 "Temporal Classification (CTC) loss, which is in the "
                 "half-opened interval [0, num_classes + 1).")
        .SetDefault(0);
    AddAttr<bool>("norm_by_times",
                  "(bool, default: false), whether to "
                  "normalize the gradients by the number of time-step, "
                  "which is also the sequence's length.")
        .SetDefault(false);
    AddComment(R"DOC(
An operator integrating the open-source
[warp-ctc](https://github.com/baidu-research/warp-ctc) library, which is used in
[Deep Speech 2: End-toEnd Speech Recognition in English and Mandarin](
https://arxiv.org/pdf/1512.02595v1.pdf),
to compute Connectionist Temporal Classification (CTC) loss.
It can be aliased as softmax with ctc, since a native softmax activation is
interated to the warp-ctc library, to to normlize values for each row of the
input tensor.

More detail of CTC loss can be found by refering to
[Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with
Recurrent Neural Networks](
http://machinelearning.wustl.edu/mlpapers/paper_files/icml2006_GravesFGS06.pdf).
)DOC");
  }
};

class WarpCTCGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("WarpCTCGrad"),
                   "Input(WarpCTCGrad) of WarpCTCGradOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput(framework::GradVarName("Logits")),
                   "Output(Logits@GRAD) of WarpCTCGradOp should not be null.");
    ctx->SetOutputDim(framework::GradVarName("Logits"),
                      ctx->GetInputDim("Logits"));
    ctx->ShareLoD("Logits", /*->*/ framework::GradVarName("Logits"));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        framework::ToDataType(ctx.Input<Tensor>("Logits")->type()),
        ctx.device_context());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(warpctc, ops::WarpCTCOp, ops::WarpCTCOpMaker,
                  paddle::framework::DefaultGradOpDescMaker<true>);
REGISTER_OPERATOR(warpctc_grad, ops::WarpCTCGradOp);
REGISTER_OP_CPU_KERNEL(
    warpctc, ops::WarpCTCKernel<paddle::platform::CPUDeviceContext, float>);
REGISTER_OP_CPU_KERNEL(
    warpctc_grad,
    ops::WarpCTCGradKernel<paddle::platform::CPUDeviceContext, float>);
