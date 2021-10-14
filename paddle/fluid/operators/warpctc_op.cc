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

#include <memory>

#ifdef PADDLE_WITH_HIP
#include "paddle/fluid/platform/miopen_helper.h"
#endif
#ifdef PADDLE_WITH_CUDA
#include "paddle/fluid/platform/cudnn_helper.h"
#endif

namespace paddle {
namespace operators {

class WarpCTCOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("Logits"), "Input", "Logits", "WarpCTC");
    OP_INOUT_CHECK(ctx->HasInput("Label"), "Input", "Label", "WarpCTC");
    OP_INOUT_CHECK(ctx->HasOutput("WarpCTCGrad"), "Output", "WarpCTCGrad",
                   "WarpCTC");
    OP_INOUT_CHECK(ctx->HasOutput("Loss"), "Output", "Loss", "WarpCTC");

    auto logits_dims = ctx->GetInputDim("Logits");
    int blank = ctx->Attrs().Get<int>("blank");
    int sequence_width = 0;

    if (ctx->HasInput("LogitsLength")) {
      sequence_width = logits_dims[2];
    } else {
      sequence_width =
          static_cast<int>(framework::product(logits_dims) / logits_dims[0]);
    }

    PADDLE_ENFORCE_GE(
        blank, 0, platform::errors::InvalidArgument(
                      "The value of Attr(blank) should be in interval [0, %d), "
                      "but received %d",
                      blank));
    PADDLE_ENFORCE_LT(
        blank, sequence_width,
        platform::errors::InvalidArgument(
            "The value of Attr(blank) should be in interval [0, %d), "
            "but received %d",
            blank));

    // TODO(liuyiqun): it is tricky to set the wrong dimension here.
    ctx->SetOutputDim("Loss", {-1, 1});
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    framework::LibraryType library_{framework::LibraryType::kPlain};
    framework::DataLayout layout_ = framework::DataLayout::kAnyLayout;
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "Logits"), ctx.GetPlace(),
        layout_, library_);
  }
};

class WarpCTCOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Logits",
             "(2-D LoDTensor<float>) or (3-D Tensor<float>), the "
             "unscaled probabilities of variable-length sequences."
             "When is a 2-D Tensor with LoD information, "
             "it's shape is [Lp, num_classes + 1], "
             "where Lp is the sum of all input sequences' length "
             "and num_classes is the true number of classes "
             "(not including the blank label)."
             "When it is 3-D Tensor, it's shape is "
             "[max_logit_length, batch_size, num_classes + 1], "
             "where max_logit_length is the length of the longest "
             "logit sequence.");
    AddInput("Label",
             "(2-D LoDTensor<int>) or (2-D Tensor<int>), the "
             "ground truth of variable-length sequence. "
             "When it is a 2-D Tensor with LoD information, "
             "it is of the shape [Lg, 1], where Lg is th sum of "
             "all labels' length."
             "When it is a 2-D Tensor<int>, it's shape is also [Lg, 1].");
    AddInput("LogitsLength",
             "1-D Tensor<int64_t>. "
             "Input sequence length for Logits when Logits is a 3-D tensor.")
        .AsDispensable();
    AddInput("LabelLength",
             "1-D Tensor<int64_t>. "
             "Target sequence length for Label when Label is a 2-D tensor.")
        .AsDispensable();
    AddOutput("WarpCTCGrad",
              "(Tensor), a temporary "
              "output Tensor to store the gradients of warp-ctc, which is "
              "computed with loss together in one call. It is a 3-D Tensor of "
              "the shape [max_sequence_length, batch_size, num_classes + 1].")
        .AsIntermediate();
    AddOutput("Loss",
              "(Tensor), the Connectionist "
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
    AddAttr<bool>(
        "norm_by_batchsize",
        "(bool, default: false), normalize the loss by the batch size."
        "If True, supersedes norm_by_times")
        .SetDefault(false);
    AddAttr<bool>(
        "norm_by_total_logits_len",
        "(bool, default: false), normalize the loss by the total number of "
        "frames"
        "in the batch. If True, supersedes norm_by_batchsize and norm_by_times")
        .SetDefault(false);
    AddComment(R"DOC(
An operator integrating the open-source
[warp-ctc](https://github.com/baidu-research/warp-ctc) library, which is used in
[Deep Speech 2: End-toEnd Speech Recognition in English and Mandarin](
https://arxiv.org/pdf/1512.02595v1.pdf),
to compute Connectionist Temporal Classification (CTC) loss.
It can be aliased as softmax with ctc, since a native softmax activation is
interated to the warp-ctc library, to to normalize values for each row of the
input tensor.

More detail of CTC loss can be found by referring to
[Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with
Recurrent Neural Networks](
http://machinelearning.wustl.edu/mlpapers/paper_files/icml2006_GravesFGS06.pdf).
)DOC");
  }
};

template <typename T>
class WarpCTCGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("warpctc_grad");

    op->SetInput("WarpCTCGrad", this->Output("WarpCTCGrad"));
    op->SetInput("Logits", this->Input("Logits"));
    op->SetInput(framework::GradVarName("Loss"), this->OutputGrad("Loss"));

    op->SetInput("LogitsLength", this->Input("LogitsLength"));

    op->SetOutput(framework::GradVarName("Logits"), this->InputGrad("Logits"));

    op->SetAttrMap(this->Attrs());
  }
};

class WarpCTCGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("WarpCTCGrad"), "Input", "WarpCTCGrad",
                   "WarpCTCGrad");
    OP_INOUT_CHECK(ctx->HasOutput(framework::GradVarName("Logits")), "Output",
                   framework::GradVarName("Logits"), "WarpCTCGrad");
    ctx->SetOutputDim(framework::GradVarName("Logits"),
                      ctx->GetInputDim("Logits"));
    ctx->ShareLoD("Logits", /*->*/ framework::GradVarName("Logits"));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Loss")),
                                   ctx.GetPlace());
  }
};

DECLARE_NO_NEED_BUFFER_VARS_INFERER(WarpCTCGradOpNoNeedBufferVarInferer,
                                    "Logits");

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(warpctc, ops::WarpCTCOp, ops::WarpCTCOpMaker,
                  ops::WarpCTCGradOpMaker<paddle::framework::OpDesc>,
                  ops::WarpCTCGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(warpctc_grad, ops::WarpCTCGradOp,
                  ops::WarpCTCGradOpNoNeedBufferVarInferer);
REGISTER_OP_CPU_KERNEL(
    warpctc, ops::WarpCTCKernel<paddle::platform::CPUDeviceContext, float>,
    ops::WarpCTCKernel<paddle::platform::CPUDeviceContext, double>);
REGISTER_OP_CPU_KERNEL(
    warpctc_grad,
    ops::WarpCTCGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::WarpCTCGradKernel<paddle::platform::CPUDeviceContext, double>);

REGISTER_OP_VERSION(warpctc)
    .AddCheckpoint(
        R"ROC(
              Upgrade warpctc add a new attribute [norm_by_batchsize] and [norm_by_total_logits_len])ROC",
        paddle::framework::compatible::OpVersionDesc()
            .NewAttr(
                "norm_by_batchsize",
                "(bool, default: false), normalize the loss by the batch size."
                "If True, supersedes norm_by_times",
                false)
            .NewAttr("norm_by_total_logits_len",
                     "(bool, default: false), normalize the loss by the total "
                     "number of "
                     "frames"
                     "in the batch. If True, supersedes norm_by_batchsize and "
                     "norm_by_times",
                     false));