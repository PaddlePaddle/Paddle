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

#include <memory>

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/device/gpu/gpu_dnn.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/multiary.h"

namespace paddle {
namespace operators {

class WarpCTCOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

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
DECLARE_INFER_SHAPE_FUNCTOR(warpctc, WarpctcInferShapeFunctor,
                            PD_INFER_META(phi::WarpctcInferMeta));
REGISTER_OPERATOR(warpctc, ops::WarpCTCOp, ops::WarpCTCOpMaker,
                  ops::WarpCTCGradOpMaker<paddle::framework::OpDesc>,
                  ops::WarpCTCGradOpMaker<paddle::imperative::OpBase>,
                  WarpctcInferShapeFunctor);
REGISTER_OPERATOR(warpctc_grad, ops::WarpCTCGradOp,
                  ops::WarpCTCGradOpNoNeedBufferVarInferer);
