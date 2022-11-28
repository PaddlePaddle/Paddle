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

class WarpRNNTOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    framework::LibraryType library_{framework::LibraryType::kPlain};
    phi::DataLayout layout_ = phi::DataLayout::kAnyLayout;
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "Logits"),
        ctx.GetPlace(),
        layout_,
        library_);
  }
};

class WarpRNNTOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Logits",
             "(4-D Tensor<float>), the "
             "unscaled probabilities of variable-length sequences."
             "When it is 4-D Tensor, it's shape is "
             "[B, Tmax, Umax, D].");
    AddInput("Label",
             "(2-D Tensor<int>), the "
             "ground truth of variable-length sequence. "
             "When it is a 2-D Tensor<int>, it's shape is also [B, Umax].");
    AddInput("LogitsLength",
             "1-D Tensor<int64_t>. "
             "Input sequence length for Logits when Logits is a 4-D tensor.");
    AddInput("LabelLength",
             "1-D Tensor<int64_t>. "
             "Target sequence length for Label when Label is a 2-D tensor.");
    AddOutput("WarpRNNTGrad",
              "(Tensor), a temporary "
              "output Tensor to store the gradients of warp-rnnt, which is "
              "computed with loss together in one call. It is a 4-D Tensor of "
              "the shape [B, Tmax, Umax, D].")
        .AsIntermediate();
    AddOutput("Loss",
              "(Tensor), the Connectionist "
              "Temporal Classification (CTC) loss, which is a 2-D Tensor of "
              "the shape [B, 1]");
    AddAttr<int>("blank",
                 "(int, default: 0), the blank label of Connectionist "
                 "Temporal Classification (CTC) loss, which is in the "
                 "half-opened interval [0, num_classes + 1).")
        .SetDefault(0);
    AddAttr<float>("fastemit_lambda",
                   "(float, default: 0.0),"
                   "Regularization parameter for FastEmit"
                   "(https://arxiv.org/pdf/2010.11148.pdf)")
        .SetDefault(0.0);
    AddAttr<int>("num_threads",
                 "(int, default 1), thread num for cpu intra kernel.")
        .SetDefault(1);
    AddComment(R"DOC(
An operator integrating the open-source
[warp-rnnt](https://github.com/b-flo/warp-transducer) library, which is
A fast parallel implementation of RNN Transducer (Graves 2013 joint network),
on both CPU and GPU.

More detail of RNN-T loss can be found by referring to
[Sequence Transduction with Recurrent Neural Networks](https://arxiv.org/abs/1211.3711).
)DOC");
  }
};

template <typename T>
class WarpRNNTGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("warprnnt_grad");

    op->SetInput("WarpRNNTGrad", this->Output("WarpRNNTGrad"));
    op->SetInput("Logits", this->Input("Logits"));
    op->SetInput(framework::GradVarName("Loss"), this->OutputGrad("Loss"));

    op->SetInput("LogitsLength", this->Input("LogitsLength"));

    op->SetOutput(framework::GradVarName("Logits"), this->InputGrad("Logits"));

    op->SetAttrMap(this->Attrs());
  }
};

class WarpRNNTGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(
        ctx->HasInput("WarpRNNTGrad"), "Input", "WarpRNNTGrad", "WarpRNNTGrad");
    OP_INOUT_CHECK(ctx->HasOutput(framework::GradVarName("Logits")),
                   "Output",
                   framework::GradVarName("Logits"),
                   "WarpRNNTGrad");
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

DECLARE_NO_NEED_BUFFER_VARS_INFERER(WarpRNNTGradOpNoNeedBufferVarInferer,
                                    "Logits");

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
DECLARE_INFER_SHAPE_FUNCTOR(warprnnt,
                            WarprnntInferShapeFunctor,
                            PD_INFER_META(phi::WarprnntInferMeta));
REGISTER_OPERATOR(warprnnt,
                  ops::WarpRNNTOp,
                  ops::WarpRNNTOpMaker,
                  ops::WarpRNNTGradOpMaker<paddle::framework::OpDesc>,
                  ops::WarpRNNTGradOpMaker<paddle::imperative::OpBase>,
                  WarprnntInferShapeFunctor);
REGISTER_OPERATOR(warprnnt_grad,
                  ops::WarpRNNTGradOp,
                  ops::WarpRNNTGradOpNoNeedBufferVarInferer);
