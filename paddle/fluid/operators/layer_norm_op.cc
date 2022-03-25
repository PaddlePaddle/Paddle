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
#include <string>
#include "paddle/fluid/framework/op_registry.h"

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/ternary.h"

#ifdef PADDLE_WITH_MKLDNN
#include "paddle/fluid/platform/mkldnn_helper.h"
#endif

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;
using DataLayout = framework::DataLayout;

class LayerNormOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto input_data_type = OperatorWithKernel::IndicateVarDataType(ctx, "X");
    framework::LibraryType library = framework::LibraryType::kPlain;
    framework::DataLayout layout = framework::DataLayout::kAnyLayout;

#ifdef PADDLE_WITH_MKLDNN
    if (library == framework::LibraryType::kPlain &&
        this->CanMKLDNNBeUsed(ctx, input_data_type)) {
      library = framework::LibraryType::kMKLDNN;
      layout = framework::DataLayout::kMKLDNN;
    }
#endif

    return framework::OpKernelType(input_data_type, ctx.GetPlace(), layout,
                                   library);
  }
};

class LayerNormOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The input tensor.");
    AddInput("Scale",
             "(optional) Scale is a 1-dimensional tensor of size "
             "H(`begin_norm_axis` splits the tensor(`X`) to a matrix [N,H])."
             "It is applied to the output.")
        .AsDispensable();
    AddInput("Bias",
             "(optional) Bias is a 1-dimensional tensor of size "
             "H(`begin_norm_axis` splits the tensor(`X`) to a matrix [N,H])."
             "It is applied to the output.")
        .AsDispensable();
    AddOutput("Y", "Result after normalization.");
    AddOutput("Mean", "Mean of the current mini batch.").AsIntermediate();
    AddOutput("Variance", "Variance of the current mini batch.")
        .AsIntermediate();

    AddAttr<float>("epsilon",
                   "Constant for numerical stability [default 1e-5].")
        .SetDefault(1e-5)
        .AddCustomChecker([](const float &epsilon) {
          PADDLE_ENFORCE_EQ(epsilon >= 0.0f && epsilon <= 0.001f, true,
                            platform::errors::InvalidArgument(
                                "'epsilon' in Op(LayerNorm) should be between"
                                "0.0 and 0.001, But received [%s].",
                                epsilon));
        });
    AddAttr<int>("begin_norm_axis",
                 "the axis of `begin_norm_axis ... Rank(X) - 1` will be "
                 "normalized. `begin_norm_axis` splits the tensor(`X`) to a "
                 "matrix [N,H]. [default 1].")
        .SetDefault(1)
        .AddCustomChecker([](const int &begin_norm_axis) {
          PADDLE_ENFORCE_GT(begin_norm_axis, 0,
                            platform::errors::InvalidArgument(
                                "'begin_norm_axis' in Op(LayerNorm) should be"
                                "greater than zero. But received [%d].",
                                begin_norm_axis));
        });
    AddAttr<bool>("use_mkldnn",
                  "(bool, default false) Only used in mkldnn kernel")
        .SetDefault(false)
        .AsExtra();
    AddAttr<std::string>(
        "mkldnn_data_type",
        "(string, default \"float32\"). Data type of mkldnn kernel")
        .SetDefault("float32")
        .InEnum({"float32", "bfloat16"})
        .AsExtra();
    AddAttr<bool>("is_test",
                  "(bool, default false) Set to true for inference only, false "
                  "for training. Some layers may run faster when this is true.")
        .SetDefault(false)
        .AsExtra();

    AddComment(R"DOC(
Assume feature vectors exist on dimensions
:attr:`begin_norm_axis ... rank(input)` and calculate the moment statistics
along these dimensions for each feature vector :math:`a` with size
:math:`H`, then normalize each feature vector using the corresponding
statistics. After that, apply learnable gain and bias on the normalized
tensor to scale and shift if :attr:`scale` and :attr:`shift` are set.

Refer to `Layer Normalization <https://arxiv.org/pdf/1607.06450v1.pdf>`_
)DOC");
  }
};

class LayerNormGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    // check input
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "LayerNormGrad");
    OP_INOUT_CHECK(ctx->HasInput("Mean"), "Input", "Mean", "LayerNormGrad");
    OP_INOUT_CHECK(ctx->HasInput("Variance"), "Input", "Variance",
                   "LayerNormGrad");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Y")), "Input",
                   framework::GradVarName("Y"), "LayerNormGrad");

    // check output
    if (ctx->HasOutput(framework::GradVarName("X"))) {
      ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("X"));
    }
    if (ctx->HasOutput(framework::GradVarName("Scale"))) {
      ctx->SetOutputDim(framework::GradVarName("Scale"),
                        ctx->GetInputDim("Scale"));
    }
    if (ctx->HasOutput(framework::GradVarName("Bias"))) {
      ctx->SetOutputDim(framework::GradVarName("Bias"),
                        ctx->GetInputDim("Bias"));
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    const auto *var = ctx.InputVar(framework::GradVarName("Y"));
    PADDLE_ENFORCE_NOT_NULL(var, platform::errors::NotFound(
                                     "Y@GRAD of LayerNorm Op is not found."));
    const Tensor *t = nullptr;
    if (var->IsType<Tensor>()) {
      t = &var->Get<Tensor>();
    } else if (var->IsType<LoDTensor>()) {
      t = &var->Get<LoDTensor>();
    }
    PADDLE_ENFORCE_NOT_NULL(
        t, platform::errors::NotFound("Y@GRAD of LayerNorm Op is not found."));

    framework::LibraryType library = framework::LibraryType::kPlain;
    framework::DataLayout layout = framework::DataLayout::kAnyLayout;

    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"), ctx.GetPlace(),
        layout, library);
  }
};

template <typename T>
class LayerNormGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("layer_norm_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput("Mean", this->Output("Mean"));
    op->SetInput("Variance", this->Output("Variance"));
    if (this->HasInput("Scale")) {
      op->SetInput("Scale", this->Input("Scale"));
      op->SetOutput(framework::GradVarName("Scale"), this->InputGrad("Scale"));
    }

    if (this->HasInput("Bias")) {
      op->SetInput("Bias", this->Input("Bias"));
      op->SetOutput(framework::GradVarName("Bias"), this->InputGrad("Bias"));
    }

    op->SetInput(framework::GradVarName("Y"), this->OutputGrad("Y"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetAttrMap(this->Attrs());
  }
};

DECLARE_NO_NEED_BUFFER_VARS_INFERER(LayerNormGradNoNeedBufferVarInferer,
                                    "Bias");

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
DECLARE_INFER_SHAPE_FUNCTOR(layer_norm, LayerNormInferShapeFunctor,
                            PD_INFER_META(phi::LayerNormInferMeta));
REGISTER_OPERATOR(layer_norm, ops::LayerNormOp, ops::LayerNormOpMaker,
                  ops::LayerNormGradOpMaker<paddle::framework::OpDesc>,
                  ops::LayerNormGradOpMaker<paddle::imperative::OpBase>,
                  LayerNormInferShapeFunctor);
REGISTER_OPERATOR(layer_norm_grad, ops::LayerNormGradOp,
                  ops::LayerNormGradNoNeedBufferVarInferer);
