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

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/prim/api/composite_backward/composite_backward_api.h"
#include "paddle/fluid/prim/utils/static/composite_grad_desc_maker.h"
#include "paddle/fluid/prim/utils/static/desc_tensor.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/ternary.h"

namespace paddle {
namespace operators {

using DataLayout = phi::DataLayout;

class LayerNormOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto input_data_type = OperatorWithKernel::IndicateVarDataType(ctx, "X");

    return phi::KernelKey(input_data_type, ctx.GetPlace());
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
        .SetDefault(1e-5);
    AddAttr<int>("begin_norm_axis",
                 "the axis of `begin_norm_axis ... Rank(X) - 1` will be "
                 "normalized. `begin_norm_axis` splits the tensor(`X`) to a "
                 "matrix [N,H]. [default 1].")
        .SetDefault(1);
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

 protected:
  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    const auto *var = ctx.InputVar(framework::GradVarName("Y"));
    PADDLE_ENFORCE_NOT_NULL(
        var,
        platform::errors::NotFound("Y@GRAD of LayerNorm Op is not found."));
    const phi::DenseTensor *t = nullptr;
    if (var->IsType<phi::DenseTensor>()) {
      t = &var->Get<phi::DenseTensor>();
    } else if (var->IsType<phi::DenseTensor>()) {
      t = &var->Get<phi::DenseTensor>();
    }
    PADDLE_ENFORCE_NOT_NULL(
        t, platform::errors::NotFound("Y@GRAD of LayerNorm Op is not found."));

    return phi::KernelKey(OperatorWithKernel::IndicateVarDataType(ctx, "X"),
                          ctx.GetPlace());
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

class LayerNormCompositeGradOpMaker : public prim::CompositeGradOpMakerBase {
  using prim::CompositeGradOpMakerBase::CompositeGradOpMakerBase;

 public:
  void Apply() override {
    // get inputs
    paddle::Tensor x = this->GetSingleForwardInput("X");
    paddle::Tensor mean = this->GetSingleForwardOutput("Mean");
    paddle::Tensor var = this->GetSingleForwardOutput("Variance");
    paddle::Tensor y_grad = this->GetSingleOutputGrad("Y");
    paddle::optional<paddle::Tensor> scale =
        this->GetOptionalSingleForwardInput("Scale");
    paddle::optional<paddle::Tensor> bias =
        this->GetOptionalSingleForwardInput("Bias");

    // get Attrs
    auto epsilon = this->Attr<float>("epsilon");
    auto begin_norm_axis = this->Attr<int>("begin_norm_axis");

    // get outputs
    paddle::Tensor x_grad = this->GetSingleInputGrad("X");
    paddle::Tensor scale_grad = this->GetSingleInputGrad("Scale");
    paddle::Tensor bias_grad = this->GetSingleInputGrad("Bias");

    auto dx_ptr = this->GetOutputPtr(&x_grad);
    std::string dx_name = this->GetOutputName(x_grad);
    auto dscale_ptr = this->GetOutputPtr(&scale_grad);
    std::string dscale_name = this->GetOutputName(scale_grad);
    auto dbias_ptr = this->GetOutputPtr(&bias_grad);
    std::string dbias_name = this->GetOutputName(bias_grad);

    VLOG(6) << "Runing layer_norm_grad composite func";
    prim::layer_norm_grad<prim::DescTensor>(x,
                                            scale,
                                            bias,
                                            mean,
                                            var,
                                            y_grad,
                                            epsilon,
                                            begin_norm_axis,
                                            dx_ptr,
                                            dscale_ptr,
                                            dbias_ptr);

    this->RecoverOutputName(x_grad, dx_name);
    this->RecoverOutputName(scale_grad, dscale_name);
    this->RecoverOutputName(bias_grad, dbias_name);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

DECLARE_INFER_SHAPE_FUNCTOR(layer_norm,
                            LayerNormInferShapeFunctor,
                            PD_INFER_META(phi::LayerNormInferMeta));

REGISTER_OPERATOR(layer_norm,
                  ops::LayerNormOp,
                  ops::LayerNormOpMaker,
                  ops::LayerNormGradOpMaker<paddle::framework::OpDesc>,
                  ops::LayerNormGradOpMaker<paddle::imperative::OpBase>,
                  ops::LayerNormCompositeGradOpMaker,
                  LayerNormInferShapeFunctor);

DECLARE_INFER_SHAPE_FUNCTOR(layer_norm_grad,
                            LayerNormGradInferShapeFunctor,
                            PD_INFER_META(phi::LayerNormGradInferMeta));

REGISTER_OPERATOR(layer_norm_grad,
                  ops::LayerNormGradOp,
                  ops::LayerNormGradNoNeedBufferVarInferer,
                  LayerNormGradInferShapeFunctor);
