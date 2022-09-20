/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
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

#include "paddle/phi/infermeta/backward.h"
#include "paddle/phi/infermeta/ternary.h"

namespace paddle {
namespace operators {

using framework::Tensor;

class SpectralNormOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "Weight"), ctx.GetPlace());
  }
};

class SpectralNormOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Weight",
             "The input weight tensor of spectral_norm operator, "
             "This can be a 2-D, 3-D, 4-D, 5-D tensor which is the "
             "weights of fc, conv1d, conv2d, conv3d layer. "
             "The data type is float32 or float64.");
    AddInput("U",
             "The weight_u tensor of spectral_norm operator, "
             "This can be a 1-D tensor in shape [H, 1],"
             "H is the 1st dimensions of Weight after reshape"
             "corresponding by Attr(dim). As for Attr(dim) = 1"
             "in conv2d layer with weight shape [M, C, K1, K2]"
             "Weight will be reshape to [C, M*K1*K2], U will"
             "be in shape [C, 1].");
    AddInput("V",
             "The weight_v tensor of spectral_norm operator, "
             "This can be a 1-D tensor in shape [W, 1], "
             "W is the 2nd dimensions of Weight after reshape "
             "corresponding by Attr(dim). As for Attr(dim) = 1 "
             "in conv2d layer with weight shape [M, C, K1, K2] "
             "Weight will be reshape to [C, M*K1*K2], V will "
             "be in shape [M*K1*K2, 1].");
    AddOutput("Out",
              "The output weight tensor of spectral_norm operator, "
              "This tensor is in same shape with Input(Weight).");

    AddAttr<int>("dim",
                 "The index of dimension which should be permuted "
                 "to the first before reshaping Input(Weight) to "
                 "matrix, it should be set as 0 if Input(Weight) is "
                 "the weight of fc layer, and should be set as 1 if "
                 "Input(Weight) is the weight of conv layer, "
                 "default 0.")
        .SetDefault(0);
    AddAttr<int>("power_iters",
                 "number of power iterations to calculate "
                 "spectral norm, default 1.")
        .SetDefault(1);
    AddAttr<float>("eps",
                   "epsilon for numerical stability in "
                   "calculating norms, it will be added to "
                   "the denominator to aviod divide zero. "
                   "Default 1e-12.")
        .SetDefault(1e-12);

    AddComment(R"DOC(
          This layer calculates the spectral normalization value of weight of
          fc, conv1d, conv2d, conv3d layers which should be 2-D, 3-D, 4-D, 5-D
          tensor.

          Spectral normalization stabilizes the training of critic in GANs
          (Generative Adversarial Networks). This layer rescaling weight tensor
          with spectral normalize value.

          For spectral normalization calculations, we rescaling weight
          tensor with :math:`\sigma`, while :math:`\sigma{\mathbf{W}}` is

            $$\sigma(\mathbf{W}) = \max_{\mathbf{h}: \mathbf{h} \ne 0} \\frac{\|\mathbf{W} \mathbf{h}\|_2}{\|\mathbf{h}\|_2}$$

          We calculate :math:`\sigma{\mathbf{W}}` through power iterations as

            $$
            \mathbf{v} = \mathbf{W}^{T} \mathbf{u}
            $$
            $$
            \mathbf{v} = \\frac{\mathbf{v}}{\|\mathbf{v}\|_2}
            $$
            $$
            \mathbf{u} = \mathbf{W}^{T} \mathbf{v}
            $$
            $$
            \mathbf{u} = \\frac{\mathbf{u}}{\|\mathbf{u}\|_2}
            $$

          And :math:`\sigma` should be

            $$\sigma{\mathbf{W}} = \mathbf{u}^{T} \mathbf{W} \mathbf{v}$$

          For details of spectral normalization, please refer to paper:
          `Spectral Normalization <https://arxiv.org/abs/1802.05957>`_ .
         )DOC");
  }
};

template <typename T>
class SpectralNormGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("spectral_norm_grad");

    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetInput("Weight", this->Input("Weight"));
    op->SetInput("U", this->Input("U"));
    op->SetInput("V", this->Input("V"));

    op->SetOutput(framework::GradVarName("Weight"), this->InputGrad("Weight"));

    op->SetAttrMap(this->Attrs());
  }
};

class SpectralNormOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "Weight"), ctx.GetPlace());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

DECLARE_INFER_SHAPE_FUNCTOR(spectral_norm,
                            SpectralNormInferMetaFunctor,
                            PD_INFER_META(phi::SpectralNormInferMeta));
DECLARE_INFER_SHAPE_FUNCTOR(spectral_norm_grad,
                            SpectralNormGradInferMetaFunctor,
                            PD_INFER_META(phi::SpectralNormGradInferMeta));

REGISTER_OPERATOR(spectral_norm,
                  ops::SpectralNormOp,
                  ops::SpectralNormOpMaker,
                  ops::SpectralNormGradOpMaker<paddle::framework::OpDesc>,
                  ops::SpectralNormGradOpMaker<paddle::imperative::OpBase>,
                  SpectralNormInferMetaFunctor);
REGISTER_OPERATOR(spectral_norm_grad,
                  ops::SpectralNormOpGrad,
                  SpectralNormGradInferMetaFunctor);
