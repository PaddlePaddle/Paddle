/*Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/unary.h"

namespace paddle {
namespace operators {

class PixelShuffleOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
};

class PixelShuffleOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(Tensor, default Tensor<float>), "
             "the input feature data of PixelShuffleOp, the layout is [N, C, "
             "H, W] or [N, H, W, C].");
    AddOutput("Out",
              "(Tensor, default Tensor<float>), the output of "
              "PixelShuffleOp. The layout is [N, C/factor^2, H*factor, "
              "W*factor] or [N, H*factor, W*factor, C/factor^2].");
    AddAttr<int>("upscale_factor",
                 "the factor to increase spatial resolution by.")
        .SetDefault(1)
        .AddCustomChecker([](const int& upscale_factor) {
          PADDLE_ENFORCE_GE(upscale_factor,
                            1,
                            platform::errors::InvalidArgument(
                                "upscale_factor should be larger than 0."));
        });
    AddAttr<std::string>(
        "data_format",
        "An optional string from: \"NHWC\", \"NCHW\". "
        "Defaults to \"NHWC\", Specify the data format of the input data.")
        .SetDefault("NCHW");

    AddComment(R"DOC(
		Pixel Shuffle operator
		This operator rearranges elements in a tensor of shape :math:`(*, C \times r^2, H, W)`
    		to a tensor of shape :math:`(C, H \times r, W \times r)`.

		This is useful for implementing efficient sub-pixel convolution
    		with a stride of :math:`1/r`.

		Please refer to the paper:
		 `Real-Time Single Image and Video Super-Resolution Using an Efficient
		 Sub-Pixel Convolutional Neural Network <https://arxiv.org/abs/1609.05158v2>`_
    		by Shi et. al (2016) for more details.

        )DOC");
  }
};

template <typename T>
class PixelShuffleGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

  void Apply(GradOpPtr<T> op) const override {
    op->SetType("pixel_shuffle_grad");
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetAttrMap(this->Attrs());
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
  }
};

class PixelShuffleGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
DECLARE_INFER_SHAPE_FUNCTOR(pixel_shuffle,
                            PixelShuffleInferShapeFunctor,
                            PD_INFER_META(phi::PixelShuffleInferMeta));

REGISTER_OPERATOR(pixel_shuffle,
                  ops::PixelShuffleOp,
                  ops::PixelShuffleOpMaker,
                  ops::PixelShuffleGradMaker<paddle::framework::OpDesc>,
                  ops::PixelShuffleGradMaker<paddle::imperative::OpBase>,
                  PixelShuffleInferShapeFunctor);

DECLARE_INFER_SHAPE_FUNCTOR(pixel_shuffle_grad,
                            PixelShuffleGradInferShapeFunctor,
                            PD_INFER_META(phi::PixelShuffleGradInferMeta));
REGISTER_OPERATOR(pixel_shuffle_grad,
                  ops::PixelShuffleGradOp,
                  PixelShuffleGradInferShapeFunctor);

REGISTER_OP_VERSION(pixel_shuffle)
    .AddCheckpoint(
        R"ROC(
               Compatible upgrade of pixel_shuffle, add a new attribute [data_format])ROC",
        paddle::framework::compatible::OpVersionDesc().NewAttr(
            "data_format", "Specify the data format of the input data", true));
