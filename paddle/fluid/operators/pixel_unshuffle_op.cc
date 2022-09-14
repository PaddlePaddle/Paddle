// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/backward.h"
#include "paddle/phi/infermeta/unary.h"

namespace paddle {
namespace operators {

class PixelUnshuffleOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
};

class PixelUnshuffleOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(Tensor, default Tensor<float>), "
             "the input feature data of PixelUnshuffleOp, the layout is "
             "[N, C, H, W] or [N, H, W, C].");
    AddOutput("Out",
              "(Tensor, default Tensor<float>), the output of "
              "PixelUnshuffleOp. The layout is [N, C*factor^2, H/factor, "
              "W/factor] or [N, H/factor, W/factor, C*factor^2].");
    AddAttr<int>("downscale_factor",
                 "the factor to decrease spatial resolution by.")
        .SetDefault(1);
    AddAttr<std::string>(
        "data_format",
        "An optional string from: \"NHWC\", \"NCHW\". "
        "Defaults to \"NHWC\", Specify the data format of the input data.")
        .SetDefault("NCHW");

    AddComment(R"DOC(
		Pixel Unshuffle operator
		This operator rearranges elements in a tensor of shape :math:`(*, C, H, W)`
    		to a tensor of shape :math:`(*, C\times r^2, H / r, W / r)`.

		This operation is the reversion of PixelShuffle operation.

		Please refer to the paper:
		 `Real-Time Single Image and Video Super-Resolution Using an Efficient
		 Sub-Pixel Convolutional Neural Network <https://arxiv.org/abs/1609.05158v2>`_
    		by Shi et. al (2016) for more details.

        )DOC");
  }
};

template <typename T>
class PixelUnshuffleGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("pixel_unshuffle_grad");
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetAttrMap(this->Attrs());
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
  }
};

class PixelUnshuffleGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
DECLARE_INFER_SHAPE_FUNCTOR(pixel_unshuffle,
                            PixelUnshuffleInferShapeFunctor,
                            PD_INFER_META(phi::PixelUnshuffleInferMeta));

REGISTER_OPERATOR(pixel_unshuffle,
                  ops::PixelUnshuffleOp,
                  ops::PixelUnshuffleOpMaker,
                  ops::PixelUnshuffleGradOpMaker<paddle::framework::OpDesc>,
                  ops::PixelUnshuffleGradOpMaker<paddle::imperative::OpBase>,
                  PixelUnshuffleInferShapeFunctor);

DECLARE_INFER_SHAPE_FUNCTOR(pixel_unshuffle_grad,
                            PixelUnshuffleGradInferShapeFunctor,
                            PD_INFER_META(phi::PixelUnshuffleGradInferMeta));

REGISTER_OPERATOR(pixel_unshuffle_grad,
                  ops::PixelUnshuffleGradOp,
                  PixelUnshuffleGradInferShapeFunctor);
