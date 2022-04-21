// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

class ChannelShuffleOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
};

class ChannelShuffleOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(Tensor, default Tensor<float>), "
             "the input feature data of ChannelShuffleOp, the layout is "
             "[N, C, H, W] or [N, H, W, C].");
    AddOutput("Out",
              "(Tensor, default Tensor<float>), the output of "
              "ChannelShuffleOp. The layout is also [N, C, "
              "H, W] or [N, H, W, C].");
    AddAttr<int>("groups", "number of groups to divide channels in.");
    AddAttr<std::string>(
        "data_format",
        "An optional string from: \"NHWC\", \"NCHW\". "
        "Defaults to \"NHWC\", Specify the data format of the input data.")
        .SetDefault("NCHW");

    AddComment(R"DOC(
    Channel Shuffle operator
    This operator divides channels in a tensor of shape :math:`(*, C, H, W)`
        into :math:`g` groups and rearranges them as :math:`(*, C/g, g, H, W)`
        while keeping the original tensor shape.

    Please refer to the paper:
        `ShuffleNet: An Extremely Efficient Convolutional Neural Network for 
        Mobile Devices <https://arxiv.org/abs/1707.01083>`_
        by Zhang et. al (2017) for more details. 

        )DOC");
  }
};

class ChannelShuffleGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
};

template <typename T>
class ChannelShuffleGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("channel_shuffle_grad");
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetAttrMap(this->Attrs());
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
DECLARE_INFER_SHAPE_FUNCTOR(channel_shuffle, ChannelShuffleInferShapeFunctor,
                            PD_INFER_META(phi::ChannelShuffleInferMeta));

REGISTER_OPERATOR(channel_shuffle, ops::ChannelShuffleOp,
                  ops::ChannelShuffleOpMaker,
                  ops::ChannelShuffleGradOpMaker<paddle::framework::OpDesc>,
                  ops::ChannelShuffleGradOpMaker<paddle::imperative::OpBase>,
                  ChannelShuffleInferShapeFunctor);

DECLARE_INFER_SHAPE_FUNCTOR(channel_shuffle_grad,
                            ChannelShuffleGradInferShapeFunctor,
                            PD_INFER_META(phi::ChannelShuffleGradInferMeta));

REGISTER_OPERATOR(channel_shuffle_grad, ops::ChannelShuffleGradOp,
                  ChannelShuffleGradInferShapeFunctor);
