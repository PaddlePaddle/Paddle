/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 *     Unless required by applicable law or agreed to in writing, software
 *     distributed under the License is distributed on an "AS IS" BASIS,
 *     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *     See the License for the specific language governing permissions and
 *     limitations under the License. */


#include "paddle/operators/maxout_op.h"
namespace paddle {
namespace operators {

using framework::Tensor;

class MaxOutOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  MaxOutOpMaker(framework::OpProto* proto, framework::OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X",
        "(Tensor) The input tensor of maxout operator. "
        "The format of input tensor is NCHW. Where N is batch size, C is the "
        "number of channels, H and W is the height and width of feature.");
    AddOutput("Out",
        "(Tensor) The output tensor of maxout operator."
        "The format of output tensor is also NCHW."
        "Where N is batch size, C is "
        "the number of channels, H and W is the height and "
        "width of feature.");

    AddAttr<int>(
        "groups",
        R"DOC(The group number of input layer.
        )DOC");
    AddComment(R"DOC(
        - Input: NCHW.
        - Output: feature map size same as input. Channel is (input channel) / groups.
        So groups should be larger than 1, and the num of channels should be able
        to devided by groups.

    .. math::
       y_{si+j} = \max_k x_{gsi + sk + j}
       g = groups
       s = input.size / num_channels
       0 \le i < num_channels / groups
       0 \le j < s
       0 \le k < groups

    Please refer to Paper:
      - Maxout Networks: http://www.jmlr.org/proceedings/papers/v28/goodfellow13.pdf
      - Multi-digit Number Recognition from Street View \
        Imagery using Deep Convolutional Neural Networks: \
        https://arxiv.org/pdf/1312.6082v4.pdf

    The simple usage is:

    .. code-block:: python

       maxout = maxout_layer(input,
                             num_channels=128,
                             groups=4)

    :param input: The input of this layer.
    :type input: LayerOutput
    :param num_channels: The channel number of input layer. If None will be set
                     automatically from previous output.
    :type num_channels: int | None
    :param groups: The group number of input layer.
    :type groups: int
    :param name: The name of this layer. It is optional.
    :type name: None | basestring.
    :param layer_attr: Extra Layer attribute.
    :type layer_attr: ExtraLayerAttribute
    :return: LayerOutput object.
    :rtype: LayerOutput

        )DOC");
  }
};


class MaxOutOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) of maxoutOp"
                   "should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of maxoutOp should not be null.");
    auto in_x_dims = ctx->GetInputDim("X");
    int groups = ctx->Attrs().Get<int>("groups");

    // check groups > 1
    PADDLE_ENFORCE_GT(
        groups, 1,
        "in maxoutop  groups should be larger than 1");


    std::vector<int64_t> output_shape({in_x_dims[0], in_x_dims[1] / groups});
    output_shape.push_back(in_x_dims[2]);
    output_shape.push_back(in_x_dims[3]);

    ctx->SetOutputDim("Out", framework::make_ddim(output_shape));
  }
};


class MaxOutOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) must not be null.");
    PADDLE_ENFORCE(ctx->HasOutput(framework::GradVarName("X")),
    "Input(X@GRAD) should not be null.");
    ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("X"));
  }
};
}    // namespace operators
}    // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP(maxout, ops::MaxOutOp, ops::MaxOutOpMaker, maxout_grad,
                        ops::MaxOutOpGrad);


REGISTER_OP_CPU_KERNEL(maxout, ops::MaxOutKernel<paddle::platform::CPUPlace,
                       float>);
REGISTER_OP_CPU_KERNEL(maxout_grad,
                       ops::MaxOutGradKernel<paddle::platform::CPUPlace,
                       float>);
