/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/detection/box_decoder_and_assign_op.h"

namespace paddle {
namespace operators {

using LoDTensor = framework::LoDTensor;

class BoxDecoderAndAssignOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(
        ctx->HasInput("PriorBox"),
        "Input(PriorBox) of BoxDecoderAndAssignOp should not be null.");
    PADDLE_ENFORCE(
        ctx->HasInput("PriorBoxVar"),
        "Input(PriorBoxVar) of BoxDecoderAndAssignOp should not be null.");
    PADDLE_ENFORCE(
        ctx->HasInput("TargetBox"),
        "Input(TargetBox) of BoxDecoderAndAssignOp should not be null.");
    PADDLE_ENFORCE(
        ctx->HasInput("BoxScore"),
        "Input(BoxScore) of BoxDecoderAndAssignOp should not be null.");
    PADDLE_ENFORCE(
        ctx->HasOutput("DecodeBox"),
        "Output(DecodeBox) of BoxDecoderAndAssignOp should not be null.");
    PADDLE_ENFORCE(
        ctx->HasOutput("OutputAssignBox"),
        "Output(OutputAssignBox) of BoxDecoderAndAssignOp should not be null.");

    auto prior_box_dims = ctx->GetInputDim("PriorBox");
    auto prior_box_var_dims = ctx->GetInputDim("PriorBoxVar");
    auto target_box_dims = ctx->GetInputDim("TargetBox");
    auto box_score_dims = ctx->GetInputDim("BoxScore");

    PADDLE_ENFORCE_EQ(prior_box_dims.size(), 2,
                      "The rank of Input of PriorBox must be 2");
    PADDLE_ENFORCE_EQ(prior_box_dims[1], 4, "The shape of PriorBox is [N, 4]");
    PADDLE_ENFORCE_EQ(prior_box_var_dims.size(), 1,
                      "The rank of Input of PriorBoxVar must be 1");
    PADDLE_ENFORCE_EQ(prior_box_var_dims[0], 4,
                      "The shape of PriorBoxVar is [4]");
    PADDLE_ENFORCE_EQ(target_box_dims.size(), 2,
                      "The rank of Input of TargetBox must be 2");
    PADDLE_ENFORCE_EQ(box_score_dims.size(), 2,
                      "The rank of Input of BoxScore must be 2");
    if (ctx->IsRuntime()) {
      PADDLE_ENFORCE_EQ(prior_box_dims[0], target_box_dims[0],
                        "The first dim of prior_box and target_box is roi nums "
                        "and should be same!");
      PADDLE_ENFORCE_EQ(prior_box_dims[0], box_score_dims[0],
                        "The first dim of prior_box and box_score is roi nums "
                        "and should be same!");
      PADDLE_ENFORCE_EQ(
          target_box_dims[1], box_score_dims[1] * prior_box_dims[1],
          "The shape of target_box is [N, classnum * 4], The shape "
          "of box_score is [N, classnum], The shape of prior_box "
          "is [N, 4]");
    }
    ctx->SetOutputDim("DecodeBox", framework::make_ddim({target_box_dims[0],
                                                         target_box_dims[1]}));
    ctx->ShareLoD("TargetBox", /*->*/ "DecodeBox");
    ctx->SetOutputDim(
        "OutputAssignBox",
        framework::make_ddim({prior_box_dims[0], prior_box_dims[1]}));
    ctx->ShareLoD("PriorBox", /*->*/ "OutputAssignBox");
  }
};

class BoxDecoderAndAssignOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput(
        "PriorBox",
        "(Tensor, default Tensor<float>) "
        "Box list PriorBox is a 2-D Tensor with shape [N, 4] which holds N "
        "boxes and each box is represented as [xmin, ymin, xmax, ymax], "
        "[xmin, ymin] is the left top coordinate of the anchor box, "
        "if the input is image feature map, they are close to the origin "
        "of the coordinate system. [xmax, ymax] is the right bottom "
        "coordinate of the anchor box.");
    AddInput("PriorBoxVar",
             "(Tensor, default Tensor<float>, optional) "
             "PriorBoxVar is a 2-D Tensor with shape [N, 4] which holds N "
             "group of variance. PriorBoxVar will set all elements to 1 by "
             "default.")
        .AsDispensable();
    AddInput("TargetBox",
             "(LoDTensor or Tensor) "
             "This input can be a 2-D LoDTensor with shape "
             "[N, classnum*4]. It holds N targets for N boxes.");
    AddInput("BoxScore",
             "(LoDTensor or Tensor) "
             "This input can be a 2-D LoDTensor with shape "
             "[N, classnum], each box is represented as [classnum] which is "
             "the classification probabilities.");
    AddAttr<float>("box_clip",
                   "(float, default 4.135, np.log(1000. / 16.)) "
                   "clip box to prevent overflowing")
        .SetDefault(4.135f);
    AddOutput("DecodeBox",
              "(LoDTensor or Tensor) "
              "the output tensor of op with shape [N, classnum * 4] "
              "representing the result of N target boxes decoded with "
              "M Prior boxes and variances for each class.");
    AddOutput("OutputAssignBox",
              "(LoDTensor or Tensor) "
              "the output tensor of op with shape [N, 4] "
              "representing the result of N target boxes decoded with "
              "M Prior boxes and variances with the best non-background class "
              "by BoxScore.");
    AddComment(R"DOC(

Bounding Box Coder.

Decode the target bounding box with the prior_box information.

The Decoding schema is described below:

    $$
    ox = (pw \\times pxv \\times tx + px) - \\frac{tw}{2} 
    $$
    $$
    oy = (ph \\times pyv \\times ty + py) - \\frac{th}{2}
    $$
    $$
    ow = \\exp (pwv \\times tw) \\times pw + \\frac{tw}{2}
    $$
    $$
    oh = \\exp (phv \\times th) \\times ph + \\frac{th}{2}
    $$

where `tx`, `ty`, `tw`, `th` denote the target box's center coordinates, width
and height respectively. Similarly, `px`, `py`, `pw`, `ph` denote the
prior_box's (anchor) center coordinates, width and height. `pxv`, `pyv`, `pwv`,
`phv` denote the variance of the prior_box and `ox`, `oy`, `ow`, `oh` denote the
decoded coordinates, width and height in decode_box. 

decode_box is obtained after box decode, then assigning schema is described below:

For each prior_box, use the best non-background class's decoded values to 
update the prior_box locations and get output_assign_box. So, the shape of
output_assign_box is the same as PriorBox.
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(box_decoder_and_assign, ops::BoxDecoderAndAssignOp,
                  ops::BoxDecoderAndAssignOpMaker,
                  paddle::framework::EmptyGradOpMaker);
REGISTER_OP_CPU_KERNEL(
    box_decoder_and_assign,
    ops::BoxDecoderAndAssignKernel<paddle::platform::CPUDeviceContext, float>,
    ops::BoxDecoderAndAssignKernel<paddle::platform::CPUDeviceContext, double>);
