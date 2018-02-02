/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/operators/box_coder_op.h"

namespace paddle {
namespace operators {

class BoxCoderOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("PriorBox"),
                   "Input(PriorBox) of BoxCoderOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("PriorBoxVar"),
                   "Input(PriorBoxVar) of BoxCoderOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("TargetBox"),
                   "Input(TargetBox) of BoxCoderOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("OutputBox"),
                   "Output(OutputBox) of BoxCoderOp should not be null.");

    auto prior_box_dims = ctx->GetInputDim("PriorBox");
    auto prior_box_var_dims = ctx->GetInputDim("PriorBoxVar");
    auto target_box_dims = ctx->GetInputDim("TargetBox");

    PADDLE_ENFORCE_EQ(prior_box_dims.size(), 2,
                      "The rank of Input of PriorBoxVar must be 2");
    PADDLE_ENFORCE_EQ(prior_box_dims[1], 4, "The shape of PriorBox is [N, 4]");
    PADDLE_ENFORCE_EQ(prior_box_dims, prior_box_var_dims);
    PADDLE_ENFORCE_EQ(target_box_dims.size(), 2,
                      "The rank of Input of TargetBox must be 2");
    PADDLE_ENFORCE_EQ(target_box_dims[1], 4,
                      "The shape of TargetBox is [M, 4]");

    GetBoxCodeType(ctx->Attrs().Get<std::string>("code_type"));

    ctx->SetOutputDim(
        "OutputBox",
        framework::make_ddim({target_box_dims[0], prior_box_dims[0], 4}));
    ctx->ShareLoD("TargetBox", /*->*/ "OutputBox");
  }
};

class BoxCoderOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  BoxCoderOpMaker(OpProto *proto, OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput(
        "PriorBox",
        "(Tensor, default Tensor<float>) "
        "Box list PriorBox is a 2-D Tensor with shape [M, 4] holds M boxes, "
        "each box is represented as [xmin, ymin, xmax, ymax], "
        "[xmin, ymin] is the left top coordinate of the anchor box, "
        "if the input is image feature map, they are close to the origin "
        "of the coordinate system. [xmax, ymax] is the right bottom "
        "coordinate of the anchor box.");
    AddInput("PriorBoxVar",
             "(Tensor, default Tensor<float>) "
             "PriorBoxVar is a 2-D Tensor with shape [M, 4] holds M group "
             "of variance.");
    AddInput(
        "TargetBox",
        "(LoDTensor or Tensor) this input is a 2-D LoDTensor with shape "
        "[N, 4], each box is represented as [xmin, ymin, xmax, ymax], "
        "[xmin, ymin] is the left top coordinate of the box if the input "
        "is image feature map, they are close to the origin of the coordinate "
        "system. [xmax, ymax] is the right bottom coordinate of the box. "
        "This tensor can contain LoD information to represent a batch "
        "of inputs. One instance of this batch can contain different "
        "numbers of entities.");
    AddAttr<std::string>("code_type",
                         "(string, default encode_center_size) "
                         "the code type used with the target box")
        .SetDefault("encode_center_size")
        .InEnum({"encode_center_size", "decode_center_size"});
    AddOutput(
        "OutputBox",
        "(LoDTensor or Tensor) "
        "(Tensor) The output of box_coder_op, a tensor with shape [N, M, 4] "
        "representing the result of N target boxes encoded/decoded with "
        "M Prior boxes and variances.");

    AddComment(R"DOC(
Bounding Box Coder Operator.
Encode/Decode the target bounding box with the priorbox information.
The Encoding schema described below:
ox = (tx - px) / pw / pxv
oy = (ty - py) / ph / pyv
ow = log(abs(tw / pw)) / pwv 
oh = log(abs(th / ph)) / phv 
The Decoding schema described below:
ox = (pw * pxv * tx * + px) - tw / 2
oy = (ph * pyv * ty * + py) - th / 2
ow = exp(pwv * tw) * pw + tw / 2
oh = exp(phv * th) * ph + th / 2
where tx, ty, tw, th denote the target box's center coordinates, width and
height respectively. Similarly, px, py, pw, ph denote the priorbox's(anchor)
center coordinates, width and height. pxv, pyv, pwv, phv denote the variance
of the priorbox and ox, oy, ow, oh denote the encoded/decoded coordinates,
width and height.
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(box_coder, ops::BoxCoderOp, ops::BoxCoderOpMaker);
REGISTER_OP_CPU_KERNEL(box_coder, ops::BoxCoderKernel<float>,
                       ops::BoxCoderKernel<double>);
