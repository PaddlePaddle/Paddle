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

#include "paddle/fluid/operators/detection/yolov3_loss_op.h"
#include <memory>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/imperative/type_defs.h"

namespace paddle {
namespace operators {

using framework::Tensor;

class Yolov3LossOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "Yolov3LossOp");
    OP_INOUT_CHECK(ctx->HasInput("GTBox"), "Input", "GTBox", "Yolov3LossOp");
    OP_INOUT_CHECK(ctx->HasInput("GTLabel"), "Input", "GTLabel",
                   "Yolov3LossOp");
    OP_INOUT_CHECK(ctx->HasOutput("Loss"), "Output", "Loss", "Yolov3LossOp");
    OP_INOUT_CHECK(ctx->HasOutput("ObjectnessMask"), "Output", "ObjectnessMask",
                   "Yolov3LossOp");
    OP_INOUT_CHECK(ctx->HasOutput("GTMatchMask"), "Output", "GTMatchMask",
                   "Yolov3LossOp");

    auto dim_x = ctx->GetInputDim("X");
    auto dim_gtbox = ctx->GetInputDim("GTBox");
    auto dim_gtlabel = ctx->GetInputDim("GTLabel");
    auto anchors = ctx->Attrs().Get<std::vector<int>>("anchors");
    int anchor_num = anchors.size() / 2;
    auto anchor_mask = ctx->Attrs().Get<std::vector<int>>("anchor_mask");
    int mask_num = anchor_mask.size();
    auto class_num = ctx->Attrs().Get<int>("class_num");

    PADDLE_ENFORCE_EQ(dim_x.size(), 4,
                      platform::errors::InvalidArgument(
                          "Input(X) should be a 4-D tensor. But received "
                          "X dimension size(%s)",
                          dim_x.size()));
    PADDLE_ENFORCE_EQ(dim_x[2], dim_x[3],
                      platform::errors::InvalidArgument(
                          "Input(X) dim[3] and dim[4] should be euqal."
                          "But received dim[3](%s) != dim[4](%s)",
                          dim_x[2], dim_x[3]));
    PADDLE_ENFORCE_EQ(
        dim_x[1], mask_num * (5 + class_num),
        platform::errors::InvalidArgument(
            "Input(X) dim[1] should be equal to (anchor_mask_number * (5 "
            "+ class_num))."
            "But received dim[1](%s) != (anchor_mask_number * "
            "(5+class_num)(%s).",
            dim_x[1], mask_num * (5 + class_num)));
    PADDLE_ENFORCE_EQ(dim_gtbox.size(), 3,
                      platform::errors::InvalidArgument(
                          "Input(GTBox) should be a 3-D tensor, but "
                          "received gtbox dimension size(%s)",
                          dim_gtbox.size()));
    PADDLE_ENFORCE_EQ(dim_gtbox[2], 4,
                      platform::errors::InvalidArgument(
                          "Input(GTBox) dim[2] should be 4",
                          "But receive dim[2](%s) != 5. ", dim_gtbox[2]));
    PADDLE_ENFORCE_EQ(
        dim_gtlabel.size(), 2,
        platform::errors::InvalidArgument(
            "Input(GTLabel) should be a 2-D tensor,"
            "But received Input(GTLabel) dimension size(%s) != 2.",
            dim_gtlabel.size()));
    PADDLE_ENFORCE_EQ(
        dim_gtlabel[0], dim_gtbox[0],
        platform::errors::InvalidArgument(
            "Input(GTBox) dim[0] and Input(GTLabel) dim[0] should be same,"
            "But received Input(GTLabel) dim[0](%s) != "
            "Input(GTBox) dim[0](%s)",
            dim_gtlabel[0], dim_gtbox[0]));
    PADDLE_ENFORCE_EQ(
        dim_gtlabel[1], dim_gtbox[1],
        platform::errors::InvalidArgument(
            "Input(GTBox) and Input(GTLabel) dim[1] should be same,"
            "But received Input(GTBox) dim[1](%s) != Input(GTLabel) "
            "dim[1](%s)",
            dim_gtbox[1], dim_gtlabel[1]));
    PADDLE_ENFORCE_GT(anchors.size(), 0,
                      platform::errors::InvalidArgument(
                          "Attr(anchors) length should be greater then 0."
                          "But received anchors length(%s)",
                          anchors.size()));
    PADDLE_ENFORCE_EQ(anchors.size() % 2, 0,
                      platform::errors::InvalidArgument(
                          "Attr(anchors) length should be even integer."
                          "But received anchors length(%s)",
                          anchors.size()));
    for (size_t i = 0; i < anchor_mask.size(); i++) {
      PADDLE_ENFORCE_LT(
          anchor_mask[i], anchor_num,
          platform::errors::InvalidArgument(
              "Attr(anchor_mask) should not crossover Attr(anchors)."
              "But received anchor_mask[i](%s) > anchor_num(%s)",
              anchor_mask[i], anchor_num));
    }
    PADDLE_ENFORCE_GT(class_num, 0,
                      platform::errors::InvalidArgument(
                          "Attr(class_num) should be an integer greater then 0."
                          "But received class_num(%s) < 0",
                          class_num));

    if (ctx->HasInput("GTScore")) {
      auto dim_gtscore = ctx->GetInputDim("GTScore");
      PADDLE_ENFORCE_EQ(dim_gtscore.size(), 2,
                        platform::errors::InvalidArgument(
                            "Input(GTScore) should be a 2-D tensor"
                            "But received GTScore dimension(%s)",
                            dim_gtbox.size()));
      PADDLE_ENFORCE_EQ(
          dim_gtscore[0], dim_gtbox[0],
          platform::errors::InvalidArgument(
              "Input(GTBox) and Input(GTScore) dim[0] should be same"
              "But received GTBox dim[0](%s) != GTScore dim[0](%s)",
              dim_gtbox[0], dim_gtscore[0]));
      PADDLE_ENFORCE_EQ(
          dim_gtscore[1], dim_gtbox[1],
          platform::errors::InvalidArgument(
              "Input(GTBox) and Input(GTScore) dim[1] should be same"
              "But received GTBox dim[1](%s) != GTScore dim[1](%s)",
              dim_gtscore[1], dim_gtbox[1]));
    }

    std::vector<int64_t> dim_out({dim_x[0]});
    ctx->SetOutputDim("Loss", phi::make_ddim(dim_out));

    std::vector<int64_t> dim_obj_mask({dim_x[0], mask_num, dim_x[2], dim_x[3]});
    ctx->SetOutputDim("ObjectnessMask", phi::make_ddim(dim_obj_mask));

    std::vector<int64_t> dim_gt_match_mask({dim_gtbox[0], dim_gtbox[1]});
    ctx->SetOutputDim("GTMatchMask", phi::make_ddim(dim_gt_match_mask));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"),
        platform::CPUPlace());
  }
};

class Yolov3LossOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "The input tensor of YOLOv3 loss operator, "
             "This is a 4-D tensor with shape of [N, C, H, W]."
             "H and W should be same, and the second dimension(C) stores"
             "box locations, confidence score and classification one-hot"
             "keys of each anchor box");
    AddInput("GTBox",
             "The input tensor of ground truth boxes, "
             "This is a 3-D tensor with shape of [N, max_box_num, 5], "
             "max_box_num is the max number of boxes in each image, "
             "In the third dimension, stores x, y, w, h coordinates, "
             "x, y is the center coordinate of boxes and w, h is the "
             "width and height and x, y, w, h should be divided by "
             "input image height to scale to [0, 1].");
    AddInput("GTLabel",
             "The input tensor of ground truth label, "
             "This is a 2-D tensor with shape of [N, max_box_num], "
             "and each element should be an integer to indicate the "
             "box class id.");
    AddInput("GTScore",
             "The score of GTLabel, This is a 2-D tensor in same shape "
             "GTLabel, and score values should in range (0, 1). This "
             "input is for GTLabel score can be not 1.0 in image mixup "
             "augmentation.")
        .AsDispensable();
    AddOutput("Loss",
              "The output yolov3 loss tensor, "
              "This is a 1-D tensor with shape of [N]");
    AddOutput("ObjectnessMask",
              "This is an intermediate tensor with shape of [N, M, H, W], "
              "M is the number of anchor masks. This parameter caches the "
              "mask for calculate objectness loss in gradient kernel.")
        .AsIntermediate();
    AddOutput("GTMatchMask",
              "This is an intermediate tensor with shape of [N, B], "
              "B is the max box number of GT boxes. This parameter caches "
              "matched mask index of each GT boxes for gradient calculate.")
        .AsIntermediate();

    AddAttr<int>("class_num", "The number of classes to predict.");
    AddAttr<std::vector<int>>("anchors",
                              "The anchor width and height, "
                              "it will be parsed pair by pair.")
        .SetDefault(std::vector<int>{});
    AddAttr<std::vector<int>>("anchor_mask",
                              "The mask index of anchors used in "
                              "current YOLOv3 loss calculation.")
        .SetDefault(std::vector<int>{});
    AddAttr<int>("downsample_ratio",
                 "The downsample ratio from network input to YOLOv3 loss "
                 "input, so 32, 16, 8 should be set for the first, second, "
                 "and thrid YOLOv3 loss operators.")
        .SetDefault(32);
    AddAttr<float>("ignore_thresh",
                   "The ignore threshold to ignore confidence loss.")
        .SetDefault(0.7);
    AddAttr<bool>("use_label_smooth",
                  "Whether to use label smooth. Default True.")
        .SetDefault(true);
    AddAttr<float>("scale_x_y",
                   "Scale the center point of decoded bounding "
                   "box. Default 1.0")
        .SetDefault(1.);
    AddComment(R"DOC(
         This operator generates yolov3 loss based on given predict result and ground
         truth boxes.
         
         The output of previous network is in shape [N, C, H, W], while H and W
         should be the same, H and W specify the grid size, each grid point predict 
         given number bounding boxes, this given number, which following will be represented as S,
         is specified by the number of anchor clusters in each scale. In the second dimension(the channel
         dimension), C should be equal to S * (class_num + 5), class_num is the object 
         category number of source dataset(such as 80 in coco dataset), so in the 
         second(channel) dimension, apart from 4 box location coordinates x, y, w, h, 
         also includes confidence score of the box and class one-hot key of each anchor box.

         Assume the 4 location coordinates are :math:`t_x, t_y, t_w, t_h`, the box predictions
         should be as follows:

         $$
         b_x = \\sigma(t_x) + c_x
         $$
         $$
         b_y = \\sigma(t_y) + c_y
         $$
         $$
         b_w = p_w e^{t_w}
         $$
         $$
         b_h = p_h e^{t_h}
         $$

         In the equation above, :math:`c_x, c_y` is the left top corner of current grid
         and :math:`p_w, p_h` is specified by anchors.

         As for confidence score, it is the logistic regression value of IoU between
         anchor boxes and ground truth boxes, the score of the anchor box which has 
         the max IoU should be 1, and if the anchor box has IoU bigger than ignore 
         thresh, the confidence score loss of this anchor box will be ignored.

         Therefore, the yolov3 loss consists of three major parts: box location loss,
         objectness loss and classification loss. The L1 loss is used for 
         box coordinates (w, h), sigmoid cross entropy loss is used for box 
         coordinates (x, y), objectness loss and classification loss.

         Each groud truth box finds a best matching anchor box in all anchors. 
         Prediction of this anchor box will incur all three parts of losses, and
         prediction of anchor boxes with no GT box matched will only incur objectness
         loss.

         In order to trade off box coordinate losses between big boxes and small 
         boxes, box coordinate losses will be mutiplied by scale weight, which is
         calculated as follows.

         $$
         weight_{box} = 2.0 - t_w * t_h
         $$

         Final loss will be represented as follows.

         $$
         loss = (loss_{xy} + loss_{wh}) * weight_{box}
              + loss_{conf} + loss_{class}
         $$

         While :attr:`use_label_smooth` is set to be :attr:`True`, the classification
         target will be smoothed when calculating classification loss, target of 
         positive samples will be smoothed to :math:`1.0 - 1.0 / class\_num` and target of
         negetive samples will be smoothed to :math:`1.0 / class\_num`.

         While :attr:`GTScore` is given, which means the mixup score of ground truth 
         boxes, all losses incured by a ground truth box will be multiplied by its 
         mixup score.
         )DOC");
  }
};

class Yolov3LossOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("X"), true,
        platform::errors::NotFound("Input(X) should not be null"));
    PADDLE_ENFORCE_EQ(
        ctx->HasInput(framework::GradVarName("Loss")), true,
        platform::errors::NotFound("Input(Loss@GRAD) should not be null"));
    auto dim_x = ctx->GetInputDim("X");
    if (ctx->HasOutput(framework::GradVarName("X"))) {
      ctx->SetOutputDim(framework::GradVarName("X"), dim_x);
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"),
        platform::CPUPlace());
  }
};

template <typename T>
class Yolov3LossGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("yolov3_loss_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput("GTBox", this->Input("GTBox"));
    op->SetInput("GTLabel", this->Input("GTLabel"));
    op->SetInput("GTScore", this->Input("GTScore"));
    op->SetInput(framework::GradVarName("Loss"), this->OutputGrad("Loss"));
    op->SetInput("ObjectnessMask", this->Output("ObjectnessMask"));
    op->SetInput("GTMatchMask", this->Output("GTMatchMask"));

    op->SetAttrMap(this->Attrs());

    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetOutput(framework::GradVarName("GTBox"), this->EmptyInputGrad());
    op->SetOutput(framework::GradVarName("GTLabel"), this->EmptyInputGrad());
    op->SetOutput(framework::GradVarName("GTScore"), this->EmptyInputGrad());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(yolov3_loss, ops::Yolov3LossOp, ops::Yolov3LossOpMaker,
                  ops::Yolov3LossGradMaker<paddle::framework::OpDesc>,
                  ops::Yolov3LossGradMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(yolov3_loss_grad, ops::Yolov3LossOpGrad);
REGISTER_OP_CPU_KERNEL(yolov3_loss, ops::Yolov3LossKernel<float>,
                       ops::Yolov3LossKernel<double>);
REGISTER_OP_CPU_KERNEL(yolov3_loss_grad, ops::Yolov3LossGradKernel<float>,
                       ops::Yolov3LossGradKernel<double>);
