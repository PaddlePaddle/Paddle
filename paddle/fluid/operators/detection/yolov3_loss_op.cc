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

#include <memory>
#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/imperative/type_defs.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/multiary.h"

namespace paddle {
namespace operators {

using framework::Tensor;

class Yolov3LossOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

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
DECLARE_INFER_SHAPE_FUNCTOR(yolov3_loss, Yolov3LossInferShapeFunctor,
                            PD_INFER_META(phi::Yolov3LossInferMeta));
REGISTER_OPERATOR(yolov3_loss, ops::Yolov3LossOp, ops::Yolov3LossOpMaker,
                  ops::Yolov3LossGradMaker<paddle::framework::OpDesc>,
                  ops::Yolov3LossGradMaker<paddle::imperative::OpBase>,
                  Yolov3LossInferShapeFunctor);
REGISTER_OPERATOR(yolov3_loss_grad, ops::Yolov3LossOpGrad);
