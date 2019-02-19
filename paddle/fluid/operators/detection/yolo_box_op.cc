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

#include "paddle/fluid/operators/detection/yolo_box_op.h"
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

using framework::Tensor;

class YoloBoxOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of YoloBoxOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("ImgSize"),
                   "Input(ImgSize) of YoloBoxOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Boxes"),
                   "Output(Boxes) of YoloBoxOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Scores"),
                   "Output(Scores) of YoloBoxOp should not be null.");

    auto dim_x = ctx->GetInputDim("X");
    auto dim_imgsize = ctx->GetInputDim("ImgSize");
    auto anchors = ctx->Attrs().Get<std::vector<int>>("anchors");
    int anchor_num = anchors.size() / 2;
    auto class_num = ctx->Attrs().Get<int>("class_num");

    PADDLE_ENFORCE_EQ(dim_x.size(), 4, "Input(X) should be a 4-D tensor.");
    PADDLE_ENFORCE_EQ(
        dim_x[1], anchor_num * (5 + class_num),
        "Input(X) dim[1] should be equal to (anchor_mask_number * (5 "
        "+ class_num)).");
    PADDLE_ENFORCE_EQ(dim_imgsize.size(), 2,
                      "Input(ImgSize) should be a 2-D tensor.");
    PADDLE_ENFORCE_EQ(
        dim_imgsize[0], dim_x[0],
        "Input(ImgSize) dim[0] and Input(X) dim[0] should be same.");
    PADDLE_ENFORCE_EQ(dim_imgsize[1], 2, "Input(ImgSize) dim[1] should be 2.");
    PADDLE_ENFORCE_GT(anchors.size(), 0,
                      "Attr(anchors) length should be greater then 0.");
    PADDLE_ENFORCE_EQ(anchors.size() % 2, 0,
                      "Attr(anchors) length should be even integer.");
    PADDLE_ENFORCE_GT(class_num, 0,
                      "Attr(class_num) should be an integer greater then 0.");

    int box_num = dim_x[2] * dim_x[3] * anchor_num;
    std::vector<int64_t> dim_boxes({dim_x[0], box_num, 4});
    ctx->SetOutputDim("Boxes", framework::make_ddim(dim_boxes));

    std::vector<int64_t> dim_scores({dim_x[0], box_num, class_num});
    ctx->SetOutputDim("Scores", framework::make_ddim(dim_scores));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(ctx.Input<Tensor>("X")->type(),
                                   ctx.GetPlace());
  }
};

class YoloBoxOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "The input tensor of YoloBox operator, "
             "This is a 4-D tensor with shape of [N, C, H, W]."
             "H and W should be same, and the second dimention(C) stores"
             "box locations, confidence score and classification one-hot"
             "keys of each anchor box. Generally, X should be the output"
             "of YOLOv3 network.");
    AddInput("ImgSize",
             "The image size tensor of YoloBox operator, "
             "This is a 2-D tensor with shape of [N, 2]. This tensor holds"
             "height and width of each input image using for resize output"
             "box in input image scale.");
    AddOutput("Boxes",
              "The output tensor of detection boxes of YoloBox operator, "
              "This is a 3-D tensor with shape of [N, M, 4], N is the"
              "batch num, M is output box number, and the 3rd dimention"
              "stores [xmin, ymin, xmax, ymax] coordinates of boxes.");
    AddOutput("Scores",
              "The output tensor ofdetection boxes scores of YoloBox"
              "operator, This is a 3-D tensor with shape of [N, M, C],"
              "N is the batch num, M is output box number, C is the"
              "class number.");

    AddAttr<int>("class_num", "The number of classes to predict.");
    AddAttr<std::vector<int>>("anchors",
                              "The anchor width and height, "
                              "it will be parsed pair by pair.")
        .SetDefault(std::vector<int>{});
    AddAttr<int>("downsample_ratio",
                 "The downsample ratio from network input to YoloBox operator "
                 "input, so 32, 16, 8 should be set for the first, second, "
                 "and thrid YoloBox operators.")
        .SetDefault(32);
    AddAttr<float>("conf_thresh",
                   "The confidence scores threshold of detection boxes."
                   "boxes with confidence scores under threshold should"
                   "be ignored.")
        .SetDefault(0.01);
    AddComment(R"DOC(
         This operator generate YOLO detection boxes fron output of YOLOv3 network.
         
         The output of previous network is in shape [N, C, H, W], while H and W
         should be the same, specify the grid size, each grid point predict given
         number boxes, this given number is specified by anchors, it should be 
         half anchors length, which following will be represented as S. In the 
         second dimention(the channel dimention), C should be S * (class_num + 5),
         class_num is the box categoriy number of source dataset(such as coco), 
         so in the second dimention, stores 4 box location coordinates x, y, w, h 
         and confidence score of the box and class one-hot key of each anchor box.

         While the 4 location coordinates if $$tx, ty, tw, th$$, the box predictions
         correspnd to:

         $$
         b_x = \sigma(t_x) + c_x
         b_y = \sigma(t_y) + c_y
         b_w = p_w e^{t_w}
         b_h = p_h e^{t_h}
         $$

         While $$c_x, c_y$$ is the left top corner of current grid and $$p_w, p_h$$
         is specified by anchors.

         The logistic scores of the 5rd channel of each anchor prediction boxes
         represent the confidence score of each prediction scores, and the logistic
         scores of the last class_num channels of each anchor prediction boxes 
         represent the classifcation scores. Boxes with confidence scores less then
         conf_thresh should be ignored, and boxes final scores if the products result
         of confidence scores and classification scores.

         )DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(yolo_box, ops::YoloBoxOp, ops::YoloBoxOpMaker,
                  paddle::framework::EmptyGradOpMaker);
REGISTER_OP_CPU_KERNEL(yolo_box, ops::YoloBoxKernel<float>,
                       ops::YoloBoxKernel<double>);
