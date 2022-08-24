/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

class YoloBoxPostOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const {
    OP_INOUT_CHECK(ctx->HasInput("Boxes0"), "Input", "Boxes0", "yolo_box_post");
    OP_INOUT_CHECK(ctx->HasInput("Boxes1"), "Input", "Boxes1", "yolo_box_post");
    OP_INOUT_CHECK(ctx->HasInput("Boxes2"), "Input", "Boxes2", "yolo_box_post");
    OP_INOUT_CHECK(
        ctx->HasInput("ImageShape"), "Input", "ImageShape", "yolo_box_post");
    OP_INOUT_CHECK(
        ctx->HasInput("ImageScale"), "Input", "ImageScale", "yolo_box_post");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "yolo_box_post");
    OP_INOUT_CHECK(
        ctx->HasOutput("NmsRoisNum"), "Output", "NmsRoisNum", "yolo_box_post");
  }
};

class YoloBoxPostOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddInput("Boxes0", "The Boxes0 tensor");
    AddInput("Boxes1", "The Boxes1 tensor");
    AddInput("Boxes2", "The Boxes2 tensor");
    AddInput("ImageShape", "The height and width of each input image.");
    AddInput("ImageScale", "The scale factor of ImageShape.");
    AddAttr<std::vector<int>>("anchors0", "The anchors of Boxes0.");
    AddAttr<std::vector<int>>("anchors1", "The anchors of Boxes1.");
    AddAttr<std::vector<int>>("anchors2", "The anchors of Boxes2.");
    AddAttr<int>("class_num", "The number of classes to predict.");
    AddAttr<float>("conf_thresh",
                   "The confidence scores threshold of detection boxes. "
                   "Boxes with confidence scores under threshold should "
                   "be ignored.");
    AddAttr<int>("downsample_ratio0", "The downsample ratio of Boxes0.");
    AddAttr<int>("downsample_ratio1", "The downsample ratio of Boxes1.");
    AddAttr<int>("downsample_ratio2", "The downsample ratio of Boxes2.");
    AddAttr<bool>("clip_bbox",
                  "Whether clip output bonding box in Input(ImgSize) "
                  "boundary. Default true.");
    AddAttr<float>("scale_x_y",
                   "Scale the center point of decoded bounding "
                   "box. Default 1.0");
    AddAttr<float>("nms_threshold", "The threshold to be used in NMS.");
    AddOutput("Out", "The output tensor");
    AddOutput("NmsRoisNum", "The output RoIs tensor");
    AddComment(R"DOC(
        yolo_box_post Operator.
        )DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(yolo_box_post, ops::YoloBoxPostOp, ops::YoloBoxPostOpMaker);
