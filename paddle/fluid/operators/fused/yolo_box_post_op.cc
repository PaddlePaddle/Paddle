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
    OP_INOUT_CHECK(ctx->HasInput("Box0"), "Input", "Box0", "yolo_box_post");
    OP_INOUT_CHECK(ctx->HasInput("Box1"), "Input", "Box1", "yolo_box_post");
    OP_INOUT_CHECK(ctx->HasInput("Box2"), "Input", "Box2", "yolo_box_post");
    OP_INOUT_CHECK(ctx->HasInput("ImageShape"), "Input", "ImageShape",
                   "yolo_box_post");
    OP_INOUT_CHECK(ctx->HasInput("ImageScale"), "Input", "ImageScale",
                   "yolo_box_post");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "yolo_box_post");
    OP_INOUT_CHECK(ctx->HasOutput("NmsRoisNum"), "Output", "NmsRoisNum",
                   "yolo_box_post");
    ctx->SetOutputDim("Out", ctx->GetInputDim("X"));
  }
};

class YoloBoxPostOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddInput("Box0", "The Box0 tensor");
    AddInput("Box1", "The Box1 tensor");
    AddInput("Box2", "The Box2 tensor");
    AddInput("ImageShape", "The height and width of each input image.");
    AddInput("ImageScale", "The scale factor of ImageShape.");
    AddAttr<std::vector<int>>("anchors0", "The anchors of Box0.");
    AddAttr<std::vector<int>>("anchors1", "The anchors of Box1.");
    AddAttr<std::vector<int>>("anchors2", "The anchors of Box2.");
    AddAttr<int>("class_num", "The number of classes to predict.");
    AddAttr<float>("conf_thresh",
                   "The confidence scores threshold of detection boxes. "
                   "Boxes with confidence scores under threshold should "
                   "be ignored.");
    AddAttr<int>("downsample_ratio0", "The downsample ratio of Box0.");
    AddAttr<int>("downsample_ratio1", "The downsample ratio of Box1.");
    AddAttr<int>("downsample_ratio2", "The downsample ratio of Box2.");
    AddAttr<bool>("clip_bbox",
                  "Whether clip output bonding box in Input(ImgSize) "
                  "boundary. Default true.");
    AddAttr<float>("scale_x_y",
                   "Scale the center point of decoded bounding "
                   "box. Default 1.0");
    AddAttr<float>("nms_threshold", "The threshold to be used in NMS.");
    AddOutput("Out", "The output tensor");
    AddOutput("NmsRoisNum", "The output RoIs tensor");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(yolo_box_post, ops::YoloBoxPostOp, ops::YoloBoxPostOpMaker);
