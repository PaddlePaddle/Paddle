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

#include "paddle/fluid/operators/yolov3_loss_op.h"
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

using framework::Tensor;

class Yolov3LossOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of Yolov3LossOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("GTBox"),
                   "Input(GTBox) of Yolov3LossOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Loss"),
                   "Output(Loss) of Yolov3LossOp should not be null.");

    auto dim_x = ctx->GetInputDim("X");
    auto dim_gt = ctx->GetInputDim("GTBox");
    auto img_height = ctx->Attrs().Get<int>("img_height");
    auto anchors = ctx->Attrs().Get<std::vector<int>>("anchors");
    auto box_num = ctx->Attrs().Get<int>("box_num");
    auto class_num = ctx->Attrs().Get<int>("class_num");
    PADDLE_ENFORCE_EQ(dim_x.size(), 4, "Input(X) should be a 4-D tensor.");
    PADDLE_ENFORCE_EQ(dim_x[2], dim_x[3],
                      "Input(X) dim[3] and dim[4] should be euqal.");
    PADDLE_ENFORCE_EQ(dim_x[1], anchors.size() / 2 * (5 + class_num),
                      "Input(X) dim[1] should be equal to (anchor_number * (5 "
                      "+ class_num)).");
    PADDLE_ENFORCE_EQ(dim_gt.size(), 3, "Input(GTBox) should be a 3-D tensor");
    PADDLE_ENFORCE_EQ(dim_gt[2], 5, "Input(GTBox) dim[2] should be 5");
    PADDLE_ENFORCE_GT(img_height, 0,
                      "Attr(img_height) value should be greater then 0");
    PADDLE_ENFORCE_GT(anchors.size(), 0,
                      "Attr(anchors) length should be greater then 0.");
    PADDLE_ENFORCE_EQ(anchors.size() % 2, 0,
                      "Attr(anchors) length should be even integer.");
    PADDLE_ENFORCE_GT(box_num, 0,
                      "Attr(box_num) should be an integer greater then 0.");
    PADDLE_ENFORCE_GT(class_num, 0,
                      "Attr(class_num) should be an integer greater then 0.");

    std::vector<int64_t> dim_out({1});
    ctx->SetOutputDim("Loss", framework::make_ddim(dim_out));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        framework::ToDataType(ctx.Input<Tensor>("X")->type()), ctx.GetPlace());
  }
};

class Yolov3LossOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "The input tensor of bilinear interpolation, "
             "This is a 4-D tensor with shape of [N, C, H, W]");
    AddInput(
        "GTBox",
        "The input tensor of ground truth boxes, "
        "This is a 3-D tensor with shape of [N, max_box_num, 5 + class_num], "
        "max_box_num is the max number of boxes in each image, "
        "class_num is the number of classes in data set. "
        "In the third dimention, stores x, y, w, h, confidence, classes "
        "one-hot key. "
        "x, y is the center cordinate of boxes and w, h is the width and "
        "height, "
        "and all of them should be divided by input image height to scale to "
        "[0, 1].");
    AddOutput("Loss",
              "The output yolov3 loss tensor, "
              "This is a 1-D tensor with shape of [1]");

    AddAttr<int>("box_num", "The number of boxes generated in each grid.");
    AddAttr<int>("class_num", "The number of classes to predict.");
    AddAttr<std::vector<int>>("anchors",
                              "The anchor width and height, "
                              "it will be parsed pair by pair.");
    AddAttr<int>("img_height",
                 "The input image height after crop of yolov3 network.");
    AddAttr<float>("ignore_thresh",
                   "The ignore threshold to ignore confidence loss.");
    AddComment(R"DOC(
         This operator generate yolov3 loss by given predict result and ground
         truth boxes.
         )DOC");
  }
};

class Yolov3LossOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) should not be null");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Loss")),
                   "Input(Loss@GRAD) should not be null");
    auto dim_x = ctx->GetInputDim("X");
    if (ctx->HasOutput(framework::GradVarName("X"))) {
      ctx->SetOutputDim(framework::GradVarName("X"), dim_x);
    }
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        framework::ToDataType(ctx.Input<Tensor>("X")->type()), ctx.GetPlace());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(yolov3_loss, ops::Yolov3LossOp, ops::Yolov3LossOpMaker,
                  paddle::framework::DefaultGradOpDescMaker<true>);
REGISTER_OPERATOR(yolov3_loss_grad, ops::Yolov3LossOpGrad);
REGISTER_OP_CPU_KERNEL(
    yolov3_loss,
    ops::Yolov3LossKernel<paddle::platform::CPUDeviceContext, float>);
REGISTER_OP_CPU_KERNEL(
    yolov3_loss_grad,
    ops::Yolov3LossGradKernel<paddle::platform::CPUDeviceContext, float>);
