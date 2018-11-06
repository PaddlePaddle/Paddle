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
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of Yolov3LossOp should not be null.");

    // PADDLE_ENFORCE(ctx->HasAttr("img_height"),
    //                "Attr(img_height) of Yolov3LossOp should not be null. ");
    // PADDLE_ENFORCE(ctx->HasAttr("anchors"),
    //                "Attr(anchor) of Yolov3LossOp should not be null.")
    // PADDLE_ENFORCE(ctx->HasAttr("class_num"),
    //                "Attr(class_num) of Yolov3LossOp should not be null.");
    // PADDLE_ENFORCE(ctx->HasAttr(
    //     "ignore_thresh",
    //     "Attr(ignore_thresh) of Yolov3LossOp should not be null."));

    auto dim_x = ctx->GetInputDim("X");
    auto dim_gt = ctx->GetInputDim("GTBox");
    auto img_height = ctx->Attrs().Get<int>("img_height");
    auto anchors = ctx->Attrs().Get<std::vector<int>>("anchors");
    auto box_num = ctx->Attrs().Get<int>("box_num");
    auto class_num = ctx->Attrs().Get<int>("class_num");
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
    PADDLE_ENFORCE_EQ(dim_x[1], anchors.size() / 2 * (5 + class_num),
                      "Input(X) dim[1] should be equal to (anchor_number * (5 "
                      "+ class_num)).");
    PADDLE_ENFORCE_EQ(dim_gt.size(), 3, "Input(GTBox) should be a 3-D tensor");
    PADDLE_ENFORCE_EQ(dim_gt[2], 5, "Input(GTBox) dim[2] should be 5");

    std::vector<int64_t> dim_out({dim_x[0], 1});
    ctx->SetOutputDim("Out", framework::make_ddim(dim_out));
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
    AddOutput("Out",
              "The output yolo loss tensor, "
              "This is a 2-D tensor with shape of [N, 1]");

    AddAttr<int>("box_num", "The number of boxes generated in each grid.");
    AddAttr<int>("class_num", "The number of classes to predict.");
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
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "Input(Out@GRAD) should not be null");
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
