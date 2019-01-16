/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/detection/box_clip_op.h"
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

class BoxClipOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("InputBox"),
                   "Input(InputBox) of BoxClipOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("ImInfo"),
                   "Input(ImInfo) of BoxClipOp should not be null.");

    auto input_box_dims = ctx->GetInputDim("InputBox");
    auto im_info_dims = ctx->GetInputDim("ImInfo");

    if (ctx->IsRuntime()) {
      auto input_box_size = input_box_dims.size();
      PADDLE_ENFORCE_EQ(input_box_dims[input_box_size - 1], 4,
                        "The last dimension of InputBox must be 4");
      PADDLE_ENFORCE_EQ(im_info_dims.size(), 2,
                        "The rank of Input(InputBox) in BoxClipOp must be 2");
      PADDLE_ENFORCE_EQ(im_info_dims[1], 2,
                        "The last dimension of ImInfo must be 2");
    }
    ctx->ShareDim("InputBox", /*->*/ "OutputBox");
    ctx->ShareLoD("InputBox", /*->*/ "OutputBox");
  }
};

class BoxClipOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("InputBox",
             "(LoDTensor) "
             "InputBox is a LoDTensor with shape [..., 4] holds 4 points"
             "in last dimension in format [xmin, ymin, xmax, ymax]");
    AddInput("ImInfo",
             "(Tensor) Information for image reshape is in shape (N, 2), "
             "in format (height, width)");
    AddOutput("OutputBox",
              "(LoDTensor) "
              "OutputBox is a LoDTensor with the same shape as InputBox"
              "and it is the result after clip");
    AddComment(R"DOC(
  This operator clips input boxes to original input images.
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(box_clip, ops::BoxClipOp, ops::BoxClipOpMaker,
                  paddle::framework::EmptyGradOpMaker);
REGISTER_OP_CPU_KERNEL(
    box_clip, ops::BoxClipKernel<paddle::platform::CPUDeviceContext, float>,
    ops::BoxClipKernel<paddle::platform::CPUDeviceContext, double>);
