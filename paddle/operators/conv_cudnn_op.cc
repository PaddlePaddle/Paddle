/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/operators/cudnn_conv_op.h"

namespace paddle {
namespace operators {

using framework::Tensor;

inline int OutputSize(int input_size, int filter_size, int padding,
                      int stride) {
  int output_size = (input_size - filter_size + 2 * padding) / stride + 1;
  return output_size;
}

class CudnnConvOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(const framework::InferShapeContext &ctx) const override {
    auto in = ctx.Input<Tensor>("Input");
    auto filter = ctx.Input<Tensor>("Filter");
    auto out = ctx.Output<framework::Tensor>("Output");
    std::vector<int> strides = Attr<std::vector<int>>("strides");
    std::vector<int> paddings = Attr<std::vector<int>>("paddings");
    int groups = Attr<int>("groups");
    int input_channels = in->dims()[1];
    int output_channels = filter->dims()[0];

    PADDLE_ENFORCE_EQ(in->dims().size(), 4,
                      "CudnnConvOp intput should be 4-D tensor.");
    PADDLE_ENFORCE_EQ(filter->dims().size(), 4,
                      "CudnnConvOp filter should be 4-D tensor.");

    PADDLE_ENFORCE_EQ(input_channels, filter->dims()[1] * groups,
                      "The number of input channels should be equal to filter "
                      "channels * groups.");
    PADDLE_ENFORCE_EQ(
        output_channels % groups, 0,
        "The number of output channels should be divided by groups.");

    auto output_height =
        OutputSize(in->dims()[2], filter->dims()[2], paddings[0], strides[0]);
    auto output_width =
        OutputSize(in->dims()[3], filter->dims()[3], paddings[1], strides[1]);
    out->Resize(
        {in->dims()[0], filter->dims()[0], output_height, output_width});
  }
};

class CudnnConvOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  CudnnConvOpMaker(framework::OpProto *proto,
                   framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("Input", "A 4-D tensor in NCHW order.");
    AddInput("Filter", "The conv kernel");
    AddOutput("Output", "");

    AddAttr<std::vector<int>>("dilations", "").SetDefault(std::vector<int>{});
    AddAttr<std::vector<int>>("strides", "").SetDefault(std::vector<int>{});
    AddAttr<std::vector<int>>("paddings", "paddings of convolution operator.")
        .SetDefault(std::vector<int>{});
    // FIXME(typhoonzero): cudnn doesn't support "group" Attributes.
    AddAttr<int>("groups", "").SetDefault(1);

    AddComment(R"DOC()DOC");
  }
};

class CudnnConvGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(const framework::InferShapeContext &ctx) const override {
    auto in = ctx.Input<Tensor>("Input");
    auto filter = ctx.Input<Tensor>("Filter");
    auto d_in = ctx.Output<framework::Tensor>(framework::GradVarName("Input"));
    auto d_filter =
        ctx.Output<framework::Tensor>(framework::GradVarName("Filter"));
    if (d_in) d_in->Resize(in->dims());
    if (d_filter) d_filter->Resize(filter->dims());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP(cudnn_conv, ops::CudnnConvOp, ops::CudnnConvOpMaker,
            cudnn_conv_grad, ops::CudnnConvGradOp);
REGISTER_OP_CPU_KERNEL(cudnn_conv,
                       ops::CudnnConvKernel<paddle::platform::CPUPlace, float>);
REGISTER_OP_CPU_KERNEL(
    cudnn_conv_grad,
    ops::CudnnConvGradKernel<paddle::platform::CPUPlace, float>);
