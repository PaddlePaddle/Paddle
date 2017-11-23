/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 *     Unless required by applicable law or agreed to in writing, software
 *     distributed under the License is distributed on an "AS IS" BASIS,
 *     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *     See the License for the specific language governing permissions and
 *     limitations under the License. */

#include "paddle/operators/unpool_op.h"
namespace paddle {
namespace operators {

class Unpool2dOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  Unpool2dOpMaker(framework::OpProto* proto,
                  framework::OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X",
        "(Tensor) The input tensor of unpool operator. "
        "The format of input tensor is NCHW. Where N is batch size, C is the "
        "number of channels, H and W is the height and width of feature.");
    AddInput("Y",
        "(Tensor) The input tensor of the indices given out by MaxPool2d. "
        "The format of input tensor is NCHW. Where N is batch size, C is the "
        "number of channels, H and W is the height and width of feature.");
    AddOutput("Out",
        "(Tensor) The output tensor of unpool operator."
        "The format of output tensor is also NCHW."
        "Where N is batch size, C is "
        "the number of channels, H and W is the height and "
        "width of feature.");
    AddAttr<std::vector<int>>("ksize",
        "(vector), the unpooling window size(height, width) "
        "of unpooling operator.");
    AddAttr<std::vector<int>>("strides",
        "(vector, default:{1, 1}), "
        "strides (height, width) of unpooling operator.")
        .SetDefault({1, 1});
    AddAttr<std::vector<int>>("paddings",
        "(vector defalut:{0,0}), "
        "paddings (height, width) of unpooling operator.")
        .SetDefault({0, 0});
    AddAttr<std::string>("unpoolingtype",
        "(string), unpooling type, can be \"max\" for max-unpooling ")
        .InEnum({"max"});
    AddComment(R"DOC(
          "input: the input Tensor to invert
          indices: the indices given out by MaxPool2d
          ksize  – Size of the max pooling window.
          stride – Stride of the max pooling window.
                   "It is set to kernel_size by default.
          padding – Padding that was added to the input"
        )DOC");
  }
};

int OutputSize(int input_size, int ksize, int padding, int stride) {
  int output_size = (input_size -1) * stride - 2 * padding + ksize;
  return output_size;
}

class UnpoolOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) of UnpoolOp"
                   "should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Y"), "Input(Y) of UnpoolOp"
                   "should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of UnpoolOp should not be null.");

    auto in_x_dims = ctx->GetInputDim("X");
    auto in_y_dims = ctx->GetInputDim("Y");
    std::string unpoolingtype =
      ctx->Attrs().Get<std::string>("unpoolingtype");
    std::vector<int> ksize = ctx->Attrs().Get<std::vector<int>>("ksize");
    std::vector<int> strides = ctx->Attrs().Get<std::vector<int>>("strides");
    std::vector<int> paddings = ctx->Attrs().Get<std::vector<int>>("paddings");

    PADDLE_ENFORCE(in_x_dims.size() == 4,
                    "Unpooling intput must be of 4-dimensional.");
    for (int i = 0; i < 4; ++i) {
      PADDLE_ENFORCE(in_x_dims[i] == in_y_dims[i],
                     "X size must be eq Y size!");
    }


    std::vector<int64_t> output_shape({in_x_dims[0], in_x_dims[1]});
    for (size_t i = 0; i < ksize.size(); ++i) {
      output_shape.push_back(
        OutputSize(in_x_dims[i + 2], ksize[i], paddings[i], strides[i]));
    }
    ctx->SetOutputDim("Out", framework::make_ddim(output_shape));
  }
};

class UnpoolOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) must not be null.");
    PADDLE_ENFORCE(ctx->HasOutput(framework::GradVarName("X")),
                                  "Input(X@GRAD) should not be null.");
    ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("X"));
  }
};
}    // namespace operators
}    // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP(unpool, ops::UnpoolOp, ops::Unpool2dOpMaker, unpool_grad,
            ops::UnpoolOpGrad);
REGISTER_OP_CPU_KERNEL(unpool,
                       ops::UnpoolKernel<paddle::platform::CPUPlace, float>,
                       ops::UnpoolKernel<paddle::platform::CPUPlace, double>);
REGISTER_OP_CPU_KERNEL(unpool_grad,
                    ops::UnpoolGradKernel<paddle::platform::CPUPlace, float>,
                    ops::UnpoolGradKernel<paddle::platform::CPUPlace, double>);

