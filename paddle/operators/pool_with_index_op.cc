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

#include "paddle/operators/pool_with_index_op.h"

namespace paddle {
namespace operators {

inline int OutputSizeMaxPool(int input_size, int filter_size, int padding,
                             int stride) {
  int output_size = (input_size - filter_size + 2 * padding) / stride + 1;
  return output_size;
}

class MaxPoolWithIndexOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "X(Input) of Pooling should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Out(Output) of Pooling should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Mask"),
                   "Mask(Output) of Pooling should not be null.");

    auto in_x_dims = ctx->GetInputDim("X");

    std::vector<int> ksize = ctx->Attrs().Get<std::vector<int>>("ksize");
    std::vector<int> strides = ctx->Attrs().Get<std::vector<int>>("strides");
    std::vector<int> paddings = ctx->Attrs().Get<std::vector<int>>("paddings");

    PADDLE_ENFORCE(in_x_dims.size() == 4 || in_x_dims.size() == 5,
                   "Pooling intput should be 4-D or 5-D");

    if (ctx->Attrs().Get<bool>("globalPooling")) {
      ksize.resize(static_cast<size_t>(in_x_dims.size()) - 2);
      for (size_t i = 0; i < ksize.size(); ++i)
        ksize[i] = static_cast<int>(in_x_dims[i + 2]);
    }

    PADDLE_ENFORCE(in_x_dims.size() - ksize.size() == 2U,
                   "Intput size and pooling size should be consistent.");
    PADDLE_ENFORCE_EQ(ksize.size(), strides.size(),
                      "Strides size and pooling size should be the same.");
    PADDLE_ENFORCE_EQ(ksize.size(), paddings.size(),
                      "Paddings size and pooling size should be the same.");

    std::vector<int64_t> output_shape({in_x_dims[0], in_x_dims[1]});
    for (size_t i = 0; i < ksize.size(); ++i) {
      output_shape.push_back(OutputSizeMaxPool(in_x_dims[i + 2], ksize[i],
                                               paddings[i], strides[i]));
    }
    ctx->SetOutputDim("Out", framework::make_ddim(output_shape));
    ctx->SetOutputDim("Mask", framework::make_ddim(output_shape));
  }
};

class MaxPoolWithIndexOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) must not be null.");
    PADDLE_ENFORCE(ctx->HasOutput(framework::GradVarName("X")),
                   "Input(X@GRAD) should not be null.");
    ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("X"));
  }
};

class MaxPool2dWithIndexOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  MaxPool2dWithIndexOpMaker(framework::OpProto *proto,
                            framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput(
        "X",
        "The input tensor of pooling operator. "
        "The format of input tensor is NCHW. Where N is batch size, C is the "
        "number of channels, H and W is the height and width of image.");
    AddOutput("Out",
              "The output tensor of pooling operator."
              "The format of output tensor is also NCHW."
              "Where N is batch size, C is "
              "the number of channels, H and W is the height and "
              "width of image.");
    AddOutput("Mask",
              "The Mask tensor of pooling operator."
              "The format of output tensor is also NCHW."
              "Where N is batch size, C is the number of channels, H and W "
              "is the height and width of image."
              "The value in it is the index in current feature map");

    AddAttr<std::vector<int>>(
        "ksize",
        "The pooling size(height, width) of pooling operator."
        "If globalPooling = true, ksize is ignored and need not be "
        "specified.");  // TODO(Chengduo): Add checker. (Currently,
                        // TypedAttrChecker don't support vector type.)
    AddAttr<bool>(
        "globalPooling",
        "Whether to use the globalPooling."
        "Bool constant equal to false or true."
        "Default false."
        "If globalPooling = true, ksize is ignored and need not be specified.")
        .SetDefault(false);
    AddAttr<std::vector<int>>("strides",
                              "Strides(height, width) of pooling operator."
                              "Default {1,1}.")
        .SetDefault({1, 1});  // TODO(Chengduo): Add checker. (Currently,
                              // TypedAttrChecker don't support vector type.)
    AddAttr<std::vector<int>>("paddings",
                              "Paddings(height, width) of pooling operator."
                              "Default {0,0}.")
        .SetDefault({0, 0});  // TODO(Chengduo): Add checker. (Currently,
                              // TypedAttrChecker don't support vector type.)

    AddComment(R"DOC(
The maxPooling2d with index operation calculates the output and the mask
based on the input and ksize, strides, paddings parameters. Input(X) and
output(Out, Mask) are in NCHW format. Where N is batch size, C is the
number of channels, H and W is the height and width of feature.
Parameters(ksize, strides, paddings) are two elements.
These two elements represent height and width, respectively.
)DOC");
  }
};

class MaxPool3dWithIndexOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  MaxPool3dWithIndexOpMaker(framework::OpProto *proto,
                            framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput(
        "X",
        "The input tensor of pooling operator. "
        "The format of input tensor is NCDHW. Where N is batch size, C is "
        "the number of channels, D, H and W is the depth, height and width of "
        "image.");
    AddOutput("Out",
              "The output tensor of pooling operator."
              "The format of output tensor is also NCDHW."
              "Where N is batch size, C is "
              "the number of channels, D, H and W is the depth, height and "
              "width of image.");
    AddOutput("Mask",
              "The Mask tensor of pooling operator."
              "The format of output tensor is also NCDHW."
              "Where N is batch size, C is the number of channels, D, H and W "
              "is the depth, height and width of image."
              "The value in it is the index in current feature map");

    AddAttr<std::vector<int>>(
        "ksize",
        "The pooling size(depth, height, width) of pooling operator."
        "If globalPooling = true, ksize is ignored and need not be "
        "specified.");  // TODO(Chengduo): Add checker. (Currently,
                        // TypedAttrChecker don't support vector type.)
    AddAttr<bool>(
        "globalPooling",
        "Whether to use the globalPooling."
        "Bool constant equal to false or true."
        "Default false."
        "If globalPooling = true, ksize is ignored and need not be specified.")
        .SetDefault(false);
    AddAttr<std::vector<int>>(
        "strides",
        "Strides(depth, height, width) of pooling operator."
        "Default {1,1,1}.")
        .SetDefault({1, 1, 1});  // TODO(Chengduo): Add checker. (Currently,
                                 // TypedAttrChecker don't support vector type.)
    AddAttr<std::vector<int>>(
        "paddings",
        "Paddings(depth, height, width) of pooling operator."
        "Default {0,0,0}.")
        .SetDefault({0, 0, 0});  // TODO(Chengduo): Add checker. (Currently,
                                 // TypedAttrChecker don't support vector type.)

    AddComment(R"DOC(
The maxpooling3d with index operation calculates the output and the mask
based on the input and ksize, strides, paddings parameters.
Input(X) and output(Out, Mask) are in NCDHW format. Where N is batch
size, C is the number of channels, D, H and W is the depth, height and
width of feature. Parameters(ksize, strides, paddings) are three elements.
These three elements represent depth, height and width, respectively.
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP(max_pool2d_with_index, ops::MaxPoolWithIndexOp,
            ops::MaxPool2dWithIndexOpMaker, max_pool2d_with_index_grad,
            ops::MaxPoolWithIndexOpGrad);

REGISTER_OP_CPU_KERNEL(
    max_pool2d_with_index,
    ops::MaxPoolWithIndexKernel<paddle::platform::CPUPlace, float>);
REGISTER_OP_CPU_KERNEL(
    max_pool2d_with_index_grad,
    ops::MaxPoolWithIndexGradKernel<paddle::platform::CPUPlace, float>)

REGISTER_OP(max_pool3d_with_index, ops::MaxPoolWithIndexOp,
            ops::MaxPool3dWithIndexOpMaker, max_pool3d_with_index_grad,
            ops::MaxPoolWithIndexOpGrad);

REGISTER_OP_CPU_KERNEL(
    max_pool3d_with_index,
    ops::MaxPoolWithIndexKernel<paddle::platform::CPUPlace, float>);
REGISTER_OP_CPU_KERNEL(
    max_pool3d_with_index_grad,
    ops::MaxPoolWithIndexGradKernel<paddle::platform::CPUPlace, float>)
