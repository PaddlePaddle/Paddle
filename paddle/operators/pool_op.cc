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

#include "paddle/operators/pool_op.h"

namespace paddle {
namespace operators {

int OutputSizePool(int input_size, int filter_size, int padding, int stride) {
  int output_size = (input_size - filter_size + 2 * padding) / stride + 1;
  return output_size;
}

void PoolOp::InferShape(framework::InferShapeContext *ctx) const {
  PADDLE_ENFORCE(ctx->HasInput("X"), "X(Input) of Pooling should not be null.");
  PADDLE_ENFORCE(ctx->HasOutput("Out"),
                 "Out(Output) of Pooling should not be null.");

  auto in_x_dims = ctx->GetInputDim("X");

  std::string pooling_type = ctx->Attrs().Get<std::string>("poolingType");
  std::vector<int> ksize = ctx->Attrs().Get<std::vector<int>>("ksize");
  std::vector<int> strides = ctx->Attrs().Get<std::vector<int>>("strides");
  std::vector<int> paddings = ctx->Attrs().Get<std::vector<int>>("paddings");

  PADDLE_ENFORCE(in_x_dims.size() == 4 || in_x_dims.size() == 5,
                 "Pooling intput should be 4-D or 5-D tensor.");

  if (ctx->Attrs().Get<bool>("globalPooling")) {
    ksize.resize(static_cast<size_t>(in_x_dims.size()) - 2);
    for (size_t i = 0; i < ksize.size(); ++i)
      ksize[i] = static_cast<int>(in_x_dims[i + 2]);
  }

  PADDLE_ENFORCE(in_x_dims.size() - ksize.size() == 2U,
                 "Input size and pooling size should be consistent.");
  PADDLE_ENFORCE_EQ(ksize.size(), strides.size(),
                    "Strides size and pooling size should be the same.");
  PADDLE_ENFORCE_EQ(ksize.size(), paddings.size(),
                    "Paddings size and pooling size should be the same.");

  std::vector<int64_t> output_shape({in_x_dims[0], in_x_dims[1]});
  for (size_t i = 0; i < ksize.size(); ++i) {
    output_shape.push_back(
        OutputSizePool(in_x_dims[i + 2], ksize[i], paddings[i], strides[i]));
  }
  ctx->SetOutputDim("Out", framework::make_ddim(output_shape));
}

void PoolOpGrad::InferShape(framework::InferShapeContext *ctx) const {
  PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) must not be null.");
  PADDLE_ENFORCE(ctx->HasOutput(framework::GradVarName("X")),
                 "Input(X@GRAD) should not be null.");
  ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("X"));
}

Pool2dOpMaker::Pool2dOpMaker(framework::OpProto *proto,
                             framework::OpAttrChecker *op_checker)
    : OpProtoAndCheckerMaker(proto, op_checker) {
  AddInput(
      "X",
      "(Tensor) The input tensor of pooling operator. "
      "The format of input tensor is NCHW. Where N is batch size, C is the "
      "number of channels, H and W is the height and width of feature.");
  AddOutput("Out",
            "(Tensor) The output tensor of pooling operator."
            "The format of output tensor is also NCHW."
            "Where N is batch size, C is "
            "the number of channels, H and W is the height and "
            "width of feature.");

  AddAttr<std::string>("poolingType",
                       "(string), poolingType of pooling operator."
                       "Str constant equal to 'max' or 'avg'.")
      .InEnum({"max", "avg"});
  AddAttr<std::vector<int>>(
      "ksize",
      "(vector ), the pooling window size(height, width) of pooling operator."
      "If globalPooling = true, ksize is ignored and need not be "
      "specified.");  // TODO(Chengduo): Add checker. (Currently,
  // TypedAttrChecker don't support vector type.)
  AddAttr<bool>(
      "globalPooling",
      "(bool default: false), whether to use the global pooling."
      "If globalPooling = true, ksize is ignored and need not be specified.")
      .SetDefault(false);
  AddAttr<std::vector<int>>(
      "strides",
      "(vector, default:{1, 1}), strides(height, width) of pooling operator.")
      .SetDefault({1, 1});  // TODO(Chengduo): Add checker. (Currently,
  // TypedAttrChecker don't support vector type.)
  AddAttr<std::vector<int>>(
      "paddings",
      "(vector defalut:{0,0}), paddings(height, width) of pooling operator.")
      .SetDefault({0, 0});  // TODO(Chengduo): Add checker. (Currently,
  // TypedAttrChecker don't support vector type.)

  AddComment(R"DOC(
The pooling2d operation calculates the output based on
the input, poolingType and ksize, strides, paddings parameters.
Input(X) and output(Out) are in NCHW format. Where N is batch size, C is the
number of channels, H and W is the height and width of feature.
Parameters(ksize, strides, paddings) are two elements.
These two elements represent height and width, respectively.
The input(X) size and output(Out) size may be different.

Example:
  Input:
       X shape: (N, C, H_in, W_in)
  Output:
       Out shape: (N, C, H_out, W_out)
  where
       H_out = (H_in - ksize[0] + 2 * paddings[0]) / strides[0] + 1;
       W_out = (W_in - ksize[1] + 2 * paddings[1]) / strides[1] + 1;
)DOC");
}

Pool3dOpMaker::Pool3dOpMaker(framework::OpProto *proto,
                             framework::OpAttrChecker *op_checker)
    : OpProtoAndCheckerMaker(proto, op_checker) {
  AddInput(
      "X",
      "(Tensor) The input tensor of pooling operator. "
      "The format of input tensor is NCDHW. Where N is batch size, C is "
      "the number of channels, D, H and W is the depth, height and width of "
      "feature.");
  AddOutput("Out",
            "(Tensor) The output tensor of pooling operator."
            "The format of output tensor is also NCDHW."
            "Where N is batch size, C is "
            "the number of channels, D, H and W is the depth, height and "
            "width of feature.");

  AddAttr<std::string>("poolingType",
                       "(string), poolingType of pooling operator."
                       "Str constant equal to 'max' or 'avg'.")
      .InEnum({"max", "avg"});
  AddAttr<std::vector<int>>(
      "ksize",
      "(vector ), the pooling window size(depth, height, width) of pooling "
      "operator."
      "If globalPooling = true, ksize is ignored and need not be "
      "specified.");  // TODO(Chengduo): Add checker. (Currently,
                      // TypedAttrChecker don't support vector type.)
  AddAttr<bool>(
      "globalPooling",
      "(bool default: false), whether to use the global pooling."
      "If globalPooling = true, ksize is ignored and need not be specified.")
      .SetDefault(false);
  AddAttr<std::vector<int>>("strides",
                            "(vector, default:{1,1,1}), strides(depth, height, "
                            "width) of pooling operator.")
      .SetDefault({1, 1, 1});  // TODO(Chengduo): Add checker. (Currently,
                               // TypedAttrChecker don't support vector type.)
  AddAttr<std::vector<int>>("paddings",
                            "(vector defalut:{0,0,0}), paddings(depth, height, "
                            "width) of pooling operator.")
      .SetDefault({0, 0, 0});  // TODO(Chengduo): Add checker. (Currently,
                               // TypedAttrChecker don't support vector type.)

  AddComment(R"DOC(
The pooling3d operation calculates the output based on
the input, poolingType and ksize, strides, paddings parameters.
Input(X) and output(Out) are in NCDHW format. Where N is batch
size, C is the number of channels, D, H and W is the depth, height and
width of feature. Parameters(ksize, strides, paddings) are three elements.
These three elements represent depth, height and width, respectively.
The input(X) size and output(Out) size may be different.

Example:
  Input:
       X shape: (N, C, D_in, H_in, W_in)
  Output:
       Out shape: (N, C, D_out, H_out, W_out)
  where
       D_out = (D_in - ksize[0] + 2 * paddings[0]) / strides[0] + 1;
       H_out = (H_in - ksize[1] + 2 * paddings[1]) / strides[1] + 1;
       W_out = (W_in - ksize[2] + 2 * paddings[2]) / strides[2] + 1;
)DOC");
}
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP(pool2d, ops::PoolOp, ops::Pool2dOpMaker, pool2d_grad,
            ops::PoolOpGrad);

REGISTER_OP_CPU_KERNEL(pool2d,
                       ops::PoolKernel<paddle::platform::CPUPlace, float>);
REGISTER_OP_CPU_KERNEL(pool2d_grad,
                       ops::PoolGradKernel<paddle::platform::CPUPlace, float>)

REGISTER_OP(pool3d, ops::PoolOp, ops::Pool3dOpMaker, pool3d_grad,
            ops::PoolOpGrad);

REGISTER_OP_CPU_KERNEL(pool3d,
                       ops::PoolKernel<paddle::platform::CPUPlace, float>);
REGISTER_OP_CPU_KERNEL(pool3d_grad,
                       ops::PoolGradKernel<paddle::platform::CPUPlace, float>);
