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

class PoolOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "X(Input) of Pooling should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Out(Output) of Pooling should not be null.");

    auto in_x_dims = ctx->GetInputDim("X");

    std::string pooling_type = ctx->Attrs().Get<std::string>("poolingType");
    std::vector<int> ksize = ctx->Attrs().Get<std::vector<int>>("ksize");
    std::vector<int> strides = ctx->Attrs().Get<std::vector<int>>("strides");
    std::vector<int> paddings = ctx->Attrs().Get<std::vector<int>>("paddings");

    PADDLE_ENFORCE(pooling_type == "max" || pooling_type == "avg",
                   "pooling_type should be 'max' or 'avg'");
    PADDLE_ENFORCE(in_x_dims.size() == 4 || in_x_dims.size() == 5,
                   "Pooling intput should be 4-D or 5-D");

    if (ctx->Attrs().Get<bool>("globalPooling")) {
      ksize.resize(static_cast<size_t>(in_x_dims.size()) - 2);
      for (size_t i = 0; i < ksize.size(); ++i)
        ksize[i] = static_cast<int>(in_x_dims[i + 2]);
    }

    PADDLE_ENFORCE(in_x_dims.size() - ksize.size() == 2U,
                   "Input size and Pooling size should be consistent.");
    PADDLE_ENFORCE(ksize.size() == 2 || ksize.size() == 3,
                   "Pooling size should be 2 elements. or 3 elements.");
    PADDLE_ENFORCE_EQ(ksize.size(), strides.size(),
                      "strides size and pooling size should be the same.");
    PADDLE_ENFORCE_EQ(ksize.size(), paddings.size(),
                      "paddings size and pooling size should be the same.");

    std::vector<int64_t> output_shape({in_x_dims[0], in_x_dims[1]});
    for (size_t i = 0; i < ksize.size(); ++i) {
      output_shape.push_back(
          OutputSizePool(in_x_dims[i + 2], ksize[i], paddings[i], strides[i]));
    }
    ctx->SetOutputDim("Out", framework::make_ddim(output_shape));
  }
};

class PoolOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "X(Input) of Pooling should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput(framework::GradVarName("X")),
                   "Input@Grad of Pooling should not be null.");
    ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("X"));
  }
};

class Pool2dOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  Pool2dOpMaker(framework::OpProto *proto, framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput(
        "X",
        "The input tensor of pooling operator. "
        "The format of input tensor is NCHW. Where N is batch size, C is the "
        "number of channels, H and W is the height and width of feature.");
    AddOutput("Out",
              "The output tensor of pooling operator."
              "The format of output tensor is also NCHW.");

    AddAttr<std::string>("poolingType",
                         "PoolingType of pooling operator."
                         "Str constant equal to 'max' or 'avg'.")
        .InEnum({"max", "avg"});
    AddAttr<std::vector<int>>(
        "ksize",
        "Pooling size(depth, height, width) of pooling operator."
        "If globalPooling = true, ksize is ignored and need not be "
        "specified.");  // TODO(Add checker)
    AddAttr<bool>(
        "globalPooling",
        "Whether to use the globalPooling."
        "Bool constant equal to false or true."
        "Default false."
        "If globalPooling = true, ksize is ignored and need not be specified.")
        .SetDefault(false);
    AddAttr<std::vector<int>>("strides",
                              "Strides(height, width) of pooling operator."
                              "Default {1,1}")
        .SetDefault({1, 1});  // TODO(Add checker)
    AddAttr<std::vector<int>>("paddings",
                              "Paddings(height, width) of pooling operator."
                              "Default {0,0}.")
        .SetDefault({0, 0});  // TODO(Add checker)
    AddComment(R"DOC(
The pooling2d operation calculates the output based on
the input, poolingType and ksize, strides, paddings parameters.
)DOC");
  }
};

class Pool3dOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  Pool3dOpMaker(framework::OpProto *proto, framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X",
             "The input tensor of pooling operator. "
             "The format of input tensor is NCDHW. Where N is batch size, C is "
             "the "
             "number of channels, D, H and W is the depth, height and width of "
             "feature.");
    AddOutput("Out",
              "The output tensor of pooling operator."
              "The format of output tensor is also NCDHW.");

    AddAttr<std::string>("poolingType",
                         "PoolingType of pooling operator."
                         "str constant equal to 'max' or 'avg'.")
        .InEnum({"max", "avg"});
    AddAttr<std::vector<int>>(
        "ksize",
        "Pooling size(depth, height, width) of pooling operator."
        "If globalPooling = true, ksize is ignored and need not be "
        "specified.");  // TODO(Add checker)
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
        .SetDefault({1, 1, 1});  // TODO(Add checker)
    AddAttr<std::vector<int>>(
        "paddings",
        "Paddings(depth, height, width) of pooling operator."
        "Default {0,0,0}.")
        .SetDefault({0, 0, 0});  // TODO(Add checker)
    AddComment(R"DOC(
The pooling3d operation calculates the output based on
the input, poolingType and ksize, strides, paddings parameters.
)DOC");
  }
};
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
