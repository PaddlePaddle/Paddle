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

int outputSize_pool(int input_size, int filter_size, int padding, int stride) {
  int output_size = (input_size - filter_size + 2 * padding) / stride + 1;
  return output_size;
}

class PoolOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(const framework::InferShapeContext &ctx) const override {
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar("X"),
                            "X(Input) of Pooling should not be null.");
    PADDLE_ENFORCE_NOT_NULL(ctx.OutputVar("Out"),
                            "Out(Output) of Pooling should not be null.");
    //    PADDLE_ENFORCE_NOT_NULL(Attr<std::string>("poolingType"),
    //                            "pooling_type should not be null.");
    //    PADDLE_ENFORCE_NOT_NULL(Attr<std::vector<int>>("ksize"), "ksize should
    //    not be null.");
    auto in_X = ctx.Input<Tensor>("X");
    auto out = ctx.Output<framework::LoDTensor>("Out");
    int global_pooling = Attr<int>("globalPooling");
    std::string pooling_type = Attr<std::string>("poolingType");
    std::vector<int> ksize = Attr<std::vector<int>>("ksize");
    std::vector<int> strides = Attr<std::vector<int>>("strides");
    std::vector<int> paddings = Attr<std::vector<int>>("paddings");

    PADDLE_ENFORCE(pooling_type == "max" || pooling_type == "ave",
                   "pooling_type should be 'max' or 'ave'");
    PADDLE_ENFORCE(in_X->dims().size() == 4 || in_X->dims().size() == 5,
                   "Pooling intput should be 4-D or 5-D");

    if (global_pooling == 1) {
      ksize.resize(static_cast<size_t>(in_X->dims().size()) - 2);
      for (size_t i = 0; i < ksize.size(); ++i)
        ksize[i] = static_cast<int>(in_X->dims()[i + 2]);
    }

    if (ksize.size() == 2) {
      PADDLE_ENFORCE_EQ(strides.size(), 2, "Pool2DOp strides should be 2-D.");
      PADDLE_ENFORCE_EQ(paddings.size(), 2, "Pool2DOp paddings should be 2-D.");
    } else {
      PADDLE_ENFORCE_EQ(strides.size(), 3, "Pool3DOp strides should be 3-D.");
      PADDLE_ENFORCE_EQ(paddings.size(), 3, "Pool3DOp paddings should be 3-D.");
    }
    std::vector<int64_t> output_shape({in_X->dims()[0], in_X->dims()[1]});
    for (size_t i = 0; i < ksize.size(); ++i) {
      output_shape.push_back(outputSize_pool(in_X->dims()[i + 2], ksize[i],
                                             paddings[i], strides[i]));
    }
    out->Resize(framework::make_ddim(output_shape));
  }
};

class PoolOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(const framework::InferShapeContext &ctx) const override {
    auto in = ctx.Input<Tensor>("X");
    auto d_in = ctx.Output<framework::LoDTensor>(framework::GradVarName("X"));
    if (d_in) d_in->Resize(in->dims());
  }
};

class Pool3dOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  Pool3dOpMaker(framework::OpProto *proto, framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput(
        "X",
        "The input tensor of pooling operator. "
        "The format of input tensor is NCDHW. Where N is batch size, C is the "
        "number of channels, D, H and W is the depth, height and width of "
        "image.");
    AddOutput("Out",
              "The output tensor of pooling operator."
              "The format of output tensor is also NCDHW.");

    AddAttr<std::string>("poolingType",
                         "poolingType of pooling operator.['max' or 'ave']");
    AddAttr<std::vector<int>>(
        "ksize", "pooling size(depth, height, width) of pooling operator.");
    AddAttr<int>("globalPooling",
                 "default 0"
                 "whether to use the globalPooling.")
        .SetDefault(0);
    AddAttr<std::vector<int>>(
        "strides",
        "default {1,1,1}"
        "strides(depth, height, width) of pooling operator.")
        .SetDefault({1, 1, 1});
    AddAttr<std::vector<int>>(
        "paddings",
        "default {0,0,0}"
        "paddings(depth, height, width) of pooling operator.")
        .SetDefault({0, 0, 0});
    AddComment(R"DOC(
The pooling3d operation calculates the output based on
the input, poolingType and ksize, strides, paddings parameters.
)DOC");
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
        "number of channels, H and W is the height and width of image.");
    AddOutput("Out",
              "The output tensor of pooling operator."
              "The format of output tensor is also NCHW.");

    AddAttr<std::string>("poolingType",
                         "poolingType of pooling operator.['max' or 'ave']");
    AddAttr<std::vector<int>>(
        "ksize", "pooling size(height, width) of pooling operator.");
    AddAttr<int>("globalPooling",
                 "default 0"
                 "whether to use the globalPooling.[0 or 1]")
        .SetDefault(0);
    AddAttr<std::vector<int>>("strides",
                              "default {1, 1}"
                              "strides(height, width) of pooling operator.")
        .SetDefault({1, 1});
    AddAttr<std::vector<int>>("paddings",
                              "default {0, 0}"
                              "paddings(height, width) of pooling operator.")
        .SetDefault({0, 0});
    AddComment(R"DOC(
The pooling2d operation calculates the output based on
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
