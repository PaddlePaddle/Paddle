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

int outputSize(int input_size, int filter_size, int padding, int stride) {
  int output_size = (input_size - filter_size + 2 * padding) / stride + 1;
  return output_size;
}

class PoolOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(const framework::InferShapeContext &ctx) const override {
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar("Input"),
                            "Input(Input) of Pooling should not be null.");
    PADDLE_ENFORCE_NOT_NULL(ctx.OutputVar("Output"),
                            "Output(Output) of Pooling should not be null.");
    //    PADDLE_ENFORCE_NOT_NULL(Attr<std::string>("pooling_type"),
    //                            "pooling_type should not be null.");
    //    PADDLE_ENFORCE_NOT_NULL(Attr<std::vector<int>>("ksize"), "ksize should
    //    not be null.");
    auto input = ctx.Input<Tensor>("Input");
    auto output = ctx.Output<framework::LoDTensor>("Output");
    int global_pooling = Attr<int>("global_pooling");
    std::string pooling_type = Attr<std::string>("pooling_type");
    std::vector<int> ksize = Attr<std::vector<int>>("ksize");
    std::vector<int> strides = Attr<std::vector<int>>("strides");
    std::vector<int> paddings = Attr<std::vector<int>>("paddings");

    PADDLE_ENFORCE(pooling_type == "max" || pooling_type == "ave",
                   "pooling_type should be 'max' or 'ave'");
    PADDLE_ENFORCE(ksize.size() == 2 || ksize.size() == 3,
                   "Pooling ksize should be 2-D or 3-D");

    if (global_pooling == 1) {
      for (size_t i = 0; i < ksize.size(); ++i) ksize[i] = input->dims()[i + 2];
    }
    if (ksize.size() == 2) {
      PADDLE_ENFORCE_EQ(input->dims().size(), 4,
                        "Pool2DOp intput should be 4-D.");
      PADDLE_ENFORCE_EQ(strides.size(), 2, "Pool2DOp strides should be 2-D.");
      PADDLE_ENFORCE_EQ(paddings.size(), 2, "Pool2DOp paddings should be 2-D.");
    } else {
      PADDLE_ENFORCE_EQ(input->dims().size(), 5,
                        "Pool3DOp intput should be 5-D.");
      PADDLE_ENFORCE_EQ(strides.size(), 3, "Pool3DOp strides should be 3-D.");
      PADDLE_ENFORCE_EQ(paddings.size(), 3, "Pool3DOp paddings should be 3-D.");
    }
    std::vector<int64_t> output_shape({input->dims()[0], input->dims()[1]});
    for (size_t i = 0; i < ksize.size(); ++i) {
      output_shape.push_back(
          outputSize(input->dims()[i + 2], ksize[i], paddings[i], strides[i]));
    }
    output->Resize(framework::make_ddim(output_shape));
  }
};

class PoolOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(const framework::InferShapeContext &ctx) const override {
    auto in = ctx.Input<Tensor>("Input");
    auto d_in =
        ctx.Output<framework::LoDTensor>(framework::GradVarName("Input"));
    if (d_in) d_in->Resize(in->dims());
  }
};

class Pool3dOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  Pool3dOpMaker(framework::OpProto *proto, framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput(
        "Input",
        "The input tensor of pooling operator. "
        "The format of input tensor is NCDHW. Where N is batch size, C is the "
        "number of channels, D, H and W is the depth, height and width of "
        "image.");
    AddOutput("Output",
              "The output tensor of pooling operator."
              "The format of output tensor is also NCDHW.");

    AddAttr<std::string>("pooling_type",
                         "pooling_type of pooling operator.['max' or 'ave']");
    AddAttr<std::vector<int>>("ksize", "strides of pooling operator.");
    AddAttr<int>("global_pooling", "whether to use the global_pooling.")
        .SetDefault(0);
    AddAttr<std::vector<int>>("strides", "strides of pooling operator.")
        .SetDefault({1, 1, 1});
    AddAttr<std::vector<int>>("paddings", "paddings of pooling operator.")
        .SetDefault({0, 0, 0});
    AddComment(R"DOC(
The pooling3d operation calculates the output based on
the input, pooling_type and ksize, strides, paddings parameters.
)DOC");
  }
};

class Pool2dOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  Pool2dOpMaker(framework::OpProto *proto, framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput(
        "Input",
        "The input tensor of pooling operator. "
        "The format of input tensor is NCHW. Where N is batch size, C is the "
        "number of channels, H and W is the height and width of image.");
    AddOutput("Output",
              "The output tensor of pooling operator."
              "The format of output tensor is also NCHW.");

    AddAttr<std::string>("pooling_type",
                         "pooling_type of pooling operator.['max' or 'ave']");
    AddAttr<std::vector<int>>("ksize", "strides of pooling operator.");
    AddAttr<int>("global_pooling", "whether to use the global_pooling.")
        .SetDefault(0);
    AddAttr<std::vector<int>>("strides", "strides of pooling operator.")
        .SetDefault({1, 1});
    AddAttr<std::vector<int>>("paddings", "paddings of pooling operator.")
        .SetDefault({0, 0});
    AddComment(R"DOC(
The pooling2d operation calculates the output based on
the input, pooling_type and ksize, strides, paddings parameters.
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
