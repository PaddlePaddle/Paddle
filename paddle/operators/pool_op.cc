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
  void InferShape(const framework::InferShapeContext &ctx) const override {
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar("X"),
                            "X(Input) of Pooling should not be null.");
    PADDLE_ENFORCE_NOT_NULL(ctx.OutputVar("Out"),
                            "Out(Output) of Pooling should not be null.");

    auto in_x = ctx.Input<Tensor>("X");
    auto out = ctx.Output<Tensor>("Out");
    int global_pooling = Attr<int>("globalPooling");
    std::string pooling_type = Attr<std::string>("poolingType");
    std::vector<int> ksize = Attr<std::vector<int>>("ksize");
    std::vector<int> strides = Attr<std::vector<int>>("strides");
    std::vector<int> paddings = Attr<std::vector<int>>("paddings");

    PADDLE_ENFORCE(pooling_type == "max" || pooling_type == "avg",
                   "pooling_type should be 'max' or 'avg'");
    PADDLE_ENFORCE(in_x->dims().size() == 4 || in_x->dims().size() == 5,
                   "Pooling intput should be 4-D or 5-D");
    PADDLE_ENFORCE(ksize.size() == 2 || ksize.size() == 3,
                   "Pooling size should be 2 elements. or 3 elements.");
    PADDLE_ENFORCE_EQ(ksize.size(), strides.size(),
                      "strides size and pooling size should be the same.");
    PADDLE_ENFORCE_EQ(ksize.size(), paddings.size(),
                      "paddings size and pooling size should be the same.");

    if (global_pooling == 1) {
      ksize.resize(static_cast<size_t>(in_x->dims().size()) - 2);
      for (size_t i = 0; i < ksize.size(); ++i)
        ksize[i] = static_cast<int>(in_x->dims()[i + 2]);
    }

    std::vector<int64_t> output_shape({in_x->dims()[0], in_x->dims()[1]});
    for (size_t i = 0; i < ksize.size(); ++i) {
      output_shape.push_back(OutputSizePool(in_x->dims()[i + 2], ksize[i],
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
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar("X"),
                            "X(Input) of Pooling should not be null.");
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar("Out"),
                            "Out(Output) of Pooling should not be null.");
    PADDLE_ENFORCE_NOT_NULL(ctx.Output<Tensor>(framework::GradVarName("X")),
                            "Input@Grad of Pooling should not be null.");

    auto in = ctx.Input<Tensor>("X");
    auto d_in = ctx.Output<Tensor>(framework::GradVarName("X"));
    d_in->Resize(in->dims());
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
                         "poolingType of pooling operator."
                         "str constant equal to 'max' or 'avg'");
    AddAttr<std::vector<int>>(
        "ksize", "pooling size(height, width) of pooling operator.")
        .AddCustomChecker(GreaterThanChecker_pool({0, 0}));
    AddAttr<int>(
        "globalPooling",
        "whether to use the globalPooling."
        "int constant equal to 0 or 1"
        "default 0"
        "If globalPooling = 1, ksize is ignored and need not be specified.")
        .SetDefault(0);
    AddAttr<std::vector<int>>("strides",
                              "strides(height, width) of pooling operator."
                              "default {1,1}")
        .SetDefault({1, 1})
        .AddCustomChecker(GreaterThanChecker_pool({0, 0}));
    AddAttr<std::vector<int>>("paddings",
                              "paddings(height, width) of pooling operator."
                              "default {0,0}")
        .SetDefault({0, 0})
        .AddCustomChecker(EqualGreaterThanChecker_pool({0, 0}));
    AddComment(R"DOC(
The pooling2d operation calculates the output based on
the input, poolingType and ksize, strides, paddings parameters.
)DOC");
  }

 private:
  struct GreaterThanChecker_pool {
   public:
    explicit GreaterThanChecker_pool(std::vector<int> lower_bound)
        : lower_bound_(lower_bound) {}
    void operator()(std::vector<int> &value) const {
      PADDLE_ENFORCE(value.size() == lower_bound_.size(), "equal check fails.");
      for (size_t i = 0; i < value.size(); ++i) {
        PADDLE_ENFORCE(value[i] > lower_bound_[i], "larger_than check fails.");
      }
    }

   private:
    std::vector<int> lower_bound_;
  };

  struct EqualGreaterThanChecker_pool {
   public:
    explicit EqualGreaterThanChecker_pool(std::vector<int> lower_bound)
        : lower_bound_(lower_bound) {}
    void operator()(std::vector<int> &value) const {
      PADDLE_ENFORCE(value.size() == lower_bound_.size(), "equal check fails.");
      for (size_t i = 0; i < value.size(); ++i) {
        PADDLE_ENFORCE(value[i] >= lower_bound_[i], "larger_than check fails.");
      }
    }

   private:
    std::vector<int> lower_bound_;
  };
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
                         "poolingType of pooling operator."
                         "str constant equal to 'max' or 'avg'");
    AddAttr<std::vector<int>>(
        "ksize", "pooling size(depth, height, width) of pooling operator.")
        .AddCustomChecker(GreaterThanChecker_pool({0, 0, 0}));
    AddAttr<int>(
        "globalPooling",
        "whether to use the globalPooling."
        "int constant equal to 0 or 1"
        "default 0"
        "If globalPooling = 1, ksize is ignored and need not be specified.")
        .SetDefault(0);
    AddAttr<std::vector<int>>(
        "strides",
        "strides(depth, height, width) of pooling operator."
        "default {1,1,1}")
        .SetDefault({1, 1, 1})
        .AddCustomChecker(GreaterThanChecker_pool({0, 0, 0}));
    AddAttr<std::vector<int>>(
        "paddings",
        "paddings(depth, height, width) of pooling operator."
        "default {0,0,0}")
        .SetDefault({0, 0, 0})
        .AddCustomChecker(EqualGreaterThanChecker_pool({0, 0, 0}));
    AddComment(R"DOC(
The pooling3d operation calculates the output based on
the input, poolingType and ksize, strides, paddings parameters.
)DOC");
  }

 private:
  struct GreaterThanChecker_pool {
   public:
    explicit GreaterThanChecker_pool(std::vector<int> lower_bound)
        : lower_bound_(lower_bound) {}
    void operator()(std::vector<int> &value) const {
      PADDLE_ENFORCE(value.size() == lower_bound_.size(), "equal check fails.");
      for (size_t i = 0; i < value.size(); ++i) {
        PADDLE_ENFORCE(value[i] > lower_bound_[i], "larger_than check fails.");
      }
    }

   private:
    std::vector<int> lower_bound_;
  };

  struct EqualGreaterThanChecker_pool {
   public:
    explicit EqualGreaterThanChecker_pool(std::vector<int> lower_bound)
        : lower_bound_(lower_bound) {}
    void operator()(std::vector<int> &value) const {
      PADDLE_ENFORCE(value.size() == lower_bound_.size(), "equal check fails.");
      for (size_t i = 0; i < value.size(); ++i) {
        PADDLE_ENFORCE(value[i] >= lower_bound_[i], "larger_than check fails.");
      }
    }

   private:
    std::vector<int> lower_bound_;
  };
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
