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

#include "paddle/operators/split_op.h"
#include "paddle/operators/net_op.h"

namespace paddle {
namespace operators {
using framework::Tensor;

class SplitOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(const framework::InferShapeContext &ctx) const override {
    // infershape
    auto *in = ctx.Input<framework::Tensor>("X");
    auto outs = ctx.MultiOutput<framework::LoDTensor>("Out");
    size_t axis = static_cast<size_t>(ctx.Attr<int>("axis"));
    size_t num = static_cast<size_t>(ctx.Attr<int>("num"));
    std::vector<int> sections =
        static_cast<std::vector<int>>(ctx.Attr<std::vector<int>>("sections"));
    const size_t n = outs.size();

    if (num > 0) {
      int64_t in_axis_dim = in->dims()[axis];
      PADDLE_ENFORCE_EQ(in_axis_dim % num, 0,
                        "tensor split does not result"
                        " in an equal division");
      size_t out_axis_dim = in_axis_dim / num;
      for (size_t i = 0; i < n; ++i) {
        auto dim = in->dims();
        dim[axis] = out_axis_dim;
        outs[i]->Resize(dim);
      }
    } else if (sections.size() > 0) {
      PADDLE_ENFORCE_EQ(sections.size(), n,
                        "tensor split sections size"
                        "should be equal to output size.");
      for (size_t i = 0; i < n; ++i) {
        auto dim = in->dims();
        dim[axis] = sections[i];
        outs[i]->Resize(dim);
      }
    } else {
      PADDLE_ENFORCE_NOT_NULL(nullptr, "split operator should",
                              " specify indices or sections.");
    }
  }
};

class SplitOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  SplitOpMaker(framework::OpProto *proto, framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "the input tensor of split operator.");
    AddOutput("Out", "the output tensors of split operator.").AsDuplicable();
    AddComment(R"DOC(
      Split the input tensor into multiple sub-tensors.
      Example:
        Input = [[1,2],
                 [3,4],
                 [5,6]]
        sections = [2,1]
        axis = 0
        Output[0] = [[1,2],
                     [3,4]]
        Output[1] = [[5,6]]

    )DOC");
    AddAttr<std::vector<int>>("sections",
                              "the length for each"
                              "output along with the specify axis.")
        .SetDefault(std::vector<int>{});
    AddAttr<int>("num",
                 "number of the sub-tensors, it must evenly divide "
                 "Input.dims()[axis]")
        .SetDefault(0);
    AddAttr<int>("axis", "The axis which the input will be splited on.")
        .SetDefault(0);
  }
};

class SplitOpGrad : public NetOp {
 public:
  SplitOpGrad(const std::string &type, const framework::VariableNameMap &inputs,
              const framework::VariableNameMap &outputs,
              const framework::AttributeMap &attrs)
      : NetOp(type, inputs, outputs, attrs) {
    auto out_grad = Inputs(framework::GradVarName("Out"));
    auto x_grad = Output(framework::GradVarName("X"));
    AppendOp(framework::OpRegistry::CreateOp("concat", {{"X", out_grad}},
                                             {{"Out", {x_grad}}}, attrs));
    CompleteAddOp(false);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
USE_CPU_ONLY_OP(concat);
REGISTER_OP(split, ops::SplitOp, ops::SplitOpMaker, split_grad,
            ops::SplitOpGrad);
REGISTER_OP_CPU_KERNEL(split,
                       ops::SplitKernel<paddle::platform::CPUPlace, float>);
