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

namespace paddle {
namespace operators {
using framework::Tensor;

class SplitOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(const framework::InferShapeContext &ctx) const override {
    printf("Infershape ... \n");
    // infershape
    auto *in = ctx.Input<framework::Tensor>("X");
    auto outs = ctx.MultiOutput<framework::Tensor>("Out");
    size_t indices = static_cast<size_t>(ctx.Attr<int>("indices"));
    size_t axis = static_cast<size_t>(ctx.Attr<int>("axis"));
    size_t n = outs.size();

    std::vector<int> axis_dim;
    std::vector<int> sections =
        static_cast<std::vector<int>>(ctx.Attr<std::vector<int>>("sections"));
    if (indices > 0) {
      PADDLE_ENFORCE_EQ(in->dims()[axis] % indices, 0,
                        "tensor split does not result in an equal division.");
      size_t out_size = in->dims()[axis] / indices;
      printf("out_size=%ld\n", out_size);
      for (size_t i = 0; i < n; ++i) {
        axis_dim.push_back(indices);
      }
    } else if (sections.size() > 0) {
      // TODO(Yancey1989)
    } else {
      // throw exception
    }
    for (size_t i = 0; i < outs.size(); i++) {
      auto dim = in->dims();
      // dim[axis] = axis_dim[i];
      // outs[i]->Resize(dim);
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
        indices = 1
        axis = 0
        Output[0] = [[1,2]]
        Output[1] = [[3,4]]
        Output[2] = [[5,6]] 
    )DOC");
    AddAttr<std::vector<int>>("sections", "The length of each output.")
        .SetDefault(std::vector<int>{});
    AddAttr<int>("axis", "The axis which the input will be splited on")
        .SetDefault(0);
    AddAttr<int>("indices",
                 "The input will be divided into N equal array"
                 "along with the specify axis.")
        .SetDefault(-1);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(split, ops::SplitOp, ops::SplitOpMaker)
REGISTER_OP_CPU_KERNEL(split,
                       ops::SplitKernel<paddle::platform::CPUPlace, float>)
