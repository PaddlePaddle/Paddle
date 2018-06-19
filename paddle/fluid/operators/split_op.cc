/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/split_op.h"

namespace paddle {
namespace operators {
using framework::Tensor;

class SplitOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of SplitOp should not be null.");
    PADDLE_ENFORCE_GE(ctx->Outputs("Out").size(), 1UL,
                      "Outputs(Out) of SplitOp should not be empty.");
    auto in_dims = ctx->GetInputDim("X");
    auto outs_names = ctx->Outputs("Out");
    size_t axis = static_cast<size_t>(ctx->Attrs().Get<int>("axis"));
    size_t num = static_cast<size_t>(ctx->Attrs().Get<int>("num"));
    std::vector<int> sections = static_cast<std::vector<int>>(
        ctx->Attrs().Get<std::vector<int>>("sections"));
    const size_t outs_number = outs_names.size();
    std::vector<framework::DDim> outs_dims;
    outs_dims.reserve(outs_number);

    if (num > 0) {
      int64_t in_axis_dim = in_dims[axis];
      PADDLE_ENFORCE_EQ(in_axis_dim % num, 0,
                        "tensor split does not result"
                        " in an equal division");
      size_t out_axis_dim = in_axis_dim / num;
      for (size_t i = 0; i < outs_number; ++i) {
        auto dim = in_dims;
        dim[axis] = out_axis_dim;
        outs_dims.push_back(dim);
      }
    } else if (sections.size() > 0) {
      PADDLE_ENFORCE_EQ(sections.size(), outs_number,
                        "tensor split sections size"
                        "should be equal to output size.");
      for (size_t i = 0; i < outs_number; ++i) {
        auto dim = in_dims;
        dim[axis] = sections[i];
        outs_dims.push_back(dim);
      }
    }
    ctx->SetOutputsDim("Out", outs_dims);
    if (axis != 0) {
      // Only pass LoD when not spliting along the first dim.
      for (size_t i = 0; i < outs_number; ++i) {
        ctx->ShareLoD("X", "Out", 0, i);
      }
    }
  }
};

class SplitOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor) Input tensor of the split operator.");
    AddOutput("Out", "(Tensor) Output tensors of the split operator.")
        .AsDuplicable();
    AddComment(R"DOC(
Split operator

This operator splits the input tensor into multiple sub-tensors.

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
                              "(vector<int>) "
                              "the length of each output along the "
                              "specified axis.")
        .SetDefault(std::vector<int>{});
    AddAttr<int>("num",
                 "(int, default 0)"
                 "Number of sub-tensors. This must evenly divide "
                 "Input.dims()[axis]")
        .SetDefault(0);
    AddAttr<int>("axis",
                 "(int, default 0) "
                 "The axis which the input will be splited on.")
        .SetDefault(0);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
USE_CPU_ONLY_OP(concat);

REGISTER_OPERATOR(split, ops::SplitOp, ops::SplitOpMaker, ops::SplitGradMaker);
REGISTER_OP_CPU_KERNEL(split,
                       ops::SplitOpKernel<paddle::platform::CPUPlace, double>,
                       ops::SplitOpKernel<paddle::platform::CPUPlace, float>,
                       ops::SplitOpKernel<paddle::platform::CPUPlace, int64_t>,
                       ops::SplitOpKernel<paddle::platform::CPUPlace, int>);
