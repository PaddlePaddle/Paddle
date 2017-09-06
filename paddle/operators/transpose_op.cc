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

#include "paddle/operators/transpose_op.h"
#include <vector>
#include "paddle/framework/ddim.h"

namespace paddle {
namespace operators {

using framework::Tensor;

class TransposeOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(const framework::InferShapeContext &ctx) const override {
    auto in_dim = ctx.Input<Tensor>("X")->dims();
    auto axis = ctx.GetAttr<std::vector<int>>("axis");
    size_t in_dim_size = in_dim.size();
    size_t axis_size = axis.size();
    PADDLE_ENFORCE_EQ(
        in_dim_size, axis_size,
        "the input tensor dimensions should be equal to the axis size");

    std::vector<int> axis_sorted(axis);
    std::sort(axis_sorted.begin(), axis_sorted.end());
    for (size_t i = 0; i < axis_sorted.size(); i++) {
      PADDLE_ENFORCE_EQ(axis_sorted[i], (int)i,
                        "the sorted axis should be [0, 1, ... dims - 1], "
                        "the dims equals to the input tensor dimensions");
    }
    //
    framework::DDim out_dim(in_dim);
    for (size_t i = 0; i < axis.size(); i++) {
      out_dim[i] = in_dim[axis[i]];
    }
    ctx.Output<Tensor>("Out")->Resize(out_dim);
  }
};

class TransposeOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  TransposeOpMaker(framework::OpProto *proto,
                   framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "The input of transpose op");
    AddOutput("Out", "The output of transpose op");
    AddAttr<std::vector<int>>(
        "axis",
        "a list of integers, and the num of integers should be "
        "the same with the input tensor dimensions");
    AddComment(R"DOC(
Transpose the input tensor. 
For example, input tensor shape(N, C, H, W) and axis {0, 2, 3, 1},
the output tensor shape will be (N, H, W, C)
)DOC");
  }
};

class TransposeOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(const framework::InferShapeContext &ctx) const override {
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar("X"), "Input(X) should not be null");
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar(framework::GradVarName("Out")),
                            "Input(Out@GRAD) should not be null");
    auto x_dims = ctx.Input<Tensor>("X")->dims();
    auto *x_grad = ctx.Output<Tensor>(framework::GradVarName("X"));

    auto out_grad_dims =
        ctx.Input<Tensor>(framework::GradVarName("Out"))->dims();
    auto out_dims = ctx.Input<Tensor>("Out")->dims();

    PADDLE_ENFORCE(out_grad_dims == out_dims,
                   "Out@GRAD dims must equal to Input(X) dims");

    x_grad->Resize(x_dims);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP(transpose, ops::TransposeOp, ops::TransposeOpMaker, transpose_grad,
            ops::TransposeOpGrad);
REGISTER_OP_CPU_KERNEL(transpose,
                       ops::TransposeKernel<paddle::platform::CPUPlace, float>);
REGISTER_OP_CPU_KERNEL(
    transpose_grad,
    ops::TransposeGradKernel<paddle::platform::CPUPlace, float>);
