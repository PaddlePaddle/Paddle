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

#include "paddle/operators/crop_op.h"
#include <boost/lexical_cast.hpp>

namespace paddle {
namespace operators {

using framework::Tensor;

class CropOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of CropOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of CropOp should not be null.");
    auto x_dim = ctx->GetInputDim("X");
    if (!ctx->HasInput("Y")) {
      auto shape = ctx->Attrs().Get<std::vector<int>>("shape");
      PADDLE_ENFORCE_EQ(
          int64_t(shape.size()), x_dim.size(),
          "Shape size should be equal to dimention size of input tensor.");
      std::vector<int64_t> tensor_shape(shape.size());
      for (size_t i = 0; i < shape.size(); ++i) {
        tensor_shape[i] = static_cast<int64_t>(shape[i]);
      }
      ctx->SetOutputDim("Out", framework::make_ddim(tensor_shape));
    } else {
      auto y_dim = ctx->GetInputDim("Y");
      PADDLE_ENFORCE_EQ(framework::arity(x_dim), framework::arity(y_dim),
                        "Tensor rank of both CropOp's "
                        "inputs must be same.");
      ctx->SetOutputDim("Out", y_dim);
    }
  }
};

class CropOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  CropOpMaker(framework::OpProto* proto, framework::OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X",
             "The input of pad op. "
             "The input should be a k-D tensor(k > 0 and k < 7).");
    AddInput("Y",
             "The input used as reference for cropping, "
             "which is of the same dimensions as X.")
        .AsDispensable();
    AddOutput("Out",
              "The output of crop op, "
              "which is of the same dimensions as X.");
    AddAttr<std::vector<int>>("offsets",
                              "A list<int> describing offsets to be cropped. "
                              "The size of offsets list should be the same as "
                              "the dimension size of input X.");
    AddAttr<std::vector<int>>("shape",
                              "A list<int> describing the shape of output. "
                              "The size of shape list should be the same as "
                              "the dimension size of input X.")
        .SetDefault(std::vector<int>());
    AddComment(R"DOC(
Crop Operator.

Crop input into output, as specified by offsets and shape.

There are two ways to set shape:
1. reference input: crop input X into the same shape as reference input.
                    The dimension of reference input should
                    be the same as the dimension of input X.
2. shape list: crop input X into the shape described by a list<int>.
               The size of shape list should be the same as
               the dimension size of input X.

The input should be a k-D tensor(k > 0 and k < 7). As an example:

Given:

    X = [[0, 1, 2, 0, 0]
         [0, 3, 4, 0, 0]
         [0, 0, 0, 0, 0]],

and

    offsets = [0, 1],

and

    shape = [2, 2],

we get:

    Out = [[1, 2],
           [3, 4]].

)DOC");
  }
};

class CropOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) should not be null");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "Input(Out@GRAD) should not be null");
    auto x_dims = ctx->GetInputDim("X");
    auto x_grad_name = framework::GradVarName("X");
    if (ctx->HasOutput(x_grad_name)) {
      ctx->SetOutputDim(x_grad_name, x_dims);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP(crop, ops::CropOp, ops::CropOpMaker, crop_grad, ops::CropOpGrad);
REGISTER_OP_CPU_KERNEL(crop, ops::CropKernel<float>);
REGISTER_OP_CPU_KERNEL(crop_grad,
                       ops::CropGradKernel<paddle::platform::CPUPlace, float>);
