/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/crop_op.h"
#include <memory>
#include <string>
#include <vector>

namespace paddle {
namespace operators {

using framework::Tensor;

class CropOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "Crop");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "Crop");
    auto x_dim = ctx->GetInputDim("X");
    if (!ctx->HasInput("Y")) {
      auto shape = ctx->Attrs().Get<std::vector<int>>("shape");
      PADDLE_ENFORCE_EQ(
          int64_t(shape.size()), x_dim.size(),
          platform::errors::InvalidArgument(
              "The number of elements (%d) of CropOp's "
              "'shape' attribute should be equal to the number of dimensions "
              "(%d) of the Input(X).",
              shape.size(), x_dim.size()));
      std::vector<int64_t> tensor_shape(shape.size());
      for (size_t i = 0; i < shape.size(); ++i) {
        tensor_shape[i] = static_cast<int64_t>(shape[i]);
      }
      ctx->SetOutputDim("Out", phi::make_ddim(tensor_shape));
    } else {
      auto y_dim = ctx->GetInputDim("Y");
      PADDLE_ENFORCE_EQ(phi::arity(x_dim), phi::arity(y_dim),
                        platform::errors::InvalidArgument(
                            "The number of dimensions (%d) of CropOp's input(X)"
                            " must be equal to that (%d) of input(Y).",
                            phi::arity(x_dim), phi::arity(y_dim)));
      ctx->SetOutputDim("Out", y_dim);
    }
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"),
        ctx.device_context());
  }
};

class CropOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "The input of pad op. "
             "The input should be a k-D tensor(k > 0 and k < 7).");
    AddInput("Y",
             "The input used as reference for cropping, "
             "which is of the same dimensions as X.")
        .AsDispensable();
    AddInput("Offsets",
             "The input used to describe offsets in runtime, which is a "
             "1-D vector whose size equals to the rank of input 'X'. The "
             "elements data type must be int.")
        .AsDispensable();
    AddOutput("Out",
              "The output of crop op, "
              "which is of the same dimensions as X.");
    AddAttr<std::vector<int>>("offsets",
                              "A list<int> describing offsets to be cropped. "
                              "The size of offsets list should be the same as "
                              "the dimension size of input X.")
        .SetDefault(std::vector<int>());
    AddAttr<std::vector<int>>("shape",
                              "A list<int> describing the shape of output. "
                              "The size of shape list should be the same as "
                              "the dimension size of input X.")
        .SetDefault(std::vector<int>());
    AddComment(R"DOC(
Crop Operator.

Crop input into output, as specified by offsets and shape.

There are two ways to set the offsets:
1. In runtime: Using the input 'Offsets', which is a Vairbale and can be 
               output of other operators. This way is suitable for 
               dynamic offsets.
2. In network configuration: Using the attribute 'offsets', which will be 
                             set in Python configure script. This way is 
                             suitable for fixed offsets.
You CANNOT use these two ways at the same time. An exception will be raised 
if input 'Offset' is configured and meanwhile the attribute 'offsets' is 
not empty.

There are two ways to set shape:
1. reference input: crop input X into the same shape as reference input.
                    The dimension of reference input should
                    be the same as the dimension of input X.
2. shape list: crop input X into the shape described by a list<int>.
               The size of shape list should be the same as
               the dimension size of input X.

The input should be a k-D tensor(k > 0 and k < 7). As an example:

Case 1:
Given

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


Case 2:
Given

    X = [[0, 1, 2, 5, 0]
         [0, 3, 4, 6, 0]
         [0, 0, 0, 0, 0]],

and

    offsets = [0, 1],

and

    Y = [[0, 0, 0]
         [0, 0, 0]],

we get:

    Out = [[1, 2, 5],
           [3, 4, 6]].
)DOC");
  }
};

class CropOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "CropGrad");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")), "Input",
                   framework::GradVarName("Out"), "CropGrad");
    auto x_dims = ctx->GetInputDim("X");
    auto x_grad_name = framework::GradVarName("X");
    if (ctx->HasOutput(x_grad_name)) {
      ctx->SetOutputDim(x_grad_name, x_dims);
    }
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Out")),
                                   ctx.device_context());
  }
};

template <typename T>
class CropGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("crop_grad");
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetInput("X", this->Input("X"));
    if (this->HasInput("Offsets")) {
      op->SetInput("Offsets", this->Input("Offsets"));
    }
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetAttrMap(this->Attrs());
  }
};

DECLARE_NO_NEED_BUFFER_VARS_INFERER(GropNoNeedBufferVarInferer, "Y");

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(crop, ops::CropOp, ops::CropOpMaker,
                  ops::CropGradOpMaker<paddle::framework::OpDesc>,
                  ops::CropGradOpMaker<paddle::imperative::OpBase>,
                  ops::GropNoNeedBufferVarInferer);
REGISTER_OPERATOR(crop_grad, ops::CropOpGrad);
REGISTER_OP_CPU_KERNEL(
    crop, ops::CropKernel<paddle::platform::CPUDeviceContext, float>,
    ops::CropKernel<paddle::platform::CPUDeviceContext, double>);
REGISTER_OP_CPU_KERNEL(
    crop_grad, ops::CropGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::CropGradKernel<paddle::platform::CPUDeviceContext, double>);

REGISTER_OP_CUDA_KERNEL(
    crop, ops::CropKernel<paddle::platform::CUDADeviceContext, float>,
    ops::CropKernel<paddle::platform::CUDADeviceContext, double>);
REGISTER_OP_CUDA_KERNEL(
    crop_grad, ops::CropGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::CropGradKernel<paddle::platform::CUDADeviceContext, double>);
