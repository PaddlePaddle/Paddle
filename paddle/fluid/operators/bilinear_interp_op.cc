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

#include "paddle/fluid/operators/bilinear_interp_op.h"
#include <vector>
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

using framework::Tensor;

class BilinearInterpOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of BilinearInterOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of BilinearInterOp should not be null.");

    auto dim_x = ctx->GetInputDim("X");  // NCHW format
    int out_h = ctx->Attrs().Get<int>("out_h");
    int out_w = ctx->Attrs().Get<int>("out_w");
    PADDLE_ENFORCE_EQ(dim_x.size(), 4, "X's dimension must be 4");

    if (ctx->HasInput("OutSize")) {
      auto out_size_dim = ctx->GetInputDim("OutSize");
      PADDLE_ENFORCE_EQ(out_size_dim.size(), 1,
                        "OutSize's dimension size must be 1");
      PADDLE_ENFORCE_EQ(out_size_dim[0], 2, "OutSize's dim[0] must be 2");
    }
    std::vector<int64_t> dim_out({dim_x[0], dim_x[1], out_h, out_w});
    ctx->SetOutputDim("Out", framework::make_ddim(dim_out));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        framework::ToDataType(ctx.Input<Tensor>("X")->type()), ctx.GetPlace());
  }
};

class BilinearInterpOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "The input tensor of bilinear interpolation, "
             "This is a 4-D tensor with shape of (N x C x h x w)");
    AddInput("OutSize",
             "This is a 1-D tensor with two number. "
             "The first number is height and the second number is width.")
        .AsDispensable();
    AddOutput("Out", "The dimension of output is (N x C x out_h x out_w)");

    AddAttr<int>("out_h", "output height of bilinear interpolation op.");
    AddAttr<int>("out_w", "output width of bilinear interpolation op.");
    AddComment(R"DOC(
          Bilinear interpolation is an extension of linear interpolation for 
          interpolating functions of two variables (e.g. H-direction and 
          W-direction in this op) on a rectilinear 2D grid. 
          
          The key idea is to perform linear interpolation first in one 
          direction, and then again in the other direction.
            
          For details, please refer to Wikipedia: 
          https://en.wikipedia.org/wiki/Bilinear_interpolation
         )DOC");
  }
};

class BilinearInterpOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) should not be null");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "Input(Out@GRAD) should not be null");
    auto dim_x = ctx->GetInputDim("X");
    if (ctx->HasOutput(framework::GradVarName("X"))) {
      ctx->SetOutputDim(framework::GradVarName("X"), dim_x);
    }
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        framework::ToDataType(ctx.Input<Tensor>("X")->type()), ctx.GetPlace());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(bilinear_interp, ops::BilinearInterpOp,
                  ops::BilinearInterpOpMaker,
                  paddle::framework::DefaultGradOpDescMaker<true>);
REGISTER_OPERATOR(bilinear_interp_grad, ops::BilinearInterpOpGrad);
REGISTER_OP_CPU_KERNEL(bilinear_interp, ops::BilinearInterpKernel<float>,
                       ops::BilinearInterpKernel<uint8_t>);
REGISTER_OP_CPU_KERNEL(bilinear_interp_grad,
                       ops::BilinearInterpGradKernel<float>);
