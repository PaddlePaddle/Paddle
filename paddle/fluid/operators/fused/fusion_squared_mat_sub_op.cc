/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. */

#include "paddle/fluid/operators/fused/fusion_squared_mat_sub_op.h"

#include <string>
#include <vector>

#include "paddle/fluid/operators/jit/kernels.h"

namespace paddle {
namespace operators {

void FusionSquaredMatSubOp::InferShape(
    framework::InferShapeContext* ctx) const {
  OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "FusionSquaredMatSub");
  OP_INOUT_CHECK(ctx->HasInput("Y"), "Input", "Y", "FusionSquaredMatSub");
  OP_INOUT_CHECK(
      ctx->HasOutput("SquaredX"), "SquaredX", "Out", "FusionSquaredMatSub");
  OP_INOUT_CHECK(
      ctx->HasOutput("SquaredY"), "SquaredY", "Out", "FusionSquaredMatSub");
  OP_INOUT_CHECK(
      ctx->HasOutput("SquaredXY"), "SquaredXY", "Out", "FusionSquaredMatSub");
  OP_INOUT_CHECK(ctx->HasOutput("Out"), "Out", "Out", "FusionSquaredMatSub");
  auto x_dims = ctx->GetInputDim("X");
  auto y_dims = ctx->GetInputDim("Y");
  PADDLE_ENFORCE_EQ(
      x_dims.size(),
      y_dims.size(),
      platform::errors::InvalidArgument("The input tensor X's dims size should "
                                        "be equal to Y's. But received X's "
                                        "dims size = %d, Y's dims size = %d.",
                                        x_dims.size(),
                                        y_dims.size()));
  PADDLE_ENFORCE_EQ(x_dims.size(),
                    2,
                    platform::errors::InvalidArgument(
                        "The input tensor X's dims size should be 2. But "
                        "received X's dims size = %d.",
                        x_dims.size()));
  PADDLE_ENFORCE_EQ(
      x_dims[1],
      y_dims[0],
      platform::errors::InvalidArgument("The input tensor X's dims[1] should "
                                        "be equal to Y's dims[0]. But received "
                                        "X's dims[1] = %d, Y's dims[0] = %d.",
                                        x_dims[1],
                                        y_dims[0]));
  ctx->SetOutputDim("SquaredX", x_dims);
  ctx->SetOutputDim("SquaredY", y_dims);
  ctx->SetOutputDim("SquaredXY", {x_dims[0], y_dims[1]});
  ctx->SetOutputDim("Out", {x_dims[0], y_dims[1]});
}

framework::OpKernelType FusionSquaredMatSubOp::GetExpectedKernelType(
    const framework::ExecutionContext& ctx) const {
  return framework::OpKernelType(
      OperatorWithKernel::IndicateVarDataType(ctx, "X"), ctx.GetPlace());
}

void FusionSquaredMatSubOpMaker::Make() {
  AddInput("X", "(phi::DenseTensor) Input Mat A of this operator.");
  AddInput("Y", "(phi::DenseTensor) Input Mat B of this operator.");
  AddOutput("SquaredX", "(phi::DenseTensor) Squared X.").AsIntermediate();
  AddOutput("SquaredY", "(phi::DenseTensor) Squared Y.").AsIntermediate();
  AddOutput("SquaredXY", "(phi::DenseTensor) Squared X*Y.").AsIntermediate();
  AddOutput("Out", "(phi::DenseTensor) Output tensor of concat operator.");
  AddAttr<float>("scalar", "The scalar on output matrix.").SetDefault(1.f);
  AddComment(R"DOC(
    Fusion Squared Matrix and substrct operator.

    ( (X * Y).^2 - (X.^2 * Y.^2) ) .* scalar
)DOC");
}

template <typename T>
class FusionSquaredMatSubKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto x = ctx.Input<phi::DenseTensor>("X");
    auto y = ctx.Input<phi::DenseTensor>("Y");
    auto* squared_x = ctx.Output<phi::DenseTensor>("SquaredX");
    auto* squared_y = ctx.Output<phi::DenseTensor>("SquaredY");
    auto* squared_xy = ctx.Output<phi::DenseTensor>("SquaredXY");
    auto* out = ctx.Output<phi::DenseTensor>("Out");
    auto place = ctx.GetPlace();
    T scalar = static_cast<T>(ctx.Attr<float>("scalar"));

    auto x_dims = x->dims();
    auto y_dims = y->dims();
    jit::matmul_attr_t attr;
    attr.m = x_dims[0];
    attr.k = x_dims[1];
    attr.n = y_dims[1];
    int o_numel = attr.m * attr.n;

    auto vsquare_x =
        jit::KernelFuncs<jit::VSquareTuple<T>, platform::CPUPlace>::Cache().At(
            attr.m * attr.k);
    auto vsquare_y =
        jit::KernelFuncs<jit::VSquareTuple<T>, platform::CPUPlace>::Cache().At(
            attr.k * attr.n);
    auto vsquare_xy =
        jit::KernelFuncs<jit::VSquareTuple<T>, platform::CPUPlace>::Cache().At(
            o_numel);
    auto vsub =
        jit::KernelFuncs<jit::VSubTuple<T>, platform::CPUPlace>::Cache().At(
            o_numel);
    auto vscal =
        jit::KernelFuncs<jit::VScalTuple<T>, platform::CPUPlace>::Cache().At(
            o_numel);
    auto matmul =
        jit::KernelFuncs<jit::MatMulTuple<T>, platform::CPUPlace>::Cache().At(
            attr);

    const T* x_data = x->data<T>();
    const T* y_data = y->data<T>();
    T* squared_x_data = squared_x->mutable_data<T>(place);
    T* squared_y_data = squared_y->mutable_data<T>(place);
    T* squared_xy_data = squared_xy->mutable_data<T>(place);
    T* o_data = out->mutable_data<T>(place);

    matmul(x_data, y_data, squared_xy_data, &attr);
    vsquare_xy(squared_xy_data, squared_xy_data, o_numel);

    vsquare_x(x_data, squared_x_data, attr.m * attr.k);
    vsquare_y(y_data, squared_y_data, attr.k * attr.n);
    matmul(squared_x_data, squared_y_data, o_data, &attr);

    vsub(squared_xy_data, o_data, o_data, o_numel);
    vscal(&scalar, o_data, o_data, o_numel);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(fusion_squared_mat_sub,
                  ops::FusionSquaredMatSubOp,
                  ops::FusionSquaredMatSubOpMaker);

REGISTER_OP_CPU_KERNEL(fusion_squared_mat_sub,
                       ops::FusionSquaredMatSubKernel<float>,
                       ops::FusionSquaredMatSubKernel<double>);
