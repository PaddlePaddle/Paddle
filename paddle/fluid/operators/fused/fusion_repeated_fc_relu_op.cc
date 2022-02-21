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

#include "paddle/fluid/operators/fused/fusion_repeated_fc_relu_op.h"
#include <string>
#include <vector>
#include "paddle/fluid/operators/jit/kernels.h"

namespace paddle {
namespace operators {

void FusionRepeatedFCReluOp::InferShape(
    framework::InferShapeContext* ctx) const {
  OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "FusionRepeatedFCRelu");
  auto sz = ctx->Inputs("W").size();
  PADDLE_ENFORCE_GT(sz, 1UL, platform::errors::InvalidArgument(
                                 "Inputs(W) of FusionRepeatedFCReluOp should "
                                 "be greater than 1, but received value is %d.",
                                 sz));
  PADDLE_ENFORCE_EQ(
      ctx->Inputs("Bias").size(), sz,
      platform::errors::InvalidArgument(
          "Size of inputs(Bias) of FusionRepeatedFCReluOp should be "
          "equal to inputs size %d, but received value is %d.",
          sz, ctx->Inputs("Bias").size()));
  PADDLE_ENFORCE_EQ(
      ctx->Outputs("ReluOut").size(), sz - 1,
      platform::errors::InvalidArgument(
          "Size of output(ReluOut) of FusionRepeatedFCReluOp should "
          "be equal to inputs size minus one %d, but received value is %d",
          sz - 1, ctx->Outputs("ReluOut").size()));
  OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out",
                 "FusionRepeatedFCRelu");

  auto i_dims = ctx->GetInputDim("X");
  PADDLE_ENFORCE_EQ(
      i_dims.size(), 2,
      platform::errors::InvalidArgument(
          "Input shape size should be 2, but received value is %d.",
          i_dims.size()));

  auto w_dims = ctx->GetInputsDim("W");
  auto b_dims = ctx->GetInputsDim("Bias");
  PADDLE_ENFORCE_EQ(w_dims.size(), b_dims.size(),
                    platform::errors::InvalidArgument(
                        "Shape size of weight and bias should be equal, but "
                        "weight size is %d, bias size is %d.",
                        w_dims.size(), b_dims.size()));
  PADDLE_ENFORCE_EQ(i_dims[1], w_dims[0][0],
                    platform::errors::InvalidArgument(
                        "input width should be equal to weight height, but "
                        "input width is %d, weight height is %d.",
                        i_dims[1], w_dims[0][0]));

  for (size_t i = 1; i < sz; ++i) {
    PADDLE_ENFORCE_EQ(w_dims[i].size(), 2,
                      platform::errors::InvalidArgument(
                          "Every weight shape size should be 2, but received "
                          "w_dims[%d].size() = %d.",
                          i, w_dims[i].size()));
    PADDLE_ENFORCE_EQ(
        phi::product(b_dims[i]), w_dims[i][1],
        platform::errors::InvalidArgument(
            "The length of Bias must be equal with w_dims[1], but received "
            "product(b_dims[%d]) = %d, w_dims[%d][1] = %d.",
            i, phi::product(b_dims[i]), i, w_dims[i][1]));
  }
  ctx->SetOutputDim("Out", {i_dims[0], w_dims[sz - 1][1]});
  ctx->ShareLoD("X", /*->*/ "Out");
}

framework::OpKernelType FusionRepeatedFCReluOp::GetExpectedKernelType(
    const framework::ExecutionContext& ctx) const {
  return framework::OpKernelType(
      OperatorWithKernel::IndicateVarDataType(ctx, "X"), ctx.GetPlace());
}

void FusionRepeatedFCReluOpMaker::Make() {
  AddInput("X", "(LoDTensor) Input tensors of this operator.");
  AddInput("W", "(Tensor) The weight tensors of this operator.").AsDuplicable();
  AddInput("Bias", "(Tensor) The bias tensors of this operator.")
      .AsDuplicable();
  AddOutput("ReluOut", "(Tensor) The output tensor of each relu operator.")
      .AsDuplicable()
      .AsIntermediate();
  AddOutput("Out", "(LoDTensor) Output tensor of this operator.");
  AddComment(R"DOC(
  Fusion Repeated FC with Relu Operator.
)DOC");
}

template <typename T>
static void fc_relu(const T* x, const T* w, const T* b, T* y,
                    const jit::matmul_attr_t& attr) {
  auto matmul =
      jit::KernelFuncs<jit::MatMulTuple<T>, platform::CPUPlace>::Cache().At(
          attr);
  auto addbias_relu =
      jit::KernelFuncs<jit::VAddReluTuple<T>, platform::CPUPlace>::Cache().At(
          attr.n);
  matmul(x, w, y, &attr);
  T* dst = y;
  for (int i = 0; i < attr.m; ++i) {
    addbias_relu(b, dst, dst, attr.n);
    dst += attr.n;
  }
}

template <typename T>
class FusionRepeatedFCReluKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto in = ctx.Input<Tensor>("X");
    auto weights = ctx.MultiInput<Tensor>("W");
    auto biases = ctx.MultiInput<Tensor>("Bias");
    auto relus = ctx.MultiOutput<Tensor>("ReluOut");
    auto* out = ctx.Output<Tensor>("Out");
    auto place = ctx.GetPlace();
    int weight_sz = static_cast<int>(weights.size());

    auto i_dims = in->dims();
    auto w_dims = weights[0]->dims();
    jit::matmul_attr_t attr;
    attr.m = i_dims[0];
    attr.n = w_dims[1];
    attr.k = w_dims[0];
    relus[0]->Resize({attr.m, attr.n});
    fc_relu(in->data<T>(), weights[0]->data<T>(), biases[0]->data<T>(),
            relus[0]->mutable_data<T>(place), attr);

    for (int i = 1; i < weight_sz - 1; ++i) {
      auto i_dims = relus[i - 1]->dims();
      auto w_dims = weights[i]->dims();
      attr.m = i_dims[0];
      attr.n = w_dims[1];
      attr.k = w_dims[0];
      relus[i]->Resize({attr.m, attr.n});
      fc_relu(relus[i - 1]->data<T>(), weights[i]->data<T>(),
              biases[i]->data<T>(), relus[i]->mutable_data<T>(place), attr);
    }

    auto i_dims_last = relus[weight_sz - 2]->dims();
    auto w_dims_last = weights[weight_sz - 1]->dims();
    attr.m = i_dims_last[0];
    attr.n = w_dims_last[1];
    attr.k = w_dims_last[0];
    fc_relu(relus[weight_sz - 2]->data<T>(), weights[weight_sz - 1]->data<T>(),
            biases[weight_sz - 1]->data<T>(), out->mutable_data<T>(place),
            attr);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(fusion_repeated_fc_relu, ops::FusionRepeatedFCReluOp,
                  ops::FusionRepeatedFCReluOpMaker);

REGISTER_OP_CPU_KERNEL(fusion_repeated_fc_relu,
                       ops::FusionRepeatedFCReluKernel<float>,
                       ops::FusionRepeatedFCReluKernel<double>);
