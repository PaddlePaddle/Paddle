/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/fused/self_dp_attention_op.h"
#include "paddle/fluid/operators/fused/scaled_dp_attention.h"

namespace paddle {
namespace operators {

void SelfDPAttenOp::InferShape(framework::InferShapeContext* ctx) const {
  OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "SelfDPAtten");
  OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "SelfDPAtten");

  auto dim_input = ctx->GetInputDim("X");
  PADDLE_ENFORCE_EQ(dim_input.size(),
                    5,
                    platform::errors::InvalidArgument(
                        "The size of input X dims should be 5, "
                        "[batchsize, tokensize, 3, nhead, headsize] "
                        ", but now Input X dim is:[%s] ",
                        dim_input));
  framework::DDim out_dims(
      {dim_input[0], dim_input[1], dim_input[3], dim_input[4]});
  ctx->SetOutputDim("Out", out_dims);
  ctx->ShareLoD("X", /*->*/ "Out");
}

phi::KernelKey SelfDPAttenOp::GetExpectedKernelType(
    const framework::ExecutionContext& ctx) const {
  return phi::KernelKey(OperatorWithKernel::IndicateVarDataType(ctx, "X"),
                        ctx.GetPlace());
}

void SelfDPAttenOpMaker::Make() {
  AddInput("X", "(LoDTensor) Input tensors of this operator.");
  AddOutput("Out", "(LoDTensor) Output tensor of this operator.");
  AddAttr<float>("alpha", "The scale of Out").SetDefault(1.0f);
  AddAttr<int>("head_number", "The number of heads of the matrix")
      .SetDefault(1);
  AddComment(R"DOC(
  Multihead Self-scaled-dp-Attention Operator.
)DOC");
}

template <typename T>
class SelfDPAttenKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    using DeviceContext = phi::CPUContext;
    auto* in = ctx.Input<Tensor>("X");
    auto* out = ctx.Output<LoDTensor>("Out");
    auto place = ctx.GetPlace();
    auto* input_d = in->data<T>();
    auto* output_d = out->mutable_data<T>(place);
    float scale = static_cast<float>(ctx.Attr<float>("alpha"));
    int head_number = ctx.Attr<int>("head_number");
    auto input_dims = in->dims();
    // in shouble be (batch * seq * 3 * head_num * head_size)
    // out shouble be (batch * seq * head_num * head_size)
    int batch_size = input_dims[0];
    int seq_len = input_dims[1];
    int head_size = input_dims[4];

    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    phi::DenseTensor temp1 =
        ctx.AllocateTmpTensor<T, DeviceContext>(input_dims, dev_ctx);
    float* trans_input = temp1.mutable_data<float>(place);
    phi::DenseTensor temp2 =
        ctx.AllocateTmpTensor<T, DeviceContext>(input_dims, dev_ctx);
    float* trans_output = temp2.mutable_data<float>(place);

    transpose_before_bmm1<T, float>(
        input_d, trans_input, batch_size, seq_len, head_number, head_size);
    float* query = trans_input;
    float* key = trans_input + batch_size * head_number * seq_len * head_size;
    float* value =
        trans_input + batch_size * head_number * seq_len * head_size * 2;

    scaled_dp_attention(query,
                        key,
                        value,
                        scale,
                        batch_size,
                        seq_len,
                        seq_len,
                        head_number,
                        head_size,
                        trans_output);
    transpose_after_bmm2<float, T>(
        trans_output, output_d, batch_size, seq_len, head_number, head_size);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(self_dp_attention,
                  ops::SelfDPAttenOp,
                  ops::SelfDPAttenOpMaker);

REGISTER_OP_KERNEL(self_dp_attention,
                   CPU,
                   phi::CPUPlace,
                   ops::SelfDPAttenKernel<float>,
                   ops::SelfDPAttenKernel<double>);
