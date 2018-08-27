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

#include "paddle/fluid/operators/fusion_seq_concat_fc_op.h"
#include <string>
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/math/cpu_vec.h"
#include "paddle/fluid/operators/math/fc_compute.h"
#include "paddle/fluid/platform/cpu_info.h"

namespace paddle {
namespace operators {

void FusionSeqConcatFCOp::InferShape(framework::InferShapeContext* ctx) const {
  PADDLE_ENFORCE(ctx->HasInput("X"),
                 "Input(X) of FusionSeqConcatFC should not be null.");
  PADDLE_ENFORCE(ctx->HasInput("FCWeight"),
                 "Input(FCWeight) of FusionSeqConcatFC should not be null.");

  PADDLE_ENFORCE(ctx->HasOutput("Out"),
                 "Output(Out) of FusionSeqConcatFC should not be null.");
  PADDLE_ENFORCE(ctx->HasOutput("FCOut"),
                 "Output(FCOut) of FusionSeqConcatFC should not be null.");

  // need check fc height = all inputs width sum

  auto x_dims = ctx->GetInputDim("X");
  const int M = x_dims[1];
  PADDLE_ENFORCE_EQ(x_dims.size(), 2, "Input(X)'s rank must be 2.");

  auto w_dims = ctx->GetInputDim("LSTMWeight");
  const int D = w_dims[1] / 4;
  PADDLE_ENFORCE_EQ(w_dims.size(), 2, "Input(LSTMWeight)'s rank must be 2.");
  PADDLE_ENFORCE_EQ(w_dims[0], D + M,
                    "LSTMWeight dims should be (%d + %d) * %d.", D + M, 4 * D);

  auto b_dims = ctx->GetInputDim("LSTMBias");
  PADDLE_ENFORCE_EQ(b_dims.size(), 2, "Input(LSTMBias)'s rank must be 2.");
  PADDLE_ENFORCE_EQ(b_dims[0], 1, "LSTMBias dims should be 1 x %d.", 4 * D);
  PADDLE_ENFORCE_EQ(b_dims[1], 4 * D, "LSTMBias dims should be 1 x %d.", 4 * D);

  auto c_dims = ctx->GetInputDim("C0");
  PADDLE_ENFORCE_EQ(c_dims.size(), 2, "Input(C0)'s rank must be 2.");
  PADDLE_ENFORCE_EQ(c_dims[1], D, "C0 dims should be N x %d.", D);
  if (ctx->HasInput("H0")) {
    auto h_dims = ctx->GetInputDim("H0");
    PADDLE_ENFORCE(h_dims == c_dims,
                   "The dimension of Input(H0) and Input(C0) "
                   "should be the same.");
  }

  auto atten_w_dims = ctx->GetInputDim("AttentionWeight");
  PADDLE_ENFORCE_EQ(atten_w_dims.size(), 2,
                    "Input(AttentionWeight)'s rank must be 2.");
  PADDLE_ENFORCE_EQ(atten_w_dims[0], M + D,
                    "AttentionWeight shapes must be (%d + %d) * 1.", M, D);
  PADDLE_ENFORCE_EQ(atten_w_dims[1], 1,
                    "AttentionWeight shapes must be (%d + %d) * 1.", M, D);
  if (ctx->HasInput("AttentionBias")) {
    auto atten_b_dims = ctx->GetInputDim("AttentionBias");
    PADDLE_ENFORCE_EQ(atten_b_dims.size(), 2,
                      "Input(AttentionBias)'s rank must be 2.");
    PADDLE_ENFORCE_EQ(atten_b_dims[0], 1,
                      "AttentionBias shapes must be 1 * 1.");
    PADDLE_ENFORCE_EQ(atten_b_dims[1], 1,
                      "AttentionBias shapes must be 1 * 1.");
  }

  if (ctx->HasInput("AttentionScalar")) {
    auto dims = ctx->GetInputDim("AttentionScalar");
    PADDLE_ENFORCE_EQ(dims.size(), 2,
                      "Input(AttentionScalar)'s rank must be 2.");
    PADDLE_ENFORCE_EQ(dims[0], 1, "AttentionScalar shapes must be 1 * 1.");
    PADDLE_ENFORCE_EQ(dims[1], 1, "AttentionScalar shapes must be 1 * 1.");
  }

  if (ctx->HasInput("AttentionScalarBias")) {
    auto dims = ctx->GetInputDim("AttentionScalarBias");
    PADDLE_ENFORCE(
        ctx->HasInput("AttentionScalar"),
        "AttentionScalar should not be null when have AttentionScalarBias.");
    PADDLE_ENFORCE_EQ(dims.size(), 2,
                      "Input(AttentionScalarBias)'s rank must be 2.");
    PADDLE_ENFORCE_EQ(dims[0], 1, "AttentionScalarBias shapes must be 1 * 1.");
    PADDLE_ENFORCE_EQ(dims[1], 1, "AttentionScalarBias shapes must be 1 * 1.");
  }

  framework::DDim out_dims({x_dims[0], D});
  ctx->SetOutputDim("Hidden", out_dims);
  ctx->SetOutputDim("Cell", out_dims);
  ctx->SetOutputDim("AttentionedX", {x_dims[0], 1});
  ctx->SetOutputDim("LSTMX", {1, M});
  ctx->SetOutputDim("LSTMOUT", {1, 4 * D});
  // AttentionFCOut should be reshape as (maxseqlen,1) in runtime
  ctx->ShareLoD("X", "Hidden");
  ctx->ShareLoD("X", "Cell");

  ctx->SetOutputDim("Out", out_dims);
  ctx->ShareLoD("X", /*->*/ "Out");
}

framework::OpKernelType FusionSeqConcatFCOp::GetExpectedKernelType(
    const framework::ExecutionContext& ctx) const {
  return framework::OpKernelType(
      framework::ToDataType(ctx.Input<framework::LoDTensor>("X")->type()),
      ctx.device_context());
}

void FusionSeqConcatFCOpMaker::Make() {
  AddInput("X",
           "(LoDTensor) input LodDTensors, the first one must be have ref lod "
           "for sequence expand, and the rest input should have same lod.")
      .AsDuplicable();
  AddInput("FCWeight", "(Tensor) the weights of fc.");
  AddInput("FCBias", "(Tensor, optional) the bias of fc.").AsDispensable();
  AddOutput("Out", "(LoDTensor) Output LodTensor.");
  AddOutput(
      "FCOut",
      "(Tensor) the intermediate tensor to keep the result of fc."
      "Shape is (N x D), where N is the batch size, D is the output dim of fc")
      .AsIntermediate();
  AddAttr<std::string>("fc_activation",
                       "(string, default: identity)"
                       "The activation for the result of fc."
                       "`identity` by default.")
      .SetDefault("identity")
      .InEnum({"sigmoid", "tanh", "relu", "identity"});
  AddComment(R"DOC(
Fusion Sequence expand + concat + fc Operator.

All below conditions should be meet:

The ref_level of seq_expand should be 0.

The ref lod of seq_expand level is the first input of concat.

The other inputs should have same lod and same batch size of ref lod.

The seq len of other inputs should be 1.

The concat axis should be 1.

)DOC");
}

// y[i] = (x[i] + bias[0]) > 0 ? (x[i] + bias[0]) : 0;
template <typename T>
inline void bias_relu(const int n, const T* x, const T* bias, T* y) {
  if (bias) {
    math::vec_add_bias<T, platform::jit::avx>(n, *bias, x, y);
    math::vec_relu<T, platform::jit::avx>(n, y, y);
  } else {
    math::vec_relu<T, platform::jit::avx>(n, x, y);
  }
}

template <typename T>
inline void vec_softmax(const int n, const T* x, T* y) {
  T scalar = x[0];
  // max
  for (int i = 1; i < n; ++i) {
    scalar = scalar < x[i] ? x[i] : scalar;
  }
  math::vec_add_bias<T, platform::jit::avx>(n, -scalar, x, y);  // sub
  math::vec_exp<T>(n, y, y);                                    // exp
  // sum
  scalar = T(0);
  for (int i = 0; i < n; ++i) {
    scalar += y[i];
  }
  math::vec_scal<T>(n, static_cast<T>(1) / scalar, y);  // scale
}

template <typename T>
class FusionSeqConcatFCKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    using DeviceContext = paddle::platform::CPUDeviceContext;
    auto* ins = ctx.Input<LoDTensor>("X");
    auto* w = ctx.Input<Tensor>("FCWeight");
    auto* b = ctx.Input<Tensor>("FCBias");

    auto* out = ctx.Output<LoDTensor>("Out");
    auto* fc_out = ctx.Output<Tensor>("FCOUT");

    std::function<void(const int, const T*, T*)> fc_act;
    auto& fc_act_str = ctx.Attr<std::string>("fc_activation");
    if (platform::jit::MayIUse(platform::jit::avx)) {
      math::VecActivations<T, platform::jit::avx> act_functor;
      fc_act = act_functor(fc_act_str);
    } else {
      math::VecActivations<T, platform::jit::isa_any> act_functor;
      fc_act = act_functor(fc_act_str);
    }

    PADDLE_ENFORCE_GT(ins.size(), 1, "Input(X)'s size must larger than 1.");
    auto* ref_in = ins[0];
    auto ref_in_lod = ref_in->lod();
    const int N = ref_in_lod[0].size() - 1;
    auto ref_in_dims = ref_in->dims();  // T x M0
    auto w_dims = w->dims();            // (M0+M1+M2+..) x D
    const int total_T = ref_in_dims[0];
    const int M0 = ref_in_dims[1];
    const int M1 = ins[1]->dims()[1];
    const int D = w_dims[1];

    const T* ref_in_data =
        ref_in->data<T>();  // size should be check at infershape
    const T* in1_data = ins[1]->data<T>();
    const T* w_data = w->data<T>();
    T* out_data = out->mutable_data<T>(ctx.GetPlace());
    T* fc_out_data = fc_out->mutable_data<T>(ctx.GetPlace());

    auto blas = math::GetBlas<DeviceContext, T>(ctx);
    math::FCCompute<DeviceContext, T>(blas, total_T, D, M0, ref_in_data, w_data,
                                      out_data, b ? b->data<T>() : NULL);
    w_data = w_data + M0 * D;

    // first one use write on
    blas.MatMul(N, D, M1, in1_data, w_data, fc_out_data);
    w_data = w_data + M1 * D;
    for (int i = 2; i < ins.size(); ++i) {
      // add on
      const T* in_data = ins[i]->data<T>();
      const int K = ins[i]->dims()[1];
      blas.GEMM(CblasNoTrans, CblasNoTrans, N, D, K, static_cast<T>(1), in_data,
                K, w_data, D, static_cast<T>(1), fc_out_data, D);
      w_data = w_data + K * D;
    }

    for (int i = 0; i < N; ++i) {
      int seq_len = ref_in_lod[0][i + 1] - ref_in_lod[0][i];
      T* src = fc_out_data + i * D;
      for (int step = 0; step < seq_len; ++step) {
        blas.VADD(D, out_data, src, out_data);
        out_data = out_data + D;
      }
    }

    fc_act(out_dims[0] * out_dims[1], out_data, out_data);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(fusion_seq_concat_fc, ops::FusionSeqConcatFCOp,
                  ops::FusionSeqConcatFCOpMaker,
                  paddle::framework::DefaultGradOpDescMaker<true>);

REGISTER_OP_CPU_KERNEL(fusion_seq_concat_fc,
                       ops::FusionSeqConcatFCKernel<float>,
                       ops::FusionSeqConcatFCKernel<double>);
