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

#include "paddle/fluid/operators/attention_lstm_op.h"
#include <string>
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/math/detail/activation_functions.h"
#include "paddle/fluid/operators/math/fc_compute.h"
#include "paddle/fluid/operators/math/lstm_compute.h"
#include "paddle/fluid/operators/math/sequence2batch.h"

#include "paddle/fluid/operators/math/cpu_vec.h"

namespace paddle {
namespace operators {

void AttentionLSTMOp::InferShape(framework::InferShapeContext* ctx) const {
  PADDLE_ENFORCE(ctx->HasInput("X"),
                 "Input(X) of AttentionLSTM should not be null.");
  PADDLE_ENFORCE(ctx->HasInput("C0"),
                 "Input(C0) of AttentionLSTM should not be null.");
  PADDLE_ENFORCE(ctx->HasInput("LSTMWeight"),
                 "Input(LSTMWeight) of AttentionLSTM should not be null.");
  PADDLE_ENFORCE(ctx->HasInput("LSTMBias"),
                 "Input(LSTMBias) of AttentionLSTM should not be null.");
  PADDLE_ENFORCE(ctx->HasInput("AttentionWeight"),
                 "Input(AttentionWeight) of AttentionLSTM should not be null.");

  PADDLE_ENFORCE(ctx->HasOutput("Hidden"),
                 "Output(Hidden) of AttentionLSTM should not be null.");
  PADDLE_ENFORCE(ctx->HasOutput("Cell"),
                 "Output(Cell) of AttentionLSTM should not be null.");
  PADDLE_ENFORCE(ctx->HasOutput("AttentionedX"),
                 "Output(AttentionedX) of AttentionLSTM should not be null.");
  PADDLE_ENFORCE(ctx->HasOutput("AttentionFCOut"),
                 "Output(AttentionFCOut) of AttentionLSTM should not be null.");
  PADDLE_ENFORCE(ctx->HasOutput("LSTMX"),
                 "Output(LSTMX) of AttentionLSTM should not be null.");
  PADDLE_ENFORCE(ctx->HasOutput("LSTMOUT"),
                 "Output(LSTMOUT) of AttentionLSTM should not be null.");

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
  PADDLE_ENFORCE_EQ(b_dims[0], 1, "LSTMBias dims should be 1 x (%d + %d).", M,
                    D);
  PADDLE_ENFORCE_EQ(b_dims[1], M + D, "LSTMBias dims should be 1 x (%d + %d).",
                    M, D);

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
}

framework::OpKernelType AttentionLSTMOp::GetExpectedKernelType(
    const framework::ExecutionContext& ctx) const {
  return framework::OpKernelType(
      framework::ToDataType(ctx.Input<framework::LoDTensor>("X")->type()),
      ctx.device_context());
}

void AttentionLSTMOpMaker::Make() {
  AddInput("X",
           "(LoDTensor) the input is a LodTensor, which support "
           "variable-time length input sequence. The underlying tensor in "
           "this LoDTensor is a matrix with shape (T X M), where T is the "
           "total time steps in this mini-batch, M is the dim size of x.");
  AddInput("C0",
           "(Tensor) LSTM C0"
           "This is a tensor with shape (N x D), where N is the batch size, D "
           "is the gate size."
           "C0 is necessary because of attention.");
  AddInput("H0",
           "(Tensor, optional) LSTM H0"
           "This is a tensor with shape (N x D), where N is the "
           "batch size and D is the gate size.")
      .AsDispensable();
  AddInput("AttentionWeight",
           "(Tensor) the weights of attention fc. Always relu the fc result."
           "The shape is ((M+D) x 1), where M is the dim size of x, D is the "
           "gate size of LSTM.");
  AddInput("AttentionBias, optional",
           "(Tensor) the bias of attention fc."
           "The shape is (1 x 1)")
      .AsDispensable();
  AddInput("AttentionScalar",
           "(Tensor, optional) the scalar on the result of attentioned fc. "
           "Always relu the Scalar."
           "The shape is (1 x 1)")
      .AsDispensable();
  AddInput("AttentionScalarBias",
           "(Tensor, optional) the scalar bias of attention fc."
           "The shape is (1 x 1)")
      .AsDispensable();
  AddInput("LSTMWeight",
           "(Tensor) the combined weight of LSTM"
           " - The shape is ((D+M) x 4D), where D is the hidden gate size, M "
           "is the dim size of x"
           " - Weight = {W_forget, W_input, W_output, W_cell}");
  AddInput("LSTMBias",
           "(Tensor) the combined bias of LSTM, shape (1x4D)."
           "Note: we should add the bias of hidden and context accorindg to "
           "the same gate: "
           "{B_forget, B_input, B_output, B_cell}");
  AddOutput("Hidden",
            "(LoDTensor) (same as LSTMOp) the hidden state of LSTM operator. "
            "The shape is (T x D), and lod is the same with the `Input`.");
  AddOutput("Cell",
            "(LoDTensor) (same as LSTMOp) the cell state of LSTM operator. "
            "The shape is (T x D), and lod is the same with the `Input`.");
  AddOutput("AttentionedX",
            "(Tensor) shape is (T x 1), the result after X * AttentionWeight,"
            " where T is the total time steps in this mini-batch,"
            " D is the hidden size.")
      .AsIntermediate();
  AddOutput("AttentionFCOut",
            "(Tensor) (max_seq_len, 1), compute at each step.")
      .AsIntermediate();
  AddOutput("LSTMX",
            "(Tensor) the input X of LSTM for each step."
            "Shape is (1 x M), where M is the x frame size")
      .AsIntermediate();
  AddOutput(
      "LSTMOUT",
      "(Tensor) the output of LSTM X(1*(D+M))* weight((D+M)*4D) for each step."
      "Shape is (1 x 4D), where M is the x frame size")
      .AsIntermediate();
  // TODO(TJ): InEnum({"sigmoid", "tanh", "relu", "identity"});
  AddAttr<std::string>("gate_activation",
                       "(string, default: sigmoid)"
                       "The activation for input gate, forget gate and output "
                       "gate, `sigmoid` by default.")
      .SetDefault("sigmoid")
      .InEnum({"sigmoid"});
  AddAttr<std::string>("cell_activation",
                       "(string, default: tanh)"
                       "The activation for cell output, `tanh` by defalut.")
      .SetDefault("tanh")
      .InEnum({"tanh"});
  AddAttr<std::string>("candidate_activation",
                       "(string, default: tanh)"
                       "The activation for candidate hidden state, "
                       "`tanh` by default.")
      .SetDefault("tanh")
      .InEnum({"tanh"});
  AddComment(R"DOC(
Attention Long-Short Term Memory (LSTM) Operator.

Attention part:
concat( x(seqlen * M), expand( cell_t-1(1,D) ) ) => tmp(seqlen*(M+D))

tmp(seqlen*(M+D)) * fc((M+D)*1) => fcout(seqlen*1) with bias, relu

fcout(seqlen*1) * scalar => fcout(seqlen*1) with bias, relu

dotmul and sum pool ( fcout(seqlen*1), x(seqlen * M) ) => lstm_x_t(1, M) 

LSTM part:
use lstm_x_t as input and compute as standard LSTM.

)DOC");
}

// y[i] = (x[i] + bias[0]) > 0 ? (x[i] + bias[0]) : 0;
template <typename T>
inline void bias_relu(const int n, const T* x, const T* bias, T* y) {
  if (bias) {
    for (int i = 0; i < n; ++i) {
      y[i] = x[i] + bias[0];
    }
    vec_relu(n, y, y);
  } else {
    vec_relu(n, x, y);
  }
}

template <typename DeviceContext, typename T>
inline void vec_softmax(const BlasT<DeviceContext, T>& blas, const int n,
                        const T* x, T* y) {
  T scalar = x[0];
  // max
  for (int i = 1; i < n; ++i) {
    scalar = scalar < x[i] ? x[i] : scalar;
  }

  // sub
  for (int i = 0; i < n; ++i) {
    y[c] = x[c] - alpha;
  }

  // exp
  blas.VEXP(n, y, y);

  // sum
  scalar = T(0);
  for (int i = 0; i < n; ++i) {
    scalar += y[i];
  }

  // scale
  blas.VSCAL(n, static_cast<T>(1) / scalar, y);
}

__m256 exp(__m256 a) { return exp256_ps(a); }

__m256 log(__m256 a) { return log256_ps(a); }

__m256 sin(__m256 a) { return sin256_ps(a); }

__m256 cos(__m256 a) { return cos256_ps(a); }

__m256 relu(const __m256 a) {
  __m256 tmp = _mm256_set1_ps(0.0f);
  return _mm256_max_ps(a, tmp);
}

__m256 sigmoid(const __m256 a) {
  __m256 max = _mm256_set1_ps(SIGMOID_THRESHOLD_MAX);
  __m256 min = _mm256_set1_ps(SIGMOID_THRESHOLD_MIN);
  __m256 tmp = _mm256_max_ps(a, min);
  tmp = _mm256_min_ps(tmp, max);
  tmp = _mm256_sub_ps(_mm256_set1_ps(0.0f), tmp);
  tmp = exp(tmp);
  tmp = _mm256_add_ps(_mm256_set1_ps(1.0f), tmp);
  tmp = _mm256_div_ps(_mm256_set1_ps(1.0f), tmp);
  return tmp;
}

__m256 tanh(const __m256 a) {
  __m256 max = _mm256_set1_ps(EXP_MAX_INPUT);
  __m256 tmp = _mm256_mul_ps(_mm256_set1_ps(-2.0f), a);
  tmp = _mm256_min_ps(tmp, max);
  tmp = exp(tmp);
  return _mm256_sub_ps(_mm256_div_ps(_mm256_set1_ps(2.0f),
                                     _mm256_add_ps(_mm256_set1_ps(1.0f), tmp)),
                       _mm256_set1_ps(1.0f));
}

__m256 linear(const __m256 a) { return a; }

inline void vec_sigmoid(const T* x, T* y) {
  const real min = SIGMOID_THRESHOLD_MIN;
  const real max = SIGMOID_THRESHOLD_MAX;
  real tmp = (a < min) ? min : ((a > max) ? max : a);
  return 1.0 / (1.0 + exp(-tmp));
}

template <typename DeviceContext, typename T>
class AttentionLSTMKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<LoDTensor>("X");                        // T x M
    auto* h0 = ctx.Input<Tensor>("H0");                         // N x D
    auto* c0 = ctx.Input<Tensor>("C0");                         // N x D
    auto* atten_w = ctx.Input<Tensor>("AttentionWeight");       // (M+D) x 1
    auto* atten_b = ctx.Input<Tensor>("AttentionBias");         // 1x1
    auto* atten_scalar = ctx.Input<Tensor>("AttentionScalar");  // 1x1
    auto* atten_scalar_bias = ctx.Input<Tensor>("AttentionScalar");  // 1x1
    auto* lstm_w = ctx.Input<Tensor>("LSTMWeight");  // (D+M) x D*4
    auto* lstm_b = ctx.Input<Tensor>("LSTMBias");    // 1 x D*4

    auto* hidden_out = ctx.Output<LoDTensor>("Hidden");   // TxD
    auto* cell_out = ctx.Output<LoDTensor>("Cell");       // TxD
    auto* atted_x = ctx.Output<Tensor>("AttentionedX");   // T x 1
    auto* fc_out = ctx.Output<Tensor>('AttentionFCOut');  // max_seq_len x 1
    auto* lstm_x = ctx.Output<Tensor>("LSTMX");           // 1 x M
    auto* lstm_out = ctx.Output<Tensor>("LSTMOUT");       // 1 x 4D

    // some shape should be reshape here since infershape can not get lod info
    auto x_lod = x->lod();
    const int N = x_lod[0].size() - 1;  // batch size
    auto x_dims = x->dims();            // T x M
    auto w_dims = w->dims();            // (D+M) x 4D
    const int M = x_dims[1];            // x frame size
    const int D = w_dims[1] / 4;        // gate frame size
    const int D2 = D * 2;
    const int D3 = D * 3;
    const int D4 = w_dims[1];
    int max_seq_len = x_lod[0][1];
    for (int i = 1; i < N; ++i) {
      int len = x_lod[0][i + 1] - x_lod[0][i];
      max_seq_len = max_seq_len < len ? len : max_seq_len;
    }
    PADDLE_ENFORCE_EQ(x_lod.size(), 1, "Input(X)'s lod size must be 1.");
    PADDLE_ENFORCE_EQ(c0->dims()[0], N, "C0 dims should be %d x %d.", N, D);
    fc_out->Resize({max_seq_len, 1});

    const T* x_data = x->data<T>();
    const T* h0_data = h0->data<T>();
    const T* c0_data = c0->data<T>();
    const T* lstm_w_data = lstm_w->data<T>();
    const T* lstm_b_data = lstm_b->data<T>();
    const T* atten_w_data = atten_w->data<T>();
    const T* atten_b_data = atten_b ? atten_b->data<T>() : NULL;
    const T* atten_scalar_data = atten_scalar ? atten_scalar->data<T>() : NULL;
    const T* atten_scalar_bias_data =
        atten_scalar_bias ? atten_scalar_bias->data<T>() : NULL;

    T* hidden_out_data = hidden_out->mutable_data<T>();
    T* cell_out_data = cell_out->mutable_data<T>();
    T* atted_x_data = atted_x->mutable_data<T>();
    T* fc_out_data = fc_out->mutable_data<T>();
    T* lstm_x_data = lstm_x->mutable_data<T>();
    T* lstm_out_data = lstm_out->mutable_data<T>();

    // x(TxM) * fc (Mx1) part of atten_wgt(M+D)x1
    auto blas = math::GetBlas<DeviceContext, T>(ctx);
    math::FCCompute<DeviceContext, T>(blas, T, 1, M, x_data, atten_w_data,
                                      atted_x_data, atten_b_data);

    const T* cur_x_data = x_data;
    const T* prev_cell_data = NULL;
    const T* prev_hidden_data = NULL;
    T* cur_cell_out_data = cell_out_data;
    T* cur_hidden_out_data = hidden_out_data;
    for (int i = 0; i < N; ++i) {
      int seq_len = x_lod[0][i + 1];
      prev_cell_data = c0_data + i * D;
      prev_hidden_data = h0 ? h0_data + i * D : NULL;

      for (int step = 0; step < seq_len; ++step) {
        /// compute attention vector
        // prev_cell(1xD) * fc(D) rest part of atten_wgt
        // T  = cblas_dot();
        T prev_cell_bias = blas.DOT(D, prev_cell_data, atten_w_data + M);
        // add cell bias and relu
        bias_relu<T>(seq_len, atted_x_data, &prev_cell_bias, fc_out_data);
        // fc2: scalar
        if (atten_scalar_data) {
          // x = a*x
          blas.SCAL(seq_len, atten_scalar_data, fc_out_data);
          bias_relu<T>(seq_len, fc_out_data, atten_scalar_bias_data,
                       fc_out_data);
        }
        vec_softmax<DeviceContext, T>(blas, seq_len, fc_out_data, fc_out_data);
        // mul x(seq_len*M) and sum pool
        math::FCCompute<DeviceContext, T>(blas, 1, M, seq_len, fc_out_data,
                                          cur_x_data, lstm_x_data);

        /// compute LSTM step
        // lstm weight : concat[forget , input , output , tilde]
        // shape : (D + M) x (4 * D)
        // fc inputX(1xM) * weightX(M*(4D))  => 1 x 4D
        blas.MatMul(1, D4, M, lstm_x_data, lstm_w_data + D * D4, lstm_out_data);
        if (prev_hidden_data) {
          blas.GEMM(CblasNoTrans, CblasNoTrans, 1, D4, D, static_cast<T>(1),
                    prev_hidden_data, D, lstm_w_data, D4, static_cast<T>(1),
                    lstm_out_data, D4);
        }
        // since input is 1xM, so can use add bias
        blas.VADD(D4, lstm_b_data, lstm_out_data, lstm_out_data);

        // gate act: sigmoid
        vec_sigmoid(D3, lstm_out_data, lstm_out_data);
        // candicate act: tanh
        vec_tanh(D, lstm_out_data + D3, lstm_out_data + D3);

        // a = forget * prev_cell
        blas.VMUL(D, lstm_out_data, prev_cell_data, lstm_out_data);

        // b = input * tilde
        blas.VMUL(D, lstm_out_data + D, lstm_out + D3, lstm_out_data + D);

        // cell_out = a + b
        blas.VADD(D, lstm_out_data, lstm_out_data + D, cur_cell_out_data);

        // state act tanh(cell_out) * output_gate
        vec_tanh(D, cur_cell_out_data, lstm_out_data);
        blas.VMUL(D, lstm_out_data, lstm_out + D2, cur_hidden_out_data);

        prev_hidden_data = hidden_out + i * gate_size;
        prev_cell_data = cur_cell_out_data;
        cur_cell_out_data = cur_cell_out_data + D;
        cur_hidden_out_data = cur_hidden_out_data + D;
      }
      cur_x_data = cur_x_data + seq_len * M;
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(attention_lstm, ops::AttentionLSTMOp,
                  ops::AttentionLSTMOpMaker,
                  paddle::framework::DefaultGradOpDescMaker<true>);

REGISTER_OP_CPU_KERNEL(
    attention_lstm,
    ops::AttentionLSTMKernel<paddle::platform::CPUDeviceContext, float>,
    ops::AttentionLSTMKernel<paddle::platform::CPUDeviceContext, double>);
