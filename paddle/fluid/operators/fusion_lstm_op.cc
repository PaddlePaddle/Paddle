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

#include "paddle/fluid/operators/fusion_lstm_op.h"
#include <string>
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/math/detail/activation_functions.h"
#include "paddle/fluid/operators/math/lstm_compute.h"
#include "paddle/fluid/operators/math/sequence2batch.h"
DECLARE_int32(paddle_num_threads);

namespace paddle {
namespace operators {

void FusionLSTMOp::InferShape(framework::InferShapeContext* ctx) const {
  PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) of LSTM should not be null.");
  PADDLE_ENFORCE(ctx->HasInput("WeightX"),
                 "Input(WeightX) of LSTM should not be null.");
  PADDLE_ENFORCE(ctx->HasInput("WeightH"),
                 "Input(WeightH) of LSTM should not be null.");
  PADDLE_ENFORCE(ctx->HasInput("Bias"),
                 "Input(Bias) of LSTM should not be null.");

  PADDLE_ENFORCE(ctx->HasOutput("XX"),
                 "Output(XX) of LSTM should not be null.");
  PADDLE_ENFORCE(ctx->HasOutput("Hidden"),
                 "Output(Hidden) of LSTM should not be null.");
  PADDLE_ENFORCE(ctx->HasOutput("Cell"),
                 "Output(Cell) of LSTM should not be null.");
  PADDLE_ENFORCE(ctx->HasOutput("BatchedGate"),
                 "Output(BatchedGate) of LSTM should not be null.");
  PADDLE_ENFORCE(ctx->HasOutput("BatchCellPreAct"),
                 "Output(BatchedGate) of LSTM should not be null.");

  auto x_dims = ctx->GetInputDim("X");
  PADDLE_ENFORCE_EQ(x_dims.size(), 2, "Input(X)'s rank must be 2.");

  if (ctx->HasInput("H0")) {
    PADDLE_ENFORCE(ctx->HasInput("C0"),
                   "Input(Cell) and Input(Hidden) of LSTM should not "
                   "be null at the same time.");
    auto h_dims = ctx->GetInputDim("H0");
    auto c_dims = ctx->GetInputDim("C0");
    PADDLE_ENFORCE(h_dims == c_dims,
                   "The dimension of Input(H0) and Input(C0) "
                   "should be the same.");
  }

  auto wx_dims = ctx->GetInputDim("WeightX");
  PADDLE_ENFORCE_EQ(wx_dims.size(), 2,
                    "The rank of Input(WeightX) should be 2.");
  PADDLE_ENFORCE_EQ(wx_dims[0], x_dims[1],
                    "The first dimension of Input(WeightX) "
                    "should be %d.",
                    x_dims[1]);

  int frame_size = wx_dims[1] / 4;
  auto wh_dims = ctx->GetInputDim("WeightH");
  PADDLE_ENFORCE_EQ(wh_dims.size(), 2,
                    "The rank of Input(WeightH) should be 2.");
  PADDLE_ENFORCE_EQ(wh_dims[0], frame_size,
                    "The first dimension of Input(WeightH) "
                    "should be %d.",
                    frame_size);
  PADDLE_ENFORCE_EQ(wh_dims[1], 4 * frame_size,
                    "The second dimension of Input(WeightH) "
                    "should be 4 * %d.",
                    frame_size);

  auto b_dims = ctx->GetInputDim("Bias");
  PADDLE_ENFORCE_EQ(b_dims.size(), 2, "The rank of Input(Bias) should be 2.");
  PADDLE_ENFORCE_EQ(b_dims[0], 1,
                    "The first dimension of Input(Bias) should be 1.");

  PADDLE_ENFORCE(!ctx->Attrs().Get<bool>("use_peepholes"),
                 "Do not support peephole yet.");
  PADDLE_ENFORCE_EQ(b_dims[1], 4 * frame_size,
                    "The second dimension of Input(Bias) should be "
                    "4 * %d if disable peepholes connection",
                    frame_size);

  framework::DDim out_dims({x_dims[0], frame_size});
  ctx->SetOutputDim("Hidden", out_dims);
  ctx->SetOutputDim("Cell", out_dims);
  ctx->SetOutputDim("BatchedGate", {x_dims[0], wx_dims[1]});
  ctx->SetOutputDim("BatchCellPreAct", out_dims);
  ctx->ShareLoD("X", "Hidden");
  ctx->ShareLoD("X", "Cell");

  int xx_width = x_dims[1] > wx_dims[1] ? wx_dims[1] : x_dims[1];
  ctx->SetOutputDim("XX", {x_dims[0], xx_width});
  ctx->ShareLoD("X", "XX");
}

framework::OpKernelType FusionLSTMOp::GetExpectedKernelType(
    const framework::ExecutionContext& ctx) const {
  return framework::OpKernelType(
      framework::ToDataType(ctx.Input<framework::LoDTensor>("X")->type()),
      ctx.device_context());
}

void FusionLSTMOpMaker::Make() {
  AddInput("X",
           "(LoDTensor) the input is a LodTensor, which support "
           "variable-time length input sequence. The underlying tensor in "
           "this LoDTensor is a matrix with shape (T X M), where T is the "
           "total time steps in this mini-batch, M is the dim size of x.");
  AddInput("WeightX",
           "(Tensor) the learnable weights of X."
           " - The shape is (M x 4D), where M is the dim size of x, D is the "
           "hidden size. "
           " - Weight = {W_cx, W_ix, W_fx, W_ox}");
  AddInput("WeightH",
           "(Tensor) same as LSTMOp, the learnable hidden-hidden weights."
           " - The shape is (D x 4D), where D is the hidden size. "
           " - Weight = {W_ch, W_ih, W_fh, W_oh}");
  AddInput("Bias",
           "(Tensor) the learnable weights. Almost same as LSTMOp"
           "Note: we should add the fc bias into this (1x4D) in bias."
           "input-hidden bias weight and peephole connections weight if "
           "setting `use_peepholes` True. "
           "1. `use_peepholes = False` "
           " - The shape is (1 x 4D). "
           " - Bias = {b_c, b_i, b_f, b_o}."
           "2. `use_peepholes = True` "
           " - The shape is (1 x 7D). "
           " - Bias = {b_c, b_i, b_f, b_o, W_ic, W_fc, W_oc}.");
  AddInput("H0",
           "(Tensor, optional) (same as LSTMOp) the initial hidden state is an "
           "optional "
           "input. This is a tensor with shape (N x D), where N is the "
           "batch size and D is the hidden size.")
      .AsDispensable();
  AddInput("C0",
           "(Tensor, optional) (same as LSTMOp) (the initial cell state is an "
           "optional "
           "input. This is a tensor with shape (N x D), where N is the "
           "batch size. `H0` and `C0` can be NULL but only at the same time.")
      .AsDispensable();
  AddOutput("Hidden",
            "(LoDTensor) (same as LSTMOp) the hidden state of LSTM operator. "
            "The shape is (T x D), and lod is the same with the `Input`.");
  AddOutput("Cell",
            "(LoDTensor) (same as LSTMOp) the cell state of LSTM operator. "
            "The shape is (T x D), and lod is the same with the `Input`.");
  AddOutput("XX",
            "(LoDTensor) the result after X * WeightX (size is T x 4D)"
            " or batched_X (size is T x M), this will be automatically chosen,"
            " where T is the total time steps in this mini-batch,"
            " D is the hidden size, M is the dim size of x input.")
      .AsIntermediate();
  AddOutput("BatchedGate", "(LoDTensor) (same as LSTMOp).").AsIntermediate();
  AddOutput("BatchCellPreAct", "(LoDTensor) (same as LSTMOp).")
      .AsIntermediate();
  AddAttr<bool>("use_peepholes",
                "(bool, defalut: True) "
                "whether to enable diagonal/peephole connections.")
      .SetDefault(true);
  AddAttr<bool>("is_reverse",
                "(bool, defalut: False) "
                "whether to compute reversed LSTM.")
      .SetDefault(false);
  AddAttr<std::string>("gate_activation",
                       "(string, default: sigmoid)"
                       "The activation for input gate, forget gate and output "
                       "gate, `sigmoid` by default.")
      .SetDefault("sigmoid")
      .InEnum({"sigmoid", "tanh", "relu", "identity"});
  AddAttr<std::string>("cell_activation",
                       "(string, default: tanh)"
                       "The activation for cell output, `tanh` by defalut.")
      .SetDefault("tanh")
      .InEnum({"sigmoid", "tanh", "relu", "identity"});
  AddAttr<std::string>("candidate_activation",
                       "(string, default: tanh)"
                       "The activation for candidate hidden state, "
                       "`tanh` by default.")
      .SetDefault("tanh")
      .InEnum({"sigmoid", "tanh", "relu", "identity"});
  AddComment(R"DOC(
Fusion Long-Short Term Memory (LSTM) Operator.
This operator fuse the X into LSTM, more details can refer to LSTM op.
)DOC");
}

template <typename DeviceContext, typename T>
inline void ReorderInitState(const DeviceContext& ctx,
                             const framework::Tensor& src,
                             framework::Vector<size_t> index_lod,
                             framework::Tensor* dst, bool indexed_src) {
  math::CopyMatrixRowsFunctor<DeviceContext, T> row_shuffle;
  dst->mutable_data<T>(src.dims(), ctx.GetPlace());
  // TODO(TJ): check mem copy perf
  row_shuffle(ctx, src, index_lod, dst, indexed_src);
}

// TODO(TJ): can move to math::details
template <typename DeviceContext, typename T>
inline void SimpleFC(const math::BlasT<DeviceContext, T>& blas, const int M,
                     const int N, const int K, const T* A, const T* B, T* C,
                     const T* bias_data = NULL) {
  blas.GEMM(CblasNoTrans, CblasNoTrans, M, N, K, static_cast<T>(1), A, B,
            static_cast<T>(0), C);
  if (bias_data) {
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for if (FLAGS_paddle_num_threads > 1)
#endif
    for (int i = 0; i < M; i++) {
      blas.AXPY(N, static_cast<T>(1), bias_data, C + i * N);
    }
  }
}

template <typename DeviceContext, typename T>
class FuisonLSTMKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<LoDTensor>("X");
    auto* wx = ctx.Input<Tensor>("WeightX");
    auto* wh = ctx.Input<Tensor>("WeightH");
    auto* bias = ctx.Input<Tensor>("Bias");
    auto* hidden_t0 = ctx.Input<Tensor>("H0");
    auto* cell_t0 = ctx.Input<Tensor>("C0");

    auto* xx = ctx.Output<LoDTensor>("XX");
    auto* batched_gate = ctx.Output<LoDTensor>("BatchedGate");
    auto* hidden_out = ctx.Output<LoDTensor>("Hidden");
    auto* cell_out = ctx.Output<LoDTensor>("Cell");
    bool is_reverse = ctx.Attr<bool>("is_reverse");

    T* xx_data = xx->mutable_data<T>(ctx.GetPlace());
    T* batched_gate_data = batched_gate->mutable_data<T>(ctx.GetPlace());
    hidden_out->mutable_data<T>(ctx.GetPlace());
    cell_out->mutable_data<T>(ctx.GetPlace());

    const T* x_data = x->data<T>();
    const T* wx_data = wx->data<T>();
    auto x_dims = x->dims();
    auto wx_dims = wx->dims();

    math::LoDTensor2BatchFunctor<DeviceContext, T> to_batch;
    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    auto blas = math::GetBlas<DeviceContext, T>(dev_ctx);
    if (x_dims[1] > wx_dims[1]) {
      SimpleFC<DeviceContext, T>(blas, x_dims[0], wx_dims[1], x_dims[1], x_data,
                                 wx_data, xx_data, bias->data<T>());
      to_batch(dev_ctx, *xx, batched_gate, true, is_reverse);
    } else {
      to_batch(dev_ctx, *x, xx, true, is_reverse);
      SimpleFC<DeviceContext, T>(blas, x_dims[0], wx_dims[1], x_dims[1],
                                 xx_data, wx_data, batched_gate_data,
                                 bias->data<T>());
    }

    int frame_size = static_cast<int>(wx_dims[1] / 4);
    framework::DDim out_dims({x_dims[0], frame_size});
    math::LstmMetaValue<T> lstm_value;
    // no peephole
    lstm_value.check_ig = nullptr;
    lstm_value.check_fg = nullptr;
    lstm_value.check_og = nullptr;
    lstm_value.prev_state_value = nullptr;
    Tensor ordered_c0;

    framework::Vector<size_t> order(batched_gate->lod()[2]);

    if (cell_t0) {
      // Since the batch computing for LSTM reorders the input sequence
      // according to their length. The initialized cell state also needs
      // to reorder.
      ReorderInitState<DeviceContext, T>(dev_ctx, *cell_t0, order, &ordered_c0,
                                         true);
      lstm_value.prev_state_value = ordered_c0.data<T>();
    }

    // Use the local variable as here.
    LoDTensor batch_hidden, batch_cell;
    auto* batch_cell_pre_act = ctx.Output<LoDTensor>("BatchCellPreAct");
    batch_hidden.mutable_data<T>(out_dims, ctx.GetPlace());
    batch_cell.mutable_data<T>(out_dims, ctx.GetPlace());
    batch_cell_pre_act->mutable_data<T>(out_dims, ctx.GetPlace());

    auto batch_starts = batched_gate->lod()[0];
    size_t max_seq_len = batch_starts.size() - 1;
    auto gate_act = math::detail::GetActivationType(
        ctx.Attr<std::string>("gate_activation"));
    auto cell_act = math::detail::GetActivationType(
        ctx.Attr<std::string>("cell_activation"));
    auto cand_act = math::detail::GetActivationType(
        ctx.Attr<std::string>("candidate_activation"));

    for (size_t n = 0; n < max_seq_len; n++) {
      int bstart = static_cast<int>(batch_starts[n]);
      int bend = static_cast<int>(batch_starts[n + 1]);

      Tensor gate_t = batched_gate->Slice(bstart, bend);
      Tensor out_t = batch_hidden.Slice(bstart, bend);
      Tensor cell_t = batch_cell.Slice(bstart, bend);
      Tensor cell_pre_act_t = batch_cell_pre_act->Slice(bstart, bend);

      int cur_batch_size = bend - bstart;

      if (n > 0) {
        int pre_h_start = static_cast<int>(batch_starts[n - 1]);
        int pre_h_end = pre_h_start + cur_batch_size;
        auto pre_hidden_t = batch_hidden.Slice(pre_h_start, pre_h_end);
        // TODO(TJ): use gemm directly
        blas.MatMul(pre_hidden_t, false, *wh, false, static_cast<T>(1.0),
                    &gate_t, static_cast<T>(1.0));
      } else if (hidden_t0) {
        // TODO(TJ): move h0 outside for
        // If n == 0 and there is no initialized hidden state, that is to say
        // the H0 is zeros, the calculation W_h * H0 will be skiped.
        // If n == 0 and there is initialized hidden state, calculate W_h * H0.

        // Since the batch computing for LSTM reorders the input sequence
        // according to their length. The initialized hidden state also needs
        // to reorder.
        Tensor ordered_h0;
        ReorderInitState<DeviceContext, T>(dev_ctx, *hidden_t0, order,
                                           &ordered_h0, true);
        // TODO(TJ): use gemm directly
        blas.MatMul(ordered_h0, false, *wh, false, static_cast<T>(1.0), &gate_t,
                    static_cast<T>(1.0));
      }

      lstm_value.gate_value = gate_t.data<T>();
      lstm_value.output_value = out_t.data<T>();
      lstm_value.state_value = cell_t.data<T>();
      lstm_value.state_active_value = cell_pre_act_t.data<T>();
      math::LstmUnitFunctor<DeviceContext, T>::compute(
          dev_ctx, lstm_value, frame_size, cur_batch_size, gate_act, cell_act,
          cand_act);
      lstm_value.prev_state_value = lstm_value.state_value;
    }

    math::Batch2LoDTensorFunctor<DeviceContext, T> to_seq;
    batch_hidden.set_lod(batched_gate->lod());
    // restore the output hidden in LoDTensor from the batch hidden
    to_seq(dev_ctx, batch_hidden, hidden_out);

    batch_cell.set_lod(batched_gate->lod());
    // restore the output cell state in LoDTensor from the batch cell
    to_seq(dev_ctx, batch_cell, cell_out);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(fusion_lstm, ops::FusionLSTMOp, ops::FusionLSTMOpMaker,
                  paddle::framework::DefaultGradOpDescMaker<true>);

REGISTER_OP_CPU_KERNEL(
    fusion_lstm,
    ops::FuisonLSTMKernel<paddle::platform::CPUDeviceContext, float>,
    ops::FuisonLSTMKernel<paddle::platform::CPUDeviceContext, double>);
