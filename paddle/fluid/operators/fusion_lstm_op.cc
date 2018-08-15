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

namespace paddle {
namespace operators {

void FusionLSTMOp::InferShape(framework::InferShapeContext* ctx) const {
  PADDLE_ENFORCE(ctx->HasInput("Input"),
                 "Input(Input) of LSTM should not be null.");
  PADDLE_ENFORCE(ctx->HasInput("Weight"),
                 "Input(Weight) of LSTM should not be null.");
  PADDLE_ENFORCE(ctx->HasInput("Bias"),
                 "Input(Bias) of LSTM should not be null.");

  PADDLE_ENFORCE(ctx->HasOutput("Hidden"),
                 "Output(Hidden) of LSTM should not be null.");
  PADDLE_ENFORCE(ctx->HasOutput("Cell"),
                 "Output(Cell) of LSTM should not be null.");
  PADDLE_ENFORCE(ctx->HasOutput("BatchGate"),
                 "Output(BatchGate) of LSTM should not be null.");
  PADDLE_ENFORCE(ctx->HasOutput("BatchCellPreAct"),
                 "Output(BatchGate) of LSTM should not be null.");

  auto in_dims = ctx->GetInputDim("Input");
  PADDLE_ENFORCE_EQ(in_dims.size(), 2, "Input(X)'s rank must be 2.");

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

  int frame_size = in_dims[1] / 4;
  auto w_dims = ctx->GetInputDim("Weight");
  PADDLE_ENFORCE_EQ(w_dims.size(), 2, "The rank of Input(Weight) should be 2.");
  PADDLE_ENFORCE_EQ(w_dims[0], frame_size,
                    "The first dimension of Input(Weight) "
                    "should be %d.",
                    frame_size);
  PADDLE_ENFORCE_EQ(w_dims[1], 4 * frame_size,
                    "The second dimension of Input(Weight) "
                    "should be 4 * %d.",
                    frame_size);

  auto b_dims = ctx->GetInputDim("Bias");
  PADDLE_ENFORCE_EQ(b_dims.size(), 2, "The rank of Input(Bias) should be 2.");
  PADDLE_ENFORCE_EQ(b_dims[0], 1,
                    "The first dimension of Input(Bias) should be 1.");

  if (ctx->Attrs().Get<bool>("use_peepholes")) {
    PADDLE_ENFORCE_EQ(b_dims[1], 7 * frame_size,
                      "The second dimension of Input(Bias) should be "
                      "7 * %d if enable peepholes connection",
                      frame_size);
  } else {
    PADDLE_ENFORCE_EQ(b_dims[1], 4 * frame_size,
                      "The second dimension of Input(Bias) should be "
                      "4 * %d if disable peepholes connection",
                      frame_size);
  }

  framework::DDim out_dims({in_dims[0], frame_size});
  ctx->SetOutputDim("Hidden", out_dims);
  ctx->SetOutputDim("Cell", out_dims);
  ctx->SetOutputDim("BatchGate", in_dims);
  ctx->SetOutputDim("BatchCellPreAct", out_dims);
  ctx->ShareLoD("Input", "Hidden");
  ctx->ShareLoD("Input", "Cell");
}

framework::OpKernelType FusionLSTMOp::GetExpectedKernelType(
    const framework::ExecutionContext& ctx) const {
  return framework::OpKernelType(
      framework::ToDataType(ctx.Input<framework::LoDTensor>("Input")->type()),
      ctx.device_context());
}

void FusionLSTMOpMaker::Make() {
  AddInput("Input",
           "(LoDTensor) the first input is a LodTensor, which support "
           "variable-time length input sequence. The underlying tensor in "
           "this LoDTensor is a matrix with shape (T X 4D), where T is the "
           "total time steps in this mini-batch, D is the hidden size.");
  AddInput("H0",
           "(Tensor, optional) the initial hidden state is an optional "
           "input. This is a tensor with shape (N x D), where N is the "
           "batch size and D is the hidden size.")
      .AsDispensable();
  AddInput("C0",
           "(Tensor, optional) the initial cell state is an optional "
           "input. This is a tensor with shape (N x D), where N is the "
           "batch size. `H0` and `C0` can be NULL but only at the same time.")
      .AsDispensable();
  AddInput("Weight",
           "(Tensor) the learnable hidden-hidden weights."
           " - The shape is (D x 4D), where D is the hidden size. "
           " - Weight = {W_ch, W_ih, W_fh, W_oh}");
  AddInput("Bias",
           "(Tensor) the learnable weights, which contains two parts: "
           "input-hidden bias weight and peephole connections weight if "
           "setting `use_peepholes` True. "
           "1. `use_peepholes = False` "
           " - The shape is (1 x 4D). "
           " - Bias = {b_c, b_i, b_f, b_o}."
           "2. `use_peepholes = True` "
           " - The shape is (1 x 7D). "
           " - Bias = {b_c, b_i, b_f, b_o, W_ic, W_fc, W_oc}.");
  AddOutput("Hidden",
            "(LoDTensor) the hidden state of LSTM operator. "
            "The shape is (T x D), and lod is the same with the `Input`.");
  AddOutput("Cell",
            "(LoDTensor) the cell state of LSTM operator. "
            "The shape is (T x D), and lod is the same with the `Input`.");
  AddOutput("BatchGate",
            "(LoDTensor) This LoDTensor contains input gate, forget gate "
            "and output gate after the nonlinear computation. This "
            "LoDTensor has the same shape as the reorganized input, which "
            "is also be called batch input. The LoD size is 2. The first "
            "LoD is the batch offsets and the second LoD contains the "
            "indexes, which denote the position of reorganized sequence "
            "in the raw input.")
      .AsIntermediate();
  AddOutput("BatchCellPreAct",
            "(LoDTensor) This LoDTensor is obtained in the forward and used "
            "in the backward.")
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
Long-Short Term Memory (LSTM) Operator.

The defalut implementation is diagonal/peephole connection
(https://arxiv.org/pdf/1402.1128.pdf), the formula is as follows:

$$ i_t = \\sigma(W_{ix}x_{t} + W_{ih}h_{t-1} + W_{ic}c_{t-1} + b_i) $$

$$ f_t = \\sigma(W_{fx}x_{t} + W_{fh}h_{t-1} + W_{fc}c_{t-1} + b_f) $$

$$ \\tilde{c_t} = act_g(W_{cx}x_t + W_{ch}h_{t-1} + b_c) $$

$$ o_t = \\sigma(W_{ox}x_{t} + W_{oh}h_{t-1} + W_{oc}c_t + b_o) $$

$$ c_t = f_t \\odot c_{t-1} + i_t \\odot \\tilde{c_t} $$

$$ h_t = o_t \\odot act_h(c_t) $$

- W terms denote weight matrices (e.g. $W_{xi}$ is the matrix
  of weights from the input gate to the input), $W_{ic}, W_{fc}, W_{oc}$
  are diagonal weight matrices for peephole connections. In our implementation,
  we use vectors to reprenset these diagonal weight matrices.
- The b terms denote bias vectors ($b_i$ is the input gate bias vector).
- $\sigma$ is the non-line activations, such as logistic sigmoid function.
- $i, f, o$ and $c$ are the input gate, forget gate, output gate,
  and cell activation vectors, respectively, all of which have the same size as
  the cell output activation vector $h$.
- The $\odot$ is the element-wise product of the vectors.
- $act_g$ and $act_h$ are the cell input and cell output activation functions
  and `tanh` is usually used for them.
- $\tilde{c_t}$ is also called candidate hidden state,
  which is computed based on the current input and the previous hidden state.

Set `use_peepholes` False to disable peephole connection. The formula
is omitted here, please refer to the paper
http://www.bioinf.jku.at/publications/older/2604.pdf for details.

Note that these $W_{xi}x_{t}, W_{xf}x_{t}, W_{xc}x_{t}, W_{xo}x_{t}$
operations on the input $x_{t}$ are NOT included in this operator.
Users can choose to use fully-connect operator before LSTM operator.

)DOC");
}

template <typename DeviceContext, typename T>
inline void ReorderInitState(const DeviceContext& ctx,
                             const framework::Tensor& src,
                             framework::Vector<size_t> index_lod,
                             framework::Tensor* dst, bool indexed_src) {
  math::CopyMatrixRowsFunctor<DeviceContext, T> row_shuffle;
  dst->mutable_data<T>(src.dims(), ctx.GetPlace());
  row_shuffle(ctx, src, index_lod, dst, indexed_src);
}

template <typename DeviceContext, typename T>
class LSTMKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<LoDTensor>("Input");
    auto* weight = ctx.Input<Tensor>("Weight");
    auto* bias = ctx.Input<Tensor>("Bias");

    auto* hidden_t0 = ctx.Input<Tensor>("H0");
    auto* cell_t0 = ctx.Input<Tensor>("C0");

    auto* batch_gate = ctx.Output<LoDTensor>("BatchGate");
    batch_gate->mutable_data<T>(ctx.GetPlace());
    auto* hidden_out = ctx.Output<LoDTensor>("Hidden");
    hidden_out->mutable_data<T>(ctx.GetPlace());
    auto* cell_out = ctx.Output<LoDTensor>("Cell");
    cell_out->mutable_data<T>(ctx.GetPlace());

    bool is_reverse = ctx.Attr<bool>("is_reverse");
    math::LoDTensor2BatchFunctor<DeviceContext, T> to_batch;
    auto& device_ctx = ctx.template device_context<DeviceContext>();
    to_batch(device_ctx, *input, batch_gate, true, is_reverse);

    auto in_dims = input->dims();
    int frame_size = static_cast<int>(in_dims[1] / 4);
    framework::DDim dims({in_dims[0], frame_size});

    if (bias) {
      Tensor b = *bias;
      b.Resize({bias->numel(), 1});
      Tensor gate_bias = b.Slice(0, 4 * frame_size);
      math::RowwiseAdd<DeviceContext, T> add_bias;
      add_bias(device_ctx, *batch_gate, gate_bias, batch_gate);
    }

    math::LstmMetaValue<T> lstm_value;
    if (bias && ctx.Attr<bool>("use_peepholes")) {
      T* bias_data = const_cast<T*>(bias->data<T>());
      // the code style in LstmMetaValue will be updated later.

      lstm_value.check_ig = bias_data + 4 * frame_size;
      lstm_value.check_fg = lstm_value.check_ig + frame_size;
      lstm_value.check_og = lstm_value.check_fg + frame_size;
    } else {
      lstm_value.check_ig = nullptr;
      lstm_value.check_fg = nullptr;
      lstm_value.check_og = nullptr;
    }
    lstm_value.prev_state_value = nullptr;
    Tensor ordered_c0;

    framework::Vector<size_t> order(batch_gate->lod()[2]);

    if (cell_t0) {
      // Since the batch computing for LSTM reorders the input sequence
      // according to their length. The initialized cell state also needs
      // to reorder.
      ReorderInitState<DeviceContext, T>(device_ctx, *cell_t0, order,
                                         &ordered_c0, true);
      lstm_value.prev_state_value = ordered_c0.data<T>();
    }

    // Use the local variable as here.
    LoDTensor batch_hidden, batch_cell;
    auto* batch_cell_pre_act = ctx.Output<LoDTensor>("BatchCellPreAct");
    batch_hidden.mutable_data<T>(dims, ctx.GetPlace());
    batch_cell.mutable_data<T>(dims, ctx.GetPlace());
    batch_cell_pre_act->mutable_data<T>(dims, ctx.GetPlace());

    auto batch_starts = batch_gate->lod()[0];
    size_t num_batch = batch_starts.size() - 1;
    auto gate_act = math::detail::GetActivationType(
        ctx.Attr<std::string>("gate_activation"));
    auto cell_act = math::detail::GetActivationType(
        ctx.Attr<std::string>("cell_activation"));
    auto cand_act = math::detail::GetActivationType(
        ctx.Attr<std::string>("candidate_activation"));

    auto blas = math::GetBlas<DeviceContext, T>(device_ctx);
    for (size_t n = 0; n < num_batch; n++) {
      int bstart = static_cast<int>(batch_starts[n]);
      int bend = static_cast<int>(batch_starts[n + 1]);

      Tensor gate_t = batch_gate->Slice(bstart, bend);
      Tensor out_t = batch_hidden.Slice(bstart, bend);
      Tensor cell_t = batch_cell.Slice(bstart, bend);
      Tensor cell_pre_act_t = batch_cell_pre_act->Slice(bstart, bend);

      int cur_batch_size = bend - bstart;

      if (n > 0) {
        int pre_h_start = static_cast<int>(batch_starts[n - 1]);
        int pre_h_end = pre_h_start + cur_batch_size;
        auto pre_hidden_t = batch_hidden.Slice(pre_h_start, pre_h_end);
        blas.MatMul(pre_hidden_t, false, *weight, false, static_cast<T>(1.0),
                    &gate_t, static_cast<T>(1.0));
      } else if (hidden_t0) {
        // If n == 0 and there is no initialized hidden state, that is to say
        // the H0 is zeros, the calculation W_h * H0 will be skiped.
        // If n == 0 and there is initialized hidden state, calculate W_h * H0.

        // Since the batch computing for LSTM reorders the input sequence
        // according to their length. The initialized hidden state also needs
        // to reorder.
        Tensor ordered_h0;
        ReorderInitState<DeviceContext, T>(device_ctx, *hidden_t0, order,
                                           &ordered_h0, true);
        blas.MatMul(ordered_h0, false, *weight, false, static_cast<T>(1.0),
                    &gate_t, static_cast<T>(1.0));
      }

      lstm_value.gate_value = gate_t.data<T>();
      lstm_value.output_value = out_t.data<T>();
      lstm_value.state_value = cell_t.data<T>();
      lstm_value.state_active_value = cell_pre_act_t.data<T>();
      math::LstmUnitFunctor<DeviceContext, T>::compute(
          device_ctx, lstm_value, frame_size, cur_batch_size, gate_act,
          cell_act, cand_act);
      lstm_value.prev_state_value = lstm_value.state_value;
    }

    math::Batch2LoDTensorFunctor<DeviceContext, T> to_seq;
    batch_hidden.set_lod(batch_gate->lod());
    // restore the output hidden in LoDTensor from the batch hidden
    to_seq(device_ctx, batch_hidden, hidden_out);

    batch_cell.set_lod(batch_gate->lod());
    // restore the output cell state in LoDTensor from the batch cell
    to_seq(device_ctx, batch_cell, cell_out);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(lstm, ops::LSTMOp, ops::LSTMOpMaker,
                  paddle::framework::DefaultGradOpDescMaker<true>);

REGISTER_OP_CPU_KERNEL(
    fusion_lstm, ops::LSTMKernel<paddle::platform::CPUDeviceContext, float>,
    ops::LSTMKernel<paddle::platform::CPUDeviceContext, double>);
