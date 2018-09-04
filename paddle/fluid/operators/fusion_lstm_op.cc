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
#include "paddle/fluid/operators/math/cpu_vec.h"
#include "paddle/fluid/operators/math/fc_compute.h"
#include "paddle/fluid/operators/math/sequence2batch.h"
#include "paddle/fluid/platform/cpu_info.h"

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
  PADDLE_ENFORCE(ctx->HasOutput("BatchedInput"),
                 "Output(BatchedInput) of LSTM should not be null.");
  PADDLE_ENFORCE(ctx->HasOutput("BatchedHidden"),
                 "Output(BatchedHidden) of LSTM should not be null.");
  PADDLE_ENFORCE(ctx->HasOutput("BatchedCell"),
                 "Output(BatchedCell) of LSTM should not be null.");
  PADDLE_ENFORCE(ctx->HasOutput("ReorderedH0"),
                 "Output(ReorderedH0) of LSTM should not be null.");
  PADDLE_ENFORCE(ctx->HasOutput("ReorderedC0"),
                 "Output(ReorderedC0) of LSTM should not be null.");

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
  ctx->SetOutputDim("BatchedInput", {x_dims[0], wx_dims[1]});
  ctx->SetOutputDim("BatchedHidden", out_dims);
  ctx->SetOutputDim("BatchedCell", out_dims);
  ctx->ShareLoD("X", "Hidden");
  ctx->ShareLoD("X", "Cell");

  int xx_width;
  if (ctx->Attrs().Get<bool>("use_seq")) {
    xx_width = wx_dims[1];
  } else {
    xx_width = x_dims[1] > wx_dims[1] ? wx_dims[1] : x_dims[1];
  }
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
  AddOutput("BatchedInput", "(LoDTensor) (T x 4D).").AsIntermediate();
  AddOutput("BatchedHidden", "(LoDTensor) (T x D).").AsIntermediate();
  AddOutput("BatchedCell", "(LoDTensor) (T x D).").AsIntermediate();
  AddOutput("ReorderedH0", "(LoDTensor) (N x D).").AsIntermediate();
  AddOutput("ReorderedC0", "(LoDTensor) (N x D).").AsIntermediate();
  AddAttr<bool>("use_peepholes",
                "(bool, defalut: True) "
                "whether to enable diagonal/peephole connections.")
      .SetDefault(true);
  AddAttr<bool>("is_reverse",
                "(bool, defalut: False) "
                "whether to compute reversed LSTM.")
      .SetDefault(false);
  AddAttr<bool>("use_seq",
                "(bool, defalut: True) "
                "whether to use seq mode to compute.")
      .SetDefault(true);
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

template <typename T>
class FuisonLSTMKernel : public framework::OpKernel<T> {
 public:
#define INIT_VEC_FUNC                                                          \
  std::function<void(const int, const T *, T *)> act_gate, act_cell, act_cand; \
  auto& act_gate_str = ctx.Attr<std::string>("gate_activation");               \
  auto& act_cell_str = ctx.Attr<std::string>("cell_activation");               \
  auto& act_cand_str = ctx.Attr<std::string>("candidate_activation");          \
  if (platform::jit::MayIUse(platform::jit::avx)) {                            \
    math::VecActivations<T, platform::jit::avx> act_functor;                   \
    act_gate = act_functor(act_gate_str);                                      \
    act_cell = act_functor(act_cell_str);                                      \
    act_cand = act_functor(act_cand_str);                                      \
  } else {                                                                     \
    math::VecActivations<T, platform::jit::isa_any> act_functor;               \
    act_gate = act_functor(act_gate_str);                                      \
    act_cell = act_functor(act_cell_str);                                      \
    act_cand = act_functor(act_cand_str);                                      \
  }

#define INIT_BASE_INPUT_OUTPUT                        \
  auto* x = ctx.Input<LoDTensor>("X");                \
  auto* h0 = ctx.Input<Tensor>("H0");                 \
  auto* c0 = ctx.Input<Tensor>("C0");                 \
  auto* wx = ctx.Input<Tensor>("WeightX");            \
  auto* wh = ctx.Input<Tensor>("WeightH");            \
  auto* bias = ctx.Input<Tensor>("Bias");             \
  auto* xx = ctx.Output<LoDTensor>("XX");             \
  auto* hidden_out = ctx.Output<LoDTensor>("Hidden"); \
  auto* cell_out = ctx.Output<LoDTensor>("Cell");     \
  bool is_reverse = ctx.Attr<bool>("is_reverse");

#define INIT_BASE_SIZES                  \
  auto x_dims = x->dims();   /* T x M*/  \
  auto wh_dims = wh->dims(); /* D x 4D*/ \
  const int M = x_dims[1];               \
  const int D = wh_dims[0];              \
  const int D2 = D * 2;                  \
  const int D3 = D * 3;                  \
  const int D4 = wh_dims[1];

  void SeqCompute(const framework::ExecutionContext& ctx) const {
    using DeviceContext = paddle::platform::CPUDeviceContext;
    INIT_BASE_INPUT_OUTPUT
    INIT_BASE_SIZES
    INIT_VEC_FUNC

    auto x_lod = x->lod();
    const int total_T = x_dims[0];
    const int N = x_lod[0].size() - 1;  // batch size

    const T* x_data = x->data<T>();
    const T* h0_data = h0 ? h0->data<T>() : nullptr;
    const T* c0_data = c0 ? c0->data<T>() : nullptr;
    const T* wx_data = wx->data<T>();
    const T* wh_data = wh->data<T>();
    T* xx_data = xx->mutable_data<T>(ctx.GetPlace());
    T* hidden_out_data = hidden_out->mutable_data<T>(ctx.GetPlace());
    T* cell_out_data = cell_out->mutable_data<T>(ctx.GetPlace());

    auto blas = math::GetBlas<DeviceContext, T>(ctx);
    math::FCCompute<DeviceContext, T>(blas, total_T, D4, M, x_data, wx_data,
                                      xx_data, bias->data<T>());
    int xx_offset = D4;
    int gate_offset = D;
    if (is_reverse) {
      const int offset = (total_T - 1) * D;
      xx_data = xx_data + offset * 4;
      hidden_out_data = hidden_out_data + offset;
      cell_out_data = cell_out_data + offset;
      xx_offset = -D4;
      gate_offset = -D;
    }

    auto move_step = [&]() {
      xx_data = xx_data + xx_offset;
      hidden_out_data = hidden_out_data + gate_offset;
      cell_out_data = cell_out_data + gate_offset;
    };

    for (int i = 0; i < N; ++i) {
      int bid = is_reverse ? N - 1 - i : i;
      int seq_len = x_lod[0][bid + 1] - x_lod[0][bid];
      const T* prev_c_data = nullptr;
      const T* prev_h_data = nullptr;
      int tstart = 0;
      if (h0_data) {
        prev_h_data = h0_data + bid * D;
        prev_c_data = c0_data + bid * D;
      } else {
        // W_ch, W_ih, W_fh, W_oh
        act_gate(D3, xx_data + D, xx_data + D);
        act_cand(D, xx_data, xx_data);
        // cell out= input*tilde
        blas.VMUL(D, xx_data, xx_data + D, cell_out_data);
        // hidden out= act_state(cellout) * outgate
        act_cell(D, cell_out_data, xx_data + D2);
        blas.VMUL(D, xx_data + D2, xx_data + D3, hidden_out_data);

        // prev
        prev_h_data = hidden_out_data;
        prev_c_data = cell_out_data;
        tstart = 1;

        move_step();
      }
      for (int step = tstart; step < seq_len; ++step) {
        blas.GEMM(CblasNoTrans, CblasNoTrans, 1, D4, D, static_cast<T>(1),
                  prev_h_data, D, wh_data, D4, static_cast<T>(1), xx_data, D4);

        // W_ch, W_ih, W_fh, W_oh
        act_gate(D3, xx_data + D, xx_data + D);
        act_cand(D, xx_data, xx_data);

        // a = forget * prev_cell
        blas.VMUL(D, xx_data + D2, prev_c_data, xx_data + D2);

        // b = input * tilde
        blas.VMUL(D, xx_data, xx_data + D, xx_data + D);

        // cell out= a+b
        blas.VADD(D, xx_data + D, xx_data + D2, cell_out_data);

        // hidden out= act_state(cellout) * outgate
        act_cell(D, cell_out_data, xx_data + D2);
        blas.VMUL(D, xx_data + D2, xx_data + D3, hidden_out_data);

        // prev
        prev_h_data = hidden_out_data;
        prev_c_data = cell_out_data;

        move_step();
      }
    }
  }

  void BatchCompute(const framework::ExecutionContext& ctx) const {
    using DeviceContext = platform::CPUDeviceContext;
    INIT_BASE_INPUT_OUTPUT
    if (x->lod()[0].size() == 2) {
      SeqCompute(ctx);
      return;
    }
    INIT_BASE_SIZES
    INIT_VEC_FUNC

    auto* reordered_h0 = ctx.Output<Tensor>("ReorderedH0");
    auto* reordered_c0 = ctx.Output<Tensor>("ReorderedC0");
    auto* batched_input = ctx.Output<LoDTensor>("BatchedInput");
    auto* batched_c_out = ctx.Output<LoDTensor>("BatchedCell");
    auto* batched_h_out = ctx.Output<LoDTensor>("BatchedHidden");

    const T* x_data = x->data<T>();
    const T* wx_data = wx->data<T>();
    const T* wh_data = wh->data<T>();
    auto place = ctx.GetPlace();
    T* xx_data = xx->mutable_data<T>(place);
    T* batched_input_data = batched_input->mutable_data<T>(place);
    T* batched_c_out_data = batched_c_out->mutable_data<T>(place);
    T* batched_h_out_data = batched_h_out->mutable_data<T>(place);
    hidden_out->mutable_data<T>(place);
    cell_out->mutable_data<T>(place);

    math::LoDTensor2BatchFunctor<DeviceContext, T> to_batch;
    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    auto blas = math::GetBlas<DeviceContext, T>(dev_ctx);
    if (M > D4) {
      math::FCCompute<DeviceContext, T>(blas, x_dims[0], D4, M, x_data, wx_data,
                                        xx_data, bias->data<T>());
      to_batch(dev_ctx, *xx, batched_input, true, is_reverse);
    } else {
      to_batch(dev_ctx, *x, xx, true, is_reverse);
      batched_input->set_lod(xx->lod());
      math::FCCompute<DeviceContext, T>(blas, x_dims[0], D4, M, xx_data,
                                        wx_data, batched_input_data,
                                        bias->data<T>());
    }

    auto batched_lod = batched_input->lod();
    const auto& seq_order = batched_lod[2];
    const int max_bs = seq_order.size();
    reordered_h0->Resize({max_bs, D});
    reordered_c0->Resize({max_bs, D});

    int tstart = 0;
    T* prev_h_data = nullptr;
    T* prev_c_data = nullptr;
    if (h0) {
      // reorder h0, c0
      T* reordered_h0_data = reordered_h0->mutable_data<T>(place);
      T* reordered_c0_data = reordered_c0->mutable_data<T>(place);
      const T* h0_data = h0->data<T>();
      const T* c0_data = c0->data<T>();
      prev_h_data = reordered_h0_data;
      prev_c_data = reordered_c0_data;
      size_t sz = sizeof(T) * D;
      for (int i = 0; i < max_bs; ++i) {
        std::memcpy(reordered_h0_data, h0_data + seq_order[i] * D, sz);
        std::memcpy(reordered_c0_data, c0_data + seq_order[i] * D, sz);
        reordered_h0_data += D;
        reordered_c0_data += D;
      }
    } else {
      // compute without h0, c0
      T* cur_in_data = batched_input_data;
      T* cur_h_out_data = batched_h_out_data;
      T* cur_c_out_data = batched_c_out_data;
      // W_ch, W_ih, W_fh, W_oh
      for (int i = 0; i < max_bs; ++i) {
        act_gate(D3, cur_in_data + D, cur_in_data + D);
        act_cand(D, cur_in_data, cur_in_data);
        // cell out= input*tilde
        blas.VMUL(D, cur_in_data, cur_in_data + D, cur_c_out_data);
        // hidden out= act_state(cellout) * outgate
        act_cell(D, cur_c_out_data, cur_in_data + D2);
        blas.VMUL(D, cur_in_data + D2, cur_in_data + D3, cur_h_out_data);

        // add offset
        cur_in_data += D4;
        cur_c_out_data += D;
        cur_h_out_data += D;
      }
      tstart = 1;
      prev_h_data = batched_h_out_data;
      prev_c_data = batched_c_out_data;
    }
    // Then start from next
    const auto& batch_starts = batched_lod[0];
    const int max_seq_len = batch_starts.size() - 1;
    const int offset = tstart * max_bs * D;
    batched_input_data = batched_input_data + offset * 4;
    batched_h_out_data = batched_h_out_data + offset;
    batched_c_out_data = batched_c_out_data + offset;
    for (int step = tstart; step < max_seq_len; ++step) {
      const int cur_bs = batch_starts[step + 1] - batch_starts[step];
      blas.GEMM(CblasNoTrans, CblasNoTrans, cur_bs, D4, D, static_cast<T>(1),
                prev_h_data, D, wh_data, D4, static_cast<T>(1),
                batched_input_data, D4);

      T* cur_in_data = batched_input_data;
      T* cur_prev_c_data = prev_c_data;
      T* cur_c_out_data = batched_c_out_data;
      T* cur_h_out_data = batched_h_out_data;
      for (int i = 0; i < cur_bs; ++i) {
        // W_ch, W_ih, W_fh, W_oh
        act_gate(D3, cur_in_data + D, cur_in_data + D);
        act_cand(D, cur_in_data, cur_in_data);
        // a = forget * prev_cell
        blas.VMUL(D, cur_in_data + D2, cur_prev_c_data, cur_in_data + D2);
        // b = input * tilde
        blas.VMUL(D, cur_in_data, cur_in_data + D, cur_in_data + D);
        // cell out= a+b
        blas.VADD(D, cur_in_data + D, cur_in_data + D2, cur_c_out_data);
        // hidden out= act_state(cellout) * outgate
        act_cell(D, cur_c_out_data, cur_in_data + D2);
        blas.VMUL(D, cur_in_data + D2, cur_in_data + D3, cur_h_out_data);

        cur_in_data += D4;
        cur_prev_c_data += D;
        cur_c_out_data += D;
        cur_h_out_data += D;
      }

      prev_c_data = batched_c_out_data;
      prev_h_data = batched_h_out_data;
      batched_c_out_data = cur_c_out_data;
      batched_h_out_data = cur_h_out_data;
      batched_input_data = cur_in_data;
    }

    math::Batch2LoDTensorFunctor<DeviceContext, T> to_seq;
    batched_h_out->set_lod(batched_lod);
    to_seq(dev_ctx, *batched_h_out, hidden_out);
    batched_c_out->set_lod(batched_lod);
    to_seq(dev_ctx, *batched_c_out, cell_out);
  }

  void Compute(const framework::ExecutionContext& ctx) const override {
    if (ctx.Attr<bool>("use_seq")) {
      SeqCompute(ctx);
    } else {
      BatchCompute(ctx);
    }
  }
#undef INIT_BASE_SIZES
#undef INIT_BASE_INPUT_OUTPUT
#undef INIT_VEC_FUNC
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(fusion_lstm, ops::FusionLSTMOp, ops::FusionLSTMOpMaker,
                  paddle::framework::DefaultGradOpDescMaker<true>);

REGISTER_OP_CPU_KERNEL(fusion_lstm, ops::FuisonLSTMKernel<float>,
                       ops::FuisonLSTMKernel<double>);
