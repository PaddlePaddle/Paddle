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

#include "paddle/fluid/operators/fused/fused_embedding_fc_lstm_op.h"

#include <string>

#include "paddle/fluid/platform/cpu_info.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/cpu_vec.h"
#include "paddle/phi/kernels/funcs/sequence2batch.h"

namespace paddle {
namespace operators {

void FusedEmbeddingFCLSTMOp::InferShape(
    framework::InferShapeContext* ctx) const {
  OP_INOUT_CHECK(ctx->HasInput("Embeddings"),
                 "Input",
                 "Embeddings",
                 "fused_embedding_fc_lstm");
  OP_INOUT_CHECK(
      ctx->HasInput("WeightH"), "Input", "WeightH", "fused_embedding_fc_lstm");
  OP_INOUT_CHECK(
      ctx->HasInput("Bias"), "Input", "Bias", "fused_embedding_fc_lstm");
  OP_INOUT_CHECK(
      ctx->HasOutput("XX"), "Output", "XX", "fused_embedding_fc_lstm");
  OP_INOUT_CHECK(
      ctx->HasOutput("Hidden"), "Output", "Hidden", "fused_embedding_fc_lstm");
  OP_INOUT_CHECK(
      ctx->HasOutput("Cell"), "Output", "Cell", "fused_embedding_fc_lstm");
  OP_INOUT_CHECK(
      ctx->HasInput("Ids"), "Input", "Ids", "fused_embedding_fc_lstm");

  auto table_dims = ctx->GetInputDim("Embeddings");
  auto ids_dims = ctx->GetInputDim("Ids");
  int ids_rank = ids_dims.size();

  PADDLE_ENFORCE_EQ(
      table_dims.size(),
      2,
      platform::errors::InvalidArgument(
          "The Embeddings's rank should be 2, but received value is:%d.",
          table_dims.size()));
  PADDLE_ENFORCE_EQ(ids_dims[ids_rank - 1],
                    1,
                    platform::errors::InvalidArgument(
                        "The last dimension of the 'Ids' tensor must be 1, but "
                        "received value is:%d.",
                        ids_dims[ids_rank - 1]));

  auto x_dims = ctx->GetInputDim("Ids");
  PADDLE_ENFORCE_EQ(
      x_dims.size(),
      2,
      platform::errors::InvalidArgument(
          "Input(Ids)'s rank must be 2, but received value is:%d.",
          x_dims.size()));

  if (ctx->HasInput("H0")) {
    PADDLE_ENFORCE_EQ(ctx->HasInput("C0"),
                      true,
                      platform::errors::InvalidArgument(
                          "Input(Cell) and Input(Hidden) of LSTM should exist "
                          "at the same time."));
    auto h_dims = ctx->GetInputDim("H0");
    auto c_dims = ctx->GetInputDim("C0");
    PADDLE_ENFORCE_EQ(
        h_dims,
        c_dims,
        platform::errors::InvalidArgument(
            "The dimension of Input(H0) and Input(C0) "
            "should be the same, but received H0 dim is:[%s], C0 dim is[%s]",
            h_dims,
            c_dims));
  }

  auto wh_dims = ctx->GetInputDim("WeightH");
  int frame_size = wh_dims[1] / 4;
  PADDLE_ENFORCE_EQ(
      wh_dims.size(),
      2,
      platform::errors::InvalidArgument(
          "The rank of Input(WeightH) should be 2, but received value is:%d.",
          wh_dims.size()));
  PADDLE_ENFORCE_EQ(wh_dims[0],
                    frame_size,
                    platform::errors::InvalidArgument(
                        "The first dimension of Input(WeightH) should equal to "
                        "frame size:%d, but received value is:%d.",
                        frame_size,
                        wh_dims[0]));
  PADDLE_ENFORCE_EQ(wh_dims[1],
                    4 * frame_size,
                    platform::errors::InvalidArgument(
                        "The second dimension of Input(WeightH) should equal "
                        "to 4 * %d, but received value is:%d.",
                        frame_size,
                        wh_dims[1]));

  auto b_dims = ctx->GetInputDim("Bias");
  PADDLE_ENFORCE_EQ(
      b_dims.size(),
      2,
      platform::errors::InvalidArgument(
          "The rank of Input(Bias) should be 2, but received value is:%d.",
          b_dims.size()));
  PADDLE_ENFORCE_EQ(b_dims[0],
                    1,
                    platform::errors::InvalidArgument(
                        "The first dimension of Input(Bias) "
                        "should be 1, but received value is:%d.",
                        b_dims[0]));
  PADDLE_ENFORCE_EQ(
      b_dims[1],
      (ctx->Attrs().Get<bool>("use_peepholes") ? 7 : 4) * frame_size,
      platform::errors::InvalidArgument(
          "The second dimension of Input(Bias) should be "
          "7 * %d if enable peepholes connection or"
          "4 * %d if disable peepholes, bias dim is:%d, use_peepholes:%d",
          frame_size,
          frame_size,
          b_dims[1],
          ctx->Attrs().Get<bool>("use_peepholes")));

  framework::DDim out_dims({x_dims[0], frame_size});
  ctx->SetOutputDim("Hidden", out_dims);
  ctx->SetOutputDim("Cell", out_dims);
  ctx->ShareLoD("Ids", "Hidden");
  ctx->ShareLoD("Ids", "Cell");
  if (!ctx->Attrs().Get<bool>("use_seq")) {
    OP_INOUT_CHECK(ctx->HasOutput("BatchedInput"),
                   "Output",
                   "BatchedInput",
                   "fused_embedding_fc_lstm");
    OP_INOUT_CHECK(ctx->HasOutput("BatchedHidden"),
                   "Output",
                   "BatchedHidden",
                   "fused_embedding_fc_lstm");
    OP_INOUT_CHECK(ctx->HasOutput("BatchedCell"),
                   "Output",
                   "BatchedCell",
                   "fused_embedding_fc_lstm");
    OP_INOUT_CHECK(ctx->HasOutput("ReorderedH0"),
                   "Output",
                   "ReorderedH0",
                   "fused_embedding_fc_lstm");
    OP_INOUT_CHECK(ctx->HasOutput("ReorderedC0"),
                   "Output",
                   "ReorderedC0",
                   "fused_embedding_fc_lstm");

    ctx->SetOutputDim("BatchedInput", {x_dims[0], wh_dims[1]});
    ctx->SetOutputDim("BatchedHidden", out_dims);
    ctx->SetOutputDim("BatchedCell", out_dims);
  }
  ctx->SetOutputDim("XX", {x_dims[0], wh_dims[1]});
  ctx->ShareLoD("Ids", "XX");
}

framework::OpKernelType FusedEmbeddingFCLSTMOp::GetExpectedKernelType(
    const framework::ExecutionContext& ctx) const {
  return framework::OpKernelType(
      OperatorWithKernel::IndicateVarDataType(ctx, "Embeddings"),
      ctx.device_context());
}

void FusedEmbeddingFCLSTMOpMaker::Make() {
  AddInput("Ids",
           "An input with type int32 or int64 "
           "contains the ids to be looked up in W. "
           "The last dimension size must be 1.");
  AddInput("Embeddings",
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
                "(bool, default: True) "
                "whether to enable diagonal/peephole connections.")
      .SetDefault(true);
  AddAttr<bool>("is_reverse",
                "(bool, default: False) "
                "whether to compute reversed LSTM.")
      .SetDefault(false);
  AddAttr<bool>("use_seq",
                "(bool, default: True) "
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
                       "The activation for cell output, `tanh` by default.")
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
class FusedEmbeddingFCLSTMKernel : public framework::OpKernel<T> {
 public:
#define INIT_VEC_FUNC                                                        \
  std::function<void(const int, const T*, T*)> act_gate, act_cell, act_cand; \
  auto& act_gate_str = ctx.Attr<std::string>("gate_activation");             \
  auto& act_cell_str = ctx.Attr<std::string>("cell_activation");             \
  auto& act_cand_str = ctx.Attr<std::string>("candidate_activation");        \
  if (platform::MayIUse(platform::avx)) {                                    \
    phi::funcs::VecActivations<T, platform::avx> act_functor;                \
    act_gate = act_functor(act_gate_str);                                    \
    act_cell = act_functor(act_cell_str);                                    \
    act_cand = act_functor(act_cand_str);                                    \
  } else {                                                                   \
    phi::funcs::VecActivations<T, platform::isa_any> act_functor;            \
    act_gate = act_functor(act_gate_str);                                    \
    act_cell = act_functor(act_cell_str);                                    \
    act_cand = act_functor(act_cand_str);                                    \
  }

#define INIT_BASE_INPUT_OUTPUT                                  \
  auto* ids = ctx.Input<LoDTensor>("Ids");                      \
  auto* h0 = ctx.Input<phi::DenseTensor>("H0");                 \
  auto* c0 = ctx.Input<phi::DenseTensor>("C0");                 \
  auto* embeddings = ctx.Input<phi::DenseTensor>("Embeddings"); \
  auto* wh = ctx.Input<phi::DenseTensor>("WeightH");            \
  auto* bias = ctx.Input<phi::DenseTensor>("Bias");             \
  auto* xx = ctx.Output<LoDTensor>("XX");                       \
  auto* hidden_out = ctx.Output<LoDTensor>("Hidden");           \
  auto* cell_out = ctx.Output<LoDTensor>("Cell");               \
  bool is_reverse = ctx.Attr<bool>("is_reverse");               \
  bool use_peepholes = ctx.Attr<bool>("use_peepholes");

#define INIT_BASE_SIZES                                \
  auto ids_dims = ids->dims();             /* T x M*/  \
  auto ids_numel = phi::product(ids_dims); /* T x 1*/  \
  auto wh_dims = wh->dims();               /* D x 4D*/ \
  const int D = wh_dims[0];                            \
  const int D2 = D * 2;                                \
  const int D3 = D * 3;                                \
  int64_t row_number = embeddings->dims()[0];          \
  int64_t row_width = embeddings->dims()[1];           \
  const int D4 = wh_dims[1];

#define INIT_BASE_INPUT_DATAS                                        \
  const int64_t* ids_data = ids->data<int64_t>();                    \
  const T* embeddings_data = embeddings->data<T>();                  \
  const T* wh_data = wh->data<T>();                                  \
  /* diagonal weight*/                                               \
  const T* wc_data = bias->data<T>() + D4;                           \
  /* for peephole only*/                                             \
  Tensor checked_cell;                                               \
  T* checked_cell_data = nullptr;                                    \
  auto place = ctx.GetPlace();                                       \
  if (use_peepholes) {                                               \
    /* w_ic * Ct-1, w_fc * Ct-1  ; w_oc * Ct => ih*/                 \
    checked_cell_data = checked_cell.mutable_data<T>({2, D}, place); \
  }

/// Compute LSTM
#define GEMM_WH_ADDON(bs, prev, out) \
  blas.GEMM(CblasNoTrans,            \
            CblasNoTrans,            \
            bs,                      \
            D4,                      \
            D,                       \
            static_cast<T>(1),       \
            prev,                    \
            D,                       \
            wh_data,                 \
            D4,                      \
            static_cast<T>(1),       \
            out,                     \
            D4)

// gates: W_ch, W_ih, W_fh, W_oh
#define GET_Ct(ct_1, gates, ct)                   \
  /* C_t = C_t-1 * fgated + cand_gated * igated*/ \
  act_cand(D, gates, gates);                      \
  blas.VMUL(D, gates, gates + D, gates + D);      \
  blas.VMUL(D, ct_1, gates + D2, gates + D2);     \
  blas.VADD(D, gates + D, gates + D2, ct)

#define GET_Ht(ct, gates, ht)        \
  /* H_t = act_cell(C_t) * ogated */ \
  act_cell(D, ct, gates + D2);       \
  blas.VMUL(D, gates + D2, gates + D3, ht)

#define GET_Ct_NOH0C0(gates, ct)     \
  /* C_t = igated * cgated*/         \
  act_gate(D, gates + D, gates + D); \
  act_cand(D, gates, gates);         \
  blas.VMUL(D, gates, gates + D, ct)

#define COMPUTE_CtHt_NOH0C0(gates, ct, ht) \
  GET_Ct_NOH0C0(gates, ct);                \
  act_gate(D, gates + D3, gates + D3);     \
  GET_Ht(ct, gates, ht)

#define COMPUTE_CtHt_PEEPHOLE_NOH0C0(gates, ct, ht) \
  GET_Ct_NOH0C0(gates, ct);                         \
  /* get outgated, put W_oc * C_t on igated */      \
  blas.VMUL(D, wc_data + D2, ct, gates + D);        \
  blas.VADD(D, gates + D, gates + D3, gates + D3);  \
  act_gate(D, gates + D3, gates + D3);              \
  GET_Ht(ct, gates, ht)

#define COMPUTE_CtHt(gates, ct_1, ct, ht) \
  act_gate(D3, gates + D, gates + D);     \
  GET_Ct(ct_1, gates, ct);                \
  GET_Ht(ct, gates, ht)

#define COMPUTE_CtHt_PEEPHOLE(gates, ct_1, ct, ht)        \
  /* get fgated and igated*/                              \
  blas.VMUL(D, wc_data, ct_1, checked_cell_data);         \
  blas.VMUL(D, wc_data + D, ct_1, checked_cell_data + D); \
  blas.VADD(D2, checked_cell_data, gates + D, gates + D); \
  act_gate(D2, gates + D, gates + D);                     \
  GET_Ct(ct_1, gates, ct);                                \
  /* get ogated*/                                         \
  blas.VMUL(D, wc_data + D2, ct, gates + D);              \
  blas.VADD(D, gates + D, gates + D3, gates + D3);        \
  act_gate(D, gates + D3, gates + D3);                    \
  GET_Ht(ct, gates, ht)

  void SeqCompute(const framework::ExecutionContext& ctx) const {
    using DeviceContext = phi::CPUContext;
    INIT_BASE_INPUT_OUTPUT
    INIT_BASE_SIZES
    INIT_VEC_FUNC
    INIT_BASE_INPUT_DATAS

    // log(INFO) << "====> SeqCompute" << "\n";
    auto ids_lod = ids->lod();
    const int total_T = ids_dims[0];
    const int N = ids_lod[0].size() - 1;
    const T* h0_data = h0 ? h0->data<T>() : nullptr;
    const T* c0_data = c0 ? c0->data<T>() : nullptr;
    T* xx_data = xx->mutable_data<T>(place);
    T* h_out_data = hidden_out->mutable_data<T>(place);
    T* c_out_data = cell_out->mutable_data<T>(place);
    auto blas = phi::funcs::GetBlas<DeviceContext, T>(ctx);

    for (int64_t i = 0; i < ids_numel; ++i) {
      PADDLE_ENFORCE_LT(
          ids_data[i],
          row_number,
          platform::errors::OutOfRange(
              "Value of Ids %d should less than dict size %d.", i, row_number));
      PADDLE_ENFORCE_GE(ids_data[i],
                        0,
                        platform::errors::OutOfRange(
                            "Value of Ids %d should greater than ZERO.", i));
      memcpy(xx_data + i * row_width,
             embeddings_data + ids_data[i] * row_width,
             row_width * sizeof(T));
    }

    int xx_offset = D4;
    int gate_offset = D;
    if (is_reverse) {
      const int offset = (total_T - 1) * D;
      xx_data = xx_data + offset * 4;
      h_out_data = h_out_data + offset;
      c_out_data = c_out_data + offset;
      xx_offset = -D4;
      gate_offset = -D;
    }

#define MOVE_ONE_STEP                    \
  prev_h_data = h_out_data;              \
  prev_c_data = c_out_data;              \
  xx_data = xx_data + xx_offset;         \
  h_out_data = h_out_data + gate_offset; \
  c_out_data = c_out_data + gate_offset

#define PROCESS_H0C0_DEFINES                           \
  int bid = is_reverse ? N - 1 - i : i;                \
  int seq_len = ids_lod[0][bid + 1] - ids_lod[0][bid]; \
  const T* prev_c_data = nullptr;                      \
  const T* prev_h_data = nullptr;                      \
  int tstart = 0

#define PROCESS_H0C0_PEEPHOLE                                      \
  PROCESS_H0C0_DEFINES;                                            \
  if (h0_data) {                                                   \
    prev_h_data = h0_data + bid * D;                               \
    prev_c_data = c0_data + bid * D;                               \
  } else {                                                         \
    COMPUTE_CtHt_PEEPHOLE_NOH0C0(xx_data, c_out_data, h_out_data); \
    MOVE_ONE_STEP;                                                 \
    tstart = 1;                                                    \
  }

#define PROCESS_H0C0                                      \
  PROCESS_H0C0_DEFINES;                                   \
  if (h0_data) {                                          \
    prev_h_data = h0_data + bid * D;                      \
    prev_c_data = c0_data + bid * D;                      \
  } else {                                                \
    COMPUTE_CtHt_NOH0C0(xx_data, c_out_data, h_out_data); \
    MOVE_ONE_STEP;                                        \
    tstart = 1;                                           \
  }

    if (use_peepholes) {
      for (int i = 0; i < N; ++i) {
        PROCESS_H0C0_PEEPHOLE
        for (int step = tstart; step < seq_len; ++step) {
          GEMM_WH_ADDON(1, prev_h_data, xx_data);
          COMPUTE_CtHt_PEEPHOLE(xx_data, prev_c_data, c_out_data, h_out_data);
          MOVE_ONE_STEP;
        }
      }
    } else {
      for (int i = 0; i < N; ++i) {
        PROCESS_H0C0
        for (int step = tstart; step < seq_len; ++step) {
          GEMM_WH_ADDON(1, prev_h_data, xx_data);
          COMPUTE_CtHt(xx_data, prev_c_data, c_out_data, h_out_data);
          MOVE_ONE_STEP;
        }
      }
    }
#undef PROCESS_H0C0_DEFINES
#undef PROCESS_H0C0_PEEPHOLE
#undef PROCESS_H0C0
#undef MOVE_ONE_STEP
  }

  void BatchCompute(const framework::ExecutionContext& ctx) const {
    using DeviceContext = phi::CPUContext;
    INIT_BASE_INPUT_OUTPUT
    if (ids->lod()[0].size() == 2) {
      SeqCompute(ctx);
      return;
    }
    INIT_BASE_SIZES
    INIT_VEC_FUNC
    INIT_BASE_INPUT_DATAS

    auto* reordered_h0 = ctx.Output<phi::DenseTensor>("ReorderedH0");
    auto* reordered_c0 = ctx.Output<phi::DenseTensor>("ReorderedC0");
    auto* batched_input = ctx.Output<LoDTensor>("BatchedInput");
    auto* batched_c_out = ctx.Output<LoDTensor>("BatchedCell");
    auto* batched_h_out = ctx.Output<LoDTensor>("BatchedHidden");
    T* xx_data = xx->mutable_data<T>(place);
    T* batched_input_data = batched_input->mutable_data<T>(place);
    T* batched_c_out_data = batched_c_out->mutable_data<T>(place);
    T* batched_h_out_data = batched_h_out->mutable_data<T>(place);
    hidden_out->mutable_data<T>(place);
    cell_out->mutable_data<T>(place);

    phi::funcs::LoDTensor2BatchFunctor<DeviceContext, T> to_batch;
    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    auto blas = phi::funcs::GetBlas<DeviceContext, T>(dev_ctx);

    for (int64_t i = 0; i < ids_numel; ++i) {
      PADDLE_ENFORCE_LT(
          ids_data[i],
          row_number,
          platform::errors::OutOfRange(
              "Value of Ids %d should less than dict size %d.", i, row_number));
      PADDLE_ENFORCE_GE(ids_data[i],
                        0,
                        platform::errors::OutOfRange(
                            "Value of Ids %d should greater than ZERO.", i));
      memcpy(xx_data + i * row_width,
             embeddings_data + ids_data[i] * row_width,
             row_width * sizeof(T));
    }

    to_batch(dev_ctx, *xx, batched_input, true, is_reverse);

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
      for (int i = 0; i < max_bs; ++i) {
        GET_Ct_NOH0C0(cur_in_data, cur_c_out_data);
        if (use_peepholes) {
          blas.VMUL(D, wc_data + D2, cur_c_out_data, cur_in_data + D);
          blas.VADD(D, cur_in_data + D, cur_in_data + D3, cur_in_data + D3);
        }
        act_gate(D, cur_in_data + D3, cur_in_data + D3);
        GET_Ht(cur_c_out_data, cur_in_data, cur_h_out_data);
        cur_in_data += D4;
        cur_c_out_data += D;
        cur_h_out_data += D;
      }
      tstart = 1;
      prev_h_data = batched_h_out_data;
      prev_c_data = batched_c_out_data;
    }
    const auto& batch_starts = batched_lod[0];
    const int max_seq_len = batch_starts.size() - 1;
    const int offset = tstart * max_bs * D;
    batched_input_data = batched_input_data + offset * 4;
    batched_h_out_data = batched_h_out_data + offset;
    batched_c_out_data = batched_c_out_data + offset;

#define DEFINE_CUR                        \
  T* cur_in_data = batched_input_data;    \
  T* cur_prev_c_data = prev_c_data;       \
  T* cur_c_out_data = batched_c_out_data; \
  T* cur_h_out_data = batched_h_out_data

#define MOVE_ONE_BATCH  \
  cur_in_data += D4;    \
  cur_prev_c_data += D; \
  cur_c_out_data += D;  \
  cur_h_out_data += D

#define MOVE_ONE_STEP                  \
  prev_c_data = batched_c_out_data;    \
  prev_h_data = batched_h_out_data;    \
  batched_c_out_data = cur_c_out_data; \
  batched_h_out_data = cur_h_out_data; \
  batched_input_data = cur_in_data

    if (use_peepholes) {
      for (int step = tstart; step < max_seq_len; ++step) {
        const int cur_bs = batch_starts[step + 1] - batch_starts[step];
        GEMM_WH_ADDON(cur_bs, prev_h_data, batched_input_data);
        DEFINE_CUR;
        for (int i = 0; i < cur_bs; ++i) {
          COMPUTE_CtHt_PEEPHOLE(
              cur_in_data, cur_prev_c_data, cur_c_out_data, cur_h_out_data);
          MOVE_ONE_BATCH;
        }
        MOVE_ONE_STEP;
      }
    } else {
      for (int step = tstart; step < max_seq_len; ++step) {
        const int cur_bs = batch_starts[step + 1] - batch_starts[step];
        GEMM_WH_ADDON(cur_bs, prev_h_data, batched_input_data);
        DEFINE_CUR;
        for (int i = 0; i < cur_bs; ++i) {
          COMPUTE_CtHt(
              cur_in_data, cur_prev_c_data, cur_c_out_data, cur_h_out_data);
          MOVE_ONE_BATCH;
        }
        MOVE_ONE_STEP;
      }
    }
#undef MOVE_ONE_STEP
#undef MOVE_ONE_BATCH
#undef DEFINE_CUR

    phi::funcs::Batch2LoDTensorFunctor<DeviceContext, T> to_seq;
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

#undef COMPUTE_CtHt_PEEPHOLE
#undef COMPUTE_CtHt
#undef GET_Ct_NOH0C0
#undef COMPUTE_CtHt_NOH0C0
#undef COMPUTE_CtHt_PEEPHOLE_NOH0C0
#undef GET_Ht
#undef GET_Ct
#undef GEMM_WH_ADDON
#undef INIT_BASE_INPUT_DATAS
#undef INIT_BASE_SIZES
#undef INIT_BASE_INPUT_OUTPUT
#undef INIT_VEC_FUNC
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(fused_embedding_fc_lstm,
                  ops::FusedEmbeddingFCLSTMOp,
                  ops::FusedEmbeddingFCLSTMOpMaker);

REGISTER_OP_CPU_KERNEL(fused_embedding_fc_lstm,
                       ops::FusedEmbeddingFCLSTMKernel<float>,
                       ops::FusedEmbeddingFCLSTMKernel<double>);
