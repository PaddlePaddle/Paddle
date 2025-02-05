// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <string>

#include "paddle/phi/backends/cpu/cpu_info.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/cpu_vec.h"
#include "paddle/phi/kernels/funcs/sequence2batch.h"
#include "paddle/utils/optional.h"

namespace phi {

#define OP_PARAM                                                             \
  dev_ctx, ids_in, embeddings_in, weight_h_in, bias_in, h0_in, c0_in,        \
      use_peepholes, is_reverse, use_seq, gate_activation, cell_activation,  \
      candidate_activation, hidden_out, cell_out, xx_out, batched_input_out, \
      batched_hidden_out, batched_cell_out, reordered_h0_out, reordered_c0_out
#define OP_PARAM_DECLARE                                                     \
  const Context &dev_ctx, const DenseTensor &ids_in,                         \
      const DenseTensor &embeddings_in, const DenseTensor &weight_h_in,      \
      const DenseTensor &bias_in, const paddle::optional<DenseTensor>&h0_in, \
      const paddle::optional<DenseTensor>&c0_in, bool use_peepholes,         \
      bool is_reverse, bool use_seq, const std::string &gate_activation,     \
      const std::string &cell_activation,                                    \
      const std::string &candidate_activation, DenseTensor *hidden_out,      \
      DenseTensor *cell_out, DenseTensor *xx_out,                            \
      DenseTensor *batched_input_out, DenseTensor *batched_hidden_out,       \
      DenseTensor *batched_cell_out, DenseTensor *reordered_h0_out,          \
      DenseTensor *reordered_c0_out

template <typename T, typename Context>
class FusedEmbeddingFCLSTMKernel {
 public:
#define INIT_VEC_FUNC                                                        \
  std::function<void(const int, const T*, T*)> act_gate, act_cell, act_cand; \
  auto& act_gate_str = gate_activation;                                      \
  auto& act_cell_str = cell_activation;                                      \
  auto& act_cand_str = candidate_activation;                                 \
  if (phi::backends::cpu::MayIUse(phi::backends::cpu::avx)) {                \
    phi::funcs::VecActivations<T, phi::backends::cpu::avx> act_functor;      \
    act_gate = act_functor(act_gate_str);                                    \
    act_cell = act_functor(act_cell_str);                                    \
    act_cand = act_functor(act_cand_str);                                    \
  } else {                                                                   \
    phi::funcs::VecActivations<T, phi::backends::cpu::isa_any> act_functor;  \
    act_gate = act_functor(act_gate_str);                                    \
    act_cell = act_functor(act_cell_str);                                    \
    act_cand = act_functor(act_cand_str);                                    \
  }

#define INIT_BASE_INPUT_OUTPUT       \
  auto* ids = &ids_in;               \
  auto* h0 = h0_in.get_ptr();        \
  auto* c0 = c0_in.get_ptr();        \
  auto* embeddings = &embeddings_in; \
  auto* wh = &weight_h_in;           \
  auto* bias = &bias_in;             \
  auto* xx = xx_out;

#define INIT_BASE_SIZES                                   \
  auto ids_dims = ids->dims();                /* T x M*/  \
  auto ids_numel = common::product(ids_dims); /* T x 1*/  \
  auto wh_dims = wh->dims();                  /* D x 4D*/ \
  const int D = wh_dims[0];                               \
  const int D2 = D * 2;                                   \
  const int D3 = D * 3;                                   \
  int64_t row_number = embeddings->dims()[0];             \
  int64_t row_width = embeddings->dims()[1];              \
  const int D4 = wh_dims[1];

#define INIT_BASE_INPUT_DATAS                                     \
  const int64_t* ids_data = ids->data<int64_t>();                 \
  const T* embeddings_data = embeddings->data<T>();               \
  const T* wh_data = wh->data<T>();                               \
  /* diagonal weight*/                                            \
  const T* wc_data = bias->data<T>() + D4;                        \
  /* for peephole only*/                                          \
  phi::DenseTensor checked_cell;                                  \
  T* checked_cell_data = nullptr;                                 \
  if (use_peepholes) {                                            \
    /* w_ic * Ct-1, w_fc * Ct-1  ; w_oc * Ct => ih*/              \
    checked_cell.Resize({2, D});                                  \
    checked_cell_data = dev_ctx.template Alloc<T>(&checked_cell); \
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

  void SeqCompute(OP_PARAM_DECLARE) const {
    INIT_BASE_INPUT_OUTPUT
    INIT_BASE_SIZES
    INIT_VEC_FUNC
    INIT_BASE_INPUT_DATAS

    // log(INFO) << "====> SeqCompute" << "\n";
    auto ids_lod = ids->lod();
    const int total_T = static_cast<int>(ids_dims[0]);
    const int N = static_cast<int>(ids_lod[0].size() - 1);
    const T* h0_data = h0 ? h0->data<T>() : nullptr;
    const T* c0_data = c0 ? c0->data<T>() : nullptr;
    T* xx_data = dev_ctx.template Alloc<T>(xx);
    T* h_out_data = dev_ctx.template Alloc<T>(hidden_out);
    T* c_out_data = dev_ctx.template Alloc<T>(cell_out);
    auto blas = phi::funcs::GetBlas<Context, T>(dev_ctx);

    for (int64_t i = 0; i < ids_numel; ++i) {
      PADDLE_ENFORCE_LT(
          ids_data[i],
          row_number,
          common::errors::OutOfRange(
              "Value of Ids %d should less than dict size %d.", i, row_number));
      PADDLE_ENFORCE_GE(ids_data[i],
                        0,
                        common::errors::OutOfRange(
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

  void BatchCompute(OP_PARAM_DECLARE) const {
    INIT_BASE_INPUT_OUTPUT
    if (ids->lod()[0].size() == 2) {
      SeqCompute(OP_PARAM);
      return;
    }
    INIT_BASE_SIZES
    INIT_VEC_FUNC
    INIT_BASE_INPUT_DATAS

    auto* reordered_h0 = reordered_h0_out;
    auto* reordered_c0 = reordered_c0_out;
    auto* batched_input = batched_input_out;
    auto* batched_c_out = batched_cell_out;
    auto* batched_h_out = batched_hidden_out;
    T* xx_data = dev_ctx.template Alloc<T>(xx);
    T* batched_input_data = dev_ctx.template Alloc<T>(batched_input);
    T* batched_c_out_data = dev_ctx.template Alloc<T>(batched_c_out);
    T* batched_h_out_data = dev_ctx.template Alloc<T>(batched_h_out);
    dev_ctx.template Alloc<T>(hidden_out);
    dev_ctx.template Alloc<T>(cell_out);

    phi::funcs::DenseTensor2BatchFunctor<Context, T> to_batch;
    auto blas = phi::funcs::GetBlas<Context, T>(dev_ctx);

    for (int64_t i = 0; i < ids_numel; ++i) {
      PADDLE_ENFORCE_LT(
          ids_data[i],
          row_number,
          common::errors::OutOfRange(
              "Value of Ids %d should less than dict size %d.", i, row_number));
      PADDLE_ENFORCE_GE(ids_data[i],
                        0,
                        common::errors::OutOfRange(
                            "Value of Ids %d should greater than ZERO.", i));
      memcpy(xx_data + i * row_width,
             embeddings_data + ids_data[i] * row_width,
             row_width * sizeof(T));
    }

    to_batch(dev_ctx, *xx, batched_input, true, is_reverse);

    auto batched_lod = batched_input->lod();
    const auto& seq_order = batched_lod[2];
    const int max_bs = static_cast<int>(seq_order.size());
    reordered_h0->Resize({max_bs, D});
    reordered_c0->Resize({max_bs, D});

    int tstart = 0;
    T* prev_h_data = nullptr;
    T* prev_c_data = nullptr;
    if (h0) {
      // reorder h0, c0
      T* reordered_h0_data = dev_ctx.template Alloc<T>(reordered_h0);
      T* reordered_c0_data = dev_ctx.template Alloc<T>(reordered_c0);
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
    const int max_seq_len = static_cast<int>(batch_starts.size() - 1);
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
        const int cur_bs =
            static_cast<int>(batch_starts[step + 1] - batch_starts[step]);
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
        const int cur_bs =
            static_cast<int>(batch_starts[step + 1] - batch_starts[step]);
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

    phi::funcs::Batch2DenseTensorFunctor<Context, T> to_seq;
    batched_h_out->set_lod(batched_lod);
    to_seq(dev_ctx, *batched_h_out, hidden_out);
    batched_c_out->set_lod(batched_lod);
    to_seq(dev_ctx, *batched_c_out, cell_out);
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

template <typename T, typename Context>
void FusedEmbeddingFCLSTMKernelWrapper(OP_PARAM_DECLARE) {
  auto obj = FusedEmbeddingFCLSTMKernel<T, Context>();
  if (use_seq) {
    obj.SeqCompute(OP_PARAM);
  } else {
    obj.BatchCompute(OP_PARAM);
  }
}

#undef OP_PARAM
#undef OP_PARAM_DECLARE
}  // namespace phi

PD_REGISTER_KERNEL(fused_embedding_fc_lstm,
                   CPU,
                   ALL_LAYOUT,
                   phi::FusedEmbeddingFCLSTMKernelWrapper,
                   float,
                   double) {}
