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

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/fc_functor.h"
#include "paddle/phi/kernels/funcs/jit/kernels.h"
#include "paddle/phi/kernels/funcs/sequence2batch.h"

namespace phi {

#define INIT_BASE_DEFINES                \
  auto *x = &x_in;                       \
  auto *h0 = h0_in.get_ptr();            \
  auto *c0 = c0_in.get_ptr();            \
  auto *wx = &weight_x_in;               \
  auto *wh = &weight_h_in;               \
  auto *bias = &bias_in;                 \
  auto *hidden_out = hidden;             \
  auto *cell_out = cell;                 \
  auto x_dims = x->dims();   /* T x M*/  \
  auto wh_dims = wh->dims(); /* D x 4D*/ \
  const int M = x_dims[1];               \
  const int D = wh_dims[0];              \
  const int D4 = wh_dims[1]

#define INIT_OTHER_DEFINES                                             \
  const T *x_data = x->data<T>();                                      \
  const T *wx_data = wx->data<T>();                                    \
  const T *wh_data = wh->data<T>();                                    \
  /* diagonal weight*/                                                 \
  const T *wp_data = bias->data<T>() + D4;                             \
  /* for peephole only*/                                               \
  T *checked_cell_data = nullptr;                                      \
  if (use_peepholes) {                                                 \
    /* w_ic * Ct-1, w_fc * Ct-1  ; w_oc * Ct => ih*/                   \
    checked_cell_data = dev_ctx.template Alloc<T>(checked_cell);       \
  }                                                                    \
  const phi::jit::lstm_attr_t attr(                                    \
      D,                                                               \
      phi::jit::to_kerneltype(gate_activation),                        \
      phi::jit::to_kerneltype(candidate_activation),                   \
      phi::jit::to_kerneltype(cell_activation),                        \
      use_peepholes);                                                  \
  phi::jit::lstm_t one_step;                                           \
  one_step.wp = wp_data;                                               \
  one_step.checked = checked_cell_data;                                \
  auto ComputeC1H1 = phi::jit::KernelFuncs<phi::jit::LSTMC1H1Tuple<T>, \
                                           phi::CPUPlace>::Cache()     \
                         .At(attr);                                    \
  auto ComputeCtHt = phi::jit::KernelFuncs<phi::jit::LSTMCtHtTuple<T>, \
                                           phi::CPUPlace>::Cache()     \
                         .At(attr)

// Wh GEMM
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

template <typename T, typename Context>
void SeqCompute(const Context &dev_ctx,
                const DenseTensor &x_in,
                const DenseTensor &weight_x_in,
                const DenseTensor &weight_h_in,
                const DenseTensor &bias_in,
                const paddle::optional<DenseTensor> &h0_in,
                const paddle::optional<DenseTensor> &c0_in,
                bool use_peepholes,
                bool is_reverse,
                bool use_seq,
                const std::string &gate_activation,
                const std::string &cell_activation,
                const std::string &candidate_activation,
                float scale_data,
                float shift_data,
                const std::vector<float> &scale_weights,
                bool force_fp32_output,
                DenseTensor *hidden,
                DenseTensor *cell,
                DenseTensor *xx,
                DenseTensor *batched_input,
                DenseTensor *batched_hidden,
                DenseTensor *batched_cell,
                DenseTensor *reordered_h0,
                DenseTensor *reordered_c0,
                DenseTensor *checked_cell) {
  INIT_BASE_DEFINES;
  INIT_OTHER_DEFINES;
  auto x_lod = x->lod();
  const int total_T = static_cast<int>(x_dims[0]);
  const int N = static_cast<int>(x_lod[0].size() - 1);
  const T *h0_data = h0 ? h0->data<T>() : nullptr;
  const T *c0_data = c0 ? c0->data<T>() : nullptr;
  T *xx_data = dev_ctx.template Alloc<T>(xx);
  T *h_out_data = dev_ctx.template Alloc<T>(hidden_out);
  T *c_out_data = dev_ctx.template Alloc<T>(cell_out);
  auto blas = phi::funcs::GetBlas<Context, T>(dev_ctx);

  phi::funcs::FCFunctor<Context, T> fc;
  fc(dev_ctx, total_T, D4, M, x_data, wx_data, xx_data, bias->data<T>());

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

  for (int i = 0; i < N; ++i) {
    int bid = is_reverse ? N - 1 - i : i;
    int seq_len = static_cast<int>(x_lod[0][bid + 1] - x_lod[0][bid]);
    const T *prev_c_data = nullptr;
    const T *prev_h_data = nullptr;
    int tstart = 0;
    if (h0_data) {
      prev_h_data = h0_data + bid * D;
      prev_c_data = c0_data + bid * D;
    } else {
      one_step.gates = xx_data;
      one_step.ct = c_out_data;
      one_step.ht = h_out_data;
      ComputeC1H1(&one_step, &attr);
      tstart = 1;
      // move one step
      prev_h_data = h_out_data;
      prev_c_data = c_out_data;
      xx_data = xx_data + xx_offset;
      h_out_data = h_out_data + gate_offset;
      c_out_data = c_out_data + gate_offset;
    }
    for (int step = tstart; step < seq_len; ++step) {
      GEMM_WH_ADDON(1, prev_h_data, xx_data);

      one_step.gates = xx_data;
      one_step.ct_1 = prev_c_data;
      one_step.ct = c_out_data;
      one_step.ht = h_out_data;
      ComputeCtHt(&one_step, &attr);
      // move one step
      prev_h_data = h_out_data;
      prev_c_data = c_out_data;
      xx_data = xx_data + xx_offset;
      h_out_data = h_out_data + gate_offset;
      c_out_data = c_out_data + gate_offset;
    }
  }
}

template <typename T, typename Context>
void BatchCompute(const Context &dev_ctx,
                  const DenseTensor &x_in,
                  const DenseTensor &weight_x_in,
                  const DenseTensor &weight_h_in,
                  const DenseTensor &bias_in,
                  const paddle::optional<DenseTensor> &h0_in,
                  const paddle::optional<DenseTensor> &c0_in,
                  bool use_peepholes,
                  bool is_reverse,
                  bool use_seq,
                  const std::string &gate_activation,
                  const std::string &cell_activation,
                  const std::string &candidate_activation,
                  float scale_data,
                  float shift_data,
                  const std::vector<float> &scale_weights,
                  bool force_fp32_output,
                  DenseTensor *hidden,
                  DenseTensor *cell,
                  DenseTensor *xx,
                  DenseTensor *batched_input,
                  DenseTensor *batched_hidden,
                  DenseTensor *batched_cell,
                  DenseTensor *reordered_h0,
                  DenseTensor *reordered_c0,
                  DenseTensor *checked_cell) {
  INIT_BASE_DEFINES;
  if (x->lod()[0].size() == 2) {
    xx->Resize({x_dims[0], D4});
    SeqCompute<T, Context>(dev_ctx,
                           x_in,
                           weight_x_in,
                           weight_h_in,
                           bias_in,
                           h0_in,
                           c0_in,
                           use_peepholes,
                           is_reverse,
                           use_seq,
                           gate_activation,
                           cell_activation,
                           candidate_activation,
                           scale_data,
                           shift_data,
                           scale_weights,
                           force_fp32_output,
                           hidden,
                           cell,
                           xx,
                           batched_input,
                           batched_hidden,
                           batched_cell,
                           reordered_h0,
                           reordered_c0,
                           checked_cell);
    return;
  }
  INIT_OTHER_DEFINES;

  auto *batched_c_out = batched_cell;
  auto *batched_h_out = batched_hidden;
  T *xx_data = dev_ctx.template Alloc<T>(xx);
  T *batched_input_data = dev_ctx.template Alloc<T>(batched_input);
  T *batched_c_out_data = dev_ctx.template Alloc<T>(batched_c_out);
  T *batched_h_out_data = dev_ctx.template Alloc<T>(batched_h_out);
  dev_ctx.template Alloc<T>(hidden_out);
  dev_ctx.template Alloc<T>(cell_out);

  phi::funcs::DenseTensor2BatchFunctor<Context, T> to_batch;
  auto blas = phi::funcs::GetBlas<Context, T>(dev_ctx);
  phi::funcs::FCFunctor<Context, T> fc;
  if (M > D4) {
    fc(dev_ctx, x_dims[0], D4, M, x_data, wx_data, xx_data, bias->data<T>());
    to_batch(dev_ctx, *xx, batched_input, true, is_reverse);
  } else {
    to_batch(dev_ctx, *x, xx, true, is_reverse);
    batched_input->set_lod(xx->lod());
    fc(dev_ctx,
       x_dims[0],
       D4,
       M,
       xx_data,
       wx_data,
       batched_input_data,
       bias->data<T>());
  }

  auto batched_lod = batched_input->lod();
  const auto &seq_order = batched_lod[2];
  const int max_bs = static_cast<int>(seq_order.size());
  reordered_h0->Resize({max_bs, D});
  reordered_c0->Resize({max_bs, D});

  int tstart = 0;
  T *prev_h_data = nullptr;
  T *prev_c_data = nullptr;
  if (h0) {
    // reorder h0, c0
    T *reordered_h0_data = dev_ctx.template Alloc<T>(reordered_h0);
    T *reordered_c0_data = dev_ctx.template Alloc<T>(reordered_c0);
    const T *h0_data = h0->data<T>();
    const T *c0_data = c0->data<T>();
    prev_h_data = reordered_h0_data;
    prev_c_data = reordered_c0_data;
    size_t sz = D;
    for (int i = 0; i < max_bs; ++i) {
      blas.VCOPY(sz, h0_data + seq_order[i] * D, reordered_h0_data);
      blas.VCOPY(sz, c0_data + seq_order[i] * D, reordered_c0_data);
      reordered_h0_data += D;
      reordered_c0_data += D;
    }
  } else {
    // compute without h0, c0
    T *cur_in_data = batched_input_data;
    T *cur_h_out_data = batched_h_out_data;
    T *cur_c_out_data = batched_c_out_data;
    for (int i = 0; i < max_bs; ++i) {
      one_step.gates = cur_in_data;
      one_step.ct = cur_c_out_data;
      one_step.ht = cur_h_out_data;
      ComputeC1H1(&one_step, &attr);

      cur_in_data += D4;
      cur_c_out_data += D;
      cur_h_out_data += D;
    }
    tstart = 1;
    prev_h_data = batched_h_out_data;
    prev_c_data = batched_c_out_data;
  }

  // compute kernel part
  const auto &batch_starts = batched_lod[0];
  const int max_seq_len = static_cast<int>(batch_starts.size() - 1);
  const int offset = tstart * max_bs * D;
  batched_input_data = batched_input_data + offset * 4;
  batched_h_out_data = batched_h_out_data + offset;
  batched_c_out_data = batched_c_out_data + offset;
  for (int step = tstart; step < max_seq_len; ++step) {
    const int cur_bs =
        static_cast<int>(batch_starts[step + 1] - batch_starts[step]);
    GEMM_WH_ADDON(cur_bs, prev_h_data, batched_input_data);
    T *cur_in_data = batched_input_data;
    T *cur_prev_c_data = prev_c_data;
    T *cur_c_out_data = batched_c_out_data;
    T *cur_h_out_data = batched_h_out_data;
    for (int i = 0; i < cur_bs; ++i) {
      one_step.gates = cur_in_data;
      one_step.ct_1 = cur_prev_c_data;
      one_step.ct = cur_c_out_data;
      one_step.ht = cur_h_out_data;
      ComputeCtHt(&one_step, &attr);

      // move one batch
      cur_in_data += D4;
      cur_prev_c_data += D;
      cur_c_out_data += D;
      cur_h_out_data += D;
    }
    // move one step
    prev_c_data = batched_c_out_data;
    prev_h_data = batched_h_out_data;
    batched_c_out_data = cur_c_out_data;
    batched_h_out_data = cur_h_out_data;
    batched_input_data = cur_in_data;
  }

  phi::funcs::Batch2DenseTensorFunctor<Context, T> to_seq;
  batched_h_out->set_lod(batched_lod);
  to_seq(dev_ctx, *batched_h_out, hidden_out);
  batched_c_out->set_lod(batched_lod);
  to_seq(dev_ctx, *batched_c_out, cell_out);
}

template <typename T, typename Context>
void FusionLSTMKernel(const Context &dev_ctx,
                      const DenseTensor &x_in,
                      const DenseTensor &weight_x_in,
                      const DenseTensor &weight_h_in,
                      const DenseTensor &bias_in,
                      const paddle::optional<DenseTensor> &h0_in,
                      const paddle::optional<DenseTensor> &c0_in,
                      bool use_peepholes,
                      bool is_reverse,
                      bool use_seq,
                      const std::string &gate_activation,
                      const std::string &cell_activation,
                      const std::string &candidate_activation,
                      float scale_data,
                      float shift_data,
                      const std::vector<float> &scale_weights,
                      bool force_fp32_output,
                      DenseTensor *hidden,
                      DenseTensor *cell,
                      DenseTensor *xx,
                      DenseTensor *batched_input,
                      DenseTensor *batched_hidden,
                      DenseTensor *batched_cell,
                      DenseTensor *reordered_h0,
                      DenseTensor *reordered_c0,
                      DenseTensor *checked_cell) {
  if (use_seq) {
    SeqCompute<T, Context>(dev_ctx,
                           x_in,
                           weight_x_in,
                           weight_h_in,
                           bias_in,
                           h0_in,
                           c0_in,
                           use_peepholes,
                           is_reverse,
                           use_seq,
                           gate_activation,
                           cell_activation,
                           candidate_activation,
                           scale_data,
                           shift_data,
                           scale_weights,
                           force_fp32_output,
                           hidden,
                           cell,
                           xx,
                           batched_input,
                           batched_hidden,
                           batched_cell,
                           reordered_h0,
                           reordered_c0,
                           checked_cell);
  } else {
    BatchCompute<T, Context>(dev_ctx,
                             x_in,
                             weight_x_in,
                             weight_h_in,
                             bias_in,
                             h0_in,
                             c0_in,
                             use_peepholes,
                             is_reverse,
                             use_seq,
                             gate_activation,
                             cell_activation,
                             candidate_activation,
                             scale_data,
                             shift_data,
                             scale_weights,
                             force_fp32_output,
                             hidden,
                             cell,
                             xx,
                             batched_input,
                             batched_hidden,
                             batched_cell,
                             reordered_h0,
                             reordered_c0,
                             checked_cell);
  }
}

#undef GEMM_WH_ADDON
#undef INIT_OTHER_DEFINES
#undef INIT_BASE_DEFINES

}  // namespace phi

PD_REGISTER_KERNEL(
    fusion_lstm, CPU, ALL_LAYOUT, phi::FusionLSTMKernel, float, double) {}
