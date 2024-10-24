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
#include "paddle/phi/kernels/funcs/fc_functor.h"
#include "paddle/utils/optional.h"

namespace phi {

// y[i] = (x[i] + bias[0]) > 0 ? (x[i] + bias[0]) : 0;
template <typename T>
inline void bias_relu(const int n, const T* x, const T* bias, T* y) {
  if (bias) {
    phi::funcs::vec_add_bias<T, phi::backends::cpu::avx>(n, *bias, x, y);
    phi::funcs::vec_relu<T, phi::backends::cpu::avx>(n, y, y);
  } else {
    phi::funcs::vec_relu<T, phi::backends::cpu::avx>(n, x, y);
  }
}

template <typename T>
inline void vec_softmax(const int n, const T* x, T* y) {
  T scalar = x[0];
  // max
  for (int i = 1; i < n; ++i) {
    scalar = scalar < x[i] ? x[i] : scalar;
  }
  phi::funcs::vec_add_bias<T, phi::backends::cpu::avx>(
      n, -scalar, x, y);            // sub
  phi::funcs::vec_exp<T>(n, y, y);  // exp
  // sum
  scalar = T(0);
  for (int i = 0; i < n; ++i) {
    scalar += y[i];
  }
  phi::funcs::vec_scal<T>(n, static_cast<T>(1) / scalar, y);  // scale
}

template <typename T, typename Context>
void AttentionLSTMKernel(
    const Context& dev_ctx,
    const DenseTensor& x_in,
    const DenseTensor& c0_in,
    const paddle::optional<DenseTensor>& h0_in,
    const DenseTensor& attention_weight_in,
    const paddle::optional<DenseTensor>& attention_bias_in,
    const paddle::optional<DenseTensor>& attention_scalar_in,
    const paddle::optional<DenseTensor>& attention_scalar_bias_in,
    const DenseTensor& lstm_weight_in,
    const DenseTensor& lstm_bias_in,
    const std::string& gate_activation,
    const std::string& cell_activation,
    const std::string& candidate_activation,
    DenseTensor* hidden,
    DenseTensor* cell,
    DenseTensor* attentioned_x,
    DenseTensor* attention_fc_out,
    DenseTensor* lstm_x,
    DenseTensor* lstm_out) {
  auto* x = &x_in;
  auto* h0 = h0_in.get_ptr();
  auto* c0 = &c0_in;
  auto* atten_w = &attention_weight_in;
  auto* atten_b = attention_bias_in.get_ptr();
  auto* atten_scalar = attention_scalar_in.get_ptr();
  auto* atten_scalar_bias = attention_scalar_bias_in.get_ptr();
  auto* lstm_w = &lstm_weight_in;
  auto* lstm_b = &lstm_bias_in;

  auto* hidden_out = hidden;
  auto* cell_out = cell;
  auto* atted_x = attentioned_x;
  auto* fc_out = attention_fc_out;

  // some shape should be reshape here since infershape can not get lod info
  auto x_lod = x->lod();
  const int N = static_cast<int>(x_lod[0].size() - 1);  // batch size
  auto x_dims = x->dims();                              // T x M
  auto w_dims = lstm_w->dims();                         // (D+M) x 4D
  const int total_T = static_cast<int>(x_dims[0]);
  const int M = static_cast<int>(x_dims[1]);      // x frame size
  const int D = static_cast<int>(w_dims[1] / 4);  // gate frame size
  const int D2 = static_cast<int>(D * 2);
  const int D3 = static_cast<int>(D * 3);
  const int D4 = static_cast<int>(w_dims[1]);
  int max_seq_len = static_cast<int>(x_lod[0][1]);
  for (int i = 1; i < N; ++i) {
    int len = static_cast<int>(x_lod[0][i + 1] - x_lod[0][i]);
    max_seq_len = max_seq_len < len ? len : max_seq_len;
  }
  PADDLE_ENFORCE_EQ(
      x_lod.size(),
      1UL,
      common::errors::InvalidArgument("Input(X)'s lod size must be 1."));
  PADDLE_ENFORCE_EQ(
      c0->dims()[0],
      N,
      common::errors::InvalidArgument("C0 dims should be %d x %d.", N, D));
  fc_out->Resize({max_seq_len, 1});

  std::function<void(const int, const T*, T*)> act_gate, act_cell, act_cand;
  auto& act_gate_str = gate_activation;
  auto& act_cell_str = cell_activation;
  auto& act_cand_str = candidate_activation;
  if (phi::backends::cpu::MayIUse(phi::backends::cpu::avx)) {
    phi::funcs::VecActivations<T, phi::backends::cpu::avx> act_functor;
    act_gate = act_functor(act_gate_str);
    act_cell = act_functor(act_cell_str);
    act_cand = act_functor(act_cand_str);
  } else {
    phi::funcs::VecActivations<T, phi::backends::cpu::isa_any> act_functor;
    act_gate = act_functor(act_gate_str);
    act_cell = act_functor(act_cell_str);
    act_cand = act_functor(act_cand_str);
  }

  const T* x_data = x->data<T>();
  const T* h0_data = h0 ? h0->data<T>() : NULL;
  const T* c0_data = c0->data<T>();
  const T* lstm_w_data = lstm_w->data<T>();
  const T* lstm_b_data = lstm_b->data<T>();
  const T* atten_w_data = atten_w->data<T>();
  const T* atten_b_data = atten_b ? atten_b->data<T>() : NULL;
  const T* atten_scalar_data = atten_scalar ? atten_scalar->data<T>() : NULL;
  const T* atten_scalar_bias_data =
      atten_scalar_bias ? atten_scalar_bias->data<T>() : NULL;

  T* hidden_out_data = dev_ctx.template Alloc<T>(hidden_out);
  T* cell_out_data = dev_ctx.template Alloc<T>(cell_out);
  T* atted_x_data = dev_ctx.template Alloc<T>(atted_x);
  T* fc_out_data = dev_ctx.template Alloc<T>(fc_out);
  T* lstm_x_data = dev_ctx.template Alloc<T>(lstm_x);
  T* lstm_out_data = dev_ctx.template Alloc<T>(lstm_out);

  // x(TxM) * fc (Mx1) part of atten_wgt(M+D)x1
  auto blas = phi::funcs::GetBlas<phi::CPUContext, T>(dev_ctx);

  phi::funcs::FCFunctor<Context, T> fc;
  fc(dev_ctx, total_T, 1, M, x_data, atten_w_data, atted_x_data, atten_b_data);

  const T* cur_atten_x_data = atted_x_data;
  const T* cur_x_data = x_data;
  const T* prev_cell_data = NULL;
  const T* prev_hidden_data = NULL;
  T* cur_cell_out_data = cell_out_data;
  T* cur_hidden_out_data = hidden_out_data;
  for (int i = 0; i < N; ++i) {
    int seq_len = static_cast<int>(x_lod[0][i + 1] - x_lod[0][i]);
    prev_cell_data = c0_data + i * D;
    prev_hidden_data = h0_data ? h0_data + i * D : NULL;
    for (int step = 0; step < seq_len; ++step) {
      /// 1. compute attention vector
      // 1a. prev_cell(1xD) * fc(D) rest part of atten_wgt
      T prev_cell_bias = blas.DOT(D, prev_cell_data, atten_w_data + M);
      // 1b. add cell bias and relu
      bias_relu<T>(seq_len, cur_atten_x_data, &prev_cell_bias, fc_out_data);
      // 1c. fc scalar
      if (atten_scalar_data) {
        blas.SCAL(seq_len, *atten_scalar_data, fc_out_data);
        bias_relu<T>(seq_len, fc_out_data, atten_scalar_bias_data, fc_out_data);
      }
      // 1d. softmax
      vec_softmax<T>(seq_len, fc_out_data, fc_out_data);
      // mul x(seq_len*M) and sum pool
      fc(dev_ctx, 1, M, seq_len, fc_out_data, cur_x_data, lstm_x_data);

      /// 2. compute LSTM step
      // lstm weight : concat[forget , input , output , tilde]
      // shape : (D + M) x (4 * D)
      // fc inputX(1xM) * weightX(M*(4D))  => 1 x 4D
      blas.MatMul(1, D4, M, lstm_x_data, lstm_w_data + D * D4, lstm_out_data);
      if (prev_hidden_data) {
        blas.GEMM(CblasNoTrans,
                  CblasNoTrans,
                  1,
                  D4,
                  D,
                  static_cast<T>(1),
                  prev_hidden_data,
                  D,
                  lstm_w_data,
                  D4,
                  static_cast<T>(1),
                  lstm_out_data,
                  D4);
      }
      // since input is 1xM, so can use add bias
      blas.VADD(D4, lstm_b_data, lstm_out_data, lstm_out_data);

      // gate act: sigmoid
      act_gate(D3, lstm_out_data, lstm_out_data);
      // candidate act: tanh
      act_cand(D, lstm_out_data + D3, lstm_out_data + D3);

      // a = forget * prev_cell
      blas.VMUL(D, lstm_out_data, prev_cell_data, lstm_out_data);

      // b = input * tilde
      blas.VMUL(D, lstm_out_data + D, lstm_out_data + D3, lstm_out_data + D);

      // cell_out = a + b
      blas.VADD(D, lstm_out_data, lstm_out_data + D, cur_cell_out_data);

      // state act tanh(cell_out) * output_gate
      act_cell(D, cur_cell_out_data, lstm_out_data);
      blas.VMUL(D, lstm_out_data, lstm_out_data + D2, cur_hidden_out_data);

      prev_hidden_data = cur_hidden_out_data;
      prev_cell_data = cur_cell_out_data;
      cur_cell_out_data = cur_cell_out_data + D;
      cur_hidden_out_data = cur_hidden_out_data + D;
    }
    cur_x_data = cur_x_data + seq_len * M;
    cur_atten_x_data = cur_atten_x_data + seq_len;
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(
    attention_lstm, CPU, ALL_LAYOUT, phi::AttentionLSTMKernel, float, double) {}
