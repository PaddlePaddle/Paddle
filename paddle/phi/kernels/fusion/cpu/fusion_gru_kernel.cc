// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include <cstring>  // for memcpy
#include <string>
#include <vector>

#include "paddle/common/errors.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/fc_functor.h"
#include "paddle/phi/kernels/funcs/jit/kernels.h"
#include "paddle/phi/kernels/funcs/sequence2batch.h"

namespace phi::fusion {

#define INIT_BASE_DEFINES                                  \
  auto x_lod = x.lod();                                    \
  auto x_dims = x.dims(); /* T x M*/                       \
  auto x_mat_dims = (x_dims.size() == 3 && x_dims[1] == 1) \
                        ? common::flatten_to_2d(x_dims, 1) \
                        : x_dims;                          \
  auto wh_dims = weight_h.dims(); /* D x 3D*/              \
  const int total_T = x_mat_dims[0];                       \
  const int D3 = wh_dims[1]

#define INIT_OTHER_DEFINES                                                   \
  const int M = x_mat_dims[1];                                               \
  const int D = wh_dims[0];                                                  \
  const int D2 = D * 2;                                                      \
  const phi::jit::gru_attr_t attr(D,                                         \
                                  phi::jit::to_kerneltype(gate_activation),  \
                                  phi::jit::to_kerneltype(activation));      \
  phi::jit::gru_t one_step;                                                  \
  auto ComputeH1 =                                                           \
      phi::jit::KernelFuncs<phi::jit::GRUH1Tuple<T>, phi::CPUPlace>::Cache() \
          .At(attr);                                                         \
  auto ComputeHtPart1 = phi::jit::KernelFuncs<phi::jit::GRUHtPart1Tuple<T>,  \
                                              phi::CPUPlace>::Cache()        \
                            .At(attr);                                       \
  auto ComputeHtPart2 = phi::jit::KernelFuncs<phi::jit::GRUHtPart2Tuple<T>,  \
                                              phi::CPUPlace>::Cache()        \
                            .At(attr);                                       \
  const T* x_data = x.data<T>();                                             \
  const T* wx_data = weight_x.data<T>();                                     \
  const T* wh_data = weight_h.data<T>();                                     \
  T* xx_data = dev_ctx.template Alloc<T>(xx)

template <typename T, typename Context>
void SeqCompute(const Context& dev_ctx,
                const DenseTensor& x,
                const paddle::optional<DenseTensor>& h0,
                const DenseTensor& weight_x,
                const DenseTensor& weight_h,
                const paddle::optional<DenseTensor>& bias,
                const std::string& activation,
                const std::string& gate_activation,
                const bool is_reverse,
                const bool use_seq,
                DenseTensor* reordered_h0,
                DenseTensor* xx,
                DenseTensor* batched_input,
                DenseTensor* batched_out,
                DenseTensor* hidden) {
  INIT_BASE_DEFINES;
  INIT_OTHER_DEFINES;
  const int N = static_cast<int>(x_lod[0].size() - 1);
  const T* h0_data = h0 ? h0->data<T>() : nullptr;
  const T* wh_state_data = wh_data + D * D2;
  T* hidden_out_data = dev_ctx.template Alloc<T>(hidden);

  auto blas = phi::funcs::GetBlas<Context, T>(dev_ctx);

  phi::funcs::FCFunctor<Context, T> fc;
  fc(dev_ctx,
     total_T,
     D3,
     M,
     x_data,
     wx_data,
     xx_data,
     bias ? bias->data<T>() : nullptr);

  int xx_offset = D3;
  int gate_offset = D;
  if (is_reverse) {
    const int offset = (total_T - 1) * D;
    xx_data = xx_data + offset * 3;
    hidden_out_data = hidden_out_data + offset;
    xx_offset = -D3;
    gate_offset = -D;
  }
  auto move_step = [&]() {
    xx_data = xx_data + xx_offset;
    hidden_out_data = hidden_out_data + gate_offset;
  };
  for (int i = 0; i < N; ++i) {
    int bid = is_reverse ? N - 1 - i : i;
    int seq_len = static_cast<int>(x_lod[0][bid + 1] - x_lod[0][bid]);
    const T* prev_hidden_data = nullptr;
    int tstart = 0;
    if (h0_data) {
      prev_hidden_data = h0_data + bid * D;
    } else {
      one_step.gates = xx_data;
      one_step.ht = hidden_out_data;
      ComputeH1(&one_step, &attr);
      prev_hidden_data = hidden_out_data;
      tstart = 1;
      move_step();
    }
    for (int step = tstart; step < seq_len; ++step) {
      // gemm prev * (Wu + Wr)
      blas.GEMM(CblasNoTrans,
                CblasNoTrans,
                1,
                D2,
                D,
                static_cast<T>(1),
                prev_hidden_data,
                D,
                wh_data,
                D2,
                static_cast<T>(1),
                xx_data,
                D3);
      one_step.gates = xx_data;
      one_step.ht_1 = prev_hidden_data;
      one_step.ht = hidden_out_data;
      ComputeHtPart1(&one_step, &attr);
      // gemm rt * Ws
      blas.GEMM(CblasNoTrans,
                CblasNoTrans,
                1,
                D,
                D,
                static_cast<T>(1),
                hidden_out_data,
                D,
                wh_state_data,
                D,
                static_cast<T>(1),
                xx_data + D2,
                D3);
      ComputeHtPart2(&one_step, &attr);
      // save prev
      prev_hidden_data = hidden_out_data;
      move_step();
    }
  }
}

template <typename T, typename Context>
void BatchCompute(const Context& dev_ctx,
                  const DenseTensor& x,
                  const paddle::optional<DenseTensor>& h0,
                  const DenseTensor& weight_x,
                  const DenseTensor& weight_h,
                  const paddle::optional<DenseTensor>& bias,
                  const std::string& activation,
                  const std::string& gate_activation,
                  const bool is_reverse,
                  const bool use_seq,
                  DenseTensor* reordered_h0,
                  DenseTensor* xx,
                  DenseTensor* batched_input,
                  DenseTensor* batched_out,
                  DenseTensor* hidden) {
  INIT_BASE_DEFINES;
  if (x_lod[0].size() == 2) {
    xx->Resize({total_T, D3});
    SeqCompute<T, Context>(dev_ctx,
                           x,
                           h0,
                           weight_x,
                           weight_h,
                           bias,
                           activation,
                           gate_activation,
                           is_reverse,
                           use_seq,
                           reordered_h0,
                           xx,
                           batched_input,
                           batched_out,
                           hidden);
    return;
  }
  INIT_OTHER_DEFINES;
  T* batched_input_data = dev_ctx.template Alloc<T>(batched_input);
  T* batched_out_data = dev_ctx.template Alloc<T>(batched_out);
  dev_ctx.template Alloc<T>(hidden);
  auto blas = phi::funcs::GetBlas<Context, T>(dev_ctx);
  phi::funcs::DenseTensor2BatchFunctor<Context, T> to_batch;

  phi::funcs::FCFunctor<Context, T> fc;
  if (M > D3) {
    fc(dev_ctx,
       total_T,
       D3,
       M,
       x_data,
       wx_data,
       xx_data,
       bias ? bias->data<T>() : nullptr);
    to_batch(dev_ctx, *xx, batched_input, true, is_reverse);
  } else {
    to_batch(dev_ctx, x, xx, true, is_reverse);
    batched_input->set_lod(xx->lod());
    fc(dev_ctx,
       total_T,
       D3,
       M,
       xx_data,
       wx_data,
       batched_input_data,
       bias ? bias->data<T>() : nullptr);
  }

  auto batched_lod = batched_input->lod();
  const auto& seq_order = batched_lod[2];
  const int max_bs = static_cast<int>(seq_order.size());
  reordered_h0->Resize({max_bs, D});

  int tstart = 0;
  T* prev_hidden_data = nullptr;
  if (h0) {
    // reorder h0
    T* reordered_h0_data = dev_ctx.template Alloc<T>(reordered_h0);
    const T* h0_data = h0->data<T>();
    prev_hidden_data = reordered_h0_data;
    size_t sz = sizeof(T) * D;
    for (int i = 0; i < max_bs; ++i) {
      std::memcpy(reordered_h0_data, h0_data + seq_order[i] * D, sz);
      reordered_h0_data += D;
    }
  } else {
    // compute without h0
    T* cur_in_data = batched_input_data;
    T* cur_out_data = batched_out_data;
    // W: {W_update, W_reset; W_state}
    for (int i = 0; i < max_bs; ++i) {
      one_step.gates = cur_in_data;
      one_step.ht = cur_out_data;
      ComputeH1(&one_step, &attr);
      // add offset
      cur_in_data += D3;
      cur_out_data += D;
    }
    tstart = 1;
    prev_hidden_data = batched_out_data;
  }
  // Then start from next
  const T* wh_state_data = wh_data + D * D2;
  const auto& batch_starts = batched_lod[0];
  const int max_seq_len = static_cast<int>(batch_starts.size() - 1);
  batched_input_data = batched_input_data + tstart * max_bs * D3;
  batched_out_data = batched_out_data + tstart * max_bs * D;
  for (int step = tstart; step < max_seq_len; ++step) {
    const int cur_bs =
        static_cast<int>(batch_starts[step + 1] - batch_starts[step]);
    // gemm prev * (Wu + Wr)
    blas.GEMM(CblasNoTrans,
              CblasNoTrans,
              cur_bs,
              D2,
              D,
              static_cast<T>(1),
              prev_hidden_data,
              D,
              wh_data,
              D2,
              static_cast<T>(1),
              batched_input_data,
              D3);

    T* cur_batched_data = batched_input_data;
    T* cur_out_data = batched_out_data;
    T* cur_prev_hidden_data = prev_hidden_data;
    for (int i = 0; i < cur_bs; ++i) {
      one_step.gates = cur_batched_data;
      one_step.ht_1 = cur_prev_hidden_data;
      one_step.ht = cur_out_data;
      ComputeHtPart1(&one_step, &attr);

      cur_batched_data += D3;
      cur_prev_hidden_data += D;
      cur_out_data += D;
    }

    cur_batched_data = batched_input_data;
    cur_out_data = batched_out_data;
    blas.GEMM(CblasNoTrans,
              CblasNoTrans,
              cur_bs,
              D,
              D,
              static_cast<T>(1),
              cur_out_data,
              D,
              wh_state_data,
              D,
              static_cast<T>(1),
              cur_batched_data + D2,
              D3);

    cur_prev_hidden_data = prev_hidden_data;
    for (int i = 0; i < cur_bs; ++i) {
      one_step.gates = cur_batched_data;
      one_step.ht_1 = cur_prev_hidden_data;
      one_step.ht = cur_out_data;
      ComputeHtPart2(&one_step, &attr);
      cur_batched_data += D3;
      cur_prev_hidden_data += D;
      cur_out_data += D;
    }
    prev_hidden_data = batched_out_data;
    batched_out_data = cur_out_data;
    batched_input_data = cur_batched_data;
  }

  phi::funcs::Batch2DenseTensorFunctor<Context, T> to_seq;
  batched_out->set_lod(batched_lod);
  to_seq(dev_ctx, *batched_out, hidden);
}

template <typename T, typename Context>
void FusionGRUKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const paddle::optional<DenseTensor>& h0,
                     const DenseTensor& weight_x,
                     const DenseTensor& weight_h,
                     const paddle::optional<DenseTensor>& bias,
                     const std::string& activation,
                     const std::string& gate_activation,
                     const bool is_reverse,
                     const bool use_seq,
                     const bool origin_mode,
                     const bool force_fp32_output,
                     DenseTensor* reordered_h0,
                     DenseTensor* xx,
                     DenseTensor* batched_input,
                     DenseTensor* batched_out,
                     DenseTensor* hidden) {
  if (use_seq) {
    SeqCompute<T, Context>(dev_ctx,
                           x,
                           h0,
                           weight_x,
                           weight_h,
                           bias,
                           activation,
                           gate_activation,
                           is_reverse,
                           use_seq,
                           reordered_h0,
                           xx,
                           batched_input,
                           batched_out,
                           hidden);
  } else {
    BatchCompute<T, Context>(dev_ctx,
                             x,
                             h0,
                             weight_x,
                             weight_h,
                             bias,
                             activation,
                             gate_activation,
                             is_reverse,
                             use_seq,
                             reordered_h0,
                             xx,
                             batched_input,
                             batched_out,
                             hidden);
  }
}

}  // namespace phi::fusion

PD_REGISTER_KERNEL(
    fusion_gru, CPU, ALL_LAYOUT, phi::fusion::FusionGRUKernel, float, double) {}
