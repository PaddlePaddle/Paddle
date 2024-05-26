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

#pragma once
#include "paddle/phi/kernels/funcs/detail/activation_functions.h"

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/gru_compute.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/sequence2batch.h"
#include "paddle/utils/optional.h"

namespace phi {

template <typename Context, typename T>
void ReorderInitState(const Context &dev_ctx,
                      const phi::DenseTensor &src,
                      phi::Vector<size_t> index_lod,
                      phi::DenseTensor *dst,
                      bool indexed_src) {
  phi::funcs::CopyMatrixRowsFunctor<Context, T> row_shuffle;
  dst->Resize(src.dims());
  dev_ctx.template Alloc<T>(dst);
  row_shuffle(dev_ctx, src, index_lod, dst, indexed_src);
}

template <typename T, typename Context>
void GRUGradKernel(const Context &dev_ctx,
                   const DenseTensor &input,
                   const paddle::optional<DenseTensor> &h0_param,
                   const DenseTensor &weight,
                   const paddle::optional<DenseTensor> &bias,
                   const DenseTensor &batch_gate,
                   const DenseTensor &batch_reset_hidden_prev,
                   const DenseTensor &batch_hidden,
                   const DenseTensor &hidden,
                   const DenseTensor &hidden_grad,
                   const std::string &activation,
                   const std::string &gate_activation,
                   bool is_reverse,
                   bool origin_mode,
                   bool is_test,
                   DenseTensor *input_grad,
                   DenseTensor *h0_grad,
                   DenseTensor *weight_grad,
                   DenseTensor *bias_grad) {
  auto *h0 = h0_param.get_ptr();
  const T *weight_data = weight.data<T>();

  auto gate_dims = batch_gate.dims();
  auto hidden_dims = hidden.dims();
  int frame_size = hidden_dims[1];

  phi::funcs::LoDTensor2BatchFunctor<Context, T> to_batch;
  phi::DenseTensor batch_hidden_grad, batch_gate_grad,
      batch_reset_hidden_prev_grad;
  batch_hidden_grad.Resize(hidden_dims);
  batch_gate_grad.Resize(gate_dims);
  batch_reset_hidden_prev_grad.Resize(hidden_dims);
  dev_ctx.template Alloc<T>(&batch_hidden_grad);
  dev_ctx.template Alloc<T>(&batch_gate_grad);
  dev_ctx.template Alloc<T>(&batch_reset_hidden_prev_grad);

  phi::funcs::SetConstant<Context, T> zero;
  zero(dev_ctx, &batch_hidden_grad, static_cast<T>(0.0));
  zero(dev_ctx, &batch_gate_grad, static_cast<T>(0.0));
  zero(dev_ctx, &batch_reset_hidden_prev_grad, static_cast<T>(0.0));

  phi::DenseTensor ordered_h0, ordered_h0_grad;

  phi::Vector<size_t> order(batch_gate.lod()[2]);

  if (h0) {
    ReorderInitState<Context, T>(dev_ctx, *h0, order, &ordered_h0, true);
  }
  if (h0_grad) {
    ordered_h0_grad.Resize(h0_grad->dims());
    dev_ctx.template Alloc<T>(&ordered_h0_grad);
    zero(dev_ctx, &ordered_h0_grad, static_cast<T>(0.0));
  }

  batch_hidden_grad.set_lod(batch_hidden.lod());
  to_batch(dev_ctx, hidden_grad, &batch_hidden_grad, false, is_reverse);

  phi::funcs::GRUMetaValue<T> gru_value;
  gru_value.gate_weight = const_cast<T *>(weight_data);
  gru_value.state_weight =
      const_cast<T *>(weight_data + 2 * frame_size * frame_size);

  phi::funcs::GRUMetaGrad<T> gru_grad;
  if (weight_grad) {
    gru_grad.gate_weight_grad = dev_ctx.template Alloc<T>(weight_grad);
    zero(dev_ctx, weight_grad, static_cast<T>(0.0));
    gru_grad.state_weight_grad =
        weight_grad->data<T>() + 2 * frame_size * frame_size;
  } else {
    gru_grad.gate_weight_grad = nullptr;
    gru_grad.state_weight_grad = nullptr;
  }

  auto batch_starts = batch_hidden_grad.lod()[0];
  size_t num_batch = batch_starts.size() - 1;
  auto active_node = phi::funcs::detail::GetActivationType(activation);
  auto active_gate = phi::funcs::detail::GetActivationType(gate_activation);
  for (int n = static_cast<int>(num_batch) - 1; n >= 0; n--) {
    int bstart = static_cast<int>(batch_starts[n]);
    int bend = static_cast<int>(batch_starts[n + 1]);
    int cur_batch_size = bend - bstart;

    phi::DenseTensor gate_t = batch_gate.Slice(bstart, bend);
    gru_value.gate_value = gate_t.data<T>();
    phi::DenseTensor reset_hidden_prev_t =
        batch_reset_hidden_prev.Slice(bstart, bend);
    gru_value.reset_output_value = reset_hidden_prev_t.data<T>();

    phi::DenseTensor hidden_grad_t = batch_hidden_grad.Slice(bstart, bend);
    gru_grad.output_grad = hidden_grad_t.data<T>();
    phi::DenseTensor gate_grad_t = batch_gate_grad.Slice(bstart, bend);
    gru_grad.gate_grad = gate_grad_t.data<T>();
    phi::DenseTensor reset_hidden_prev_grad_t =
        batch_reset_hidden_prev_grad.Slice(bstart, bend);
    gru_grad.reset_output_grad = reset_hidden_prev_grad_t.data<T>();
    if (n == 0) {
      gru_value.prev_out_value = h0 ? ordered_h0.data<T>() : nullptr;
      gru_grad.prev_out_grad =
          h0 && h0_grad ? ordered_h0_grad.data<T>() : nullptr;
    } else {
      int bstart_pre = static_cast<int>(batch_starts[n - 1]);
      phi::DenseTensor hidden_prev_t = batch_hidden.Slice(bstart_pre, bstart);
      gru_value.prev_out_value = hidden_prev_t.data<T>();
      phi::DenseTensor hidden_prev_grad_t =
          batch_hidden_grad.Slice(bstart_pre, bstart);
      gru_grad.prev_out_grad = hidden_prev_grad_t.data<T>();
    }
    gru_value.output_value = nullptr;
    phi::funcs::GRUUnitGradFunctor<Context, T>::compute(dev_ctx,
                                                        gru_value,
                                                        gru_grad,
                                                        frame_size,
                                                        cur_batch_size,
                                                        active_node,
                                                        active_gate,
                                                        origin_mode);
  }
  if (input_grad) {
    dev_ctx.template Alloc<T>(input_grad);
    phi::funcs::Batch2LoDTensorFunctor<Context, T> to_seq;
    batch_gate_grad.set_lod(batch_gate.lod());
    to_seq(dev_ctx, batch_gate_grad, input_grad);
  }
  if (bias_grad) {
    dev_ctx.template Alloc<T>(bias_grad);
    phi::funcs::ColwiseSum<Context, T> col_sum;
    col_sum(dev_ctx, batch_gate_grad, bias_grad);
  }
  if (h0_param && h0_grad) {
    ReorderInitState<Context, T>(
        dev_ctx, ordered_h0_grad, order, h0_grad, false);
  }
}
}  // namespace phi
