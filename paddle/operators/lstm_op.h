/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
#include "paddle/framework/op_registry.h"
#include "paddle/operators/math/detail/activation_functions.h"
#include "paddle/operators/math/lstm_compute.h"
#include "paddle/operators/math/math_function.h"
#include "paddle/operators/math/sequence2batch.h"

namespace paddle {
namespace operators {

using LoDTensor = framework::LoDTensor;
using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
inline void ReorderInitState(const DeviceContext& ctx,
                             const framework::Tensor& src,
                             framework::Vector<size_t> index_lod,
                             framework::Tensor* dst, bool indexed_src) {
  math::CopyMatrixRowsFunctor<DeviceContext, T> row_shuffle;
  dst->mutable_data<T>(src.dims(), ctx.GetPlace());
  row_shuffle(ctx, src, index_lod, *dst, indexed_src);
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
    to_batch(device_ctx, *input, *batch_gate, true, is_reverse);

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
        math::matmul<DeviceContext, T>(device_ctx, pre_hidden_t, false, *weight,
                                       false, static_cast<T>(1.0), &gate_t,
                                       static_cast<T>(1.0));
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
        math::matmul<DeviceContext, T>(device_ctx, ordered_h0, false, *weight,
                                       false, static_cast<T>(1.0), &gate_t,
                                       static_cast<T>(1.0));
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
    to_seq(device_ctx, batch_hidden, *hidden_out);

    batch_cell.set_lod(batch_gate->lod());
    // restore the output cell state in LoDTensor from the batch cell
    to_seq(device_ctx, batch_cell, *cell_out);
  }
};

template <typename DeviceContext, typename T>
class LSTMGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<LoDTensor>("Input");
    auto* weight = ctx.Input<Tensor>("Weight");
    auto* bias = ctx.Input<Tensor>("Bias");

    auto* hidden_out = ctx.Input<LoDTensor>("Hidden");
    auto* cell_out = ctx.Input<LoDTensor>("Cell");

    auto* batch_gate = ctx.Input<LoDTensor>("BatchGate");
    auto* batch_cell_pre_act = ctx.Input<LoDTensor>("BatchCellPreAct");

    auto* hidden_g = ctx.Input<LoDTensor>(framework::GradVarName("Hidden"));

    auto* in_g = ctx.Output<LoDTensor>(framework::GradVarName("Input"));
    auto* weight_g = ctx.Output<Tensor>(framework::GradVarName("Weight"));
    auto* bias_g = ctx.Output<Tensor>(framework::GradVarName("Bias"));

    auto* h0 = ctx.Input<Tensor>("H0");
    auto* c0 = ctx.Input<Tensor>("C0");

    auto* h0_g = ctx.Output<Tensor>(framework::GradVarName("H0"));
    auto* c0_g = ctx.Output<Tensor>(framework::GradVarName("C0"));

    auto& device_ctx = ctx.template device_context<DeviceContext>();
    math::SetConstant<DeviceContext, T> zero;
    if (weight_g) {
      weight_g->mutable_data<T>(ctx.GetPlace());
      zero(device_ctx, weight_g, static_cast<T>(0.0));
    }

    // ordered_h0/c0 is the reordered hidden/cell initialization.
    // ordered_h0_g/c0_g is the reordered gradient of hidden/cell
    // initialization.
    Tensor ordered_h0, ordered_c0, ordered_h0_g, ordered_c0_g;
    framework::Vector<size_t> order(batch_gate->lod()[2]);

    if (c0) {
      ReorderInitState<DeviceContext, T>(device_ctx, *c0, order, &ordered_c0,
                                         true);
    }
    if (c0 && c0_g) {
      ordered_c0_g.mutable_data<T>(c0_g->dims(), ctx.GetPlace());
    }

    auto in_dims = input->dims();
    auto out_dims = hidden_g->dims();
    int frame_size = static_cast<int>(in_dims[1] / 4);
    PADDLE_ENFORCE_EQ(frame_size, out_dims[1]);

    math::LstmMetaValue<T> lstm_value;
    if (bias && ctx.Attr<bool>("use_peepholes")) {
      T* bias_data = const_cast<T*>(bias->data<T>());
      lstm_value.check_ig = bias_data + 4 * frame_size;
      lstm_value.check_fg = lstm_value.check_ig + frame_size;
      lstm_value.check_og = lstm_value.check_fg + frame_size;
    } else {
      lstm_value.check_ig = nullptr;
      lstm_value.check_fg = nullptr;
      lstm_value.check_og = nullptr;
    }

    math::LstmMetaGrad<T> lstm_grad;

    if (bias && bias_g) {
      bias_g->mutable_data<T>(ctx.GetPlace());
      zero(device_ctx, bias_g, static_cast<T>(0.0));
    }
    if (bias && bias_g && ctx.Attr<bool>("use_peepholes")) {
      T* bias_g_data = bias_g->data<T>();
      lstm_grad.check_ig_grad = bias_g_data + 4 * frame_size;
      lstm_grad.check_fg_grad = lstm_grad.check_ig_grad + frame_size;
      lstm_grad.check_og_grad = lstm_grad.check_fg_grad + frame_size;
    } else {
      lstm_grad.check_ig_grad = nullptr;
      lstm_grad.check_fg_grad = nullptr;
      lstm_grad.check_og_grad = nullptr;
    }

    math::LoDTensor2BatchFunctor<DeviceContext, T> to_batch;

    auto ToBatch = [&batch_gate, &to_batch](
        const DeviceContext& ctx, const framework::LoDTensor& src,
        const framework::DDim& dims, framework::LoDTensor& dst) {
      dst.mutable_data<T>(dims, ctx.GetPlace());
      dst.set_lod(batch_gate->lod());
      to_batch(ctx, src, dst, false);
    };

    LoDTensor batch_hidden, batch_hidden_g, batch_cell;
    ToBatch(device_ctx, *hidden_out, out_dims, batch_hidden);
    ToBatch(device_ctx, *hidden_g, out_dims, batch_hidden_g);
    ToBatch(device_ctx, *cell_out, out_dims, batch_cell);

    LoDTensor batch_cell_g, batch_gate_g;
    batch_cell_g.mutable_data<T>(out_dims, ctx.GetPlace());
    // TODO(qingqing) support the case output cell has gradient.
    // to_batch(device_ctx, *cell_g, batch_cell_g, false);
    zero(device_ctx, &batch_cell_g, static_cast<T>(0.0));
    batch_gate_g.mutable_data<T>(batch_gate->dims(), ctx.GetPlace());
    batch_gate_g.set_lod(batch_gate->lod());

    auto gate_act = math::detail::GetActivationType(
        ctx.Attr<std::string>("gate_activation"));
    auto cell_act = math::detail::GetActivationType(
        ctx.Attr<std::string>("cell_activation"));
    auto cand_act = math::detail::GetActivationType(
        ctx.Attr<std::string>("candidate_activation"));

    auto batch_starts = batch_gate->lod()[0];
    size_t num_batch = batch_starts.size() - 1;
    for (int n = static_cast<int>(num_batch) - 1; n >= 0; n--) {
      int bstart = static_cast<int>(batch_starts[n]);
      int bend = static_cast<int>(batch_starts[n + 1]);

      Tensor gate = batch_gate->Slice(bstart, bend);
      Tensor cell = batch_cell.Slice(bstart, bend);
      Tensor cell_pre_act = batch_cell_pre_act->Slice(bstart, bend);
      lstm_value.gate_value = gate.data<T>();
      lstm_value.state_value = cell.data<T>();
      lstm_value.state_active_value = cell_pre_act.data<T>();

      Tensor out_g = batch_hidden_g.Slice(bstart, bend);
      Tensor gate_g = batch_gate_g.Slice(bstart, bend);
      Tensor cell_g = batch_cell_g.Slice(bstart, bend);
      lstm_grad.state_grad = cell_g.data<T>();
      lstm_grad.gate_grad = gate_g.data<T>();
      lstm_grad.output_grad = out_g.data<T>();

      if (n > 0) {
        int bstart_pre = static_cast<int>(batch_starts[n - 1]);
        Tensor cell_pre = batch_cell.Slice(bstart_pre, bstart);
        Tensor cell_pre_g = batch_cell_g.Slice(bstart_pre, bstart);
        lstm_value.prev_state_value = cell_pre.data<T>();
        lstm_grad.prev_state_grad = cell_pre_g.data<T>();
      } else {
        lstm_value.prev_state_value = c0 ? ordered_c0.data<T>() : nullptr;
        lstm_grad.prev_state_grad = c0_g ? ordered_c0_g.data<T>() : nullptr;
      }

      int cur_batch_size = bend - bstart;
      math::LstmUnitGradFunctor<DeviceContext, T>::compute(
          device_ctx, lstm_value, lstm_grad, frame_size, cur_batch_size,
          gate_act, cell_act, cand_act);

      if (n > 0) {
        int pre_h_start = static_cast<int>(batch_starts[n - 1]);
        int pre_h_end = pre_h_start + cur_batch_size;
        auto pre_hidden_g = batch_hidden_g.Slice(pre_h_start, pre_h_end);
        math::matmul<DeviceContext, T>(device_ctx, gate_g, false, *weight, true,
                                       static_cast<T>(1.0), &pre_hidden_g,
                                       static_cast<T>(1.0));
        if (weight_g) {
          /* backward weight */
          auto pre_hidden = batch_hidden.Slice(pre_h_start, pre_h_end);
          math::matmul<DeviceContext, T>(device_ctx, pre_hidden, true, gate_g,
                                         false, static_cast<T>(1.0), weight_g,
                                         static_cast<T>(1.0));
        }
      } else {
        if (h0 && weight_g) {
          ReorderInitState<DeviceContext, T>(device_ctx, *h0, order,
                                             &ordered_h0, true);
          math::matmul<DeviceContext, T>(device_ctx, ordered_h0, true, gate_g,
                                         false, static_cast<T>(1.0), weight_g,
                                         static_cast<T>(1.0));
        }
        if (h0 && h0_g) {
          ordered_h0_g.mutable_data<T>(h0_g->dims(), ctx.GetPlace());
          math::matmul<DeviceContext, T>(device_ctx, gate_g, false, *weight,
                                         true, static_cast<T>(1.0),
                                         &ordered_h0_g, static_cast<T>(0.0));
        }
      }
    }

    math::Batch2LoDTensorFunctor<DeviceContext, T> to_seq;
    if (in_g) {
      /* backward data */
      in_g->mutable_data<T>(ctx.GetPlace());
      to_seq(device_ctx, batch_gate_g, *in_g);
    }
    if (bias && bias_g) {
      /* backward bias */
      Tensor b_g = *bias_g;
      b_g.Resize({bias_g->numel(), 1});
      Tensor gate_bias_g = b_g.Slice(0, 4 * frame_size);
      math::ColwiseSum<DeviceContext, T> col_sum;
      col_sum(device_ctx, batch_gate_g, &gate_bias_g);
    }

    if (h0 && h0_g) {
      ReorderInitState<DeviceContext, T>(device_ctx, ordered_h0_g, order, h0_g,
                                         false);
    }
    if (c0 && c0_g) {
      ReorderInitState<DeviceContext, T>(device_ctx, ordered_c0_g, order, c0_g,
                                         false);
    }
  }
};

}  // namespace operators
}  // namespace paddle
