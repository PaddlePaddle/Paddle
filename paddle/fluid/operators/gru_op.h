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

#include "paddle/operators/math/detail/activation_functions.h"
#include "paddle/operators/math/gru_compute.h"
#include "paddle/operators/math/math_function.h"
#include "paddle/operators/math/sequence2batch.h"

#include "paddle/framework/eigen.h"
#include "paddle/framework/op_registry.h"

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
class GRUKernel : public framework::OpKernel<T> {
 public:
  void BatchCompute(const framework::ExecutionContext& context) const {
    auto* input = context.Input<LoDTensor>("Input");
    auto* h0 = context.Input<Tensor>("H0");
    auto* weight = context.Input<Tensor>("Weight");
    const T* weight_data = weight->data<T>();
    auto* bias = context.Input<Tensor>("Bias");
    auto* batch_gate = context.Output<LoDTensor>("BatchGate");
    batch_gate->mutable_data<T>(context.GetPlace());
    auto* batch_reset_hidden_prev =
        context.Output<LoDTensor>("BatchResetHiddenPrev");
    batch_reset_hidden_prev->mutable_data<T>(context.GetPlace());
    auto* batch_hidden = context.Output<LoDTensor>("BatchHidden");
    batch_hidden->mutable_data<T>(context.GetPlace());
    auto* hidden = context.Output<LoDTensor>("Hidden");
    hidden->mutable_data<T>(context.GetPlace());

    context.ShareLoD("Input", "Hidden");

    auto hidden_dims = hidden->dims();

    bool is_reverse = context.Attr<bool>("is_reverse");
    math::LoDTensor2BatchFunctor<DeviceContext, T> to_batch;
    auto& dev_ctx = context.template device_context<DeviceContext>();
    to_batch(dev_ctx, *input, *batch_gate, true, is_reverse);

    if (bias) {
      math::RowwiseAdd<DeviceContext, T> add_bias;
      add_bias(dev_ctx, *batch_gate, *bias, batch_gate);
    }

    int frame_size = hidden_dims[1];
    math::GRUMetaValue<T> gru_value;
    gru_value.gate_weight = const_cast<T*>(weight_data);
    gru_value.state_weight =
        const_cast<T*>(weight_data + 2 * frame_size * frame_size);
    Tensor ordered_h0;

    framework::Vector<size_t> order(batch_gate->lod()[2]);

    if (h0) {
      // Since the batch computing for GRU reorders the input sequences
      // according to their length. The initialized cell state also needs
      // to reorder.
      ReorderInitState<DeviceContext, T>(
          context.template device_context<DeviceContext>(), *h0, order,
          &ordered_h0, true);
      gru_value.prev_out_value = ordered_h0.data<T>();
    } else {
      gru_value.prev_out_value = nullptr;
    }
    auto batch_starts = batch_gate->lod()[0];
    size_t num_batch = batch_starts.size() - 1;
    auto active_node = math::detail::GetActivationType(
        context.Attr<std::string>("activation"));
    auto active_gate = math::detail::GetActivationType(
        context.Attr<std::string>("gate_activation"));
    for (size_t n = 0; n < num_batch; n++) {
      int bstart = static_cast<int>(batch_starts[n]);
      int bend = static_cast<int>(batch_starts[n + 1]);
      int cur_batch_size = bend - bstart;

      Tensor gate_t = batch_gate->Slice(bstart, bend);
      Tensor reset_hidden_prev_t = batch_reset_hidden_prev->Slice(bstart, bend);
      Tensor hidden_t = batch_hidden->Slice(bstart, bend);
      gru_value.output_value = hidden_t.data<T>();
      gru_value.gate_value = gate_t.data<T>();
      gru_value.reset_output_value = reset_hidden_prev_t.data<T>();
      math::GRUUnitFunctor<DeviceContext, T>::compute(
          dev_ctx, gru_value, frame_size, cur_batch_size, active_node,
          active_gate);
      gru_value.prev_out_value = gru_value.output_value;
    }

    math::Batch2LoDTensorFunctor<DeviceContext, T> to_seq;
    batch_hidden->set_lod(batch_gate->lod());
    to_seq(dev_ctx, *batch_hidden, *hidden);
  }

  void Compute(const framework::ExecutionContext& context) const override {
    BatchCompute(context);
  }
};

template <typename DeviceContext, typename T>
class GRUGradKernel : public framework::OpKernel<T> {
 public:
  void BatchCompute(const framework::ExecutionContext& context) const {
    auto* h0 = context.Input<Tensor>("H0");
    auto* weight = context.Input<Tensor>("Weight");
    const T* weight_data = weight->data<T>();
    auto* batch_gate = context.Input<LoDTensor>("BatchGate");
    auto* batch_reset_hidden_prev =
        context.Input<LoDTensor>("BatchResetHiddenPrev");
    auto* batch_hidden = context.Input<LoDTensor>("BatchHidden");
    auto* hidden = context.Input<LoDTensor>("Hidden");
    auto* hidden_grad =
        context.Input<LoDTensor>(framework::GradVarName("Hidden"));
    auto* input_grad =
        context.Output<LoDTensor>(framework::GradVarName("Input"));
    auto* h0_grad = context.Output<Tensor>(framework::GradVarName("H0"));
    auto* weight_grad =
        context.Output<Tensor>(framework::GradVarName("Weight"));
    auto* bias_grad = context.Output<Tensor>(framework::GradVarName("Bias"));

    auto gate_dims = batch_gate->dims();
    auto hidden_dims = hidden->dims();
    int frame_size = hidden_dims[1];

    math::LoDTensor2BatchFunctor<DeviceContext, T> to_batch;
    LoDTensor batch_hidden_grad, batch_gate_grad, batch_reset_hidden_prev_grad;
    batch_hidden_grad.mutable_data<T>(hidden_dims, context.GetPlace());
    batch_gate_grad.mutable_data<T>(gate_dims, context.GetPlace());
    batch_reset_hidden_prev_grad.mutable_data<T>(hidden_dims,
                                                 context.GetPlace());
    math::SetConstant<DeviceContext, T> zero;
    auto& dev_ctx = context.template device_context<DeviceContext>();
    zero(dev_ctx, &batch_hidden_grad, static_cast<T>(0.0));
    zero(dev_ctx, &batch_gate_grad, static_cast<T>(0.0));
    zero(dev_ctx, &batch_reset_hidden_prev_grad, static_cast<T>(0.0));

    Tensor ordered_h0, ordered_h0_grad;

    framework::Vector<size_t> order(batch_gate->lod()[2]);

    if (h0) {
      ReorderInitState<DeviceContext, T>(dev_ctx, *h0, order, &ordered_h0,
                                         true);
    }
    if (h0_grad) {
      ordered_h0_grad.mutable_data<T>(h0_grad->dims(), context.GetPlace());
      zero(context.template device_context<DeviceContext>(), &ordered_h0_grad,
           static_cast<T>(0.0));
    }

    bool is_reverse = context.Attr<bool>("is_reverse");
    batch_hidden_grad.set_lod(batch_hidden->lod());
    to_batch(dev_ctx, *hidden_grad, batch_hidden_grad, false, is_reverse);

    math::GRUMetaValue<T> gru_value;
    gru_value.gate_weight = const_cast<T*>(weight_data);
    gru_value.state_weight =
        const_cast<T*>(weight_data + 2 * frame_size * frame_size);

    math::GRUMetaGrad<T> gru_grad;
    if (weight_grad) {
      gru_grad.gate_weight_grad =
          weight_grad->mutable_data<T>(context.GetPlace());
      zero(dev_ctx, weight_grad, static_cast<T>(0.0));
      gru_grad.state_weight_grad =
          weight_grad->data<T>() + 2 * frame_size * frame_size;
    } else {
      gru_grad.gate_weight_grad = nullptr;
      gru_grad.state_weight_grad = nullptr;
    }

    auto batch_starts = batch_hidden_grad.lod()[0];
    size_t num_batch = batch_starts.size() - 1;
    auto active_node = math::detail::GetActivationType(
        context.Attr<std::string>("activation"));
    auto active_gate = math::detail::GetActivationType(
        context.Attr<std::string>("gate_activation"));
    for (int n = static_cast<int>(num_batch) - 1; n >= 0; n--) {
      int bstart = static_cast<int>(batch_starts[n]);
      int bend = static_cast<int>(batch_starts[n + 1]);
      int cur_batch_size = bend - bstart;

      Tensor gate_t = batch_gate->Slice(bstart, bend);
      gru_value.gate_value = gate_t.data<T>();
      Tensor reset_hidden_prev_t = batch_reset_hidden_prev->Slice(bstart, bend);
      gru_value.reset_output_value = reset_hidden_prev_t.data<T>();

      Tensor hidden_grad_t = batch_hidden_grad.Slice(bstart, bend);
      gru_grad.output_grad = hidden_grad_t.data<T>();
      Tensor gate_grad_t = batch_gate_grad.Slice(bstart, bend);
      gru_grad.gate_grad = gate_grad_t.data<T>();
      Tensor reset_hidden_prev_grad_t =
          batch_reset_hidden_prev_grad.Slice(bstart, bend);
      gru_grad.reset_output_grad = reset_hidden_prev_grad_t.data<T>();
      if (n == 0) {
        gru_value.prev_out_value = h0 ? ordered_h0.data<T>() : nullptr;
        gru_grad.prev_out_grad =
            h0 && h0_grad ? ordered_h0_grad.data<T>() : nullptr;
      } else {
        int bstart_pre = static_cast<int>(batch_starts[n - 1]);
        Tensor hidden_prev_t = batch_hidden->Slice(bstart_pre, bstart);
        gru_value.prev_out_value = hidden_prev_t.data<T>();
        Tensor hidden_prev_grad_t = batch_hidden_grad.Slice(bstart_pre, bstart);
        gru_grad.prev_out_grad = hidden_prev_grad_t.data<T>();
      }

      math::GRUUnitGradFunctor<DeviceContext, T>::compute(
          dev_ctx, gru_value, gru_grad, frame_size, cur_batch_size, active_node,
          active_gate);
    }
    if (input_grad) {
      input_grad->mutable_data<T>(context.GetPlace());
      math::Batch2LoDTensorFunctor<DeviceContext, T> to_seq;
      batch_gate_grad.set_lod(batch_gate->lod());
      to_seq(dev_ctx, batch_gate_grad, *input_grad);
    }
    if (bias_grad) {
      bias_grad->mutable_data<T>(context.GetPlace());
      math::ColwiseSum<DeviceContext, T> col_sum;
      col_sum(dev_ctx, batch_gate_grad, bias_grad);
    }
    if (h0 && h0_grad) {
      ReorderInitState<DeviceContext, T>(dev_ctx, ordered_h0_grad, order,
                                         h0_grad, false);
    }
  }

  void Compute(const framework::ExecutionContext& context) const override {
    BatchCompute(context);
  }
};

}  // namespace operators
}  // namespace paddle
