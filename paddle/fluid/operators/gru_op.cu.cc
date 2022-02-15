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

#include "paddle/fluid/operators/gru_op.h"

namespace paddle {
namespace platform {
class CUDADeviceContext;

}  // namespace platform
}  // namespace paddle

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class GRUKernel : public framework::OpKernel<T> {
 public:
  void BatchCompute(const framework::ExecutionContext& context) const {
    using LodTensorPtr = LoDTensor*;

    bool is_test = context.Attr<bool>("is_test");
    bool origin_mode = context.Attr<bool>("origin_mode");
    auto* input = context.Input<LoDTensor>("Input");
    auto* h0 = context.Input<Tensor>("H0");
    auto* weight = context.Input<Tensor>("Weight");
    const T* weight_data = weight->data<T>();
    auto* bias = context.Input<Tensor>("Bias");
    auto* hidden = context.Output<LoDTensor>("Hidden");
    hidden->mutable_data<T>(context.GetPlace());

    auto input_dims = input->dims();
    auto hidden_dims = hidden->dims();

    LodTensorPtr batch_gate, batch_reset_hidden_prev, batch_hidden;
    LoDTensor batch_gate_tmp, batch_reset_hidden_prev_tmp, batch_hidden_tmp;
    if (is_test) {
      batch_gate = &batch_gate_tmp;
      batch_gate->Resize(input_dims);

      batch_reset_hidden_prev = &batch_reset_hidden_prev_tmp;
      batch_reset_hidden_prev->Resize(hidden_dims);

      batch_hidden = &batch_hidden_tmp;
      batch_hidden->Resize(hidden_dims);
    } else {
      batch_gate = context.Output<LoDTensor>("BatchGate");
      batch_hidden = context.Output<LoDTensor>("BatchHidden");
      batch_reset_hidden_prev =
          context.Output<LoDTensor>("BatchResetHiddenPrev");
    }
    batch_gate->mutable_data<T>(context.GetPlace());
    batch_reset_hidden_prev->mutable_data<T>(context.GetPlace());
    batch_hidden->mutable_data<T>(context.GetPlace());

    bool is_reverse = context.Attr<bool>("is_reverse");
    math::LoDTensor2BatchFunctor<DeviceContext, T> to_batch;
    auto& dev_ctx = context.template device_context<DeviceContext>();
    to_batch(dev_ctx, *input, batch_gate, true, is_reverse);

    if (bias) {
      pten::funcs::RowwiseAdd<DeviceContext, T> add_bias;
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
          active_gate, origin_mode);
      gru_value.prev_out_value = gru_value.output_value;
    }

    math::Batch2LoDTensorFunctor<DeviceContext, T> to_seq;
    batch_hidden->set_lod(batch_gate->lod());
    to_seq(dev_ctx, *batch_hidden, hidden);
  }

  void Compute(const framework::ExecutionContext& context) const override {
    BatchCompute(context);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    gru, ops::GRUKernel<paddle::platform::CUDADeviceContext, float>,
    ops::GRUKernel<paddle::platform::CUDADeviceContext, double>);
REGISTER_OP_CUDA_KERNEL(
    gru_grad, ops::GRUGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::GRUGradKernel<paddle::platform::CUDADeviceContext, double>);
