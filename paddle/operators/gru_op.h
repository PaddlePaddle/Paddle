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

#include "paddle/operators/math/gru_compute.h"
#include "paddle/operators/math/math_function.h"
#include "paddle/operators/math/sequence2batch.h"

#include "paddle/framework/eigen.h"
#include "paddle/framework/op_registry.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;

template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenMatrix = framework::EigenMatrix<T, MajorType, IndexType>;

template <typename Place, typename T>
class GRUKernel : public framework::OpKernel<T> {
 public:
  void BatchCompute(const framework::ExecutionContext& context) const {
    auto* input = context.Input<LoDTensor>("Input");
    auto* h0 = context.Input<Tensor>("H0");
    const T* h0_data = h0 ? h0->data<T>() : nullptr;
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
    math::LoDTensor2BatchFunctor<Place, T> to_batch;
    to_batch(context.device_context(), *input, *batch_gate, true, is_reverse);

    int frame_size = hidden_dims[1];
    int batch_size = hidden_dims[0];
    auto g = EigenMatrix<T>::From(*batch_gate);
    auto place = context.GetEigenDevice<Place>();
    if (bias) {
      auto b = EigenMatrix<T>::From(*bias);
      g.device(place) = g +
                        b.reshape(Eigen::array<int, 2>({{1, frame_size * 3}}))
                            .broadcast(Eigen::array<int, 2>({{batch_size, 1}}));
    }

    math::hl_gru_value<T> gru_value;
    gru_value.gateWeight = const_cast<T*>(weight_data);
    gru_value.stateWeight =
        const_cast<T*>(weight_data + 2 * frame_size * frame_size);
    gru_value.prevOutValue = const_cast<T*>(h0_data);
    auto batch_starts = batch_gate->lod()[0];
    size_t num_batch = batch_starts.size() - 1;
    for (size_t n = 0; n < num_batch; n++) {
      int bstart = static_cast<int>(batch_starts[n]);
      int bend = static_cast<int>(batch_starts[n + 1]);
      int cur_batch_size = bend - bstart;

      Tensor gate_t = batch_gate->Slice(bstart, bend);
      Tensor reset_hidden_prev_t = batch_reset_hidden_prev->Slice(bstart, bend);
      Tensor hidden_t = batch_hidden->Slice(bstart, bend);
      gru_value.outputValue = hidden_t.data<T>();
      gru_value.gateValue = gate_t.data<T>();
      gru_value.resetOutputValue = reset_hidden_prev_t.data<T>();
      math::GRUUnitFunctor<Place, T>::compute(
          context.device_context(), gru_value, frame_size, cur_batch_size,
          math::ActiveType(context.Attr<std::string>("activation")),
          math::ActiveType(context.Attr<std::string>("gate_activation")));
      gru_value.prevOutValue = gru_value.outputValue;
    }

    math::Batch2LoDTensorFunctor<Place, T> to_seq;
    batch_hidden->set_lod(batch_gate->lod());
    to_seq(context.device_context(), *batch_hidden, *hidden);
  }

  void Compute(const framework::ExecutionContext& context) const override {
    BatchCompute(context);
  }
};

template <typename Place, typename T>
class GRUGradKernel : public framework::OpKernel<T> {
 public:
  void BatchCompute(const framework::ExecutionContext& context) const {
    auto* h0 = context.Input<Tensor>("H0");
    const T* h0_data = h0 ? h0->data<T>() : nullptr;
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

    math::LoDTensor2BatchFunctor<Place, T> to_batch;
    LoDTensor batch_hidden_grad, batch_gate_grad, batch_reset_hidden_prev_grad;
    batch_hidden_grad.mutable_data<T>(hidden_dims, context.GetPlace());
    batch_gate_grad.mutable_data<T>(gate_dims, context.GetPlace());
    batch_reset_hidden_prev_grad.mutable_data<T>(hidden_dims,
                                                 context.GetPlace());
    math::SetConstant<Place, T> zero;
    zero(context.device_context(), &batch_hidden_grad, static_cast<T>(0.0));
    zero(context.device_context(), &batch_gate_grad, static_cast<T>(0.0));
    zero(context.device_context(), &batch_reset_hidden_prev_grad,
         static_cast<T>(0.0));

    bool is_reverse = context.Attr<bool>("is_reverse");
    batch_hidden_grad.set_lod(batch_hidden->lod());
    to_batch(context.device_context(), *hidden_grad, batch_hidden_grad, false,
             is_reverse);

    math::hl_gru_value<T> gru_value;
    gru_value.gateWeight = const_cast<T*>(weight_data);
    gru_value.stateWeight =
        const_cast<T*>(weight_data + 2 * frame_size * frame_size);

    math::hl_gru_grad<T> gru_grad;
    if (weight_grad) {
      gru_grad.gateWeightGrad =
          weight_grad->mutable_data<T>(context.GetPlace());
      zero(context.device_context(), weight_grad, static_cast<T>(0.0));
      gru_grad.stateWeightGrad =
          weight_grad->data<T>() + 2 * frame_size * frame_size;
    } else {
      gru_grad.gateWeightGrad = nullptr;
      gru_grad.stateWeightGrad = nullptr;
    }

    auto batch_starts = batch_hidden_grad.lod()[0];
    size_t num_batch = batch_starts.size() - 1;
    for (int n = static_cast<int>(num_batch) - 1; n >= 0; n--) {
      int bstart = static_cast<int>(batch_starts[n]);
      int bend = static_cast<int>(batch_starts[n + 1]);
      int cur_batch_size = bend - bstart;

      Tensor gate_t = batch_gate->Slice(bstart, bend);
      gru_value.gateValue = gate_t.data<T>();
      Tensor reset_hidden_prev_t = batch_reset_hidden_prev->Slice(bstart, bend);
      gru_value.resetOutputValue = reset_hidden_prev_t.data<T>();

      Tensor hidden_grad_t = batch_hidden_grad.Slice(bstart, bend);
      gru_grad.outputGrad = hidden_grad_t.data<T>();
      Tensor gate_grad_t = batch_gate_grad.Slice(bstart, bend);
      gru_grad.gateGrad = gate_grad_t.data<T>();
      Tensor reset_hidden_prev_grad_t =
          batch_reset_hidden_prev_grad.Slice(bstart, bend);
      gru_grad.resetOutputGrad = reset_hidden_prev_grad_t.data<T>();
      if (n == 0) {
        gru_value.prevOutValue = const_cast<T*>(h0_data);
        if (h0_grad) {
          T* h0_grad_data = h0_grad->mutable_data<T>(context.GetPlace());
          zero(context.device_context(), h0_grad, static_cast<T>(0.0));
          gru_grad.prevOutGrad = h0_grad_data;
        } else {
          gru_grad.prevOutGrad = nullptr;
        }
      } else {
        int bstart_pre = static_cast<int>(batch_starts[n - 1]);
        Tensor hidden_prev_t = batch_hidden->Slice(bstart_pre, bstart);
        gru_value.prevOutValue = hidden_prev_t.data<T>();
        Tensor hidden_prev_grad_t = batch_hidden_grad.Slice(bstart_pre, bstart);
        gru_grad.prevOutGrad = hidden_prev_grad_t.data<T>();
      }

      math::GRUUnitGradFunctor<Place, T>::compute(
          context.device_context(), gru_value, gru_grad, frame_size,
          cur_batch_size,
          math::ActiveType(context.Attr<std::string>("activation")),
          math::ActiveType(context.Attr<std::string>("gate_activation")));
    }
    if (input_grad) {
      input_grad->mutable_data<T>(context.GetPlace());
      math::Batch2LoDTensorFunctor<Place, T> to_seq;
      batch_gate_grad.set_lod(batch_gate->lod());
      to_seq(context.device_context(), batch_gate_grad, *input_grad);
    }
    if (bias_grad) {
      bias_grad->mutable_data<T>(context.GetPlace());
      auto d_b = EigenMatrix<T>::From(*bias_grad);
      auto d_g = EigenMatrix<T>::From(batch_gate_grad);
      auto place = context.GetEigenDevice<Place>();
      d_b.device(place) = d_g.sum(Eigen::array<int, 1>({{0}}));
    }
  }

  void Compute(const framework::ExecutionContext& context) const override {
    BatchCompute(context);
  }
};

}  // namespace operators
}  // namespace paddle
