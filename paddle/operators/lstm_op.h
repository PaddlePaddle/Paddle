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
#include "paddle/operators/math/lstm_compute.h"
#include "paddle/operators/math/math_function.h"
#include "paddle/operators/math/sequence2batch.h"

namespace paddle {
namespace operators {

using LoDTensor = framework::LoDTensor;
using Tensor = framework::Tensor;

template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenMatrix = framework::EigenMatrix<T, MajorType, IndexType>;

template <typename Place, typename T>
class LSTMKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<LoDTensor>("Input");
    auto* weight = ctx.Input<Tensor>("Weight");
    auto* bias = ctx.Input<Tensor>("Bias");

    auto* batch_gate = ctx.Output<LoDTensor>("BatchGate");
    batch_gate->mutable_data<T>(ctx.GetPlace());
    auto* hidden_out = ctx.Output<LoDTensor>("Hidden");
    hidden_out->mutable_data<T>(ctx.GetPlace());
    auto* cell_out = ctx.Output<LoDTensor>("Cell");
    cell_out->mutable_data<T>(ctx.GetPlace());

    // Now the function ShareLoD in InferShape is not implemented.
    // So copy LoD here.
    ctx.ShareLoD("Input", "Hidden");
    ctx.ShareLoD("Input", "Cell");

    bool is_reverse = ctx.Attr<bool>("isReverse");
    math::LoDTensor2BatchFunctor<Place, T> to_batch;
    auto& device_ctx = ctx.device_context();
    to_batch(device_ctx, *input, *batch_gate, true, is_reverse);

    auto in_dims = input->dims();
    int frame_size = static_cast<int>(in_dims[1] / 4);
    framework::DDim dims({in_dims[0], frame_size});

    if (bias) {
      Eigen::array<int, 2> extents({{1, 4 * frame_size}});
      Eigen::array<int, 2> offsets({{0, 0}});
      auto b = EigenMatrix<T>::From(*bias);
      auto gate = EigenMatrix<T>::From(*batch_gate);
      gate.device(ctx.GetEigenDevice<Place>()) =
          gate +
          b.slice(offsets, extents)
              .reshape(Eigen::array<int, 2>({{1, frame_size * 4}}))
              .broadcast(
                  Eigen::array<int, 2>({{static_cast<int>(in_dims[0]), 1}}));
    }

    math::LstmMetaValue<T> lstm_value;
    if (bias) {
      T* bias_data = const_cast<T*>(bias->data<T>());
      // the code style in LstmMetaValue will be updated later.

      lstm_value.checkIg = bias_data + 4 * frame_size;
      lstm_value.checkFg = lstm_value.checkIg + frame_size;
      lstm_value.checkOg = lstm_value.checkFg + frame_size;
    } else {
      lstm_value.checkIg = nullptr;
      lstm_value.checkFg = nullptr;
      lstm_value.checkOg = nullptr;
    }
    lstm_value.prevStateValue = nullptr;

    // Use the local variable as here.
    LoDTensor batch_hidden, batch_cell;
    auto* batch_cell_pre_act = ctx.Output<LoDTensor>("BatchCellPreAct");
    batch_hidden.mutable_data<T>(dims, ctx.GetPlace());
    batch_cell.mutable_data<T>(dims, ctx.GetPlace());
    batch_cell_pre_act->mutable_data<T>(dims, ctx.GetPlace());

    auto batch_starts = batch_gate->lod()[0];
    size_t num_batch = batch_starts.size() - 1;
    auto gate_act = ctx.Attr<std::string>("gateActivation");
    auto cell_act = ctx.Attr<std::string>("cellActivation");
    auto cand_act = ctx.Attr<std::string>("candidateActivation");

    for (size_t n = 0; n < num_batch; n++) {
      int bstart = static_cast<int>(batch_starts[n]);
      int bend = static_cast<int>(batch_starts[n + 1]);

      Tensor gate_t = batch_gate->Slice(bstart, bend);
      Tensor out_t = batch_hidden.Slice(bstart, bend);
      Tensor cell_t = batch_cell.Slice(bstart, bend);
      Tensor cell_pre_act_t = batch_cell_pre_act->Slice(bstart, bend);

      int cur_batch_size = bend - bstart;

      if (n != 0) {
        int pre_h_start = static_cast<int>(batch_starts[n - 1]);
        int pre_h_end = pre_h_start + cur_batch_size;
        auto pre_hidden_t = batch_hidden.Slice(pre_h_start, pre_h_end);
        math::matmul<Place, T>(device_ctx, pre_hidden_t, false, *weight, false,
                               static_cast<T>(1.0), &gate_t,
                               static_cast<T>(1.0));
      }
      // else if : FIXME support the initial hidden and cell

      lstm_value.gateValue = gate_t.data<T>();
      lstm_value.outputValue = out_t.data<T>();
      lstm_value.stateValue = cell_t.data<T>();
      lstm_value.stateActiveValue = cell_pre_act_t.data<T>();
      math::LstmUnitFunctor<Place, T>::compute(device_ctx, lstm_value,
                                               frame_size, cur_batch_size,
                                               gate_act, cell_act, cand_act);
      lstm_value.prevStateValue = lstm_value.stateValue;
    }

    math::Batch2LoDTensorFunctor<Place, T> to_seq;
    batch_hidden.set_lod(batch_gate->lod());
    // restore the output hidden in LoDTensor from the batch hidden
    to_seq(device_ctx, batch_hidden, *hidden_out);

    batch_cell.set_lod(batch_gate->lod());
    // restore the output cell state in LoDTensor from the batch cell
    to_seq(device_ctx, batch_cell, *cell_out);
  }
};

template <typename Place, typename T>
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

    auto& device_ctx = ctx.device_context();
    math::SetConstant<Place, T> zero;
    if (weight_g) {
      weight_g->mutable_data<T>(ctx.GetPlace());
      zero(device_ctx, weight_g, static_cast<T>(0.0));
    }

    auto in_dims = input->dims();
    auto out_dims = hidden_g->dims();
    int frame_size = static_cast<int>(in_dims[1] / 4);
    PADDLE_ENFORCE_EQ(frame_size, out_dims[1]);

    math::LstmMetaValue<T> lstm_value;
    if (bias) {
      T* bias_data = const_cast<T*>(bias->data<T>());
      lstm_value.checkIg = bias_data + 4 * frame_size;
      lstm_value.checkFg = lstm_value.checkIg + frame_size;
      lstm_value.checkOg = lstm_value.checkFg + frame_size;
    } else {
      lstm_value.checkIg = nullptr;
      lstm_value.checkFg = nullptr;
      lstm_value.checkOg = nullptr;
    }

    math::LstmMetaGrad<T> lstm_grad;
    if (bias && bias_g) {
      T* bias_g_data = const_cast<T*>(bias_g->mutable_data<T>(ctx.GetPlace()));
      zero(device_ctx, bias_g, static_cast<T>(0.0));
      lstm_grad.checkIgGrad = bias_g_data + 4 * frame_size;
      lstm_grad.checkFgGrad = lstm_grad.checkIgGrad + frame_size;
      lstm_grad.checkOgGrad = lstm_grad.checkFgGrad + frame_size;
    } else {
      lstm_grad.checkIgGrad = nullptr;
      lstm_grad.checkFgGrad = nullptr;
      lstm_grad.checkOgGrad = nullptr;
    }

    math::LoDTensor2BatchFunctor<Place, T> to_batch;

    // use the local variable as here.
    LoDTensor batch_hidden;
    batch_hidden.mutable_data<T>(out_dims, ctx.GetPlace());
    batch_hidden.set_lod(batch_gate->lod());
    to_batch(device_ctx, *hidden_out, batch_hidden, false);

    LoDTensor batch_hidden_g;
    batch_hidden_g.mutable_data<T>(out_dims, ctx.GetPlace());
    batch_hidden_g.set_lod(batch_gate->lod());
    to_batch(device_ctx, *hidden_g, batch_hidden_g, false);

    LoDTensor batch_cell;
    batch_cell.mutable_data<T>(out_dims, ctx.GetPlace());
    batch_cell.set_lod(batch_gate->lod());
    to_batch(device_ctx, *cell_out, batch_cell, false);

    LoDTensor batch_cell_g;
    batch_cell_g.mutable_data<T>(out_dims, ctx.GetPlace());
    batch_cell_g.set_lod(batch_gate->lod());
    // TODO(qingqing) support the case output cell has gradient.
    // to_batch(device_ctx, *cell_g, batch_cell_g, false);
    zero(device_ctx, &batch_cell_g, static_cast<T>(0.0));

    LoDTensor batch_gate_g;
    batch_gate_g.mutable_data<T>(batch_gate->dims(), ctx.GetPlace());
    batch_gate_g.set_lod(batch_gate->lod());

    auto gate_act = ctx.Attr<std::string>("gateActivation");
    auto cell_act = ctx.Attr<std::string>("cellActivation");
    auto cand_act = ctx.Attr<std::string>("candidateActivation");

    auto batch_starts = batch_gate->lod()[0];
    size_t num_batch = batch_starts.size() - 1;
    for (int n = static_cast<int>(num_batch) - 1; n >= 0; n--) {
      int bstart = static_cast<int>(batch_starts[n]);
      int bend = static_cast<int>(batch_starts[n + 1]);

      Tensor gate = batch_gate->Slice(bstart, bend);
      Tensor cell = batch_cell.Slice(bstart, bend);
      Tensor cell_pre_act = batch_cell_pre_act->Slice(bstart, bend);
      lstm_value.gateValue = gate.data<T>();
      lstm_value.stateValue = cell.data<T>();
      lstm_value.stateActiveValue = cell_pre_act.data<T>();

      Tensor out_g = batch_hidden_g.Slice(bstart, bend);
      Tensor gate_g = batch_gate_g.Slice(bstart, bend);
      Tensor cell_g = batch_cell_g.Slice(bstart, bend);
      lstm_grad.stateGrad = cell_g.data<T>();
      lstm_grad.gateGrad = gate_g.data<T>();
      lstm_grad.outputGrad = out_g.data<T>();

      if (n) {
        int bstart_pre = static_cast<int>(batch_starts[n - 1]);
        Tensor cell_pre = batch_cell.Slice(bstart_pre, bstart);
        Tensor cell_pre_g = batch_cell_g.Slice(bstart_pre, bstart);
        lstm_value.prevStateValue = cell_pre.data<T>();
        lstm_grad.prevStateGrad = cell_pre_g.data<T>();
      } else {
        lstm_value.prevStateValue = nullptr;
        lstm_grad.prevStateGrad = nullptr;
      }

      int cur_batch_size = bend - bstart;
      math::LstmUnitGradFunctor<Place, T>::compute(
          device_ctx, lstm_value, lstm_grad, frame_size, cur_batch_size,
          gate_act, cell_act, cand_act);

      if (n != 0) {
        int pre_h_start = static_cast<int>(batch_starts[n - 1]);
        int pre_h_end = pre_h_start + cur_batch_size;
        auto pre_hidden_g = batch_hidden_g.Slice(pre_h_start, pre_h_end);
        math::matmul<Place, T>(device_ctx, gate_g, false, *weight, true,
                               static_cast<T>(1.0), &pre_hidden_g,
                               static_cast<T>(1.0));
        if (weight_g) {
          /* backward weight */
          auto pre_hidden = batch_hidden.Slice(pre_h_start, pre_h_end);
          math::matmul<Place, T>(device_ctx, pre_hidden, true, gate_g, false,
                                 static_cast<T>(1.0), weight_g,
                                 static_cast<T>(1.0));
        }
      }
    }

    math::Batch2LoDTensorFunctor<Place, T> to_seq;
    if (in_g) {
      /* backward data */
      in_g->mutable_data<T>(ctx.GetPlace());
      to_seq(device_ctx, batch_gate_g, *in_g);
    }
    if (bias && bias_g) {
      /* backward bias */
      int m = static_cast<int>(batch_gate_g.dims()[0]);
      int n = static_cast<int>(batch_gate_g.dims()[1]);

      Tensor ones;
      ones.mutable_data<T>({m}, ctx.GetPlace());
      math::SetConstant<Place, T> set;
      set(device_ctx, &ones, static_cast<T>(1.0));

      math::gemv<Place, T>(device_ctx, true, m, n, 1., batch_gate_g.data<T>(),
                           ones.data<T>(), 0., bias_g->data<T>());
    }
  }
};

}  // namespace operators
}  // namespace paddle
