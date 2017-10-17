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

using framework::LoDTensor;
using framework::Tensor;
template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenMatrix = framework::EigenMatrix<T, MajorType, IndexType>;

template <typename Place, typename T>
class LSTMKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<framework::LoDTensor>("Input");
    auto* weight = ctx.Input<framework::Tensor>("Weight");
    auto* bias = ctx.Input<framework::Tensor>("Bias");

    auto* batch_gate = ctx.Output<framework::LoDTensor>("BatchGate");
    batch_gate->mutable_data<T>(ctx.GetPlace());
    auto* hidden_out = ctx.Output<framework::LoDTensor>("Hidden");
    hidden_out->mutable_data<T>(ctx.GetPlace());
    auto* cell_out = ctx.Output<framework::LoDTensor>("Cell");
    cell_out->mutable_data<T>(ctx.GetPlace());

    // Now the function ShareLoD in InferShape is not implemented.
    // So copy LoD here.
    ctx.ShareLoD("Input", "Hidden");
    ctx.ShareLoD("Input", "Cell");

    bool is_reverse = ctx.Attr<bool>("isReverse");
    math::LoDTensor2BatchFunctor<Place, T> to_batch;
    to_batch(ctx.device_context(), *input, *batch_gate, is_reverse);

    auto in_dims = input->dims();
    int frame_size = in_dims[1];

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
    T* bias_data = const_cast<T*>(bias->data<T>());
    // the code styple in LstmMetaValue will be updated later.
    lstm_value.checkIg = bias_data + 4 * frame_size;
    lstm_value.checkFg = lstm_value.checkIg + frame_size;
    lstm_value.checkOg = lstm_value.checkFg + frame_size;
    lstm_value.prevStateValue = nullptr;

    framework::LoDTensor batch_out;
    batch_out.mutable_data<T>(in_dims, ctx.GetPlace());
    framework::LoDTensor batch_cell;
    batch_cell.mutable_data<T>(in_dims, ctx.GetPlace());
    framework::LoDTensor batch_cell_pre_act;
    batch_cell_pre_act.mutable_data<T>(in_dims, ctx.GetPlace());

    auto batch_lod = batch_gate->lod()[0];
    int num_batch = batch_lod.size() - 1;

    auto gate_act = ctx.Attr<std::string>("gateActivation");
    auto cell_act = ctx.Attr<std::string>("cellActivation");
    auto cand_act = ctx.Attr<std::string>("candidateActivation");

    for (int n = 0; n < num_batch; n++) {
      int bstart = batch_lod[n];
      int bend = batch_lod[n + 1];

      Tensor gate_t = batch_gate->Slice<T>(bstart, bend);
      Tensor out_t = batch_out.Slice<T>(bstart, bend);
      Tensor cell_t = batch_cell.Slice<T>(bstart, bend);
      Tensor cell_pre_act_t = batch_cell_pre_act.Slice<T>(bstart, bend);

      int cur_batch_size = bend - bstart;

      if (n != 0) {
        int pre_end = batch_lod[n - 1];
        auto pre_hidden_t = batch_out.Slice<T>(pre_end, bstart);
        math::matmul<Place, T>(ctx.device_context(), pre_hidden_t, false,
                               *weight, false, static_cast<T>(1.0), &gate_t,
                               static_cast<T>(0.0));
      }
      // else if : how to pass the state from
      // last mini-batch will be supported later

      lstm_value.gateValue = gate_t.data<T>();
      lstm_value.outputValue = out_t.data<T>();
      lstm_value.stateValue = cell_t.data<T>();
      lstm_value.stateActiveValue = cell_pre_act_t.data<T>();
      math::LstmUnitFunctor<Place, T>::compute(ctx.device_context(), lstm_value,
                                               frame_size, cur_batch_size,
                                               gate_act, cell_act, cand_act);
      lstm_value.prevStateValue = lstm_value.stateValue;
    }

    math::Batch2LoDTensorFunctor<Place, T> to_seq;
    batch_out.set_lod(batch_gate->lod());
    // restore the output hidden in LoDTensor from the batch hidden
    to_seq(ctx.device_context(), batch_out, *hidden_out);

    batch_out.set_lod(batch_gate->lod());
    // restore the output cell state in LoDTensor from the batch cell
    to_seq(ctx.device_context(), batch_cell, *cell_out);
  }
};

template <typename Place, typename T>
class LSTMGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {}
};

}  // namespace operators
}  // namespace paddle
