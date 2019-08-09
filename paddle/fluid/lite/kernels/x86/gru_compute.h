// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include <vector>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/lite/core/kernel.h"
#include "paddle/fluid/lite/core/op_registry.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/math/detail/activation_functions.h"
#include "paddle/fluid/operators/math/detail/gru_cpu_kernel.h"
#include "paddle/fluid/operators/math/detail/gru_kernel.h"
#include "paddle/fluid/operators/math/gru_compute.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/math/sequence2batch.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

template <typename DeviceContext, typename T>
inline void ReorderInitState(const lite::Tensor& src,
                             framework::Vector<size_t> index_lod,
                             lite::Tensor* dst, bool indexed_src) {
  paddle::operators::math::CopyMatrixRowsFunctor<platform::CPUDeviceContext, T>
      row_shuffle;
  dst->Resize(src.dims());
  dst->mutable_data<T>();
  row_shuffle(platform::CPUDeviceContext(), src.raw_tensor(), index_lod,
              &dst->raw_tensor(), indexed_src);
}

template <typename T>
class GruCompute : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  using param_t = operators::GruParam;

  void Run() override {
    auto& param = *param_.get_mutable<operators::GruParam>();
    // auto& context = context_->As<X86Context>();
    bool origin_mode = param.origin_mode;
    auto* input = param.x;
    auto* h0 = param.h0;
    auto weight = param.weight;
    const T* weight_data = weight->data<T>();
    auto* bias = param.bias;
    auto* batch_gate = param.batchGate;
    batch_gate->mutable_data<T>();
    auto* batch_reset_hidden_prev = param.batchResetHiddenPrev;
    batch_reset_hidden_prev->mutable_data<T>();
    auto* batch_hidden = param.batchHidden;
    batch_hidden->mutable_data<T>();
    auto* hidden = param.hidden;
    hidden->mutable_data<T>();
    auto hidden_dims = hidden->dims();
    bool is_reverse = param.is_reverse;
    paddle::operators::math::LoDTensor2BatchFunctor<platform::CPUDeviceContext,
                                                    T>
        to_batch;
    to_batch(platform::CPUDeviceContext(), input->raw_tensor(),
             &batch_gate->raw_tensor(), true, is_reverse);

    if (bias) {
      paddle::operators::math::RowwiseAdd<platform::CPUDeviceContext, T>
          add_bias;
      add_bias(platform::CPUDeviceContext(), batch_gate->raw_tensor(),
               bias->raw_tensor(), &batch_gate->raw_tensor());
    }

    int frame_size = hidden_dims[1];
    paddle::operators::math::GRUMetaValue<T> gru_value;
    gru_value.gate_weight = const_cast<T*>(weight_data);
    gru_value.state_weight =
        const_cast<T*>(weight_data + 2 * frame_size * frame_size);
    lite::Tensor ordered_h0;

    framework::Vector<size_t> order(batch_gate->raw_tensor().lod()[2]);

    if (h0) {
      // Since the batch computing for GRU reorders the input sequences
      // according to their length. The initialized cell state also needs
      // to reorder.
      ReorderInitState<platform::CPUDeviceContext, T>(*h0, order, &ordered_h0,
                                                      true);
      gru_value.prev_out_value = const_cast<T*>(ordered_h0.data<T>());
    } else {
      gru_value.prev_out_value = nullptr;
    }
    auto batch_starts = batch_gate->raw_tensor().lod()[0];
    size_t seq_len = batch_starts.size() - 1;
    auto active_node =
        paddle::operators::math::detail::GetActivationType(param.activation);
    auto active_gate = paddle::operators::math::detail::GetActivationType(
        param.gate_activation);

    for (size_t n = 0; n < seq_len; n++) {
      int bstart = static_cast<int>(batch_starts[n]);
      int bend = static_cast<int>(batch_starts[n + 1]);
      int cur_batch_size = bend - bstart;

      lite::Tensor gate_t;
      gate_t.ShareDataWith(batch_gate->raw_tensor().Slice(bstart, bend));
      lite::Tensor reset_hidden_prev_t;
      reset_hidden_prev_t.ShareDataWith(
          batch_reset_hidden_prev->raw_tensor().Slice(bstart, bend));
      Tensor hidden_t;
      hidden_t.ShareDataWith(batch_hidden->raw_tensor().Slice(bstart, bend));
      gru_value.output_value = const_cast<T*>(hidden_t.data<T>());
      gru_value.gate_value = const_cast<T*>(gate_t.data<T>());
      gru_value.reset_output_value =
          const_cast<T*>(reset_hidden_prev_t.data<T>());

      paddle::operators::math::GRUUnitFunctor<
          platform::CPUDeviceContext, T>::compute(platform::CPUDeviceContext(),
                                                  gru_value, frame_size,
                                                  cur_batch_size, active_node,
                                                  active_gate, origin_mode);

      gru_value.prev_out_value = gru_value.output_value;
    }

    paddle::operators::math::Batch2LoDTensorFunctor<platform::CPUDeviceContext,
                                                    T>
        to_seq;
    batch_hidden->raw_tensor().set_lod(batch_gate->raw_tensor().lod());
    to_seq(platform::CPUDeviceContext(), batch_hidden->raw_tensor(),
           &hidden->raw_tensor());
  }

  virtual ~GruCompute() = default;
};

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
