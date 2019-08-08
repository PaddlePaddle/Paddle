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
#include "paddle/fluid/lite/operators/fusion_gru_op.h"
#include "paddle/fluid/operators/jit/kernels.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/math/fc_compute.h"
#include "paddle/fluid/operators/math/sequence2batch.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

template <typename T>
class FusionGruCompute : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  using param_t = operators::FusionGruParam;
  auto& ctx = context_ -> As<X86Context>();

#define INIT_BASE_DEFINES                \
  auto* x = param.x;                     \
  auto* wh = param.weightH;              \
  auto* xx = param.XX;                   \
  auto x_lod = x->lod();                 \
  auto x_dims = x->dims();   /* T x M*/  \
  auto wh_dims = wh->dims(); /* D x 3D*/ \
  const int total_T = x_dims[0];         \
  const int D3 = wh_dims[1]

#define INIT_OTHER_DEFINES                                                   \
  auto* h0 = param.h0;                                                       \
  auto* wx = param.weightX;                                                  \
  auto* bias = param.bias;                                                   \
  auto* hidden_out = param.hidden;                                           \
  bool is_reverse = param.is_reverse;                                        \
  const int M = x_dims[1];                                                   \
  const int D = wh_dims[0];                                                  \
  const int D2 = D * 2;                                                      \
  const jit::gru_attr_t attr(D, jit::to_kerneltype(param.gate_activation),   \
                             jit::to_kerneltype(param.activation));          \
  jit::gru_t one_step;                                                       \
  auto ComputeH1 =                                                           \
      jit::KernelFuncs<jit::GRUH1Tuple<T>, platform::CPUPlace>::Cache().At(  \
          attr);                                                             \
  auto ComputeHtPart1 =                                                      \
      jit::KernelFuncs<jit::GRUHtPart1Tuple<T>, platform::CPUPlace>::Cache() \
          .At(attr);                                                         \
  auto ComputeHtPart2 =                                                      \
      jit::KernelFuncs<jit::GRUHtPart2Tuple<T>, platform::CPUPlace>::Cache() \
          .At(attr);                                                         \
  const T* x_data = x->data<T>();                                            \
  const T* wx_data = wx->data<T>();                                          \
  const T* wh_data = wh->data<T>();                                          \
  auto place = platform::CPUPlace();                                         \
  T* xx_data = xx->mutable_data<T>(place)

  void SeqCompute(const operators::FusionGruParam& param) const {
    using DeviceContext = paddle::platform::CPUDeviceContext;
    INIT_BASE_DEFINES;
    INIT_OTHER_DEFINES;
    const int N = x_lod[0].size() - 1;
    const T* h0_data = h0 ? h0->data<T>() : nullptr;
    const T* wh_state_data = wh_data + D * D2;
    T* hidden_out_data = hidden_out->mutable_data<T>(place);
    auto blas = math::GetBlas<DeviceContext, T>(ctx);
    math::FCCompute<DeviceContext, T>(blas, total_T, D3, M, x_data, wx_data,
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
      int seq_len = x_lod[0][bid + 1] - x_lod[0][bid];
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
        blas.GEMM(CblasNoTrans, CblasNoTrans, 1, D2, D, static_cast<T>(1),
                  prev_hidden_data, D, wh_data, D2, static_cast<T>(1), xx_data,
                  D3);
        one_step.gates = xx_data;
        one_step.ht_1 = prev_hidden_data;
        one_step.ht = hidden_out_data;
        ComputeHtPart1(&one_step, &attr);
        // gemm rt * Ws
        blas.GEMM(CblasNoTrans, CblasNoTrans, 1, D, D, static_cast<T>(1),
                  hidden_out_data, D, wh_state_data, D, static_cast<T>(1),
                  xx_data + D2, D3);
        ComputeHtPart2(&one_step, &attr);
        // save prev
        prev_hidden_data = hidden_out_data;
        move_step();
      }
    }
  }

  void BatchCompute(const operators::FusionGruParam& param) const {
    using DeviceContext = paddle::platform::CPUDeviceContext;
    INIT_BASE_DEFINES;
    if (x_lod[0].size() == 2) {
      xx->Resize({total_T, D3});
      SeqCompute(param);
      return;
    }
    INIT_OTHER_DEFINES;
    auto* reordered_h0 = param.reorderedH0;
    auto* batched_input = param.batchedInput;
    auto* batched_out = param.batchedOut;
    T* batched_input_data = batched_input->mutable_data<T>(place);
    T* batched_out_data = batched_out->mutable_data<T>(place);
    hidden_out->mutable_data<T>(place);
    auto& dev_ctx = paddle::platform::CPUDeviceContext();
    auto blas = math::GetBlas<DeviceContext, T>(dev_ctx);
    math::LoDTensor2BatchFunctor<DeviceContext, T> to_batch;
    if (M > D3) {
      math::FCCompute<DeviceContext, T>(blas, total_T, D3, M, x_data, wx_data,
                                        xx_data,
                                        bias ? bias->data<T>() : nullptr);
      to_batch(dev_ctx, xx->raw_tensor(), &batched_input->raw_tensor(), true,
               is_reverse);
    } else {
      to_batch(dev_ctx, x->raw_tensor, &xx->raw_tensor(), true, is_reverse);
      batched_input->set_lod(xx->lod());
      math::FCCompute<DeviceContext, T>(blas, total_T, D3, M, xx_data, wx_data,
                                        batched_input_data,
                                        bias ? bias->data<T>() : nullptr);
    }

    auto batched_lod = batched_input->lod();
    const auto& seq_order = batched_lod[2];
    const int max_bs = seq_order.size();
    std::vector<int64_t> h0_shape{max_bs, D};
    reordered_h0->Resize(lite::DDim(h0_shape));

    int tstart = 0;
    T* prev_hidden_data = nullptr;
    if (h0) {
      // reorder h0
      T* reordered_h0_data = reordered_h0->mutable_data<T>(place);
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
    const int max_seq_len = batch_starts.size() - 1;
    batched_input_data = batched_input_data + tstart * max_bs * D3;
    batched_out_data = batched_out_data + tstart * max_bs * D;
    for (int step = tstart; step < max_seq_len; ++step) {
      const int cur_bs = batch_starts[step + 1] - batch_starts[step];
      // gemm prev * (Wu + Wr)
      blas.GEMM(CblasNoTrans, CblasNoTrans, cur_bs, D2, D, static_cast<T>(1),
                prev_hidden_data, D, wh_data, D2, static_cast<T>(1),
                batched_input_data, D3);

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
      blas.GEMM(CblasNoTrans, CblasNoTrans, cur_bs, D, D, static_cast<T>(1),
                cur_out_data, D, wh_state_data, D, static_cast<T>(1),
                cur_batched_data + D2, D3);

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

    math::Batch2LoDTensorFunctor<DeviceContext, T> to_seq;
    batched_out->raw_tensor().set_lod(batched_lod);
    to_seq(dev_ctx, batched_out->raw_tensor(), &hidden_out->raw_tensor());
  }
#undef INIT_OTHER_DEFINES
#undef INIT_BASE_DEFINES

  void Run() override {
    auto& param = *param_.get_mutable<operators::FusionGruParam>();
    if (param.use_seq) {
      SeqCompute(param);
    } else {
      BatchCompute(param);
    }
  }

  virtual ~FusionGruCompute() = default;
};

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
