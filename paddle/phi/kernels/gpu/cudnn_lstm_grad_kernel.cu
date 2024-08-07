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

#include "paddle/phi/kernels/cudnn_lstm_grad_kernel.h"

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/gpu/cudnn_lstm_utils.h"

namespace phi {

template <typename T, typename Context>
void CudnnLSTMGradKernel(
    const Context &ctx,
    const DenseTensor &x,
    const DenseTensor &init_h,
    const DenseTensor &init_c,
    const paddle::optional<std::vector<const DenseTensor *>> &weight_list,
    const paddle::optional<DenseTensor> &sequence_length,
    const DenseTensor &out,
    const DenseTensor &reserve,
    const DenseTensor &state_out,
    const DenseTensor &out_grad,
    const DenseTensor &last_h_grad,
    const DenseTensor &last_c_grad,
    float dropout_prob,
    bool is_bidirec,
    int hidden_size,
    int num_layers,
    bool is_test,
    int seed,
    DenseTensor *x_grad,
    DenseTensor *init_h_grad,
    DenseTensor *init_c_grad,
    std::vector<DenseTensor *> weight_grad_list) {
  auto input_dims = x.dims();
  auto init_h_dims = init_h.dims();
  auto init_c_dims = init_c.dims();

  auto *init_h_data = init_h.data<T>();
  auto *init_c_data = init_c.data<T>();
  auto *out_data = out.data<T>();
  auto *out_grad_data = out_grad.data<T>();
  auto *last_h_grad_data = last_h_grad.data<T>();
  auto *last_c_grad_data = last_c_grad.data<T>();

  auto running_weight_list = *weight_list.get_ptr();
  int weight_numel = size_sum(running_weight_list);
  bool continuous = is_continuous<T, std::vector<const phi::DenseTensor *>>(
      running_weight_list);

  auto handle = ctx.cudnn_handle();
  auto place = ctx.GetPlace();
  auto stream = ctx.stream();
  phi::DenseTensor weight_whole;
  T *weight_data = nullptr;

  if (!continuous) {
    weight_whole.Resize({weight_numel});
    ctx.template Alloc<T>(&weight_whole);
    weight_to_tensor<T>(place, stream, running_weight_list, &weight_whole);
    weight_data = weight_whole.data<T>();
  } else {
    weight_data = const_cast<T *>(running_weight_list[0]->data<T>());
  }

  phi::DenseTensor weight_grad;
  phi::funcs::SetConstant<phi::GPUContext, T> zero;
  weight_grad.Resize({weight_numel});
  ctx.template Alloc<T>(&weight_grad);
  zero(ctx, &weight_grad, static_cast<T>(0.0));
  T *weight_grad_data = weight_grad.data<T>();

  int offset = 0;
  for (size_t i = 0; i < weight_grad_list.size(); ++i) {
    size_t len = weight_grad_list[i]->numel();
    auto dim = weight_grad_list[i]->dims();
    weight_grad_list[i]
        ->ShareDataWith(weight_grad.Slice(static_cast<int64_t>(offset),
                                          static_cast<int64_t>(offset + len)))
        .Resize(dim);
    offset += len;
  }

  x_grad->Resize(input_dims);
  ctx.template Alloc<T>(x_grad);
  auto *in_grad_data = x_grad->data<T>();

  if (init_h_grad) {
    init_h_grad->Resize(init_h_dims);
    ctx.template Alloc<T>(init_h_grad);
  }
  auto *init_h_grad_data = init_h_grad ? init_h_grad->data<T>() : nullptr;

  if (init_c_grad) {
    init_c_grad->Resize(init_c_dims);
    ctx.template Alloc<T>(init_c_grad);
  }
  auto *init_c_grad_data = init_c_grad ? init_c_grad->data<T>() : nullptr;

  auto running_seq_length = sequence_length.get_ptr();
  bool has_seq_length = running_seq_length != nullptr;
  std::vector<int> SequenceLength;
  if (has_seq_length) {
    SequenceLength = phi::GetVectorFromTensor<int>(running_seq_length);
  }

  int seq_length = input_dims[0];
  int batch_size = x.dims()[1];
  int input_size = x.dims()[2];

  size_t workspace_size;
  size_t reserve_size;

  ScopedRNNBase rnn(seq_length,
                    batch_size,
                    input_size,
                    hidden_size,
                    num_layers,
                    dropout_prob,
                    seed,
                    weight_numel,
                    true,
                    is_bidirec);

  rnn.Create<T>(handle,
                ctx.GetPlace(),
                SequenceLength,
                &workspace_size,
                &reserve_size,
                const_cast<phi::DenseTensor *>(&state_out));

  phi::DenseTensor workspace_data_;
  workspace_data_.Resize({static_cast<int64_t>(workspace_size)});
  ctx.template Alloc<uint8_t>(&workspace_data_);
  const uint8_t *reserve_data = reserve.data<uint8_t>();

#if CUDNN_VERSION >= 90000
  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnRNNBackwardData_v8(
      handle,
      rnn.rnn_desc(),
      nullptr,
      rnn.y_seq_desc(),
      out_data,
      out_grad_data,
      rnn.x_seq_desc(),
      in_grad_data,
      rnn.init_h_desc(),
      init_h_data,
      last_h_grad_data,
      init_h_grad_data,
      rnn.init_c_desc(),
      init_c_data,
      last_c_grad_data,
      init_c_grad_data,
      rnn.weights_size(),
      weight_data,
      workspace_size,
      workspace_data_.data<uint8_t>(),
      reserve_size,
      const_cast<uint8_t *>(reserve_data)));

  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnRNNBackwardWeights_v8(
      handle,
      rnn.rnn_desc(),
      CUDNN_WGRAD_MODE_ADD,
      nullptr,
      rnn.x_seq_desc(),
      x.data<T>(),
      rnn.init_h_desc(),
      init_h.data<T>(),
      rnn.y_seq_desc(),
      out.data<T>(),
      rnn.weights_size(),
      weight_grad_data,
      workspace_size,
      workspace_data_.data<uint8_t>(),
      reserve_size,
      const_cast<uint8_t *>(reserve_data)));
#else

  if (!has_seq_length) {
// This interface is used when the input/output is unpadded.
#ifdef PADDLE_WITH_HIP
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::miopenRNNBackwardData(handle,
                                            rnn.rnn_desc(),
                                            seq_length,
                                            rnn.y_descs(),
                                            out_data,
                                            rnn.y_descs(),
                                            out_grad_data,
                                            rnn.last_h_desc(),
                                            last_h_grad_data,
                                            rnn.last_c_desc(),
                                            last_c_grad_data,
                                            rnn.weight_desc(),
                                            weight_data,
                                            rnn.init_h_desc(),
                                            init_h_data,
                                            rnn.init_c_desc(),
                                            init_c_data,
                                            rnn.x_descs(),
                                            in_grad_data,
                                            rnn.init_h_desc(),
                                            init_h_grad_data,
                                            rnn.init_c_desc(),
                                            init_c_grad_data,
                                            workspace_data_.data<uint8_t>(),
                                            workspace_size,
                                            const_cast<uint8_t *>(reserve_data),
                                            reserve_size));

    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::miopenRNNBackwardWeights(
        handle,
        rnn.rnn_desc(),
        seq_length,
        rnn.x_descs(),
        x.data<T>(),
        rnn.init_h_desc(),
        init_h.data<T>(),
        rnn.y_descs(),
        out.data<T>(),
        rnn.weight_desc(),
        weight_grad_data,
        workspace_data_.data<uint8_t>(),
        workspace_size,
        const_cast<uint8_t *>(reserve_data),
        reserve_size));
#else
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::cudnnRNNBackwardData(handle,
                                           rnn.rnn_desc(),
                                           seq_length,
                                           rnn.y_descs(),
                                           out_data,
                                           rnn.y_descs(),
                                           out_grad_data,
                                           rnn.last_h_desc(),
                                           last_h_grad_data,
                                           rnn.last_c_desc(),
                                           last_c_grad_data,
                                           rnn.weight_desc(),
                                           weight_data,
                                           rnn.init_h_desc(),
                                           init_h_data,
                                           rnn.init_c_desc(),
                                           init_c_data,
                                           rnn.x_descs(),
                                           in_grad_data,
                                           rnn.init_h_desc(),
                                           init_h_grad_data,
                                           rnn.init_c_desc(),
                                           init_c_grad_data,
                                           workspace_data_.data<uint8_t>(),
                                           workspace_size,
                                           const_cast<uint8_t *>(reserve_data),
                                           reserve_size));

    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnRNNBackwardWeights(
        handle,
        rnn.rnn_desc(),
        seq_length,
        rnn.x_descs(),
        x.data<T>(),
        rnn.init_h_desc(),
        init_h.data<T>(),
        rnn.y_descs(),
        out.data<T>(),
        workspace_data_.data<uint8_t>(),
        workspace_size,
        rnn.weight_desc(),
        weight_grad_data,
        const_cast<uint8_t *>(reserve_data),
        reserve_size));
#endif
  } else {
#if !defined(PADDLE_WITH_HIP) && CUDNN_VERSION >= 7201
    // for train
    // This interface is used when the input/output is padded.
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnRNNBackwardDataEx(
        handle,
        rnn.rnn_desc(),
        rnn.y_seq_desc(),
        out_data,
        rnn.y_seq_desc(),
        out_grad_data,
        nullptr,
        nullptr,
        rnn.last_h_desc(),
        last_h_grad_data,
        rnn.last_c_desc(),
        last_c_grad_data,
        rnn.weight_desc(),
        weight_data,
        rnn.init_h_desc(),
        init_h_data,
        rnn.init_c_desc(),
        init_c_data,
        rnn.x_seq_desc(),
        in_grad_data,
        rnn.init_h_desc(),
        init_h_grad_data,
        rnn.init_c_desc(),
        init_c_grad_data,
        nullptr,
        nullptr,
        workspace_data_.data<uint8_t>(),
        workspace_size,
        const_cast<uint8_t *>(reserve_data),
        reserve_size));

    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnRNNBackwardWeightsEx(
        handle,
        rnn.rnn_desc(),
        rnn.x_seq_desc(),
        x.data<T>(),
        rnn.init_h_desc(),
        init_h.data<T>(),
        rnn.y_seq_desc(),
        out.data<T>(),
        workspace_data_.data<uint8_t>(),
        workspace_size,
        rnn.weight_desc(),
        weight_grad_data,
        const_cast<uint8_t *>(reserve_data),
        reserve_size));
#else
    PADDLE_THROW(common::errors::Unavailable(
        "The padded input of rnn is supported by cudnnRNNBackwardDataEx, "
        "cudnnRNNBackwardWeightsEx, but it only works when the version "
        "of cudnn is larger than 7.2.1"));
#endif
  }

#endif  // end CUDNN_VERSION >= 90000
}

}  // namespace phi

#ifdef PADDLE_WITH_HIP
PD_REGISTER_KERNEL(
    cudnn_lstm_grad, GPU, ALL_LAYOUT, phi::CudnnLSTMGradKernel, float) {}
#else
PD_REGISTER_KERNEL(
    cudnn_lstm_grad, GPU, ALL_LAYOUT, phi::CudnnLSTMGradKernel, float, double) {
}
#endif
