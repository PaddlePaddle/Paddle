// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/rnn_grad_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/gpu/rnn_functor.h"

#include "paddle/fluid/operators/utils.h"

namespace phi {

#ifdef PADDLE_WITH_HIP
template <typename T>
void TensorToPermutedWeight(const Place &place,
                            gpuStream_t stream,
                            const DenseTensor &tensor,
                            std::vector<DenseTensor *> *weight_grad_list,
                            const gpuRNNMode_t rnn_mode,
                            bool is_bidirec) {
  if (is_bidirec) {
    for (size_t i = 0; i < weight_grad_list->size(); i += 4) {
      auto tmp = (*weight_grad_list)[i + 1];
      (*weight_grad_list)[i + 1] = (*weight_grad_list)[i + 2];
      (*weight_grad_list)[i + 2] = tmp;
    }
  }
  size_t weight_offset = 0;
  for (size_t i = 0; i < weight_grad_list->size(); ++i) {
    auto numel_size = (*weight_grad_list)[i]->numel();
    DenseTensor temp;
    temp.Resize({numel_size});
    temp.ShareDataWith(tensor.Slice(weight_offset, weight_offset + numel_size));

    if (rnn_mode == miopenLSTM) {
      std::vector<DenseTensor> split_tensor = temp.Chunk(4, 0);
      WeightListToTensor<T>(
          place,
          stream,
          {split_tensor[0], split_tensor[1], split_tensor[3], split_tensor[2]},
          (*weight_grad_list)[i]);
    } else if (rnn_mode == miopenGRU) {
      std::vector<DenseTensor> split_tensor = temp.Chunk(3, 0);
      WeightListToTensor<T>(place,
                            stream,
                            {split_tensor[1], split_tensor[0], split_tensor[2]},
                            (*weight_grad_list)[i]);
    } else {
      WeightListToTensor<T>(place, stream, {temp}, (*weight_grad_list)[i]);
    }
    weight_offset += numel_size;
  }
  if (is_bidirec) {
    for (size_t i = 0; i < weight_grad_list->size(); i += 4) {
      auto tmp = (*weight_grad_list)[i + 1];
      (*weight_grad_list)[i + 1] = (*weight_grad_list)[i + 2];
      (*weight_grad_list)[i + 2] = tmp;
    }
  }
}
#endif

template <typename T, typename Context>
void RnnGradKernel(const Context &dev_ctx,
                   const DenseTensor &x,
                   const std::vector<const DenseTensor *> &pre_state,
                   const std::vector<const DenseTensor *> &weight_list,
                   paddle::optional<const DenseTensor &> sequence_length,
                   const DenseTensor &out,
                   const DenseTensor &dropout_state,
                   const DenseTensor &reserve,
                   const DenseTensor &out_grad,
                   const std::vector<const DenseTensor *> &state_grad,
                   float dropout_prob,
                   bool is_bidirec,
                   int input_size,
                   int hidden_size,
                   int num_layers,
                   const std::string &mode,
                   int seed,
                   bool is_test,
                   DenseTensor *x_grad,
                   std::vector<DenseTensor *> pre_state_grad,
                   std::vector<DenseTensor *> weight_grad_list) {
#ifdef PADDLE_WITH_HIP
  miopenRNNMode_t rnn_mode = miopenLSTM;
  if (mode == "LSTM")
    rnn_mode = miopenLSTM;
  else if (mode == "GRU")
    rnn_mode = miopenGRU;
  else if (mode == "RNN_RELU")
    rnn_mode = miopenRNNRELU;
  else if (mode == "RNN_TANH")
    rnn_mode = miopenRNNTANH;
#else
  cudnnRNNMode_t rnn_mode = CUDNN_LSTM;
  if (mode == "LSTM")
    rnn_mode = CUDNN_LSTM;
  else if (mode == "GRU")
    rnn_mode = CUDNN_GRU;
  else if (mode == "RNN_RELU")
    rnn_mode = CUDNN_RNN_RELU;
  else if (mode == "RNN_TANH")
    rnn_mode = CUDNN_RNN_TANH;
#endif
  else
    PADDLE_THROW(phi::errors::InvalidArgument(
        "rnn_mode should be LSTM, GRU, RNN_RELU or RNN_TANH, but received: "
        "%s.",
        mode));
  auto handle = dev_ctx.cudnn_handle();
  auto place = dev_ctx.GetPlace();
  auto weight_numel = std::accumulate(
      weight_list.begin(),
      weight_list.end(),
      0,
      [](int64_t num, const DenseTensor *t) { return num + t->numel(); });
  bool continuous =
      IsContinuous<T, std::vector<const DenseTensor *>>(weight_list);
  auto stream = dev_ctx.stream();
  DenseTensor weight_whole;
  T *weight_data = nullptr;

#ifdef PADDLE_WITH_HIP
  // Need to permute weight, set continuous to false
  continuous = false;
#endif

  if (!continuous) {
    weight_whole.Resize({weight_numel});
    dev_ctx.template Alloc<T>(&weight_whole);
#ifdef PADDLE_WITH_HIP
    // MIOPEN need to permute weight for miopenLSTM or miopenGRU
    std::vector<const DenseTensor *> weight_list_tmp = weight_list;
    WeightToPermutedTensor<T>(
        place, stream, &weight_list_tmp, &weight_whole, rnn_mode, is_bidirec);
#else
    WeightToTensor<T>(place, stream, weight_list, &weight_whole);
#endif
    weight_data = weight_whole.data<T>();
  } else {
    weight_data = const_cast<T *>(weight_list[0]->data<T>());
  }

  DenseTensor weight_grad = Full<T>(dev_ctx, {weight_numel}, 0);
  T *weight_grad_data = weight_grad.data<T>();

#ifdef PADDLE_WITH_HIP
  // MIOPEN need to permute weight_grad_list, so do not share data with
  // weight_grad
  for (size_t i = 0; i < weight_grad_list.size(); ++i) {
    dev_ctx.template Alloc<T>(weight_grad_list[i]);
  }
#else
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
#endif

  DenseTensor input_grad_value;
  if (!x_grad) {
    x_grad = &input_grad_value;
    x_grad->Resize(x.dims());
  }

  auto *init_h_data = pre_state[0]->data<T>();
  // auto *last_h_data = state[0]->data<T>();
  auto *last_h_grad_data = state_grad[0]->data<T>();
  const T *init_c_data = nullptr;
  // const T *last_c_data = nullptr;
  const T *last_c_grad_data = nullptr;
  T *init_h_grad_data = pre_state_grad.size() != 0 && pre_state_grad[0]
                            ? dev_ctx.template Alloc<T>(pre_state_grad[0])
                            : nullptr;
  T *init_c_grad_data = nullptr;
#ifdef PADDLE_WITH_HIP
  if (rnn_mode == miopenLSTM) {
#else
  if (rnn_mode == CUDNN_LSTM) {
#endif
    init_c_data = pre_state[1]->data<T>();
    // last_c_data = state[1]->data<T>();
    last_c_grad_data = state_grad[1]->data<T>();
    init_c_grad_data = pre_state_grad.size() >= 2 && pre_state_grad[1]
                           ? dev_ctx.template Alloc<T>(pre_state_grad[1])
                           : nullptr;
  }
  auto *out_data = out.data<T>();
  auto *out_grad_data = out_grad.data<T>();

  // need check exist
  T *x_grad_data = nullptr;
  if (x_grad) {
    x_grad_data = dev_ctx.template Alloc<T>(x_grad);
  }

  bool has_seq_length = sequence_length.is_initialized();
#ifdef PADDLE_WITH_HIP
  PADDLE_ENFORCE_EQ(
      has_seq_length,
      false,
      phi::errors::InvalidArgument("ROCm do not support SequenceLength yet."));
#endif
  std::vector<int> SequenceLength;
  if (has_seq_length) {
    SequenceLength =
        paddle::operators::GetDataFromTensor<int>(sequence_length.get_ptr());
  }

  auto input_dims = x.dims();
  int seq_length = input_dims[0];
  int batch_size = input_dims[1];
  int input_size_local = input_dims[2];

  size_t workspace_size;
  size_t reserve_size;

  RNNDescriptors rnn(seq_length,
                     batch_size,
                     input_size_local,
                     hidden_size,
                     num_layers,
                     dropout_prob,
                     seed,
                     weight_numel,
                     rnn_mode,
                     is_bidirec,
                     is_test);

  rnn.Create<T>(handle,
                dev_ctx.GetPlace(),
                SequenceLength,
                &workspace_size,
                &reserve_size,
                const_cast<DenseTensor *>(&dropout_state));

  DenseTensor workspace_data_ =
      Empty<uint8_t>(dev_ctx, {static_cast<int64_t>(workspace_size)});
  const uint8_t *reserve_data = reserve.data<uint8_t>();

  if (!has_seq_length) {
    if (x_grad) {
#ifdef PADDLE_WITH_HIP
      PADDLE_ENFORCE_GPU_SUCCESS(
          paddle::platform::dynload::miopenRNNBackwardData(
              handle,
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
              x_grad_data,
              rnn.init_h_desc(),
              init_h_grad_data,
              rnn.init_c_desc(),
              init_c_grad_data,
              workspace_data_.data<uint8_t>(),
              workspace_size,
              const_cast<uint8_t *>(reserve_data),
              reserve_size));
#else
      // This interface is used when the input/output is unpadded.
      PADDLE_ENFORCE_GPU_SUCCESS(
          paddle::platform::dynload::cudnnRNNBackwardData(
              handle,
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
              x_grad_data,
              rnn.init_h_desc(),
              init_h_grad_data,
              rnn.init_c_desc(),
              init_c_grad_data,
              workspace_data_.data<uint8_t>(),
              workspace_size,
              const_cast<uint8_t *>(reserve_data),
              reserve_size));
#endif
    }
    if (!weight_grad_list.empty()) {
#ifdef PADDLE_WITH_HIP
      PADDLE_ENFORCE_GPU_SUCCESS(
          paddle::platform::dynload::miopenRNNBackwardWeights(
              handle,
              rnn.rnn_desc(),
              seq_length,
              rnn.x_descs(),
              x.data<T>(),
              rnn.init_h_desc(),
              init_h_data,
              rnn.y_descs(),
              out.data<T>(),
              rnn.weight_desc(),
              weight_grad_data,
              workspace_data_.data<uint8_t>(),
              workspace_size,
              const_cast<uint8_t *>(reserve_data),
              reserve_size));
      // permute weight grad list from weight grad tensor
      TensorToPermutedWeight<T>(
          place, stream, weight_grad, &weight_grad_list, rnn_mode, is_bidirec);
#else
      PADDLE_ENFORCE_GPU_SUCCESS(
          paddle::platform::dynload::cudnnRNNBackwardWeights(
              handle,
              rnn.rnn_desc(),
              seq_length,
              rnn.x_descs(),
              x.data<T>(),
              rnn.init_h_desc(),
              init_h_data,
              rnn.y_descs(),
              out.data<T>(),
              workspace_data_.data<uint8_t>(),
              workspace_size,
              rnn.weight_desc(),
              weight_grad_data,
              const_cast<uint8_t *>(reserve_data),
              reserve_size));
#endif
    }
  } else {
#if defined(PADDLE_WITH_CUDA) && CUDNN_VERSION >= 7201
    // for train
    // This interface is used when the input/output is padded.
    if (x_grad) {
      PADDLE_ENFORCE_GPU_SUCCESS(
          paddle::platform::dynload::cudnnRNNBackwardDataEx(
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
              x_grad_data,
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
    }

    if (!weight_grad_list.empty()) {
      PADDLE_ENFORCE_GPU_SUCCESS(
          paddle::platform::dynload::cudnnRNNBackwardWeightsEx(
              handle,
              rnn.rnn_desc(),
              rnn.x_seq_desc(),
              x.data<T>(),
              rnn.init_h_desc(),
              init_h_data,
              rnn.y_seq_desc(),
              out.data<T>(),
              workspace_data_.data<uint8_t>(),
              workspace_size,
              rnn.weight_desc(),
              weight_grad_data,
              const_cast<uint8_t *>(reserve_data),
              reserve_size));
    }
#else
    PADDLE_THROW(phi::errors::Unavailable(
        "The padded input of rnn is supported by cudnnRNNBackwardDataEx, "
        "cudnnRNNBackwardWeightsEx, but it only works when the version "
        "of cudnn is larger than 7.2.1"));
#endif
  }
}

}  // namespace phi

#ifdef PADDLE_WITH_HIP
// MIOPEN do not support double
PD_REGISTER_KERNEL(rnn_grad, GPU, ALL_LAYOUT, phi::RnnGradKernel, float) {}
#else
PD_REGISTER_KERNEL(
    rnn_grad, GPU, ALL_LAYOUT, phi::RnnGradKernel, float, double) {}
#endif
