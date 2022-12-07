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

#include "paddle/phi/kernels/rnn_kernel.h"

#include "paddle/fluid/operators/utils.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/generator.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/gpu/rnn_functor.h"

namespace phi {

template <typename T>
void RNNInferece(bool has_seq_length,
                 const gpuDnnHandle_t &handle,
                 int seq_length,
                 RNNDescriptors *rnn,
                 const T *x_data,
                 const T *init_h_data,
                 const T *init_c_data,
                 const T *w_data,
                 T *out_data,
                 T *last_h_data,
                 T *last_c_data,
                 DenseTensor *workspace_data,
                 size_t workspace_size) {
  if (!has_seq_length) {
// for inference
// This interface is used when the input/output is unpadded.
#ifdef PADDLE_WITH_HIP
    PADDLE_ENFORCE_GPU_SUCCESS(
        paddle::platform::dynload::miopenRNNForwardInference(
            handle,
            rnn->rnn_desc(),
            seq_length,
            rnn->x_descs(),
            x_data,
            rnn->init_h_desc(),
            init_h_data,
            rnn->init_c_desc(),
            init_c_data,
            rnn->weight_desc(),
            w_data,
            rnn->y_descs(),
            out_data,
            rnn->last_h_desc(),
            last_h_data,
            rnn->last_c_desc(),
            last_c_data,
            workspace_data->data<uint8_t>(),
            workspace_size));
#else
    PADDLE_ENFORCE_GPU_SUCCESS(
        paddle::platform::dynload::cudnnRNNForwardInference(
            handle,
            rnn->rnn_desc(),
            seq_length,
            rnn->x_descs(),
            x_data,
            rnn->init_h_desc(),
            init_h_data,
            rnn->init_c_desc(),
            init_c_data,
            rnn->weight_desc(),
            w_data,
            rnn->y_descs(),
            out_data,
            rnn->last_h_desc(),
            last_h_data,
            rnn->last_c_desc(),
            last_c_data,
            workspace_data->data<uint8_t>(),
            workspace_size));
#endif
  } else {
#if defined(PADDLE_WITH_CUDA) && CUDNN_VERSION >= 7201
    // for inference
    // This interface is used when the input/output is padded.
    PADDLE_ENFORCE_GPU_SUCCESS(
        paddle::platform::dynload::cudnnRNNForwardInferenceEx(
            handle,
            rnn->rnn_desc(),
            rnn->x_seq_desc(),
            x_data,
            rnn->init_h_desc(),
            init_h_data,
            rnn->init_c_desc(),
            init_c_data,
            rnn->weight_desc(),
            w_data,
            rnn->y_seq_desc(),
            out_data,
            rnn->last_h_desc(),
            last_h_data,
            rnn->last_c_desc(),
            last_c_data,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            workspace_data->data<uint8_t>(),
            workspace_size));
#else
    // CUDNN VERSION has to >=7.2.1
    PADDLE_THROW(phi::errors::Unavailable(
        "The padded input is supported by "
        "cudnnRNNForwardInferenceEx, but it only works when "
        "the version of cudnn is larger than 7.2.1"));
#endif
  }
}

template <typename T, typename Context>
void RnnKernel(const Context &dev_ctx,
               const DenseTensor &x,
               const std::vector<const DenseTensor *> &pre_state,
               const std::vector<const DenseTensor *> &weight_list,
               const paddle::optional<DenseTensor> &sequence_length,
               float dropout_prob,
               bool is_bidirec,
               int input_size,
               int hidden_size,
               int num_layers,
               const std::string &mode,
               int seed,
               bool is_test,
               DenseTensor *out,
               DenseTensor *dropout_state,
               std::vector<DenseTensor *> state,
               DenseTensor *reserve) {
#ifdef PADDLE_WITH_HIP
  gpuRNNMode_t rnn_mode = miopenLSTM;
  if (mode == "LSTM")
    rnn_mode = miopenLSTM;
  else if (mode == "GRU")
    rnn_mode = miopenGRU;
  else if (mode == "RNN_RELU")
    rnn_mode = miopenRNNRELU;
  else if (mode == "RNN_TANH")
    rnn_mode = miopenRNNTANH;
#else
  gpuRNNMode_t rnn_mode = CUDNN_LSTM;
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

  if (!is_test) {
    if (seed == 0) {
      // If not specify seed, use global Generator to generate seed.
      auto gen_cuda = dev_ctx.GetGenerator();
      seed = static_cast<int>(gen_cuda->Random64());
    }
    // else use `ctx.Attr<int>("seed")` specified seed
  }

  const T *x_data = x.data<T>();
  const T *init_h_data = pre_state[0]->data<T>();
  const T *init_c_data = nullptr;
  T *out_data = dev_ctx.template Alloc<T>(out);
  T *last_h_data = dev_ctx.template Alloc<T>(state[0]);
  T *last_c_data = nullptr;
#ifdef PADDLE_WITH_HIP
  if (rnn_mode == miopenLSTM) {
#else
  if (rnn_mode == CUDNN_LSTM) {
#endif
    init_c_data = pre_state[1]->data<T>();
    last_c_data = dev_ctx.template Alloc<T>(state[1]);
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

  auto handle = dev_ctx.cudnn_handle();

  int seq_length = x.dims()[0];
  int batch_size = x.dims()[1];
  int input_size_local = x.dims()[2];

  size_t workspace_size;
  size_t reserve_size;
  DenseTensor weight_whole;
  T *w_data = nullptr;
  auto place = dev_ctx.GetPlace();
  auto stream = dev_ctx.stream();
  auto weight_numel = std::accumulate(
      weight_list.begin(),
      weight_list.end(),
      0,
      [](int64_t num, const DenseTensor *t) { return num + t->numel(); });
  bool continuous =
      IsContinuous<T, std::vector<const DenseTensor *>>(weight_list);
#ifdef PADDLE_WITH_HIP
  // Need to permute weight, set continuous to false
  continuous = false;
#endif
  if (!continuous) {
    LOG_FIRST_N(WARNING, 2)
        << "If the memory space of the Input WeightList is not continuous, "
           "less efficient calculation will be called. Please call "
           "flatten_parameters() to make the input memory continuous.";
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
    w_data = weight_whole.data<T>();
#ifndef PADDLE_WITH_HIP
    // MIOPEN need to permute weight, do not share with weight_grad
    if (is_test) {  // maybe also reset small weights' ptr for training
      int offset = 0;
      for (size_t i = 0; i < weight_list.size(); ++i) {
        size_t len = weight_list[i]->numel();
        auto dim = weight_list[i]->dims();
        const_cast<DenseTensor *>(weight_list[i])
            ->ShareDataWith(
                weight_whole.Slice(static_cast<int64_t>(offset),
                                   static_cast<int64_t>(offset + len)))
            .Resize(dim);
        offset += len;
      }
    }
#endif
  } else {
    w_data = const_cast<T *>(weight_list[0]->data<T>());
  }

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
                dev_ctx,
                SequenceLength,
                &workspace_size,
                &reserve_size,
                dropout_state);

  DenseTensor workspace_data_ =
      Empty<uint8_t>(dev_ctx, {static_cast<int64_t>(workspace_size)});

  reserve->Resize({static_cast<int64_t>(reserve_size)});
  auto *reserve_data = dev_ctx.template Alloc<uint8_t>(reserve);

  if (is_test) {
    RNNInferece(has_seq_length,
                handle,
                seq_length,
                &rnn,
                x_data,
                init_h_data,
                init_c_data,
                w_data,
                out_data,
                last_h_data,
                last_c_data,
                &workspace_data_,
                workspace_size);
  } else {
    if (!has_seq_length) {
// for train
// This interface is used when the input/output is unpadded.
#ifdef PADDLE_WITH_HIP
      PADDLE_ENFORCE_GPU_SUCCESS(
          paddle::platform::dynload::miopenRNNForwardTraining(
              handle,
              rnn.rnn_desc(),
              seq_length,
              rnn.x_descs(),
              x_data,
              rnn.init_h_desc(),
              init_h_data,
              rnn.init_c_desc(),
              init_c_data,
              rnn.weight_desc(),
              w_data,
              rnn.y_descs(),
              out_data,
              rnn.last_h_desc(),
              last_h_data,
              rnn.last_c_desc(),
              last_c_data,
              workspace_data_.data<uint8_t>(),
              workspace_size,
              reserve_data,
              reserve_size));
#else
      PADDLE_ENFORCE_GPU_SUCCESS(
          paddle::platform::dynload::cudnnRNNForwardTraining(
              handle,
              rnn.rnn_desc(),
              seq_length,
              rnn.x_descs(),
              x_data,
              rnn.init_h_desc(),
              init_h_data,
              rnn.init_c_desc(),
              init_c_data,
              rnn.weight_desc(),
              w_data,
              rnn.y_descs(),
              out_data,
              rnn.last_h_desc(),
              last_h_data,
              rnn.last_c_desc(),
              last_c_data,
              workspace_data_.data<uint8_t>(),
              workspace_size,
              reserve_data,
              reserve_size));
#endif
    } else {
#if defined(PADDLE_WITH_CUDA) && CUDNN_VERSION >= 7201
      // for train
      // This interface is used when the input/output is padded.
      PADDLE_ENFORCE_GPU_SUCCESS(
          paddle::platform::dynload::cudnnRNNForwardTrainingEx(
              handle,
              rnn.rnn_desc(),
              rnn.x_seq_desc(),
              x_data,
              rnn.init_h_desc(),
              init_h_data,
              rnn.init_c_desc(),
              init_c_data,
              rnn.weight_desc(),
              w_data,
              rnn.y_seq_desc(),
              out_data,
              rnn.last_h_desc(),
              last_h_data,
              rnn.last_c_desc(),
              last_c_data,
              nullptr,
              nullptr,
              nullptr,
              nullptr,
              nullptr,
              nullptr,
              nullptr,
              nullptr,
              workspace_data_.data<uint8_t>(),
              workspace_size,
              reserve_data,
              reserve_size));
#else
      PADDLE_THROW(phi::errors::Unavailable(
          "The padded input is supported by "
          "cudnnRNNForwardTrainingEx, but it only works when "
          "the version of cudnn is larger than 7.2.1"));
#endif
    }
  }
}

}  // namespace phi

#ifdef PADDLE_WITH_HIP
// MIOPEN do not support double
PD_REGISTER_KERNEL(rnn, GPU, ALL_LAYOUT, phi::RnnKernel, float) {}
#else
PD_REGISTER_KERNEL(rnn, GPU, ALL_LAYOUT, phi::RnnKernel, float, double) {}
#endif
