/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

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

#include <vector>

#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/backends/dynload/cudnn.h"
#include "paddle/phi/backends/gpu/forwards.h"
#include "paddle/phi/backends/gpu/gpu_dnn.h"
#include "paddle/phi/core/dense_tensor.h"

namespace phi {

class ScopedRNNBase {
 public:
  ScopedRNNBase(int seq_length,
                int batch_size,
                int input_size,
                int hidden_size,
                int num_layers,
                float dropout_prob,
                int seed,
                int weight_numel,
                bool initialized,
                bool is_bidirec)
      : seq_length_(seq_length),
        batch_size_(batch_size),
        input_size_(input_size),
        hidden_size_(hidden_size),
        num_layers_(num_layers),
        dropout_prob_(dropout_prob),
        seed_(seed),
        weight_numel_(weight_numel),
        initialized_(initialized),
        is_bidirec_(is_bidirec) {}

  template <typename T>
  void Create(const cudnnHandle_t& handle,
              const phi::Place& place,
              const std::vector<int>& sequence_length,
              size_t* workspace_size,
              size_t* reserve_size,
              phi::DenseTensor* dropout_state) {
    int numDirections = is_bidirec_ ? 2 : 1;
    cudnnDataType_t cudnn_type = phi::backends::gpu::CudnnDataType<T>::type;

    // ------------------- cudnn x, y descriptors ---------------------
    std::vector<int> dims_x = {batch_size_, input_size_, 1};
    std::vector<int> strides_x = {input_size_, 1, 1};
    std::vector<int> dims_y = {batch_size_, hidden_size_ * numDirections, 1};
    std::vector<int> strides_y = {hidden_size_ * numDirections, 1, 1};
    for (int i = 0; i < seq_length_; ++i) {
      x_descs_.emplace_back(x_desc_.descriptor<T>(dims_x, strides_x));
      y_descs_.emplace_back(y_desc_.descriptor<T>(dims_y, strides_y));
    }

#if CUDNN_VERSION >= 90000
    auto seqlen_is_empty = sequence_length.empty();
    if (seqlen_is_empty) {
      std::vector<int> seqlen_array(batch_size_);
      for (int i = 0; i < batch_size_; ++i) {
        seqlen_array[i] = seq_length_;
      }
      x_seq_desc_.descriptor<T>(
          seq_length_, batch_size_, input_size_, true, seqlen_array);
      y_seq_desc_.descriptor<T>(seq_length_,
                                batch_size_,
                                hidden_size_ * numDirections,
                                true,
                                seqlen_array);
    } else {
      x_seq_desc_.descriptor<T>(
          seq_length_, batch_size_, input_size_, true, sequence_length);
      y_seq_desc_.descriptor<T>(seq_length_,
                                batch_size_,
                                hidden_size_ * numDirections,
                                true,
                                sequence_length);
    }
#elif CUDNN_VERSION >= 7201
    if (!sequence_length.empty()) {
      x_seq_desc_.descriptor<T>(
          seq_length_, batch_size_, input_size_, true, sequence_length);
      y_seq_desc_.descriptor<T>(seq_length_,
                                batch_size_,
                                hidden_size_ * numDirections,
                                true,
                                sequence_length);
    }
#endif

    // ------------------- cudnn hx, hy, cx, cy descriptors----------
    std::vector<int> dims_hx = {
        num_layers_ * numDirections, batch_size_, hidden_size_};
    std::vector<int> strides_hx = {hidden_size_ * batch_size_, hidden_size_, 1};
    init_h_desc_.descriptor<T>(dims_hx, strides_hx);
    init_c_desc_.descriptor<T>(dims_hx, strides_hx);
    last_h_desc_.descriptor<T>(dims_hx, strides_hx);
    last_c_desc_.descriptor<T>(dims_hx, strides_hx);

    // ------------------- cudnn dropout descriptors ---------------------
    size_t state_size;
    if (!initialized_) {
      PADDLE_ENFORCE_GPU_SUCCESS(
          phi::dynload::cudnnDropoutGetStatesSize(handle, &state_size));
      phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();
      auto* dev_ctx = reinterpret_cast<phi::GPUContext*>(pool.Get(place));
      dropout_state->Resize({static_cast<int64_t>(state_size)});
      dev_ctx->template Alloc<uint8_t>(dropout_state);
    }
    dropout_desc_.descriptor(handle,
                             place,
                             initialized_,
                             dropout_prob_,
                             dropout_state,
                             seed_,
                             state_size);

    // ------------------- cudnn rnn descriptors ---------------------
#if CUDNN_VERSION >= 90000
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnSetRNNDescriptor_v8(
        rnn_desc_.desc(),
        CUDNN_RNN_ALGO_STANDARD,
        CUDNN_LSTM,
        CUDNN_RNN_DOUBLE_BIAS,
        is_bidirec_ ? CUDNN_BIDIRECTIONAL : CUDNN_UNIDIRECTIONAL,
        CUDNN_LINEAR_INPUT,
        cudnn_type,
        cudnn_type,
        CUDNN_DEFAULT_MATH,
        input_size_,
        hidden_size_,
        hidden_size_,
        num_layers_,
        dropout_desc_.desc(),
        seqlen_is_empty ? CUDNN_RNN_PADDED_IO_DISABLED
                        : CUDNN_RNN_PADDED_IO_ENABLED));
#else
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnSetRNNDescriptor_v6(
        handle,
        rnn_desc_.desc(),
        hidden_size_,
        num_layers_,
        dropout_desc_.desc(),
        CUDNN_LINEAR_INPUT,
        is_bidirec_ ? CUDNN_BIDIRECTIONAL : CUDNN_UNIDIRECTIONAL,
        CUDNN_LSTM,
        CUDNN_RNN_ALGO_STANDARD,
        cudnn_type));
#endif

#if CUDNN_VERSION < 90000 && CUDNN_VERSION >= 7201
    if (!sequence_length.empty()) {
      PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnSetRNNPaddingMode(
          rnn_desc_.desc(), CUDNN_RNN_PADDED_IO_ENABLED));
    }
#endif

    // ------------------- cudnn weights_size ---------------------
#if CUDNN_VERSION >= 90000
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnGetRNNWeightSpaceSize(
        handle, rnn_desc_.desc(), &weights_size_));
#else
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnGetRNNParamsSize(
        handle, rnn_desc_.desc(), x_descs_[0], &weights_size_, cudnn_type));
#endif

    PADDLE_ENFORCE_EQ(
        weights_size_,
        sizeof(T) * weight_numel_,
        phi::errors::InvalidArgument(
            "The cudnn lstm and setting weight size should be same."));
    // ------------------- cudnn weight descriptors ---------------------
    phi::backends::gpu::DataLayout layout =
        phi::backends::gpu::DataLayout::kNCHW;
    int dim_tmp = weights_size_ / sizeof(T);
    std::vector<int> dim_w = {dim_tmp, 1, 1};
    weight_desc_.descriptor<T>(layout, dim_w);
    // ------------------- cudnn workspace, reserve size ---------------------
#if CUDNN_VERSION >= 90000
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::cudnnGetRNNTempSpaceSizes(handle,
                                                rnn_desc_.desc(),
                                                CUDNN_FWD_MODE_TRAINING,
                                                x_seq_desc_.desc(),
                                                workspace_size,
                                                reserve_size));
#else
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::cudnnGetRNNWorkspaceSize(handle,
                                               rnn_desc_.desc(),
                                               seq_length_,
                                               x_descs_.data(),
                                               workspace_size));
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnGetRNNTrainingReserveSize(
        handle, rnn_desc_.desc(), seq_length_, x_descs_.data(), reserve_size));
#endif
  }
  cudnnTensorDescriptor_t* x_descs() { return x_descs_.data(); }
  cudnnTensorDescriptor_t* y_descs() { return y_descs_.data(); }
#if CUDNN_VERSION >= 7201
  cudnnRNNDataDescriptor_t x_seq_desc() { return x_seq_desc_.desc(); }
  cudnnRNNDataDescriptor_t y_seq_desc() { return y_seq_desc_.desc(); }
#endif
  cudnnTensorDescriptor_t init_h_desc() { return init_h_desc_.desc(); }
  cudnnTensorDescriptor_t init_c_desc() { return init_c_desc_.desc(); }
  cudnnTensorDescriptor_t last_h_desc() { return last_h_desc_.desc(); }
  cudnnTensorDescriptor_t last_c_desc() { return last_c_desc_.desc(); }
  cudnnRNNDescriptor_t rnn_desc() { return rnn_desc_.desc(); }
  cudnnDropoutDescriptor_t dropout_desc() { return dropout_desc_.desc(); }
  cudnnFilterDescriptor_t weight_desc() { return weight_desc_.desc(); }
  size_t weights_size() { return weights_size_; }

 private:
  int seq_length_;
  int batch_size_;
  int input_size_;
  int hidden_size_;
  int num_layers_;
  float dropout_prob_;
  int seed_;
  int weight_numel_;
  bool initialized_;
  bool is_bidirec_;
  size_t weights_size_;
  std::vector<cudnnTensorDescriptor_t> x_descs_;
  std::vector<cudnnTensorDescriptor_t> y_descs_;

  phi::backends::gpu::ScopedTensorDescriptor x_desc_;
  phi::backends::gpu::ScopedTensorDescriptor y_desc_;
#if CUDNN_VERSION >= 7201
  phi::backends::gpu::ScopedRNNTensorDescriptor x_seq_desc_;
  phi::backends::gpu::ScopedRNNTensorDescriptor y_seq_desc_;
#endif
  phi::backends::gpu::ScopedTensorDescriptor init_h_desc_;
  phi::backends::gpu::ScopedTensorDescriptor init_c_desc_;
  phi::backends::gpu::ScopedTensorDescriptor last_h_desc_;
  phi::backends::gpu::ScopedTensorDescriptor last_c_desc_;
  phi::backends::gpu::ScopedDropoutDescriptor dropout_desc_;
  phi::backends::gpu::ScopedFilterDescriptor weight_desc_;
  phi::backends::gpu::ScopedRNNDescriptor rnn_desc_;
};

}  // namespace phi
