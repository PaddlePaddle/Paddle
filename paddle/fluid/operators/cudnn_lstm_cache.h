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
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/platform/cudnn_helper.h"
#include "paddle/fluid/platform/dynload/cudnn.h"

namespace paddle {
namespace operators {

class ScopedRNNBase {
 public:
  ScopedRNNBase(int seq_length, int batch_size, int input_size, int hidden_size,
                int num_layers, float dropout_prob, int seed, int weight_numel,
                bool initialized, bool is_bidirec)
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
  void Create(const cudnnHandle_t& handle, const platform::Place& place,
              const std::vector<const framework::Tensor*>& sequence_length,
              size_t* workspace_size, size_t* reserve_size,
              framework::Tensor* dropout_state) {
    int numDirections = is_bidirec_ ? 2 : 1;
    cudnnDataType_t cudnn_type = platform::CudnnDataType<T>::type;

    // ------------------- cudnn x, y descriptors ---------------------
    std::vector<int> dims_x = {batch_size_, input_size_, 1};
    std::vector<int> strides_x = {input_size_, 1, 1};
    std::vector<int> dims_y = {batch_size_, hidden_size_ * numDirections, 1};
    std::vector<int> strides_y = {hidden_size_ * numDirections, 1, 1};
    for (int i = 0; i < seq_length_; ++i) {
      x_desc_.emplace_back(x_desc.descriptor<T>(dims_x, strides_x));
      y_desc_.emplace_back(y_desc.descriptor<T>(dims_y, strides_y));
    }
    if (!sequence_length.empty()) {
      x_seq_desc.descriptor<T>(seq_length_, batch_size_, input_size_, true,
                               sequence_length);
      y_seq_desc.descriptor<T>(seq_length_, batch_size_,
                               hidden_size_ * numDirections, true,
                               sequence_length);
    }

    // ------------------- cudnn hx, hy, cx, cy descriptors----------
    std::vector<int> dims_hx = {num_layers_ * numDirections, batch_size_,
                                hidden_size_};
    std::vector<int> strides_hx = {hidden_size_ * batch_size_, hidden_size_, 1};
    init_h_desc.descriptor<T>(dims_hx, strides_hx);
    init_c_desc.descriptor<T>(dims_hx, strides_hx);
    last_h_desc.descriptor<T>(dims_hx, strides_hx);
    last_c_desc.descriptor<T>(dims_hx, strides_hx);

    // ------------------- cudnn dropout descriptors ---------------------
    size_t state_size;
    if (!initialized_) {
      PADDLE_ENFORCE_CUDA_SUCCESS(
          platform::dynload::cudnnDropoutGetStatesSize(handle, &state_size));
      dropout_state->mutable_data<uint8_t>({static_cast<int64_t>(state_size)},
                                           place);
    }
    dropout_desc.descriptor(handle, place, initialized_, dropout_prob_,
                            dropout_state, seed_, state_size);

// ------------------- cudnn rnn descriptors ---------------------
#if CUDNN_VERSION >= 6000
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnSetRNNDescriptor_v6(
        handle, rnn_desc.get_desc_(), hidden_size_, num_layers_,
        dropout_desc.get_desc_(), CUDNN_LINEAR_INPUT,
        is_bidirec_ ? CUDNN_BIDIRECTIONAL : CUDNN_UNIDIRECTIONAL, CUDNN_LSTM,
        CUDNN_RNN_ALGO_STANDARD, cudnn_type));
#else
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnSetRNNDescriptor(
        rnn_desc.get_desc_(), hidden_size_, num_layers_,
        dropout_desc.get_desc_(), CUDNN_LINEAR_INPUT,
        is_bidirec_ ? CUDNN_BIDIRECTIONAL : CUDNN_UNIDIRECTIONAL, CUDNN_LSTM,
        cudnn_type));
#endif
    if (!sequence_length.empty()) {
      PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnSetRNNPaddingMode(
          rnn_desc.get_desc_(), CUDNN_RNN_PADDED_IO_ENABLED));
    }

    // ------------------- cudnn weights_size ---------------------
    size_t weights_size_;
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnGetRNNParamsSize(
        handle, rnn_desc.get_desc_(), x_desc_[0], &weights_size_, cudnn_type));
    PADDLE_ENFORCE_EQ(
        weights_size_, sizeof(T) * weight_numel_,
        platform::errors::InvalidArgument(
            "The cudnn lstm and setting weight size should be same."));
    // ------------------- cudnn weight descriptors ---------------------
    platform::DataLayout layout = platform::DataLayout::kNCHW;
    int dim_tmp = weights_size_ / sizeof(T);
    std::vector<int> dim_w = {dim_tmp, 1, 1};
    weight_desc.descriptor<T>(layout, dim_w);
    // ------------------- cudnn workspace, reserve size ---------------------
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnGetRNNWorkspaceSize(
        handle, rnn_desc.get_desc_(), seq_length_, x_desc_.data(),
        workspace_size));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cudnnGetRNNTrainingReserveSize(
            handle, rnn_desc.get_desc_(), seq_length_, x_desc_.data(),
            reserve_size));
  }
  cudnnTensorDescriptor_t* x_descs_() { return x_desc_.data(); }
  cudnnTensorDescriptor_t* y_descs_() { return y_desc_.data(); }
  cudnnRNNDataDescriptor_t x_seq_desc_() { return x_seq_desc.get_desc_(); }
  cudnnRNNDataDescriptor_t y_seq_desc_() { return y_seq_desc.get_desc_(); }
  cudnnTensorDescriptor_t init_h_desc_() { return init_h_desc.get_desc_(); }
  cudnnTensorDescriptor_t init_c_desc_() { return init_c_desc.get_desc_(); }
  cudnnTensorDescriptor_t last_h_desc_() { return last_h_desc.get_desc_(); }
  cudnnTensorDescriptor_t last_c_desc_() { return last_c_desc.get_desc_(); }
  cudnnRNNDescriptor_t rnn_desc_() { return rnn_desc.get_desc_(); }
  cudnnDropoutDescriptor_t dropout_desc_() { return dropout_desc.get_desc_(); }
  cudnnFilterDescriptor_t weight_desc_() { return weight_desc.get_desc_(); }

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
  std::vector<cudnnTensorDescriptor_t> x_desc_;
  std::vector<cudnnTensorDescriptor_t> y_desc_;

  platform::ScopedTensorDescriptor x_desc;
  platform::ScopedTensorDescriptor y_desc;
  platform::ScopedRNNTensorDescriptor x_seq_desc;
  platform::ScopedRNNTensorDescriptor y_seq_desc;
  platform::ScopedTensorDescriptor init_h_desc;
  platform::ScopedTensorDescriptor init_c_desc;
  platform::ScopedTensorDescriptor last_h_desc;
  platform::ScopedTensorDescriptor last_c_desc;
  platform::ScopedDropoutDescriptor dropout_desc;
  platform::ScopedFilterDescriptor weight_desc;
  platform::ScopedRNNDescriptor rnn_desc;
};

}  // namespace operators
}  // namespace paddle
