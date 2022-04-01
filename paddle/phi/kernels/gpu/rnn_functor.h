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

#pragma once

#include "paddle/phi/common/place.h"
#include "paddle/phi/core/dense_tensor.h"

#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/platform/device/gpu/gpu_dnn.h"

namespace phi {

#ifdef PADDLE_WITH_HIP
using gpuRNNMode_t = miopenRNNMode_t;
using gpuDnnHandle_t = miopenHandle_t;
using gpuDnnDataType_t = miopenDataType_t;
#else
using gpuRNNMode_t = cudnnRNNMode_t;
using gpuDnnHandle_t = cudnnHandle_t;
using gpuDnnDataType_t = cudnnDataType_t;
#endif

class RNNDescriptors {
 public:
  RNNDescriptors(int seq_length,
                 int batch_size,
                 int input_size,
                 int hidden_size,
                 int num_layers,
                 float dropout_prob,
                 int seed,
                 int weight_numel,
                 gpuRNNMode_t mode,
                 bool is_bidirec,
                 bool is_test)
      : seq_length_(seq_length),
        batch_size_(batch_size),
        input_size_(input_size),
        hidden_size_(hidden_size),
        num_layers_(num_layers),
        dropout_prob_(dropout_prob),
        seed_(seed),
        weight_numel_(weight_numel),
        mode_(mode),
        is_bidirec_(is_bidirec),
        is_test_(is_test) {}

  template <typename T>
  void Create(const gpuDnnHandle_t &handle,
              const Place &place,
              const std::vector<int> &sequence_length,
              size_t *workspace_size,
              size_t *reserve_size,
              DenseTensor *dropout_state) {
    int numDirections = is_bidirec_ ? 2 : 1;
    gpuDnnDataType_t cudnn_type = paddle::platform::CudnnDataType<T>::type;
    // ------------------- cudnn x, y descriptors ---------------------
    std::vector<int> dims_x = {batch_size_, input_size_, 1};
    std::vector<int> strides_x = {input_size_, 1, 1};
    std::vector<int> dims_y = {batch_size_, hidden_size_ * numDirections, 1};
    std::vector<int> strides_y = {hidden_size_ * numDirections, 1, 1};
    for (int i = 0; i < seq_length_; ++i) {
      x_descs_.emplace_back(x_desc_.descriptor<T>(dims_x, strides_x));
      y_descs_.emplace_back(y_desc_.descriptor<T>(dims_y, strides_y));
    }

#if defined(PADDLE_WITH_CUDA) && CUDNN_VERSION >= 7201
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
    bool is_initialized = dropout_state->IsInitialized();
    if (!is_test_ && !is_initialized) {
#ifdef PADDLE_WITH_HIP
      PADDLE_ENFORCE_GPU_SUCCESS(
          paddle::platform::dynload::miopenDropoutGetStatesSize(handle,
                                                                &state_size));
      dropout_state->mutable_data<uint8_t>({static_cast<int64_t>(state_size)},
                                           place);
#else
      PADDLE_ENFORCE_GPU_SUCCESS(
          paddle::platform::dynload::cudnnDropoutGetStatesSize(handle,
                                                               &state_size));
      dropout_state->mutable_data<uint8_t>({static_cast<int64_t>(state_size)},
                                           place);
#endif
    }
    dropout_desc_.descriptor(handle,
                             place,
                             is_initialized,
                             dropout_prob_,
                             is_test_ ? nullptr : dropout_state,
                             seed_,
                             state_size);

// ------------------- cudnn rnn descriptors ---------------------
#ifdef PADDLE_WITH_HIP
    PADDLE_ENFORCE_GPU_SUCCESS(
        paddle::platform::dynload::miopenSetRNNDescriptor_V2(
            rnn_desc_.desc(),
            hidden_size_,
            num_layers_,
            dropout_desc_.desc(),
            miopenRNNlinear,
            is_bidirec_ ? miopenRNNbidirection : miopenRNNunidirection,
            mode_,
            miopenRNNwithBias,
            miopenRNNdefault,
            cudnn_type));
#elif CUDNN_VERSION >= 6000
    PADDLE_ENFORCE_GPU_SUCCESS(
        paddle::platform::dynload::cudnnSetRNNDescriptor_v6(
            handle,
            rnn_desc_.desc(),
            hidden_size_,
            num_layers_,
            dropout_desc_.desc(),
            CUDNN_LINEAR_INPUT,
            is_bidirec_ ? CUDNN_BIDIRECTIONAL : CUDNN_UNIDIRECTIONAL,
            mode_,
            CUDNN_RNN_ALGO_STANDARD,
            cudnn_type));
#else
    PADDLE_ENFORCE_GPU_SUCCESS(paddle::platform::dynload::cudnnSetRNNDescriptor(
        rnn_desc_.desc(),
        hidden_size_,
        num_layers_,
        dropout_desc_.desc(),
        CUDNN_LINEAR_INPUT,
        is_bidirec_ ? CUDNN_BIDIRECTIONAL : CUDNN_UNIDIRECTIONAL,
        mode_,
        cudnn_type));
#endif

#if defined(PADDLE_WITH_CUDA) && CUDNN_VERSION >= 7201
    if (!sequence_length.empty()) {
      PADDLE_ENFORCE_GPU_SUCCESS(
          paddle::platform::dynload::cudnnSetRNNPaddingMode(
              rnn_desc_.desc(), CUDNN_RNN_PADDED_IO_ENABLED));
    }
#endif

    // ------------------- cudnn weights_size ---------------------
    size_t weights_size_;
#ifdef PADDLE_WITH_HIP
    PADDLE_ENFORCE_GPU_SUCCESS(
        paddle::platform::dynload::miopenGetRNNParamsSize(
            handle, rnn_desc_.desc(), x_descs_[0], &weights_size_, cudnn_type));
#else
    PADDLE_ENFORCE_GPU_SUCCESS(paddle::platform::dynload::cudnnGetRNNParamsSize(
        handle, rnn_desc_.desc(), x_descs_[0], &weights_size_, cudnn_type));
#endif
    PADDLE_ENFORCE_EQ(
        weights_size_,
        sizeof(T) * weight_numel_,
        phi::errors::InvalidArgument(
            "The cudnn rnn and setting weight size should be same."));
    // ------------------- cudnn weight descriptors ---------------------
    auto layout = paddle::platform::DataLayout::kNCHW;
    int dim_tmp = weights_size_ / sizeof(T);
    std::vector<int> dim_w = {dim_tmp, 1, 1};
    weight_desc_.descriptor<T>(layout, dim_w);
// ------------------- cudnn workspace, reserve size ---------------------
#ifdef PADDLE_WITH_HIP
    PADDLE_ENFORCE_GPU_SUCCESS(
        paddle::platform::dynload::miopenGetRNNWorkspaceSize(handle,
                                                             rnn_desc_.desc(),
                                                             seq_length_,
                                                             x_descs_.data(),
                                                             workspace_size));
    PADDLE_ENFORCE_GPU_SUCCESS(
        paddle::platform::dynload::miopenGetRNNTrainingReserveSize(
            handle,
            rnn_desc_.desc(),
            seq_length_,
            x_descs_.data(),
            reserve_size));
#else
    PADDLE_ENFORCE_GPU_SUCCESS(
        paddle::platform::dynload::cudnnGetRNNWorkspaceSize(handle,
                                                            rnn_desc_.desc(),
                                                            seq_length_,
                                                            x_descs_.data(),
                                                            workspace_size));
    PADDLE_ENFORCE_GPU_SUCCESS(
        paddle::platform::dynload::cudnnGetRNNTrainingReserveSize(
            handle,
            rnn_desc_.desc(),
            seq_length_,
            x_descs_.data(),
            reserve_size));
#endif
  }
#ifdef PADDLE_WITH_HIP
  miopenTensorDescriptor_t *x_descs() { return x_descs_.data(); }
  miopenTensorDescriptor_t *y_descs() { return y_descs_.data(); }
  miopenTensorDescriptor_t init_h_desc() { return init_h_desc_.desc(); }
  miopenTensorDescriptor_t init_c_desc() { return init_c_desc_.desc(); }
  miopenTensorDescriptor_t last_h_desc() { return last_h_desc_.desc(); }
  miopenTensorDescriptor_t last_c_desc() { return last_c_desc_.desc(); }
  miopenRNNDescriptor_t rnn_desc() { return rnn_desc_.desc(); }
  miopenDropoutDescriptor_t dropout_desc() { return dropout_desc_.desc(); }
  miopenTensorDescriptor_t weight_desc() { return weight_desc_.desc(); }
#else
  cudnnTensorDescriptor_t *x_descs() { return x_descs_.data(); }
  cudnnTensorDescriptor_t *y_descs() { return y_descs_.data(); }
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
#endif

 private:
  int seq_length_;
  int batch_size_;
  int input_size_;
  int hidden_size_;
  int num_layers_;
  float dropout_prob_;
  int seed_;
  int weight_numel_;
  gpuRNNMode_t mode_;
  bool is_bidirec_;
  bool is_test_;
#ifdef PADDLE_WITH_HIP
  std::vector<miopenTensorDescriptor_t> x_descs_;
  std::vector<miopenTensorDescriptor_t> y_descs_;
#else
  std::vector<cudnnTensorDescriptor_t> x_descs_;
  std::vector<cudnnTensorDescriptor_t> y_descs_;
#endif

  paddle::platform::ScopedTensorDescriptor x_desc_;
  paddle::platform::ScopedTensorDescriptor y_desc_;
#if defined(PADDLE_WITH_CUDA) && CUDNN_VERSION >= 7201
  paddle::platform::ScopedRNNTensorDescriptor x_seq_desc_;
  paddle::platform::ScopedRNNTensorDescriptor y_seq_desc_;
#endif
  paddle::platform::ScopedTensorDescriptor init_h_desc_;
  paddle::platform::ScopedTensorDescriptor init_c_desc_;
  paddle::platform::ScopedTensorDescriptor last_h_desc_;
  paddle::platform::ScopedTensorDescriptor last_c_desc_;
  paddle::platform::ScopedDropoutDescriptor dropout_desc_;
  paddle::platform::ScopedFilterDescriptor weight_desc_;
  paddle::platform::ScopedRNNDescriptor rnn_desc_;
};

template <typename T, typename Type>
bool IsContinuous(const Type &weight_list) {
  bool continuous = true;
  for (size_t i = 0; i < weight_list.size() - 1; ++i) {
    auto *in_data = weight_list[i]->template data<T>();
    auto *in_after_data = weight_list[i + 1]->template data<T>();
    auto in_size = weight_list[i]->numel();
    bool temp = in_data + in_size == in_after_data;
    continuous = continuous && temp;
  }
  return continuous;
}

template <typename T>
void WeightToTensor(const Place &place,
                    gpuStream_t stream,
                    const std::vector<const DenseTensor *> &weight_list,
                    DenseTensor *weight) {
  auto weight_data = weight->data<T>();
  int weight_offset = 0;
  for (size_t i = 0; i < weight_list.size(); ++i) {
    const T *in_data = weight_list[i]->data<T>();
    auto in_size = weight_list[i]->numel();

    paddle::memory::Copy(weight->place(),
                         weight_data + weight_offset,
                         weight_list[i]->place(),
                         in_data,
                         in_size * sizeof(T),
                         stream);
    weight_offset += in_size;
  }
}

#ifdef PADDLE_WITH_HIP
template <typename T>
void WeightListToTensor(const Place &place,
                        gpuStream_t stream,
                        const std::vector<DenseTensor> &tensor_list,
                        DenseTensor *weight_whole,
                        const size_t offset = 0UL) {
  size_t weight_offset = offset;
  auto weight_data = weight_whole->data<T>();

  for (size_t i = 0; i < tensor_list.size(); ++i) {
    const T *in_data = tensor_list[i].data<T>();
    auto in_size = tensor_list[i].numel();
    paddle::memory::Copy(weight_whole->place(),
                         weight_data + weight_offset,
                         tensor_list[i].place(),
                         in_data,
                         in_size * sizeof(T),
                         stream);
    weight_offset += in_size;
  }
}

template <typename T>
void WeightToPermutedTensor(const Place &place,
                            gpuStream_t stream,
                            std::vector<const DenseTensor *> *weight_list,
                            DenseTensor *weight_whole,
                            const gpuRNNMode_t rnn_mode,
                            const bool is_bidirec) {
  if (is_bidirec) {
    for (size_t i = 0; i < weight_list->size(); i += 4) {
      auto tmp = (*weight_list)[i + 1];
      (*weight_list)[i + 1] = (*weight_list)[i + 2];
      (*weight_list)[i + 2] = tmp;
    }
  }
  size_t weight_offset = 0;
  for (size_t i = 0; i < weight_list->size(); ++i) {
    if (rnn_mode == miopenLSTM) {
      std::vector<DenseTensor> split_tensor = (*weight_list)[i]->Chunk(4, 0);
      WeightListToTensor<T>(
          place,
          stream,
          {split_tensor[0], split_tensor[1], split_tensor[3], split_tensor[2]},
          weight_whole,
          weight_offset);
    } else if (rnn_mode == miopenGRU) {
      std::vector<DenseTensor> split_tensor = (*weight_list)[i]->Chunk(3, 0);
      WeightListToTensor<T>(place,
                            stream,
                            {split_tensor[1], split_tensor[0], split_tensor[2]},
                            weight_whole,
                            weight_offset);
    } else {
      WeightListToTensor<T>(
          place, stream, {*(*weight_list)[i]}, weight_whole, weight_offset);
    }
    weight_offset += (*weight_list)[i]->numel();
  }
}

#endif

}  // namespace phi
