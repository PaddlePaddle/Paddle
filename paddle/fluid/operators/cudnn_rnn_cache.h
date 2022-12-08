/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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
#include "paddle/fluid/platform/device/gpu/gpu_dnn.h"

namespace paddle {
namespace operators {

struct CudnnRNNCache {
  CudnnRNNCache() {
    x_desc_ = NULL;
    y_desc_ = NULL;
  }
  ~CudnnRNNCache() { release(); }

  cudnnRNNDescriptor_t rnn_desc_;
  cudnnTensorDescriptor_t *x_desc_;
  cudnnTensorDescriptor_t *y_desc_;

  cudnnTensorDescriptor_t hx_desc_;
  cudnnTensorDescriptor_t cx_desc_;
  cudnnTensorDescriptor_t hy_desc_;
  cudnnTensorDescriptor_t cy_desc_;

  cudnnTensorDescriptor_t dhx_desc_;
  cudnnTensorDescriptor_t dcx_desc_;
  cudnnTensorDescriptor_t dhy_desc_;
  cudnnTensorDescriptor_t dcy_desc_;

  cudnnTensorDescriptor_t output_x_desc_;
  cudnnTensorDescriptor_t output_y_desc_;

  cudnnDropoutDescriptor_t dropout_desc_;

  size_t weights_size_;
  cudnnFilterDescriptor_t w_desc_;
  cudnnFilterDescriptor_t dw_desc_;

  size_t workspace_size_;
  phi::DenseTensor workspace_data_;

  size_t seq_length_;

  float dropout_prob_;
  bool is_bidirec_;

  int batch_size_;
  int input_size_;
  int hidden_size_;
  int num_layers_;
  int seed_;

  void init(cudnnHandle_t handle,
            const platform::Place &place,
            size_t seq_len,
            int batch_size,
            int input_size,
            int hidden_size,
            int num_layers,
            float dropout_prob,
            bool is_bidirec,
            int seed,
            int weight_numel,
            size_t *reserve_size_,
            phi::DenseTensor *dropout_state_,
            bool initialized,
            cudnnDataType_t cudnn_type) {
    seq_length_ = seq_len;
    batch_size_ = batch_size;
    input_size_ = input_size;
    hidden_size_ = hidden_size;
    num_layers_ = num_layers;
    dropout_prob_ = dropout_prob;
    is_bidirec_ = is_bidirec;
    seed_ = seed;

    const auto numDirections = is_bidirec_ ? 2 : 1;
    auto cudnn_size =
        cudnn_type == CUDNN_DATA_FLOAT ? sizeof(float) : sizeof(double);

    x_desc_ = new cudnnTensorDescriptor_t[seq_length_];
    y_desc_ = new cudnnTensorDescriptor_t[seq_length_];
    std::vector<int> dims = {batch_size_, input_size_, 1};
    std::vector<int> strides = {input_size_, 1, 1};

    std::vector<int> dims_y = {batch_size_, hidden_size_ * numDirections, 1};
    std::vector<int> strides_y = {hidden_size_ * numDirections, 1, 1};

    for (size_t i = 0; i < seq_length_; ++i) {
      PADDLE_ENFORCE_GPU_SUCCESS(
          platform::dynload::cudnnCreateTensorDescriptor(&x_desc_[i]));
      PADDLE_ENFORCE_GPU_SUCCESS(
          platform::dynload::cudnnCreateTensorDescriptor(&y_desc_[i]));

      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cudnnSetTensorNdDescriptor(
          x_desc_[i], cudnn_type, 3, dims.data(), strides.data()));

      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cudnnSetTensorNdDescriptor(
          y_desc_[i], cudnn_type, 3, dims_y.data(), strides_y.data()));
    }

    std::vector<int> dims_hx = {
        num_layers_ * numDirections, batch_size_, hidden_size_};
    std::vector<int> strides_hx = {hidden_size_ * batch_size_, hidden_size_, 1};

    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cudnnCreateTensorDescriptor(&hx_desc_));
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cudnnCreateTensorDescriptor(&cx_desc_));
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cudnnCreateTensorDescriptor(&hy_desc_));
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cudnnCreateTensorDescriptor(&cy_desc_));
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cudnnCreateTensorDescriptor(&dhx_desc_));
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cudnnCreateTensorDescriptor(&dcx_desc_));
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cudnnCreateTensorDescriptor(&dhy_desc_));
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cudnnCreateTensorDescriptor(&dcy_desc_));

    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cudnnSetTensorNdDescriptor(
        hx_desc_, cudnn_type, 3, dims_hx.data(), strides_hx.data()));
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cudnnSetTensorNdDescriptor(
        cx_desc_, cudnn_type, 3, dims_hx.data(), strides_hx.data()));
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cudnnSetTensorNdDescriptor(
        hy_desc_, cudnn_type, 3, dims_hx.data(), strides_hx.data()));
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cudnnSetTensorNdDescriptor(
        cy_desc_, cudnn_type, 3, dims_hx.data(), strides_hx.data()));
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cudnnSetTensorNdDescriptor(
        dhx_desc_, cudnn_type, 3, dims_hx.data(), strides_hx.data()));
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cudnnSetTensorNdDescriptor(
        dcx_desc_, cudnn_type, 3, dims_hx.data(), strides_hx.data()));
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cudnnSetTensorNdDescriptor(
        dhy_desc_, cudnn_type, 3, dims_hx.data(), strides_hx.data()));
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cudnnSetTensorNdDescriptor(
        dcy_desc_, cudnn_type, 3, dims_hx.data(), strides_hx.data()));

    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cudnnCreateDropoutDescriptor(&dropout_desc_));

    size_t state_size;
    if (!initialized) {
      PADDLE_ENFORCE_GPU_SUCCESS(
          platform::dynload::cudnnDropoutGetStatesSize(handle, &state_size));
      dropout_state_->Resize({static_cast<int64_t>(state_size)});
      uint8_t *dropout_state_data =
          dropout_state_->mutable_data<uint8_t>(place);
      PADDLE_ENFORCE_GPU_SUCCESS(
          platform::dynload::cudnnSetDropoutDescriptor(dropout_desc_,
                                                       handle,
                                                       dropout_prob_,
                                                       dropout_state_data,
                                                       state_size,
                                                       seed_));
    } else {
      uint8_t *dropout_state_data = dropout_state_->data<uint8_t>();
      auto dropout_state_dims = dropout_state_->dims();
      state_size = dropout_state_dims[0];
      PADDLE_ENFORCE_GPU_SUCCESS(
          platform::dynload::cudnnRestoreDropoutDescriptor(dropout_desc_,
                                                           handle,
                                                           dropout_prob_,
                                                           dropout_state_data,
                                                           state_size,
                                                           0));
    }

    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cudnnCreateRNNDescriptor(&rnn_desc_));

    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cudnnSetRNNDescriptor_v6(
        handle,
        rnn_desc_,
        hidden_size_,
        num_layers_,
        dropout_desc_,
        CUDNN_LINEAR_INPUT,
        is_bidirec_ ? CUDNN_BIDIRECTIONAL : CUDNN_UNIDIRECTIONAL,
        CUDNN_LSTM,
        CUDNN_RNN_ALGO_STANDARD,
        cudnn_type));

    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cudnnCreateFilterDescriptor(&w_desc_));
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cudnnCreateFilterDescriptor(&dw_desc_));

    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cudnnGetRNNParamsSize(
        handle, rnn_desc_, x_desc_[0], &weights_size_, cudnn_type));

    PADDLE_ENFORCE_EQ(
        weights_size_,
        cudnn_size * weight_numel,
        platform::errors::InvalidArgument(
            "The cudnn lstm and setting weight size should be same."));

    int dim_w[3];
    dim_w[0] = weights_size_ / cudnn_size;
    dim_w[1] = 1;
    dim_w[2] = 1;
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cudnnSetFilterNdDescriptor(
        w_desc_, cudnn_type, CUDNN_TENSOR_NCHW, 3, dim_w));
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cudnnSetFilterNdDescriptor(
        dw_desc_, cudnn_type, CUDNN_TENSOR_NCHW, 3, dim_w));

    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cudnnGetRNNWorkspaceSize(
        handle, rnn_desc_, seq_length_, x_desc_, &workspace_size_));
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cudnnGetRNNTrainingReserveSize(
            handle, rnn_desc_, seq_length_, x_desc_, reserve_size_));

    workspace_data_.Resize({static_cast<int64_t>(workspace_size_)});
    workspace_data_.mutable_data<uint8_t>(place);
  }

  void release() {
    for (size_t i = 0; i < seq_length_; ++i) {
      PADDLE_ENFORCE_GPU_SUCCESS(
          platform::dynload::cudnnDestroyTensorDescriptor(x_desc_[i]));
      PADDLE_ENFORCE_GPU_SUCCESS(
          platform::dynload::cudnnDestroyTensorDescriptor(y_desc_[i]));
    }

    delete[] x_desc_;
    delete[] y_desc_;

    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cudnnDestroyTensorDescriptor(hx_desc_));
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cudnnDestroyTensorDescriptor(cx_desc_));
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cudnnDestroyTensorDescriptor(hy_desc_));
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cudnnDestroyTensorDescriptor(cy_desc_));
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cudnnDestroyTensorDescriptor(dhx_desc_));
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cudnnDestroyTensorDescriptor(dcx_desc_));
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cudnnDestroyTensorDescriptor(dhy_desc_));
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cudnnDestroyTensorDescriptor(dcy_desc_));

    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cudnnDestroyDropoutDescriptor(dropout_desc_));
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cudnnDestroyRNNDescriptor(rnn_desc_));

    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cudnnDestroyFilterDescriptor(w_desc_));
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cudnnDestroyFilterDescriptor(dw_desc_));
  }
};

}  // namespace operators
}  // namespace paddle
