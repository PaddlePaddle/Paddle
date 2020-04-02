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

#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/platform/cudnn_helper.h"

namespace paddle {
namespace operators {

struct CudnnRNNCache {
  CudnnRNNCache() {
    x_desc_ = NULL;
    y_desc_ = NULL;
    dx_desc_ = NULL;
    dy_desc_ = NULL;
  }
  ~CudnnRNNCache() { release(); }

  cudnnRNNDescriptor_t rnn_desc_;
  cudnnTensorDescriptor_t *x_desc_;
  cudnnTensorDescriptor_t *y_desc_;
  cudnnTensorDescriptor_t *dx_desc_;
  cudnnTensorDescriptor_t *dy_desc_;

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
  size_t reserve_size_;
  framework::Tensor reserve_data_;
  framework::Tensor workspace_data_;

  framework::Tensor dropout_state_;

  size_t max_length_;

  float dropout_prob_;
  bool is_bidirec_;

  int batch_size_;
  int input_size_;
  int hidden_size_;
  int num_layers_;
  int seed_;

  void init(cudnnHandle_t handle, const platform::Place &place, size_t max_len,
            int batch_size, int input_size, int hidden_size, int num_layers,
            float dropout_prob, bool is_bidirec, int seed, int weight_numel) {
    max_length_ = max_len;
    batch_size_ = batch_size;
    input_size_ = input_size;
    hidden_size_ = hidden_size;
    num_layers_ = num_layers;
    dropout_prob_ = dropout_prob;
    is_bidirec_ = is_bidirec;
    seed_ = seed;

    x_desc_ = new cudnnTensorDescriptor_t[max_length_];
    y_desc_ = new cudnnTensorDescriptor_t[max_length_];
    dx_desc_ = new cudnnTensorDescriptor_t[max_length_];
    dy_desc_ = new cudnnTensorDescriptor_t[max_length_];
    int dim_a[3];
    int stride_a[3];

    for (size_t i = 0; i < max_length_; ++i) {
      PADDLE_ENFORCE_CUDA_SUCCESS(
          platform::dynload::cudnnCreateTensorDescriptor(&x_desc_[i]));
      PADDLE_ENFORCE_CUDA_SUCCESS(
          platform::dynload::cudnnCreateTensorDescriptor(&y_desc_[i]));
      PADDLE_ENFORCE_CUDA_SUCCESS(
          platform::dynload::cudnnCreateTensorDescriptor(&dx_desc_[i]));
      PADDLE_ENFORCE_CUDA_SUCCESS(
          platform::dynload::cudnnCreateTensorDescriptor(&dy_desc_[i]));
      dim_a[0] = batch_size_;
      dim_a[1] = input_size_;
      dim_a[2] = 1;

      stride_a[0] = dim_a[2] * dim_a[1];
      stride_a[1] = dim_a[2];
      stride_a[2] = 1;
      PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnSetTensorNdDescriptor(
          x_desc_[i], CUDNN_DATA_FLOAT, 3, dim_a, stride_a));
      PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnSetTensorNdDescriptor(
          dx_desc_[i], CUDNN_DATA_FLOAT, 3, dim_a, stride_a));

      dim_a[0] = batch_size_;
      dim_a[1] = is_bidirec_ ? hidden_size_ * 2 : hidden_size_;
      dim_a[2] = 1;

      stride_a[0] = dim_a[2] * dim_a[1];
      stride_a[1] = dim_a[2];
      stride_a[2] = 1;

      PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnSetTensorNdDescriptor(
          y_desc_[i], CUDNN_DATA_FLOAT, 3, dim_a, stride_a));
      PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnSetTensorNdDescriptor(
          dy_desc_[i], CUDNN_DATA_FLOAT, 3, dim_a, stride_a));
    }

    dim_a[0] = num_layers_ * (is_bidirec_ ? 2 : 1);
    dim_a[1] = batch_size_;
    dim_a[2] = hidden_size_;

    stride_a[0] = dim_a[2] * dim_a[1];
    stride_a[1] = dim_a[2];
    stride_a[2] = 1;

    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cudnnCreateTensorDescriptor(&hx_desc_));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cudnnCreateTensorDescriptor(&cx_desc_));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cudnnCreateTensorDescriptor(&hy_desc_));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cudnnCreateTensorDescriptor(&cy_desc_));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cudnnCreateTensorDescriptor(&dhx_desc_));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cudnnCreateTensorDescriptor(&dcx_desc_));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cudnnCreateTensorDescriptor(&dhy_desc_));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cudnnCreateTensorDescriptor(&dcy_desc_));

    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnSetTensorNdDescriptor(
        hx_desc_, CUDNN_DATA_FLOAT, 3, dim_a, stride_a));
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnSetTensorNdDescriptor(
        cx_desc_, CUDNN_DATA_FLOAT, 3, dim_a, stride_a));
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnSetTensorNdDescriptor(
        hy_desc_, CUDNN_DATA_FLOAT, 3, dim_a, stride_a));
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnSetTensorNdDescriptor(
        cy_desc_, CUDNN_DATA_FLOAT, 3, dim_a, stride_a));
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnSetTensorNdDescriptor(
        dhx_desc_, CUDNN_DATA_FLOAT, 3, dim_a, stride_a));
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnSetTensorNdDescriptor(
        dcx_desc_, CUDNN_DATA_FLOAT, 3, dim_a, stride_a));
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnSetTensorNdDescriptor(
        dhy_desc_, CUDNN_DATA_FLOAT, 3, dim_a, stride_a));
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnSetTensorNdDescriptor(
        dcy_desc_, CUDNN_DATA_FLOAT, 3, dim_a, stride_a));

    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cudnnCreateDropoutDescriptor(&dropout_desc_));

    size_t state_size;
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cudnnDropoutGetStatesSize(handle, &state_size));
    dropout_state_.Resize({static_cast<int64_t>(state_size)});
    auto *dropout_state_data = dropout_state_.mutable_data<uint8_t>(place);
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnSetDropoutDescriptor(
        dropout_desc_, handle, dropout_prob_, dropout_state_data, state_size,
        seed_));

    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cudnnCreateRNNDescriptor(&rnn_desc_));

#if CUDNN_VERSION >= 6000
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnSetRNNDescriptor_v6(
        handle, rnn_desc_, hidden_size_, num_layers_, dropout_desc_,
        CUDNN_LINEAR_INPUT,
        is_bidirec_ ? CUDNN_BIDIRECTIONAL : CUDNN_UNIDIRECTIONAL, CUDNN_LSTM,
        CUDNN_RNN_ALGO_STANDARD, CUDNN_DATA_FLOAT));
#else
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnSetRNNDescriptor(
        rnn_desc_, hidden_size_, num_layers_, dropout_desc_, CUDNN_LINEAR_INPUT,
        is_bidirec_ ? CUDNN_BIDIRECTIONAL : CUDNN_UNIDIRECTIONAL, CUDNN_LSTM,
        CUDNN_DATA_FLOAT));
#endif

    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cudnnCreateFilterDescriptor(&w_desc_));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cudnnCreateFilterDescriptor(&dw_desc_));

    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnGetRNNParamsSize(
        handle, rnn_desc_, x_desc_[0], &weights_size_, CUDNN_DATA_FLOAT));

    PADDLE_ENFORCE_EQ(weights_size_, sizeof(float) * weight_numel,
                      "cudnn lstm weight size should be SAME");
    int dim_w[3];
    dim_w[0] = weights_size_ / sizeof(float);
    dim_w[1] = 1;
    dim_w[2] = 1;
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnSetFilterNdDescriptor(
        w_desc_, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 3, dim_w));
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnSetFilterNdDescriptor(
        dw_desc_, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 3, dim_w));

    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnGetRNNWorkspaceSize(
        handle, rnn_desc_, max_length_, x_desc_, &workspace_size_));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cudnnGetRNNTrainingReserveSize(
            handle, rnn_desc_, max_length_, x_desc_, &reserve_size_));

    reserve_data_.Resize({static_cast<int64_t>(reserve_size_)});
    reserve_data_.mutable_data<uint8_t>(place);

    workspace_data_.Resize({static_cast<int64_t>(workspace_size_)});
    workspace_data_.mutable_data<uint8_t>(place);
  }

  void release() {
    for (size_t i = 0; i < max_length_; ++i) {
      PADDLE_ENFORCE_CUDA_SUCCESS(
          platform::dynload::cudnnDestroyTensorDescriptor(x_desc_[i]));
      PADDLE_ENFORCE_CUDA_SUCCESS(
          platform::dynload::cudnnDestroyTensorDescriptor(y_desc_[i]));
      PADDLE_ENFORCE_CUDA_SUCCESS(
          platform::dynload::cudnnDestroyTensorDescriptor(dx_desc_[i]));
      PADDLE_ENFORCE_CUDA_SUCCESS(
          platform::dynload::cudnnDestroyTensorDescriptor(dy_desc_[i]));
    }

    delete[] x_desc_;
    delete[] y_desc_;
    delete[] dx_desc_;
    delete[] dy_desc_;

    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cudnnDestroyTensorDescriptor(hx_desc_));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cudnnDestroyTensorDescriptor(cx_desc_));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cudnnDestroyTensorDescriptor(hy_desc_));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cudnnDestroyTensorDescriptor(cy_desc_));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cudnnDestroyTensorDescriptor(dhx_desc_));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cudnnDestroyTensorDescriptor(dcx_desc_));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cudnnDestroyTensorDescriptor(dhy_desc_));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cudnnDestroyTensorDescriptor(dcy_desc_));

    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cudnnDestroyDropoutDescriptor(dropout_desc_));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cudnnDestroyRNNDescriptor(rnn_desc_));

    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cudnnDestroyFilterDescriptor(w_desc_));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cudnnDestroyFilterDescriptor(dw_desc_));
  }
};

}  // namespace operators
}  // namespace paddle
