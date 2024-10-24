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
#include "paddle/phi/backends/gpu/gpu_dnn.h"
#include "paddle/phi/core/dense_tensor.h"

namespace phi {
namespace funcs {

struct CudnnRNNCache {
  CudnnRNNCache() {
    x_desc_ = NULL;
    y_desc_ = NULL;
  }
  ~CudnnRNNCache() { release(); }

  miopenRNNDescriptor_t rnn_desc_;
  miopenTensorDescriptor_t *x_desc_;
  miopenTensorDescriptor_t *y_desc_;

  miopenTensorDescriptor_t hx_desc_;
  miopenTensorDescriptor_t cx_desc_;
  miopenTensorDescriptor_t hy_desc_;
  miopenTensorDescriptor_t cy_desc_;

  miopenTensorDescriptor_t dhx_desc_;
  miopenTensorDescriptor_t dcx_desc_;
  miopenTensorDescriptor_t dhy_desc_;
  miopenTensorDescriptor_t dcy_desc_;

  miopenTensorDescriptor_t output_x_desc_;
  miopenTensorDescriptor_t output_y_desc_;

  miopenDropoutDescriptor_t dropout_desc_;

  size_t weights_size_;
  miopenTensorDescriptor_t w_desc_;
  miopenTensorDescriptor_t dw_desc_;

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

  void init(miopenHandle_t handle,
            const phi::Place &place,
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
            miopenDataType_t miopen_type) {
    seq_length_ = seq_len;
    batch_size_ = batch_size;
    input_size_ = input_size;
    hidden_size_ = hidden_size;
    num_layers_ = num_layers;
    dropout_prob_ = dropout_prob;
    is_bidirec_ = is_bidirec;
    seed_ = seed;

    const auto numDirections = is_bidirec_ ? 2 : 1;

    PADDLE_ENFORCE_EQ(miopen_type,
                      miopenFloat,
                      common::errors::InvalidArgument(
                          "MIOPEN do not support double datatype."));
    auto miopen_size = sizeof(float);

    x_desc_ = new miopenTensorDescriptor_t[seq_length_];
    y_desc_ = new miopenTensorDescriptor_t[seq_length_];
    std::vector<int> dims = {batch_size_, input_size_, 1};
    std::vector<int> strides = {input_size_, 1, 1};

    std::vector<int> dims_y = {batch_size_, hidden_size_ * numDirections, 1};
    std::vector<int> strides_y = {hidden_size_ * numDirections, 1, 1};

    for (size_t i = 0; i < seq_length_; ++i) {
      PADDLE_ENFORCE_GPU_SUCCESS(
          phi::dynload::miopenCreateTensorDescriptor(&x_desc_[i]));
      PADDLE_ENFORCE_GPU_SUCCESS(
          phi::dynload::miopenCreateTensorDescriptor(&y_desc_[i]));

      PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::miopenSetTensorDescriptor(
          x_desc_[i],
          miopen_type,
          3,
          const_cast<int *>(dims.data()),
          const_cast<int *>(strides.data())));

      PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::miopenSetTensorDescriptor(
          y_desc_[i],
          miopen_type,
          3,
          const_cast<int *>(dims_y.data()),
          const_cast<int *>(strides_y.data())));
    }

    std::vector<int> dims_hx = {
        num_layers_ * numDirections, batch_size_, hidden_size_};
    std::vector<int> strides_hx = {hidden_size_ * batch_size_, hidden_size_, 1};

    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::miopenCreateTensorDescriptor(&hx_desc_));
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::miopenCreateTensorDescriptor(&cx_desc_));
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::miopenCreateTensorDescriptor(&hy_desc_));
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::miopenCreateTensorDescriptor(&cy_desc_));
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::miopenCreateTensorDescriptor(&dhx_desc_));
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::miopenCreateTensorDescriptor(&dcx_desc_));
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::miopenCreateTensorDescriptor(&dhy_desc_));
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::miopenCreateTensorDescriptor(&dcy_desc_));

    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::miopenSetTensorDescriptor(
        hx_desc_,
        miopen_type,
        3,
        const_cast<int *>(dims_hx.data()),
        const_cast<int *>(strides_hx.data())));
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::miopenSetTensorDescriptor(
        cx_desc_,
        miopen_type,
        3,
        const_cast<int *>(dims_hx.data()),
        const_cast<int *>(strides_hx.data())));
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::miopenSetTensorDescriptor(
        hy_desc_,
        miopen_type,
        3,
        const_cast<int *>(dims_hx.data()),
        const_cast<int *>(strides_hx.data())));
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::miopenSetTensorDescriptor(
        cy_desc_,
        miopen_type,
        3,
        const_cast<int *>(dims_hx.data()),
        const_cast<int *>(strides_hx.data())));
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::miopenSetTensorDescriptor(
        dhx_desc_,
        miopen_type,
        3,
        const_cast<int *>(dims_hx.data()),
        const_cast<int *>(strides_hx.data())));
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::miopenSetTensorDescriptor(
        dcx_desc_,
        miopen_type,
        3,
        const_cast<int *>(dims_hx.data()),
        const_cast<int *>(strides_hx.data())));
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::miopenSetTensorDescriptor(
        dhy_desc_,
        miopen_type,
        3,
        const_cast<int *>(dims_hx.data()),
        const_cast<int *>(strides_hx.data())));
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::miopenSetTensorDescriptor(
        dcy_desc_,
        miopen_type,
        3,
        const_cast<int *>(dims_hx.data()),
        const_cast<int *>(strides_hx.data())));

    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::miopenCreateDropoutDescriptor(&dropout_desc_));

    size_t state_size;
    auto *dev_ctx = phi::DeviceContextPool::Instance().Get(place);
    if (!initialized) {
      PADDLE_ENFORCE_GPU_SUCCESS(
          phi::dynload::miopenDropoutGetStatesSize(handle, &state_size));
      dropout_state_->Resize({static_cast<int64_t>(state_size)});
      uint8_t *dropout_state_data = dev_ctx->Alloc<uint8_t>(dropout_state_);
      PADDLE_ENFORCE_GPU_SUCCESS(
          phi::dynload::miopenSetDropoutDescriptor(dropout_desc_,
                                                   handle,
                                                   dropout_prob_,
                                                   dropout_state_data,
                                                   state_size,
                                                   seed_,
                                                   false,
                                                   false,
                                                   MIOPEN_RNG_PSEUDO_XORWOW));
    } else {
      uint8_t *dropout_state_data = dropout_state_->data<uint8_t>();
      auto dropout_state_dims = dropout_state_->dims();
      state_size = dropout_state_dims[0];
      PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::miopenRestoreDropoutDescriptor(
          dropout_desc_,
          handle,
          dropout_prob_,
          dropout_state_data,
          state_size,
          0,
          false,
          false,
          MIOPEN_RNG_PSEUDO_XORWOW));
    }

    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::miopenCreateRNNDescriptor(&rnn_desc_));

    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::miopenSetRNNDescriptor(
        rnn_desc_,
        hidden_size_,
        num_layers_,
        miopenRNNlinear,
        is_bidirec_ ? miopenRNNbidirection : miopenRNNunidirection,
        miopenLSTM,
        miopenRNNNoBias,
        miopenRNNdefault,
        miopen_type));

    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::miopenCreateTensorDescriptor(&w_desc_));
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::miopenCreateTensorDescriptor(&dw_desc_));

    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::miopenGetRNNParamsSize(
        handle, rnn_desc_, x_desc_[0], &weights_size_, miopen_type));

    PADDLE_ENFORCE_EQ(
        weights_size_,
        miopen_size * weight_numel,
        common::errors::InvalidArgument(
            "The miopen lstm and setting weight size should be same."));

    int dim_w[3];
    dim_w[0] = weights_size_ / miopen_size;
    dim_w[1] = 1;
    dim_w[2] = 1;

    int dim_s[2];
    dim_s[1] = 1;
    dim_s[0] = dim_w[1];

    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::miopenSetTensorDescriptor(
        w_desc_, miopen_type, 3, dim_w, dim_s));
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::miopenSetTensorDescriptor(
        dw_desc_, miopen_type, 3, dim_w, dim_s));

    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::miopenGetRNNWorkspaceSize(
        handle, rnn_desc_, seq_length_, x_desc_, &workspace_size_));
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::miopenGetRNNTrainingReserveSize(
        handle, rnn_desc_, seq_length_, x_desc_, reserve_size_));

    workspace_data_.Resize({static_cast<int64_t>(workspace_size_)});
    dev_ctx->Alloc<uint8_t>(&workspace_data_);
  }

  void release() {
    for (size_t i = 0; i < seq_length_; ++i) {
      PADDLE_ENFORCE_GPU_SUCCESS(
          phi::dynload::miopenDestroyTensorDescriptor(x_desc_[i]));
      PADDLE_ENFORCE_GPU_SUCCESS(
          phi::dynload::miopenDestroyTensorDescriptor(y_desc_[i]));
    }

    delete[] x_desc_;
    delete[] y_desc_;

    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::miopenDestroyTensorDescriptor(hx_desc_));
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::miopenDestroyTensorDescriptor(cx_desc_));
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::miopenDestroyTensorDescriptor(hy_desc_));
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::miopenDestroyTensorDescriptor(cy_desc_));
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::miopenDestroyTensorDescriptor(dhx_desc_));
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::miopenDestroyTensorDescriptor(dcx_desc_));
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::miopenDestroyTensorDescriptor(dhy_desc_));
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::miopenDestroyTensorDescriptor(dcy_desc_));

    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::miopenDestroyDropoutDescriptor(dropout_desc_));
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::miopenDestroyRNNDescriptor(rnn_desc_));

    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::miopenDestroyTensorDescriptor(w_desc_));
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::miopenDestroyTensorDescriptor(dw_desc_));
  }
};

}  // namespace funcs
}  // namespace phi
