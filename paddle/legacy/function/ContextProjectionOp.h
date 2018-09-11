/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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
#include "Function.h"

namespace paddle {

/**
 * \brief   Context Projection Forward.
 *
 * \param[in/out]  outputs           output data.
 * \param[in]      input             input data.
 * \param[in]      weight            input weight.
 * \param[in]      sequence          input data.
 * \param[in]      context_length    consecutive rows for concatenation.
 * \param[in]      context_start     context start position.
 * \param[in]      begin_pad         begining pad position.
 * \param[in]      is_padding        whether padding 0 or not.
 *
 */
template <DeviceType DType>
void ContextProjectionForward(
    typename Tensor<real, DType>::Matrix& output,
    const typename Tensor<real, DType>::Matrix& input,
    const typename Tensor<real, DType>::Matrix& weight,
    const typename Tensor<int, DType>::Vector& sequence,
    size_t context_length,
    int context_start,
    size_t begin_pad);

/**
 * \brief   Context Projection Backward.
 *
 * \param[out]  outputs           output gradient.
 * \param[in]   input             input gradient.
 * \param[in]   weight            input weight gradient.
 * \param[in]   sequence          input data.
 * \param[in]   context_length    consecutive rows for concatenation.
 * \param[in]   context_start     context start position.
 * \param[in]   begin_pad         begining pad position.
 * \param[in]   is_padding        whether padding 0 or not.
 *
 */
template <DeviceType DType>
void ContextProjectionBackward(
    const typename Tensor<real, DType>::Matrix& out_grad,
    typename Tensor<real, DType>::Matrix& in_grad,
    typename Tensor<real, DType>::Matrix& w_grad,
    const typename Tensor<int, DType>::Vector& seq_vec,
    size_t context_length,
    int context_start,
    size_t begin_pad,
    bool is_padding,
    size_t total_pad);

template <DeviceType DType>
void ContextProjectionBackwardData(
    const typename Tensor<real, DType>::Matrix& out_grad,
    typename Tensor<real, DType>::Matrix& in_grad,
    const typename Tensor<int, DType>::Vector& sequence,
    size_t context_length,
    int context_start);

template <DeviceType DType>
void ContextProjectionBackwardWeight(
    const typename Tensor<real, DType>::Matrix& out_grad,
    typename Tensor<real, DType>::Matrix& w_grad,
    const typename Tensor<int, DType>::Vector& seq_vec,
    size_t context_length,
    int context_start,
    size_t total_pad,
    size_t begin_pad);

}  // namespace paddle
