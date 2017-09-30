/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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
#include <cstring>

#include "paddle/framework/ddim.h"
#include "paddle/framework/eigen.h"
#include "paddle/framework/tensor.h"
#include "paddle/platform/place.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenVector = framework::EigenVector<T, MajorType, IndexType>;

// Implementation of CPU copy
template <typename T>
void CPUScatterUpdate(const paddle::framework::Tensor* src, const int* index,
                      const size_t index_size,
                      paddle::framework::Tensor* output) {
  paddle::framework::DDim output_dims = output->dims();

  for (size_t i = 0; i < index_size; ++i) {
    int index_ = index[i];

    paddle::framework::Tensor src_ = *src;
    paddle::framework::Tensor output_ = *output;
    if (index_size > 1) src_ = src->Slice<T>(i, i + 1);
    if (output_dims[0] > 1) output_ = output->Slice<T>(index_, index_ + 1);

    auto X = EigenVector<T>::Flatten(src_);
    auto Y = EigenVector<T>::Flatten(output_);

    Y = X + Y;
  }
}

// Implementation of GPU scatter:
template <typename T>
void GPUScatterUpdate(const T* src, const int* index, const int slice_size,
                      const int index_size, T* output);

/**
 * Return a updated tensor from source tensor, scattered according to index:
 * dst[i] += src[index[i]]
 * input[src]: type-T source Tensor
 * input[index]: type-int index Tensor (1-D)
 * return: output tensor
 */
template <typename T>
void ScatterUpdate(const platform::Place& place,
                   const paddle::framework::Tensor* src,
                   const paddle::framework::Tensor* index,
                   paddle::framework::Tensor* output) {
  // check index of shape 1-D
  PADDLE_ENFORCE(index->dims().size() == 1);
  int index_size = index->dims()[0];

  auto src_dims = src->dims();
  auto dst_dims = output->dims();

  // check src shape and dst shape should match
  for (int i = 1; i < src_dims.size(); i++)
    PADDLE_ENFORCE(src_dims[i] == dst_dims[i]);

  if (platform::is_cpu_place(place)) {
    CPUScatterUpdate<T>(src, index->data<int>(), index_size, output);
  } else {
  }
}

}  // namespace operators
}  // namespace paddle
