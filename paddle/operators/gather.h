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
#include <memory.h>
#include <cstring>

#include "paddle/framework/ddim.h"
#include "paddle/framework/eigen.h"
#include "paddle/framework/tensor.h"
#include "paddle/platform/place.h"

namespace paddle {
namespace operators {

// Implementation of CPU copy
template <typename T>
struct CPUGather {
  void operator()(const T* src, const int* indices, const int slice_size,
                  const int index_size, T* output) {
    const size_t slice_bytes = slice_size * sizeof(T);

    for (int i = 0; i < index_size; ++i) {
      int index_ = indices[i];
      memcpy(output + i * slice_size, src + index_ * slice_size, slice_bytes);
    }
  }
};

/**
 * A thin wrapper on cpu tensor
 * Return a new tensor from source tensor, gathered according to index
 * input[src]: type-T source Tensor
 * input[index]: type-int index Tensor (1-D)
 * return: output tensor
 */
template <typename T>
void CPUTGather(const platform::Place& place,
                const paddle::framework::Tensor* src,
                const paddle::framework::Tensor* index,
                paddle::framework::Tensor* output) {
  PADDLE_ENFORCE(platform::is_cpu_place(place));
  // check index of shape 1-D
  PADDLE_ENFORCE(index->dims().size() == 1);
  int index_size = index->dims()[0];

  auto src_dims = src->dims();
  framework::DDim output_dims(src_dims);
  output_dims[0] = index_size;

  // slice size
  int slice_size = 1;
  for (int i = 1; i < src_dims.size(); ++i) slice_size *= src_dims[i];

  // Gathering
  CPUGather<T> gather_functor;
  gather_functor(src->data<T>(), index->data<int>(), slice_size, index_size,
                 output->data<T>());
}

}  // namespace operators
}  // namespace paddle
