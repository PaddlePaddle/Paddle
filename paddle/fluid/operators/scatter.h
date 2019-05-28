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
#include <cstring>
#include <string>

#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/platform/place.h"
#include "unordered_set"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

/**
  * Return the updated array pointer, use blas or eigen lib to optimize time
 * cost
 */

template <typename T, typename IndexT = int>
typename std::enable_if<std::is_floating_point<T>::value>::type
elementwise_inner_add(const framework::ExecutionContext& ctx,
                      const framework::Tensor& src, framework::Tensor* dist,
                      const int& src_index, const IndexT& dist_index,
                      const int& slice_size, const size_t& slice_bytes) {
  // use blas lib to add
  auto blas = math::GetBlas<platform::CPUDeviceContext, T>(ctx);

  // new array to save tmp add result
  T* tmp_add_pointer = new T[slice_size];
  const T* src_pointer = src.data<T>();
  const T* dist_pointer = dist->data<T>();

  blas.VADD(slice_size, src_pointer + src_index * slice_size,
            dist_pointer + dist_index * slice_size, tmp_add_pointer);

  memcpy(dist->data<T>() + dist_index * slice_size, tmp_add_pointer,
         slice_bytes);
  // free tmp memory
  delete[] tmp_add_pointer;
}

template <typename T, typename IndexT = int>
typename std::enable_if<!std::is_floating_point<T>::value>::type
elementwise_inner_add(const framework::ExecutionContext& ctx,
                      const framework::Tensor& src, framework::Tensor* dist,
                      const int& src_index, const IndexT& dist_index,
                      const int& slice_size, const size_t& slice_bytes) {
  auto src_slice = src.Slice(src_index, src_index + 1);
  auto dist_slice = dist->Slice(dist_index, dist_index + 1);

  auto eigen_src = framework::EigenVector<T>::Flatten(src_slice);
  auto eigen_dist = framework::EigenVector<T>::Flatten(dist_slice);

  eigen_dist += eigen_src;
}

/**
 * Return a updated tensor from source tensor, scattered according to index:
 * dst[i] = src[index[i]]
 * input[src]: type-T source Tensor
 * input[index]: type-IndexT index Tensor (1-D)
 * return: output tensor
 */
template <typename T, typename IndexT = int>
void ScatterAssign(const platform::DeviceContext& ctx, const Tensor& src,
                   const Tensor& index, Tensor* output) {
  PADDLE_ENFORCE(platform::is_cpu_place(ctx.GetPlace()));
  // check index of shape 1-D
  PADDLE_ENFORCE(index.dims().size() == 1 ||
                 (index.dims().size() == 2 && index.dims()[1] == 1));
  int index_size = index.dims()[0];

  auto src_dims = src.dims();
  auto dst_dims = output->dims();

  const T* p_src = src.data<T>();
  const IndexT* p_index = index.data<IndexT>();
  T* p_output = output->data<T>();

  // check src shape and dst shape should match
  for (int i = 1; i < src_dims.size(); i++)
    PADDLE_ENFORCE(src_dims[i] == dst_dims[i]);

  // slice size
  size_t slice_size = 1;
  for (int i = 1; i < src_dims.size(); ++i) slice_size *= src_dims[i];

  const size_t slice_bytes = slice_size * sizeof(T);

  for (int i = 0; i < index_size; ++i) {
    IndexT index_ = p_index[i];
    memcpy(p_output + index_ * slice_size, p_src + i * slice_size, slice_bytes);
  }
}

template <typename T, typename IndexT = int>
void ScatterAssignAdd(const framework::ExecutionContext& ctx, const Tensor& src,
                      const Tensor& index, Tensor* output) {
  PADDLE_ENFORCE(platform::is_cpu_place(ctx.device_context().GetPlace()));
  // check index of shape 1-D
  PADDLE_ENFORCE(index.dims().size() == 1 ||
                 (index.dims().size() == 2 && index.dims()[1] == 1));
  int index_size = index.dims()[0];

  auto src_dims = src.dims();
  auto dst_dims = output->dims();

  const T* p_src = src.data<T>();
  const IndexT* p_index = index.data<IndexT>();

  bool overwrite = true;

  // if find duplicate index, can not use overwrite mode
  std::unordered_set<IndexT> index_count;
  for (int i = 0; i < index_size; ++i) {
    if (index_count.count(p_index[i]) > 0) {
      overwrite = false;
      break;
    } else {
      index_count.emplace(p_index[i]);
    }
  }

  T* p_output = output->data<T>();

  // check src shape and dst shape should match
  for (int i = 1; i < src_dims.size(); i++)
    PADDLE_ENFORCE(src_dims[i] == dst_dims[i]);

  // slice size
  size_t slice_size = 1;
  for (int i = 1; i < src_dims.size(); ++i) slice_size *= src_dims[i];

  // if in not in overwrite mode, need init output
  if (!overwrite) {
    memset(p_output, 0, output->numel() * sizeof(T));
  }

  const size_t& slice_bytes = slice_size * sizeof(T);

  for (int i = 0; i < index_size; ++i) {
    const IndexT& index_ = p_index[i];

    // if in overwrite mode, can copy data directly
    if (overwrite) {
      memcpy(p_output + index_ * slice_size, p_src + i * slice_size,
             slice_bytes);
    } else {
      elementwise_inner_add<T, IndexT>(ctx, src, output, i, index_, slice_size,
                                       slice_bytes);
    }
  }
}

}  // namespace operators
}  // namespace paddle
