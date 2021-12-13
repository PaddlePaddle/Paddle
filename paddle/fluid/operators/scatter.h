/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

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
                      const T* src_pointer, T* dst_pointer, size_t src_index,
                      IndexT dst_index, size_t slice_size) {
  auto blas = math::GetBlas<platform::CPUDeviceContext, T>(ctx);
  blas.VADD(slice_size, src_pointer + src_index * slice_size,
            dst_pointer + dst_index * slice_size,
            dst_pointer + dst_index * slice_size);
}

template <typename T, typename IndexT = int>
typename std::enable_if<!std::is_floating_point<T>::value>::type
elementwise_inner_add(const framework::ExecutionContext& ctx,
                      const T* src_pointer, T* dst_pointer, size_t src_index,
                      IndexT dst_index, size_t slice_size) {
  using EigenVector = typename framework::EigenTensor<T, 1>::Type;
  using ConstEigenVector = typename framework::EigenTensor<T, 1>::ConstType;

  framework::EigenDim<1>::Type dim;
  dim[0] = slice_size;

  ConstEigenVector eigen_src(src_pointer + src_index * slice_size, dim);
  EigenVector eigen_dst(dst_pointer + dst_index * slice_size, dim);
  eigen_dst += eigen_src;
}

/**
 * Return an updated tensor from source tensor, scattered according to index:
 * dst[i] = src[index[i]]
 * input[src]: type-T source Tensor
 * input[index]: type-IndexT index Tensor (1-D)
 * return: output tensor
 */
template <typename T, typename IndexT = int>
void ScatterAssign(const platform::DeviceContext& ctx, const Tensor& src,
                   const Tensor& index, Tensor* output) {
  PADDLE_ENFORCE_EQ(
      platform::is_cpu_place(ctx.GetPlace()), true,
      platform::errors::PreconditionNotMet("This kernel only runs on CPU."));
  // check index of shape 1-D
  if (index.dims().size() == 2) {
    PADDLE_ENFORCE_EQ(index.dims()[1], 1,
                      platform::errors::InvalidArgument(
                          "index.dims()[1] should be 1 when "
                          "index.dims().size() =2 in scatter_op."
                          "But received value is [%d]",
                          index.dims()[1]));
  } else {
    PADDLE_ENFORCE_EQ(index.dims().size(), 1,
                      platform::errors::InvalidArgument(
                          "index.dims().size() should be 1 or 2 in scatter_op."
                          "But received value is [%d]",
                          index.dims().size()));
  }
  int64_t index_size = index.dims()[0];

  auto src_dims = src.dims();
  auto dst_dims = output->dims();

  const T* p_src = src.data<T>();
  const IndexT* p_index = index.data<IndexT>();
  T* p_output = output->data<T>();

  // check src shape and dst shape should match
  for (int i = 1; i < src_dims.size(); i++)
    PADDLE_ENFORCE_EQ(
        src_dims[i], dst_dims[i],
        platform::errors::InvalidArgument(
            "The dimensions of the source tensor and target tensor should"
            " match, but received source tensor's %d-th dimension is %d,"
            "target tensor's %d-th dimension is %d.",
            i, src_dims[i], i, dst_dims[i]));

  // slice size
  size_t slice_size = 1;
  for (int i = 1; i < src_dims.size(); ++i) slice_size *= src_dims[i];

  const size_t slice_bytes = slice_size * sizeof(T);

  for (int64_t i = 0; i < index_size; ++i) {
    IndexT index_ = p_index[i];

    PADDLE_ENFORCE_GE(index_, 0,
                      platform::errors::OutOfRange(
                          "The index is out of bounds, "
                          "please check whether the dimensions of index and "
                          "input meet the requirements. It should "
                          "be greater than or equal to 0, but received [%d]",
                          index_));

    memcpy(p_output + index_ * slice_size, p_src + i * slice_size, slice_bytes);
  }
}

template <typename T, typename IndexT = int>
void ScatterAssignAdd(const framework::ExecutionContext& ctx, const Tensor& src,
                      const Tensor& index, Tensor* output) {
  PADDLE_ENFORCE_EQ(
      platform::is_cpu_place(ctx.device_context().GetPlace()), true,
      platform::errors::PreconditionNotMet("This kernel only runs on CPU."));
  // check index of shape 1-D
  PADDLE_ENFORCE_EQ(
      index.dims().size() == 1 ||
          (index.dims().size() == 2 && index.dims()[1] == 1),
      true, platform::errors::InvalidArgument(
                "index's shape is error, "
                "expect index'dims shape is 1 or 2 and index.dims[1] is 1"
                "but got index'dims shape is %d",
                index.dims().size()));
  int64_t index_size = index.dims()[0];

  auto src_dims = src.dims();
  auto dst_dims = output->dims();

  const T* p_src = src.data<T>();
  const IndexT* p_index = index.data<IndexT>();

  T* p_output = output->data<T>();

  // check src shape and dst shape should match
  for (int i = 1; i < src_dims.size(); i++)
    PADDLE_ENFORCE_EQ(
        src_dims[i], dst_dims[i],
        platform::errors::InvalidArgument(
            "The dimensions of the source tensor and target tensor should"
            " match, but received source tensor's %d-th dimension is %d,"
            "target tensor's %d-th dimension is %d.",
            i, src_dims[i], i, dst_dims[i]));

  // slice size
  size_t slice_size = 1;
  for (int i = 1; i < src_dims.size(); ++i) slice_size *= src_dims[i];

  const size_t& slice_bytes = slice_size * sizeof(T);

  // if not in overwrite mode, need to init output data
  auto max_index = dst_dims[0];
  for (int64_t i = 0; i < index_size; ++i) {
    const IndexT& index_val = p_index[i];
    PADDLE_ENFORCE_GE(index_val, 0,
                      platform::errors::OutOfRange(
                          "The index is out of bounds, "
                          "please check whether the dimensions of index and "
                          "input meet the requirements. It should "
                          "be greater than or equal to 0, but received [%d]",
                          index_val));
    PADDLE_ENFORCE_LT(index_val, max_index,
                      platform::errors::OutOfRange(
                          "The index is out of bounds, "
                          "please check whether the dimensions of index and "
                          "input meet the requirements. It should "
                          "be less than %d, but received %d",
                          max_index, index_val));
    memset(p_output + slice_size * index_val, 0, slice_bytes);
  }

  // if not in overwrite mode, need to init output data
  for (int64_t i = 0; i < index_size; ++i) {
    const IndexT& index_val = p_index[i];
    elementwise_inner_add<T, IndexT>(ctx, p_src, p_output, i, index_val,
                                     slice_size);
  }
}

// The function is only for scatter grad x,
// however update grad use gather
template <typename T, typename IndexT = int>
void CPUScatterGradForX(const platform::DeviceContext& ctx, const Tensor& index,
                        Tensor* output) {
  int64_t index_size = index.dims()[0];
  auto dst_dims = output->dims();
  const IndexT* p_index = index.data<IndexT>();
  T* p_output = output->data<T>();
  size_t slice_size = 1;
  for (int i = 1; i < dst_dims.size(); ++i) slice_size *= dst_dims[i];
  const size_t slice_bytes = slice_size * sizeof(T);
  for (int64_t i = 0; i < index_size; ++i) {
    const IndexT& index_ = p_index[i];
    memset(p_output + slice_size * index_, 0, slice_bytes);
  }
}

template <typename T, typename IndexT = int>
void ScatterNdAdd(const framework::ExecutionContext& ctx, const Tensor& update,
                  const Tensor& index, Tensor* output) {
  PADDLE_ENFORCE_EQ(
      platform::is_cpu_place(ctx.device_context().GetPlace()), true,
      platform::errors::PreconditionNotMet("It should be running on the CPU"));

  // update.shape = index.shape[:-1] + output.shape[index.shape[-1]:]
  auto index_dims = index.dims();
  auto index_dims_size = index_dims.size();

  auto output_dims = output->dims();
  auto output_dims_size = output_dims.size();

  const T* p_update = update.data<T>();
  const IndexT* p_index = index.data<IndexT>();
  T* p_output = output->data<T>();

  // final dim
  int64_t end_size = index_dims[index_dims_size - 1];
  // remain dim
  auto remain_ddim = framework::slice_ddim(index_dims, 0, index_dims_size - 1);
  int64_t remain_numel = framework::product(remain_ddim);
  // slice size
  int64_t slice_size = 1;
  for (int64_t i = end_size; i < output_dims_size; ++i) {
    slice_size *= output_dims[i];
  }

  for (int64_t i = 0; i < remain_numel; ++i) {
    IndexT index_val = 0;
    IndexT temp = 1;
    for (int64_t j = end_size - 1; j >= 0; --j) {
      IndexT index_value = p_index[i * end_size + j];
      PADDLE_ENFORCE_EQ(
          (index_value >= 0 && index_value < output_dims[j]), true,
          platform::errors::OutOfRange(
              "The index is out of bounds, "
              "please check whether the dimensions of index and "
              "input meet the requirements. It should "
              "be less than [%d] and greater or equal to 0, but received [%d]",
              output_dims[j], index_value));

      index_val += (index_value * temp);
      temp *= output_dims[j];
    }
    elementwise_inner_add<T, IndexT>(ctx, p_update, p_output, i, index_val,
                                     slice_size);
  }
}

}  // namespace operators
}  // namespace paddle
