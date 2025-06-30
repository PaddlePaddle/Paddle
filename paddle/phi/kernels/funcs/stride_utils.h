// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include <vector>
#include "paddle/common/array.h"
#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/dense_tensor.h"

#if defined(__NVCC__) || defined(__HIPCC__)
#ifdef __NVCC__
#include <cuda.h>
#include <cuda_runtime.h>
#elif defined(__HIPCC__)
#include <hip/hip_runtime.h>
#endif
#endif

namespace phi {

namespace funcs {

static inline std::vector<int64_t> infer_size_dimvector(
    std::vector<int64_t> a, std::vector<int64_t> b) {
  // Use ptrdiff_t to ensure signed comparison.
  auto dimsA = a.size();
  auto dimsB = b.size();
  auto ndim = dimsA > dimsB ? dimsA : dimsB;
  std::vector<int64_t> expandedSizes = std::vector<int64_t>(ndim, 0);

  for (int64_t i = ndim - 1; i >= 0; --i) {
    int64_t offset = ndim - 1 - i;
    int64_t dimA = dimsA - 1 - offset;
    int64_t dimB = dimsB - 1 - offset;
    auto sizeA = (dimA >= 0) ? a[dimA] : 1;
    auto sizeB = (dimB >= 0) ? b[dimB] : 1;

    expandedSizes[i] = sizeA == 1 ? sizeB : sizeA;
  }

  return expandedSizes;
}

static inline std::vector<int64_t> compute_strides(
    const std::vector<int64_t> input_dims,  // value_tensor
    const std::vector<int64_t> input_strides,
    const int64_t input_elesize,
    const int64_t ndim,
    const std::vector<int64_t>* shape_,
    std::vector<int64_t>* stride_size) {
  std::vector<int64_t> stride_bytes(ndim, 0);
  const auto& original_shape = input_dims;
  const auto& original_stride = input_strides;
  int64_t element_size_in_bytes = input_elesize;
  int offset = ndim - original_shape.size();

  if (offset > 0)
    stride_bytes.resize(ndim, 0);
  else
    stride_bytes.resize(ndim);
  for (int i = 0; i < original_shape.size(); i++) {
    if (original_shape[i] == 1 && (*shape_)[offset + i] != 1) {
      stride_bytes[offset + i] = 0;
    } else {
      stride_bytes[offset + i] = original_stride[i] * element_size_in_bytes;
    }
  }
  stride_size->push_back(stride_bytes.size());
  return stride_bytes;
}

static inline std::vector<int64_t> compute_shapes(
    std::vector<std::vector<int64_t>> input_dims) {
  std::vector<int64_t> shape_;
  for (size_t i = 0; i < input_dims.size(); i++) {
    auto shape = input_dims[i];
    if (shape_.empty()) {
      shape_ = shape;
    } else if (!(shape == shape_)) {
      shape_ = infer_size_dimvector(shape_, shape);
    }
  }
  return shape_;
}

template <int N>
static inline void permute_dimensions(const std::vector<int64_t> stride_size,
                                      const std::vector<int64_t> perm,
                                      std::array<int64_t*, N>* strides_array,
                                      std::vector<int64_t>* shape_) {
  auto reorder = [perm](std::vector<int64_t> data) {
    auto res = std::vector<int64_t>(data.size(), 0);
    for (int64_t i = 0; i < perm.size(); i++) {
      res[i] = data[perm[i]];
    }
    return res;
  };

  // Update shape and strides
  *shape_ = reorder(*shape_);
  static std::array<std::vector<int64_t>, N> temp_strides;
  for (int64_t i = 0; i < N; i++) {
    if ((*strides_array)[i] != nullptr) {
      std::vector<int64_t> original_data((*strides_array)[i],
                                         (*strides_array)[i] + stride_size[i]);
      temp_strides[i] = reorder(original_data);
      (*strides_array)[i] = temp_strides[i].data();
    }
  }
}

template <int N>
static inline void reorder_dimensions(const std::vector<int64_t> stride_size,
                                      std::vector<int64_t>* shape_,
                                      std::array<int64_t*, N>* strides_array) {
  // Sort the dimensions based on strides in ascending order with reduced dims
  // at the front. NOTE: that this inverts the order of C-contiguous tensors.
  // strides[0] is the fastest moving dimension instead of strides[ndim - 1].
  // See NOTE: [Computing output strides] and inline  comments for more detailed
  // description
  auto ndim = shape_->size();
  std::vector<int64_t> perm_;

  perm_.resize(ndim);
  if (ndim == 1) {
    perm_[0] = 0;
    return;
  }

  // initialize perm with n-1, n-2, ..., 1, 0
  std::iota(perm_.rbegin(), perm_.rend(), 0);
  // returns 1 if the dim0 should come after dim1, -1 if dim0 should come
  // before dim1, and 0 if the comparison is ambiguous.
  auto should_swap = [&](size_t dim0, size_t dim1) {
    for (int64_t arg = 0; arg < N; arg++) {
      // ignore undefined or incorrectly sized tensors
      if ((*strides_array)[arg] == nullptr) {
        continue;
      }
      int64_t stride0 = (*strides_array)[arg][dim0];
      int64_t stride1 = (*strides_array)[arg][dim1];
      // move on to the next input if one of the dimensions is broadcasted
      if (stride0 == 0 || stride1 == 0) {
        continue;
        // it is important to return here only with strict comparisons, for
        // equal strides we try to break the tie later by comparing
        // corresponding dimensions or if that does not work, moving on to the
        // next tensor
      } else if (stride0 < stride1) {
        return -1;
      } else if (stride0 > stride1) {
        return 1;
      } else {  // equal strides, use dimensions themselves as the tie-breaker.
        // at this point, with zero strides out of the way, we are guaranteed
        // that operand dimensions are equal to shape_
        auto t_dim0 = (*shape_)[dim0];
        auto t_dim1 = (*shape_)[dim1];
        // return only if dimensions should be swapped, otherwise move on to the
        // next tensor
        if (t_dim0 > t_dim1) {
          return 1;
        }
      }
    }
    return 0;
  };
  // insertion sort with support for ambiguous comparisons
  for (int64_t i = 0; i < ndim; i++) {
    int dim1 = i;
    for (int dim0 = i - 1; dim0 >= 0; dim0--) {
      int comparison = should_swap(perm_[dim0], perm_[dim1]);
      if (comparison > 0) {
        std::swap(perm_[dim0], perm_[dim1]);
        dim1 = dim0;
      } else if (comparison < 0) {
        break;
      }
    }
  }

  // perform re-ordering of shape and strides
  permute_dimensions<N>(stride_size, perm_, strides_array, shape_);
}

static inline std::vector<int64_t> compatible_stride(
    const std::vector<int64_t>* shape_,
    const int64_t ndim,
    const int64_t element_size) {
  std::vector<int64_t> stride;
  int64_t next_stride = element_size;

  for (int64_t dim = 0; dim < ndim; ++dim) {
    stride.push_back(next_stride);
    next_stride *= (*shape_)[dim];
  }
  return stride;
}

template <int N>
static inline void allocate_or_resize_outputs(
    const std::vector<int64_t>* shape_,
    const int64_t element_size,
    const int64_t ndim,
    std::array<int64_t*, N>* strides_array) {
  std::vector<int64_t> stride_bytes =
      compatible_stride(shape_, ndim, static_cast<int64_t>(element_size));

  if (strides_array && (*strides_array)[0]) {
    std::copy(stride_bytes.begin(), stride_bytes.end(), (*strides_array)[0]);
  }
}

template <int N>
static inline void coalesce_dimensions(const int64_t ndim,
                                       std::array<int64_t*, N>* strides_array,
                                       std::vector<int64_t>* stride_size,
                                       std::vector<int64_t>* shape_) {
  for (size_t i = 0; i < N; i++) {
    int64_t* stride_tmp = (*strides_array)[i];
  }

  if (ndim <= 1) {
    return;
  }

  // We can coalesce two adjacent dimensions if either dim has size 1 or if:
  // shape[n] * stride[n] == stride[n + 1].
  auto can_coalesce = [&](int dim0, int dim1) {
    auto shape0 = (*shape_)[dim0];
    auto shape1 = (*shape_)[dim1];
    if (shape0 == 1 || shape1 == 1) {
      return true;
    }
    for (int64_t i = 0; i < N; i++) {
      auto& stride = (*strides_array)[i];
      if (shape0 * stride[dim0] != stride[dim1]) {
        return false;
      }
    }
    return true;
  };

  // replace each operands stride at dim0 with its stride at dim1
  auto replace_stride = [&](int dim0, int dim1) {
    for (int64_t i = 0; i < N; i++) {
      auto& stride = (*strides_array)[i];
      stride[dim0] = stride[dim1];
    }
  };

  int prev_dim = 0;
  for (int64_t dim = 1; dim < ndim; dim++) {
    if (can_coalesce(prev_dim, dim)) {
      if ((*shape_)[prev_dim] == 1) {
        replace_stride(prev_dim, dim);
      }
      (*shape_)[prev_dim] *= (*shape_)[dim];
    } else {
      prev_dim++;
      if (prev_dim != dim) {
        replace_stride(prev_dim, dim);
        (*shape_)[prev_dim] = (*shape_)[dim];
      }
    }
  }
  (*shape_).resize(prev_dim + 1);
  for (int64_t i = 0; i < N; i++) {
    (*stride_size)[i] = shape_->size();
  }
}

template <int N>
static inline void IndexPutStride(
    const std::vector<int64_t> output_dims,  // value_tensor
    const std::vector<int64_t> output_strides,
    const int64_t output_elesize,
    const std::vector<int64_t> input_dims,  // input_tensor
    const std::vector<int64_t> input_strides,
    const int64_t input_elesize,
    const std::vector<int64_t> index_dims,  // index_tensor
    const std::vector<int64_t> index_strides,
    const int64_t index_elesize,
    std::vector<int64_t>* desired_shape,
    std::array<int64_t*, N>* strides_array,
    int64_t* numel,
    std::array<std::vector<int64_t>, N>& strides_vec) {  // NOLINT
  int ndim = output_dims.size();

  std::vector<int64_t> stride_size;

  *desired_shape = compute_shapes({input_dims, output_dims, index_dims});
  strides_vec[0] = compute_strides(output_dims,  // input_tensor
                                   output_strides,
                                   output_elesize,
                                   ndim,
                                   desired_shape,
                                   &stride_size);

  strides_vec[1] = compute_strides(input_dims,  // value_tensor
                                   input_strides,
                                   input_elesize,
                                   ndim,
                                   desired_shape,
                                   &stride_size);

  strides_vec[2] = compute_strides(index_dims,  // index_tensor
                                   index_strides,
                                   index_elesize,
                                   ndim,
                                   desired_shape,
                                   &stride_size);

  for (size_t i = 0; i < N; i++) {
    (*strides_array)[i] = strides_vec[i].data();
  }
  reorder_dimensions<N>(stride_size, desired_shape, strides_array);

  coalesce_dimensions<N>(ndim, strides_array, &stride_size, desired_shape);

  int num = 1;
  for (int i = 0; i < desired_shape->size(); i++) {
    num *= (*desired_shape)[i];
  }
  *numel = num;
}

template <int N>
static inline void IndexGetStride(
    const std::vector<int64_t> output_dims,
    const std::vector<int64_t> output_strides,
    const int64_t output_elesize,
    const std::vector<int64_t> input_dims,
    const std::vector<int64_t> input_strides,
    const int64_t input_elesize,
    const std::vector<int64_t> index_dims,
    const std::vector<int64_t> index_strides,
    const int64_t index_elesize,
    std::vector<int64_t>* desired_shape,
    std::array<int64_t*, N>* strides_array,
    int64_t* numel,
    std::array<std::vector<int64_t>, N>& strides_vec) {  // NOLINT
  int ndim = output_dims.size();

  std::vector<int64_t> stride_size;

  *desired_shape = compute_shapes({input_dims, output_dims, index_dims});

  strides_vec[0] = compute_strides(input_dims,
                                   input_strides,
                                   input_elesize,
                                   ndim,
                                   desired_shape,
                                   &stride_size);

  strides_vec[1] = compute_strides(output_dims,
                                   output_strides,
                                   output_elesize,
                                   ndim,
                                   desired_shape,
                                   &stride_size);

  strides_vec[2] = compute_strides(index_dims,
                                   index_strides,
                                   index_elesize,
                                   ndim,
                                   desired_shape,
                                   &stride_size);

  for (size_t i = 0; i < N; i++) {
    (*strides_array)[i] = strides_vec[i].data();
  }
  reorder_dimensions<N>(stride_size, desired_shape, strides_array);

  allocate_or_resize_outputs<N>(
      desired_shape, output_elesize, ndim, strides_array);

  coalesce_dimensions<N>(ndim, strides_array, &stride_size, desired_shape);

  int num = 1;
  for (int i = 0; i < desired_shape->size(); i++) {
    num *= (*desired_shape)[i];
  }
  *numel = num;
}

}  // namespace funcs
}  // namespace phi
