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
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/cast_kernel.h"
#include "paddle/phi/kernels/elementwise_add_kernel.h"
#include "paddle/phi/kernels/elementwise_kernel.h"
#include "paddle/phi/kernels/elementwise_multiply_kernel.h"
#include "paddle/phi/kernels/expand_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/nonzero_kernel.h"
#include "paddle/phi/kernels/slice_kernel.h"
#include "paddle/phi/kernels/transpose_kernel.h"

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
  for (size_t i = 0; i < original_shape.size(); i++) {
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
    for (size_t i = 0; i < perm.size(); i++) {
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
      for (int64_t j = 0; j < stride_size[i]; j++) {
        (*strides_array)[i][j] = temp_strides[i][j];
      }
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
  for (size_t i = 1; i < ndim; i++) {
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
static inline void CopyStride(
    const std::vector<int64_t> output_dims,  // value_tensor
    const std::vector<int64_t> output_strides,
    const int64_t output_elesize,
    const std::vector<int64_t> input_dims,  // input_tensor
    const std::vector<int64_t> input_strides,
    const int64_t input_elesize,
    std::vector<int64_t>* desired_shape,
    std::array<int64_t*, N>* strides_array,
    int64_t* numel,
    std::array<std::vector<int64_t>, N>& strides_vec) {  // NOLINT
  int ndim = output_dims.size();

  std::vector<int64_t> stride_size;

  *desired_shape = compute_shapes({input_dims, output_dims});
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

  for (size_t i = 0; i < N; i++) {
    (*strides_array)[i] = strides_vec[i].data();
  }
  reorder_dimensions<N>(stride_size, desired_shape, strides_array);

  coalesce_dimensions<N>(ndim, strides_array, &stride_size, desired_shape);

  int num = 1;
  for (size_t i = 0; i < desired_shape->size(); i++) {
    num *= (*desired_shape)[i];
  }
  *numel = num;
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
  for (size_t i = 0; i < desired_shape->size(); i++) {
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
  for (size_t i = 0; i < desired_shape->size(); i++) {
    num *= (*desired_shape)[i];
  }
  *numel = num;
}

static inline void cal_shape_stride(const std::vector<int64_t> index_dims,
                                    int64_t* num_indices,
                                    std::vector<int64_t>* shape_tmp,
                                    std::vector<int64_t>* stride_tmp) {
  std::vector<int64_t> index_dims_;
  std::vector<int64_t> index_stride_;

  bool tmp_flag = false;
  for (unsigned i = 0; i < index_dims.size(); i++) {
    if (index_dims[i] == -1) {
      if (!tmp_flag) {
        *num_indices = i;
        tmp_flag = true;
        continue;
      } else {
        break;
      }
    }

    if (!tmp_flag) {
      index_dims_.push_back(index_dims[i]);
    } else {
      shape_tmp->push_back(index_dims[i]);
    }
  }

  int shape_size = shape_tmp->size();
  stride_tmp->resize(shape_size);
  if (shape_size > 0) {
    (*stride_tmp)[shape_size - 1] = 1;
  }
  if (shape_size > 1) {
    for (int i = shape_size - 2; i >= 0; i--) {
      (*stride_tmp)[i] = (*stride_tmp)[i + 1] * (*shape_tmp)[i + 1];
    }
  }
}

template <int N>
static inline void ScatterAddStride(
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

  coalesce_dimensions<N>(ndim, strides_array, &stride_size, desired_shape);

  int num = 1;
  for (int i = 0; i < desired_shape->size(); i++) {
    num *= (*desired_shape)[i];
  }
  *numel = num;
}

static inline common::DDim infer_size_symdimvector(common::DDim a,
                                                   common::DDim b) {
  auto dimsA = a.size();
  auto dimsB = b.size();
  auto ndim = dimsA > dimsB ? dimsA : dimsB;
  common::DDim expandedSizes = common::make_ddim(std::vector<int64_t>(ndim, 0));

  for (int64_t i = ndim - 1; i >= 0; --i) {
    int64_t offset = ndim - 1 - i;
    int64_t dimA = dimsA - 1 - offset;
    int64_t dimB = dimsB - 1 - offset;
    auto sizeA = (dimA >= 0) ? a[dimA] : 1;
    auto sizeB = (dimB >= 0) ? b[dimB] : 1;

    PADDLE_ENFORCE_EQ(
        sizeA == sizeB || sizeA == 1 || sizeB == 1,
        true,
        common::errors::Fatal("The size of tensor a (",
                              sizeA,
                              ") must match the size of tensor b (",
                              sizeB,
                              ") at non-singleton dimension ",
                              i));

    expandedSizes[i] = sizeA == 1 ? sizeB : sizeA;
  }

  return expandedSizes;
}

static inline bool hasContiguousSubspace(
    const std::vector<phi::DenseTensor>& tl) {
  auto isDefined = [](const phi::DenseTensor& tensor) {
    return tensor.initialized();
  };
  auto isNull = [](const phi::DenseTensor& tensor) {
    return !tensor.initialized();
  };

  auto start = std::find_if(tl.begin(), tl.end(), isDefined);
  auto stop = std::find_if(tl.rbegin(), tl.rend(), isDefined);
  auto it = std::find_if(start, stop.base(), isNull);
  return it == stop.base();
}

#if defined(PADDLE_WITH_CUDA)

static inline std::vector<phi::DenseTensor> expandTensors(
    const phi::GPUContext& dev_ctx,
    const std::vector<const phi::DenseTensor*>& indices) {
  std::vector<phi::DenseTensor> result;
  for (const auto& index : indices) {
    if (index == nullptr) {
      result.emplace_back();
      continue;
    }

    if (index->dtype() == phi::DataType::BOOL) {
      phi::DenseTensor bool_2_idx;
      phi::NonZeroKernel<bool, phi::GPUContext>(dev_ctx, *index, &bool_2_idx);

      for (int j = 0; j < index->dims().size(); ++j) {
        phi::DenseTensor sliced_tensor;
        phi::SliceKernel<int64_t, phi::GPUContext>(
            dev_ctx, bool_2_idx, {1}, {j}, {j + 1}, {1}, {}, &sliced_tensor);
        result.emplace_back(sliced_tensor);
      }
    } else {
      result.emplace_back(*index);
    }
  }
  return result;
}

static inline std::vector<phi::DenseTensor> expand_outplace(
    const phi::GPUContext& dev_ctx, std::vector<phi::DenseTensor> to_expand) {
  bool first = true;
  phi::DDim target_shape;
  for (size_t i = 0; i < to_expand.size(); ++i) {
    if (!to_expand[i].initialized()) continue;
    if (first) {
      target_shape = to_expand[i].dims();
      first = false;
    } else {
      target_shape = infer_size_symdimvector(target_shape, to_expand[i].dims());
    }
  }

  std::vector<phi::DenseTensor> result(to_expand.size());
  for (size_t i = 0; i < to_expand.size(); ++i) {
    if (!to_expand[i].initialized()) continue;
    if (to_expand[i].dims() == target_shape) {
      result[i] = to_expand[i];
    } else {
      phi::ExpandKernel<float, phi::GPUContext>(
          dev_ctx,
          to_expand[i],
          phi::IntArray(common::vectorize<int64_t>(target_shape)),
          &result[i]);
    }
  }
  return result;
}

template <typename T>
inline std::
    tuple<phi::DenseTensor, std::vector<phi::DenseTensor>, std::vector<int64_t>>
    transposeToFrontAndInvPerm(const phi::GPUContext& dev_ctx,
                               const phi::DenseTensor& self,
                               const std::vector<phi::DenseTensor>& indices) {
  std::vector<int> dims;
  std::vector<int64_t> inv_perm;
  std::vector<phi::DenseTensor> transposed_indices;
  dims.reserve(self.dims().size());
  inv_perm.resize(self.dims().size());

  for (int64_t i = 0; i < self.dims().size(); ++i) {
    if (indices[i].initialized()) {
      dims.push_back(static_cast<int>(i));
      transposed_indices.emplace_back(indices[i]);
    }
  }

  for (int64_t i = 0; i < self.dims().size(); ++i) {
    if (!indices[i].initialized()) {
      dims.push_back(static_cast<int>(i));
      transposed_indices.emplace_back();
    }
  }

  for (int64_t i = 0; i < self.dims().size(); ++i) {
    inv_perm[dims[i]] = i;
  }

  phi::DenseTensor transposed_self;
  phi::TransposeKernel<T, phi::GPUContext>(
      dev_ctx, self, dims, &transposed_self);

  return std::make_tuple(transposed_self, transposed_indices, inv_perm);
}

static inline std::vector<int64_t> computeLinearStride(
    const phi::DenseTensor& tensor) {
  auto sizes = phi::vectorize<int64_t>(tensor.dims());
  std::vector<int64_t> stride(sizes.size());
  if (stride.empty()) {
    return stride;
  }
  stride.back() = 1;
  std::partial_sum(sizes.rbegin(),
                   sizes.rend() - 1,
                   stride.rbegin() + 1,
                   std::multiplies<int64_t>());
  return stride;
}

static inline phi::DenseTensor wrapIndexOnce(const phi::GPUContext& dev_ctx,
                                             const phi::DenseTensor& index,
                                             int64_t dim,
                                             int64_t dim_size,
                                             bool check_range) {
  phi::DenseTensor dim_size_tensor;
  dim_size_tensor.Resize(index.dims());
  dev_ctx.Alloc<int64_t>(&dim_size_tensor);

  auto* dim_size_data = dim_size_tensor.data<int64_t>();
  auto numel = index.numel();
  std::vector<int64_t> host_data(numel, dim_size);
  phi::memory_utils::Copy(dev_ctx.GetPlace(),
                          dim_size_data,
                          CPUPlace(),
                          host_data.data(),
                          numel * sizeof(int64_t),
                          dev_ctx.stream());

  return phi::Remainder<int64_t>(dev_ctx, index, dim_size_tensor);
}

static inline std::tuple<phi::DenseTensor, int64_t, int64_t, int64_t>
computeLinearIndex(const phi::GPUContext& dev_ctx,
                   const phi::DenseTensor& src,
                   const std::vector<phi::DenseTensor>& indices,
                   bool check_range) {
  std::vector<int64_t> strides = computeLinearStride(src);
  phi::DenseTensor linearIndex;
  int64_t nElemBefore = 1, nElemAfter = 1, strideBefore = 0;

  for (int64_t i = 0; i < src.dims().size(); ++i) {
    if (indices[i].initialized()) {
      auto wrapped_index =
          wrapIndexOnce(dev_ctx, indices[i], i, src.dims()[i], check_range);

      auto strides_tensor = phi::Full<int64_t, phi::GPUContext>(
          dev_ctx,
          common::vectorize<int64_t>(wrapped_index.dims()),
          phi::Scalar(strides[i]));

      auto scaled_index = phi::Multiply<int64_t, phi::GPUContext>(
          dev_ctx, wrapped_index, strides_tensor);

      if (linearIndex.initialized()) {
        phi::AddKernel<int64_t, phi::GPUContext>(
            dev_ctx, linearIndex, scaled_index, &linearIndex);
      } else {
        linearIndex = scaled_index;
        if (i > 0) {
          strideBefore = src.strides()[i - 1];
        }
      }
    } else if (linearIndex.initialized()) {
      nElemAfter *= src.dims()[i];
    } else {
      nElemBefore *= src.dims()[i];
    }
  }

  return std::make_tuple(
      std::move(linearIndex), nElemBefore, strideBefore, nElemAfter);
}

template <typename T>
static inline std::tuple<DenseTensor,
                         DenseTensor,
                         int64_t,
                         int64_t,
                         int64_t,
                         std::vector<int64_t>>
makeLinearIndex(const phi::GPUContext& dev_ctx,
                const DenseTensor& self,
                const std::vector<const DenseTensor*>& orig,
                bool check_range) {
  auto indices = expandTensors(dev_ctx, orig);
  for (auto& idx : indices) {
    if (idx.initialized() && idx.dtype() == phi::DataType::INT32) {
      idx = phi::Cast<int32_t, phi::GPUContext>(
          dev_ctx, idx, phi::DataType::INT64);
    }
  }
  indices = expand_outplace(dev_ctx, std::move(indices));

  while (indices.size() < static_cast<size_t>(self.dims().size())) {
    indices.emplace_back();
  }

  std::vector<int64_t> inverse_perm;
  DenseTensor transposed_self = self;
  std::vector<phi::DenseTensor> transposed_indices;
  std::vector<int64_t> inv_perm;
  if (!hasContiguousSubspace(indices)) {
    auto [tmp_self, tmp_indices, tmp_perm] =
        transposeToFrontAndInvPerm<T>(dev_ctx, self, indices);
    transposed_self = std::move(tmp_self);
    transposed_indices = std::move(tmp_indices);
    inv_perm = std::move(tmp_perm);
  } else {
    transposed_indices = indices;
  }

  auto [linear_index, n_elem_before, stride_before, n_elem_after] =
      computeLinearIndex(
          dev_ctx, transposed_self, transposed_indices, check_range);

  return std::make_tuple(linear_index,
                         transposed_self,
                         n_elem_before,
                         stride_before,
                         n_elem_after,
                         inv_perm);
}

#endif
inline bool are_expandable(const std::vector<int64_t>& shape1,
                           const std::vector<int64_t>& shape2) {
  size_t ndim1 = shape1.size();
  size_t ndim2 = shape2.size();
  size_t ndim = std::min(ndim1, ndim2);

  for (int64_t i = static_cast<int64_t>(ndim) - 1; i >= 0; --i) {
    auto dim1 = shape1[--ndim1];
    auto dim2 = shape2[--ndim2];
    if (dim1 == dim2 || dim1 == 1 || dim2 == 1) {
      continue;
    }
    return false;
  }
  return true;
}

inline int64_t LargestIndex(const phi::DenseTensor& tensor) {
  int64_t result = 0;
  const auto& dims = tensor.dims();
  const auto& strides = tensor.strides();

  for (int i = 0; i < dims.size(); ++i) {
    result += (dims[i] - 1) * strides[i];
  }
  return result;
}

inline int GetNumBits(uint64_t max_val) {
  if (max_val == 0) return 1;

  int num_bits = 1;
  while (max_val > 1) {
    max_val >>= 1;
    num_bits++;
  }
  return num_bits;
}
}  // namespace funcs
}  // namespace phi
