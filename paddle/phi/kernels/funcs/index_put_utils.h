// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/phi/kernels/expand_kernel.h"
#include "paddle/phi/kernels/nonzero_kernel.h"
#include "paddle/phi/kernels/reshape_kernel.h"
#include "paddle/phi/kernels/slice_kernel.h"
#include "paddle/phi/kernels/split_kernel.h"

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

static inline common::DDim infer_size_symdimvector(common::DDim a,
                                                   common::DDim b) {
  // Use ptrdiff_t to ensure signed comparison.
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

    // 1s map to the other size (even 0).
    expandedSizes[i] = sizeA == 1 ? sizeB : sizeA;
  }

  return expandedSizes;
}

template <typename T, typename Context>
std::vector<phi::DenseTensor*> expandTensors(
    const Context& dev_ctx, std::vector<phi::DenseTensor*> indices) {
  // expands bool to int tensors;
  std::vector<phi::DenseTensor*> result;
  for (auto& index : indices) {
    if (index->dtype() == paddle::DataType::BOOL) {
      phi::DenseTensor bool_2_idx(phi::DataType::INT64);
      NonZeroKernel<bool, Context>(dev_ctx, *index, &bool_2_idx);
      // auto bool_2_idx = nonzero_ad_func(index);
      for (int j = 0; j < index->dims().size(); j++) {
        phi::DenseTensor* sliced_tensor = new phi::DenseTensor();
        // slice_ad_func(bool_2_idx, {1}, {j}, {j + 1}, {1}, {1});
        SliceKernel<int64_t, Context>(
            dev_ctx, bool_2_idx, {1}, {j}, {j + 1}, {1}, {1}, sliced_tensor);
        result.emplace_back(sliced_tensor);
      }
    } else {
      result.emplace_back(index);
    }
  }
  return result;
}

template <typename T, typename Context>
std::vector<phi::DenseTensor*> expand_outplace(
    const Context& dev_ctx, std::vector<phi::DenseTensor*> to_expand) {
  // expands a list of Tensors; ignores undefined (null) tensors
  bool first = true;
  common::DDim sizes;
  for (size_t i = 0; i < to_expand.size(); i++) {
    if (!to_expand[i]->initialized()) {
      continue;
    } else if (first) {
      sizes = to_expand[i]->dims();
      first = false;
    } else {
      sizes = infer_size_symdimvector(sizes, to_expand[i]->dims());
    }
  }

  std::vector<phi::DenseTensor*> result(to_expand.size());
  for (size_t i = 0; i < to_expand.size(); i++) {
    if (!to_expand[i]->initialized()) {
      continue;
    } else if (to_expand[i]->dims() == sizes) {
      result[i] = to_expand[i];
    } else {
      if (to_expand[i]->dtype() == phi::DataType::INT32) {
        result[i] = new phi::DenseTensor(phi::DataType::INT64);
        ExpandKernel<int32_t, Context>(
            dev_ctx,
            *(to_expand[i]),
            IntArray(common::vectorize<int32_t>(sizes)),
            result[i]);
      } else if (to_expand[i]->dtype() == phi::DataType::INT64) {
        result[i] = new phi::DenseTensor(phi::DataType::INT64);
        ExpandKernel<int64_t, Context>(
            dev_ctx,
            *(to_expand[i]),
            IntArray(common::vectorize<int64_t>(sizes)),
            result[i]);
      } else {
        PADDLE_THROW(::common::errors::Unimplemented(
            "Index in Stride Mechanism must be int32_t, int64_t or bool"));
      }
    }
  }
  return result;
}

struct AdvancedIndex {
  AdvancedIndex(phi::DenseTensor src, std::vector<phi::DenseTensor*> indices);

  phi::DenseTensor src;
  std::vector<phi::DenseTensor*> indices;
  std::vector<int64_t> indexed_sizes;
  std::vector<int64_t> indexed_strides;
  std::vector<int64_t> src_sizes;
  std::vector<int64_t> src_strides;
  int64_t dims_before;
  int64_t dims_after;
  bool bool_case;
};

inline static void restride_src(std::vector<int64_t>* shape,
                                std::vector<int64_t>* strides,
                                int64_t dims_before,
                                int64_t dims_indexed,
                                std::vector<int64_t> replacement_shape) {
  int64_t end = dims_before + dims_indexed;
  shape->erase(shape->begin() + dims_before, shape->begin() + end);
  strides->erase(strides->begin() + dims_before, strides->begin() + end);
  shape->insert(shape->begin() + dims_before,
                replacement_shape.begin(),
                replacement_shape.end());
  strides->insert(strides->begin() + dims_before, replacement_shape.size(), 0);
}

// move to cuda kernel
inline static void reshape_indexer(phi::DenseTensor* index,
                                   int64_t dims_before,
                                   int64_t dims_after) {
  auto orig_shape = common::vectorize<int64_t>(index->dims());
  auto shape = std::vector<int64_t>{};
  shape.insert(shape.end(), dims_before, 1);
  shape.insert(shape.end(), orig_shape.begin(), orig_shape.end());
  shape.insert(shape.end(), dims_after, 1);
  index->Resize(common::make_ddim(shape));
}

inline AdvancedIndex::AdvancedIndex(
    phi::DenseTensor src, std::vector<phi::DenseTensor*> indices_list) {
  uint32_t element_size_bytes = phi::SizeOf(src.dtype());
  int64_t dims_before = 0, dims_after = 0, dims_indexed = 0;
  std::vector<int64_t> shape_vec = common::vectorize<int64_t>(src.dims());
  std::vector<int64_t> stride_vec = common::vectorize<int64_t>(src.strides());
  std::vector<int64_t> replacement_shape;
  std::vector<int64_t> idx_shape_vec = {};
  std::vector<int64_t> idx_stride_vec = {};
  for (size_t dim = 0; dim < indices_list.size(); dim++) {
    if (!indices_list[dim]) {
      if (dims_indexed == 0) {
        dims_before++;
      } else {
        dims_after++;
      }
    } else {
      dims_indexed++;
      replacement_shape = common::vectorize<int64_t>(indices_list[dim]->dims());

      indexed_sizes.push_back(shape_vec[dim]);
      indexed_strides.push_back(stride_vec[dim] * element_size_bytes);
    }
  }

  this->dims_before = dims_before;
  this->dims_after = dims_after;
  restride_src(
      &shape_vec, &stride_vec, dims_before, dims_indexed, replacement_shape);
  this->src_sizes = shape_vec;
  this->src_strides = stride_vec;

  // use dims_before and dims_after / move to cuda kernel
  for (auto& index : indices_list) {
    if (index) {
      reshape_indexer(index, dims_before, dims_after);
      this->indices.push_back(index);
    }
  }
}

template <typename T, typename Context>
inline AdvancedIndex make_info(
    const Context& dev_ctx,
    const phi::DenseTensor& self,
    const std::vector<const phi::DenseTensor*>& orig) {
  std::vector<phi::DenseTensor*> tmp_indices;
  for (int i = 0; i < orig.size(); i++) {
    phi::DenseTensor* tmp = new phi::DenseTensor();
    *tmp = *(const_cast<phi::DenseTensor*>(orig[i]));
    tmp_indices.push_back(tmp);
  }

  auto indices = expandTensors<T, Context>(dev_ctx, tmp_indices);
  // next broadcast all index tensors together
  indices = expand_outplace<T, Context>(dev_ctx, indices);
  // add missing null Tensors so that it matches self.dim()
  while (indices.size() < static_cast<size_t>(self.dims().size())) {
    indices.emplace_back();
  }

  std::vector<phi::DenseTensor*> indices_int64;
  // Ensure indices are on the same device as self
  for (auto& indice : indices) {
    if (indice && indice->dtype() == paddle::DataType::INT32) {
      // indice = indice->cast(paddle::DataType::INT64);  // int32 -> int64
      *indice = phi::Cast<int, Context>(dev_ctx, *indice, phi::DataType::INT64);
    }
    indices_int64.push_back(indice);
  }

  return AdvancedIndex(self, indices_int64);
}

template <typename T, typename Context>
phi::DenseTensor GetReshapeAndExpandTensor(const Context& dev_ctx,
                                           const phi::DenseTensor& tensor,
                                           const phi::DDim& res_dim,
                                           const phi::DDim& bd_dim,
                                           int index) {
  std::vector<int64_t> before_dims = common::vectorize(tensor.dims());
  std::vector<int64_t> mid_dims(res_dim.size(), 1);

  if (index == 0) {
    for (size_t i = 0; i < before_dims.size(); ++i) {
      mid_dims[bd_dim.size() - i - 1] = before_dims[before_dims.size() - i - 1];
    }
  } else {
    mid_dims[index] = before_dims[0];
  }

  phi::DenseTensor mid_tensor(tensor.dtype());
  mid_tensor.Resize(common::make_ddim(mid_dims));
  ReshapeKernel<Context>(dev_ctx, tensor, IntArray(mid_dims), &mid_tensor);

  phi::DenseTensor res_tensor(tensor.dtype());
  res_tensor.Resize(res_dim);
  ExpandKernel<T, Context>(
      dev_ctx, mid_tensor, IntArray(common::vectorize(res_dim)), &res_tensor);
  return res_tensor;
}

template <typename T, typename Context>
std::vector<const phi::DenseTensor*> DealWithBoolIndices(
    const Context& dev_ctx,
    const std::vector<const phi::DenseTensor*>& indices_v,
    std::vector<phi::DenseTensor>* tmp_indices_v) {
  std::vector<const phi::DenseTensor*> res;

  bool contains_bool_tensor = false;
  for (size_t i = 0; i < indices_v.size(); ++i) {
    if (indices_v[i]->dtype() == phi::DataType::BOOL) {
      contains_bool_tensor = true;
      break;
    }
  }

  if (contains_bool_tensor) {
    for (size_t i = 0; i < indices_v.size(); ++i) {
      if (indices_v[i]->dtype() == phi::DataType::BOOL) {
        int rank = indices_v[i]->dims().size();
        PADDLE_ENFORCE_GE(rank,
                          1UL,
                          common::errors::InvalidArgument(
                              "the only bool tensor in indices should "
                              "have number of dimension at least 1"));
        phi::DenseTensor nonzero_indices(phi::DataType::INT64);
        nonzero_indices.Resize(common::make_ddim({-1, rank}));
        NonZeroKernel<bool, Context>(dev_ctx, *indices_v[i], &nonzero_indices);

        if (nonzero_indices.numel() == 0) {
          std::vector<const phi::DenseTensor*> empty_indices;
          return empty_indices;
        }

        std::vector<phi::DenseTensor*> integer_indices(rank, nullptr);
        const int tmp_ix = tmp_indices_v->size();
        for (int i = 0; i < rank; ++i) {
          tmp_indices_v->emplace_back(
              DenseTensor(phi::DataType::INT64)
                  .Resize(common::make_ddim({nonzero_indices.dims()[0]})));
        }
        for (int i = 0; i < rank; ++i) {
          integer_indices[i] = &((*tmp_indices_v)[i + tmp_ix]);
        }
        SplitWithNumKernel<int64_t, Context>(
            dev_ctx, nonzero_indices, rank, 1, integer_indices);
#ifdef PADDLE_WITH_XPU
        auto place = dev_ctx.GetPlace();
        if (place.GetType() == phi::AllocationType::XPU) {
          auto& pool = phi::DeviceContextPool::Instance();
          auto* xpu_ctx = static_cast<phi::XPUContext*>(pool.Get(place));
          if (xpu_ctx->x_context()->xpu_stream) {
            dev_ctx.Wait();
          }
        }
#endif

      } else if ((indices_v[i]->dtype() == phi::DataType::INT64) ||
                 (indices_v[i]->dtype() == phi::DataType::INT32)) {
        tmp_indices_v->emplace_back(*indices_v[i]);
      } else {
        PADDLE_THROW(common::errors::InvalidArgument(
            "data type of tensor in indices must be int32, int64 or bool"));
      }
    }

    res.reserve(tmp_indices_v->size());
    for (size_t i = 0; i < tmp_indices_v->size(); ++i) {
      res.emplace_back(&((*tmp_indices_v)[i]));
    }
  } else {
    res = indices_v;
  }
  return res;
}

static phi::DDim BroadCastTensorsDims(
    const std::vector<const phi::DenseTensor*>& tensors) {
  int target_rank = 0;
  for (const auto& tensor : tensors) {
    target_rank = std::max(target_rank, tensor->dims().size());
  }

  PADDLE_ENFORCE_GT(target_rank,
                    0,
                    errors::InvalidArgument("BroadCastTensorsDims requires at "
                                            "least one input tensor to have "
                                            "rank greater than zero"));

  std::vector<int64_t> target_dims(target_rank, 0);
  for (int index = 0; index < target_rank; index++) {
    int target_dim_size = 1;
    for (const auto& tensor : tensors) {
      auto input_ddim = tensor->dims();
      int axis = static_cast<int>(input_ddim.size()) - index - 1;
      int dim_size = 1;
      if (axis >= 0) {
        dim_size = input_ddim[axis];
      }

      if (target_dim_size != 1 && dim_size != 1 &&
          target_dim_size != dim_size) {
        PADDLE_THROW(errors::InvalidArgument(
            "BroadCastTensorsDims inputs does not satisfy bcast semantics, "
            "please check axis = %d in reverse order",
            index));
      }

      target_dim_size = dim_size == 1 ? target_dim_size : dim_size;
    }
    target_dims[target_rank - index - 1] = target_dim_size;
  }
  return common::make_ddim(target_dims);
}

template <typename T, typename Context>
T** GetDevicePointerArray(const Context& dev_ctx,
                          const std::vector<const DenseTensor*>& indices_v,
                          phi::Allocator::AllocationPtr* holder_ptr) {
  PADDLE_ENFORCE_NOT_NULL(
      holder_ptr,
      common::errors::InvalidArgument(
          "hold_ptr should be provided when calling GetDevicePointerArray."));
  std::vector<const T*> h_indices_v(indices_v.size());
  for (size_t i = 0; i < indices_v.size(); ++i) {
    h_indices_v[i] = indices_v[i]->data<T>();
  }
  auto& d_indices_data = *holder_ptr;
  d_indices_data = phi::memory_utils::Alloc(
      dev_ctx.GetPlace(),
      h_indices_v.size() * sizeof(T*),
      phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream())));
  phi::memory_utils::Copy(dev_ctx.GetPlace(),
                          d_indices_data->ptr(),
                          phi::CPUPlace(),
                          reinterpret_cast<void*>(h_indices_v.data()),
                          h_indices_v.size() * sizeof(T*),
                          dev_ctx.stream());
  return reinterpret_cast<T**>(d_indices_data->ptr());
}

template <typename T, typename Context>
void DealWithIndices(const Context& dev_ctx,
                     const DenseTensor& x,
                     const std::vector<const phi::DenseTensor*>& int_indices_v,
                     std::vector<const phi::DenseTensor*>* res_indices_v,
                     std::vector<DenseTensor>* tmp_res_indices_v,
                     const std::vector<DenseTensor>& range_tensor_v,
                     const phi::DDim& bd_dim,
                     std::vector<int64_t>* res_dim_v) {
  size_t total_dims = x.dims().size();
  if (int_indices_v.size() < total_dims) {
    std::vector<int64_t> tmp_x_dims = common::vectorize(x.dims());
    int len_bd_dim = bd_dim.size();
    res_dim_v->insert(res_dim_v->end(),
                      tmp_x_dims.begin() + int_indices_v.size(),
                      tmp_x_dims.end());
    phi::DDim res_dim = common::make_ddim(*res_dim_v);
    for (size_t i = 0; i < int_indices_v.size(); ++i) {
      phi::DenseTensor index_tensor;
      if (int_indices_v[i]->dtype() == phi::DataType::INT32) {
        index_tensor = phi::Cast<int, Context>(
            dev_ctx, *int_indices_v[i], phi::DataType::INT64);
      } else {
        index_tensor = *int_indices_v[i];
      }
      tmp_res_indices_v->emplace_back(
          GetReshapeAndExpandTensor<int64_t, Context>(
              dev_ctx, index_tensor, res_dim, bd_dim, 0));
    }
    for (size_t i = 0; i < range_tensor_v.size(); ++i) {
      tmp_res_indices_v->emplace_back(
          GetReshapeAndExpandTensor<int64_t, Context>(
              dev_ctx, range_tensor_v[i], res_dim, bd_dim, i + len_bd_dim));
    }
    for (size_t i = 0; i < res_indices_v->size(); ++i) {
      (*res_indices_v)[i] = &(*tmp_res_indices_v)[i];
    }

  } else {
    for (size_t i = 0; i < int_indices_v.size(); ++i) {
      phi::DenseTensor index_tensor;
      phi::DenseTensor expand_index;
      if (int_indices_v[i]->dtype() == phi::DataType::INT32) {
        index_tensor = phi::Cast<int, Context>(
            dev_ctx, *int_indices_v[i], phi::DataType::INT64);
      } else {
        index_tensor = *int_indices_v[i];
      }
      if (bd_dim != int_indices_v[i]->dims()) {
        expand_index = DenseTensor(phi::DataType::INT64).Resize(bd_dim);
        ExpandKernel<int64_t, Context>(
            dev_ctx,
            index_tensor,
            IntArray(common::vectorize<int64_t>(bd_dim)),
            &expand_index);
      } else {
        expand_index = index_tensor;
      }
      tmp_res_indices_v->emplace_back(expand_index);
    }
    for (size_t i = 0; i < res_indices_v->size(); ++i) {
      (*res_indices_v)[i] = &(*tmp_res_indices_v)[i];
    }
  }
}

static void CalCompressedDimsWith1AndWithout1(
    std::vector<int64_t>* after_dims,
    std::vector<int64_t>* before_dims,
    std::vector<int64_t>* compress_dims,
    std::vector<int64_t>* dims_without_1) {
  int i = static_cast<int>(after_dims->size()) - 1;
  int j = static_cast<int>(before_dims->size()) - 1;
  if (i < j) {
    PADDLE_THROW(common::errors::InvalidArgument(
        "shape of value can't not be broadcast to shape of x[indices]"));
  }

  while ((i >= 0) && (j >= 0)) {
    if ((*after_dims)[i] == (*before_dims)[j]) {
      dims_without_1->push_back((*before_dims)[j]);
      i--;
      j--;
      continue;
    } else if ((*before_dims)[j] == 1) {
      compress_dims->push_back(i);
      i--;
      j--;
    } else {
      PADDLE_THROW(common::errors::InvalidArgument(
          "shape of value can't not be broadcast to shape of x[indices]"));
    }
  }
  while (i >= 0) {
    compress_dims->push_back(i);
    i--;
  }
}

#if defined(__NVCC__) || defined(__HIPCC__)
template <typename T>
__global__ void range_cuda_kernel(int64_t N, T* out) {
  int64_t idx = threadIdx.x + blockDim.x * blockIdx.x;

  if (idx >= N) {
    return;
  }
  out[idx] = idx;
}

template <typename T, typename Context>
phi::DenseTensor GetRangeCudaTensor(const Context& dev_ctx,
                                    int64_t N,
                                    phi::DataType dtype) {
  phi::DenseTensor res(dtype);
  res.Resize(common::make_ddim({N}));
  DenseTensor* p_res = &res;
  T* out = dev_ctx.template Alloc<T>(p_res);
  auto config = phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, N);
  range_cuda_kernel<T>
      <<<config.block_per_grid, config.thread_per_block, 0, dev_ctx.stream()>>>(
          N, out);
  return res;
}
#endif

template <typename T>
void range_kernel(int64_t N, T* out) {
  for (int64_t idx = 0; idx < N; ++idx) {
    out[idx] = idx;
  }
}

template <typename T, typename Context>
phi::DenseTensor GetRangeTensor(const Context& dev_ctx,
                                int64_t N,
                                phi::DataType dtype) {
  phi::DenseTensor res(dtype);
  res.Resize(common::make_ddim({N}));
  DenseTensor* p_res = &res;
  T* out = dev_ctx.template Alloc<T>(p_res);
  range_kernel<T>(N, out);
  return res;
}

}  // namespace funcs
}  // namespace phi
