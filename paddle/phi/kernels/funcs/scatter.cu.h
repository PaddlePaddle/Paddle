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
#include <unordered_set>
#include <vector>

#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/funcs/aligned_vector.h"
#include "paddle/phi/kernels/funcs/index_elementwise.cu.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/stride_utils.h"
#include "paddle/phi/kernels/primitive/kernel_primitives.h"

namespace phi {
namespace funcs {

template <typename T, typename IndexT = int>
__global__ void ScatterInitCUDAKernel(const IndexT* indices,
                                      T* output,
                                      int64_t output_count,
                                      size_t index_size,
                                      size_t slice_size) {
  CUDA_KERNEL_LOOP_TYPE(i, index_size * slice_size, int64_t) {
    int64_t indices_i = i / slice_size;
    int64_t slice_i = i - indices_i * slice_size;  // offset inside the slice
    IndexT scatter_i = indices[indices_i];

    PADDLE_ENFORCE(
        scatter_i >= -output_count && scatter_i < output_count,
        "The index is out of bounds, "
        "please check whether the dimensions of index and "
        "input meet the requirements. It should "
        "be less than [%ld] and greater or equal to [%ld], but received [%d]",
        output_count,
        -output_count,
        scatter_i);
    if (scatter_i < 0) {
      scatter_i += output_count;
    }

    int64_t out_i = scatter_i * slice_size + slice_i;
    *(output + out_i) = static_cast<T>(0);
  }
}

template <typename T, typename IndexT, bool Overwrite, int VecSize>
__global__ void ScatterCUDAKernel(const T* params,
                                  const IndexT* indices,
                                  T* output,
                                  int64_t output_count,
                                  size_t index_size,
                                  size_t slice_size) {
  int64_t num = index_size * slice_size;
  int64_t block_size = blockDim.x;
  int64_t i = (blockIdx.x * block_size + threadIdx.x) * VecSize;
  for (; i < num; i += gridDim.x * block_size * VecSize) {
    int64_t indices_i = i / slice_size;
    int64_t slice_i = i % slice_size;  // offset inside the slice
    IndexT scatter_i = indices[indices_i];

    PADDLE_ENFORCE(
        scatter_i >= -output_count && scatter_i < output_count,
        "The index is out of bounds, "
        "please check whether the dimensions of index and "
        "input meet the requirements. It should "
        "be less than [%d] and greater or equal to [%d], but received [%d]",
        output_count,
        -output_count,
        scatter_i);
    if (scatter_i < 0) {
      scatter_i += output_count;
    }

    int64_t out_i = scatter_i * slice_size + slice_i;
    if constexpr (Overwrite) {
      using VecType = kps::details::VectorType<T, VecSize>;
      const VecType* src = reinterpret_cast<const VecType*>(params + i);
      VecType* dst = reinterpret_cast<VecType*>(output + out_i);
      *dst = *src;
    } else {
      phi::CudaAtomicAdd(output + out_i, *(params + i));
    }
  }
}

template <typename T, typename IndexT, int VecSize>
__global__ void ScatterNdCUDAKernel(const T* update,
                                    const IndexT* indices,
                                    T* output,
                                    const Dim<DDim::kMaxRank> output_dims,
                                    size_t remain_size,
                                    size_t slice_size,
                                    size_t end_size) {
  size_t total_size = remain_size * slice_size;
  size_t idx =
      (static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x) * VecSize;
  size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x * VecSize;

#pragma unroll
  for (; idx < total_size; idx += stride) {
    size_t indices_i = idx / slice_size;
    size_t slice_i = idx % slice_size;
    size_t gather_i = 0;
    size_t gather_stride = slice_size;

#pragma unroll
    for (int j = end_size - 1; j >= 0; --j) {
      IndexT index_value = indices[indices_i * end_size + j];
      PADDLE_ENFORCE(
          index_value >= -output_dims[j] && index_value < output_dims[j],
          "The index is out of bounds, "
          "please check whether the dimensions of index and "
          "input meet the requirements. It should "
          "be less than [%ld] and greater or equal to [%ld], but received "
          "[%ld]",
          static_cast<int64_t>(output_dims[j]),
          -static_cast<int64_t>(output_dims[j]),
          static_cast<int64_t>(index_value));
      if (index_value < 0) {
        index_value += output_dims[j];
      }

      gather_i += index_value * gather_stride;
      gather_stride *= output_dims[j];
    }

    size_t output_i = gather_i + slice_i;

    using VecType = kps::details::VectorType<T, VecSize>;
    const VecType* src = reinterpret_cast<const VecType*>(&update[idx]);
    VecType* dst = reinterpret_cast<VecType*>(&output[output_i]);

#pragma unroll
    for (int k = 0; k < VecSize; ++k) {
      phi::CudaAtomicAdd(&(dst->val[k]), src->val[k]);
    }
  }
}

/**
 * A thin wrapper on gpu tensor
 * Return a new updated tensor from source tensor, scatter-assigned according to
 * index
 * input[src]: type-T source Tensor
 * input[index]: type-IndexT index Tensor (1-D)
 * return: output tensor
 */
template <typename T, typename IndexT = int>
void GPUScatterAssign(const phi::GPUContext& dev_ctx,
                      const DenseTensor& src,
                      const DenseTensor& index,
                      DenseTensor* output,
                      bool overwrite = true) {
  if (index.dims().size() == 2) {
    PADDLE_ENFORCE_EQ(
        index.dims()[1],
        1,
        common::errors::InvalidArgument("index.dims()[1] should be 1 when "
                                        "index.dims().size() = 2 in scatter_op."
                                        "But received value is [%d]",
                                        index.dims()[1]));
  } else {
    PADDLE_ENFORCE_EQ(
        index.dims().size() == 1 || index.dims().size() == 0,
        true,
        common::errors::InvalidArgument(
            "index.dims().size() should be 0, 1 or 2 in scatter_op."
            "But received value is [%d]",
            index.dims().size()));
  }

  int64_t index_size = index.dims().size() == 0 ? 1 : index.dims()[0];

  auto src_dims = src.dims();
  phi::DDim output_dims = output->dims();

  // slice size
  size_t slice_size = 1;
  if (index.dims().size() != 0) {
    for (int i = 1; i < src_dims.size(); ++i) slice_size *= src_dims[i];
  } else {
    for (int i = 0; i < src_dims.size(); ++i) slice_size *= src_dims[i];
  }
  const T* p_src = src.data<T>();
  const IndexT* p_index = index.data<IndexT>();
  T* p_output = output->data<T>();

  const size_t& slice_bytes = slice_size * sizeof(T);

  // set block and grid num
  int block = 512;
  int64_t n = slice_size * index_size;
  dim3 grid = dim3((n + block - 1) / block);
  phi::backends::gpu::LimitGridDim(dev_ctx, &grid);

  // if not overwrite mode, init data
  if (!overwrite) {
    ScatterInitCUDAKernel<T, IndexT><<<grid, block, 0, dev_ctx.stream()>>>(
        p_index, p_output, output_dims[0], index_size, slice_size);

    ScatterCUDAKernel<T, IndexT, false, 1>
        <<<grid, block, 0, dev_ctx.stream()>>>(
            p_src, p_index, p_output, output_dims[0], index_size, slice_size);
    return;
  }

  // for overwrite mode, use vectorization
  int vec_size = 8;
  vec_size = std::min(phi::GetVectorizedSize(&src), vec_size);
  vec_size = std::min(phi::GetVectorizedSize(output), vec_size);
  while (vec_size > 1 && slice_size % vec_size != 0) {
    vec_size /= 2;
  }

  constexpr int loop_count = 4;
  auto config = phi::backends::gpu::GetGpuLaunchConfig1D(
      dev_ctx, n, vec_size * loop_count);
  switch (vec_size) {
#define CASE_VEC_SIZE(__Sz)                                                \
  case __Sz:                                                               \
    ScatterCUDAKernel<T, IndexT, true, __Sz><<<config.block_per_grid,      \
                                               config.thread_per_block,    \
                                               0,                          \
                                               dev_ctx.stream()>>>(        \
        p_src, p_index, p_output, output_dims[0], index_size, slice_size); \
    break
    CASE_VEC_SIZE(8);
    CASE_VEC_SIZE(4);
    CASE_VEC_SIZE(2);
    CASE_VEC_SIZE(1);
#undef CASE_VEC_SIZE
    default:
      PADDLE_THROW(common::errors::Unimplemented(
          "Unsupported vectorized size: %d", vec_size));
  }
}

// The function is only for scatter grad x,
// however update grad use gather
template <typename T, typename IndexT = int>
void GPUScatterGradForX(const phi::GPUContext& dev_ctx,
                        const DenseTensor& index,
                        DenseTensor* output) {
  int64_t index_size = index.dims().size() == 0 ? 1 : index.dims()[0];
  auto dst_dims = output->dims();
  // slice size
  int64_t slice_size = 1;
  for (int i = 1; i < dst_dims.size(); ++i) slice_size *= dst_dims[i];
  const IndexT* p_index = index.data<IndexT>();
  T* p_output = output->data<T>();
  const size_t& slice_bytes = slice_size * sizeof(T);

  // set block and grid num
  int64_t block = 512;
  int64_t n = slice_size * index_size;
  int64_t height = (n + block - 1) / block;
  dim3 grid = dim3((n + block - 1) / block);
  phi::backends::gpu::LimitGridDim(dev_ctx, &grid);

  ScatterInitCUDAKernel<T, IndexT><<<grid, block, 0, dev_ctx.stream()>>>(
      p_index, p_output, dst_dims[0], index_size, slice_size);
}

template <typename T, typename IndexT = int>
void GPUScatterNdAdd(const phi::GPUContext& dev_ctx,
                     const DenseTensor& update,
                     const DenseTensor& index,
                     DenseTensor* output) {
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
  auto remain_ddim = common::slice_ddim(index_dims, 0, index_dims_size - 1);
  int64_t remain_numel = common::product(remain_ddim);
  // slice size
  int64_t slice_size = 1;
  for (int64_t i = end_size; i < output_dims_size; ++i) {
    slice_size *= output_dims[i];
  }
  const size_t slice_bytes = slice_size * sizeof(T);

  Dim<DDim::kMaxRank> g_output_dims;
  for (int i = 0; i < output_dims_size; ++i) {
    g_output_dims[i] = output_dims[i];
  }

  int vec_size = 8;
  vec_size = std::min(phi::GetVectorizedSize(p_update), vec_size);
  vec_size = std::min(phi::GetVectorizedSize(p_output), vec_size);
  while (vec_size > 1 && slice_size % vec_size != 0) {
    vec_size /= 2;
  }

  constexpr int loop_count = 4;
  auto config = phi::backends::gpu::GetGpuLaunchConfig1D(
      dev_ctx, remain_numel * slice_size, vec_size * loop_count);

  auto stream = dev_ctx.stream();
  switch (vec_size) {
#define CASE_VEC_SIZE(__Sz)                                              \
  case __Sz:                                                             \
    ScatterNdCUDAKernel<T, IndexT, __Sz>                                 \
        <<<config.block_per_grid, config.thread_per_block, 0, stream>>>( \
            p_update,                                                    \
            p_index,                                                     \
            p_output,                                                    \
            g_output_dims,                                               \
            remain_numel,                                                \
            slice_size,                                                  \
            end_size);                                                   \
    break
    CASE_VEC_SIZE(8);
    CASE_VEC_SIZE(4);
    CASE_VEC_SIZE(2);
    CASE_VEC_SIZE(1);
#undef CASE_VEC_SIZE
    default:
      PADDLE_THROW(common::errors::Unimplemented(
          "Unsupported vectorized size: %d", vec_size));
  }
}

inline int64_t ensure_nonempty_size(const phi::DenseTensor& t, int64_t dim) {
  return t.dims().size() == 0 ? 1 : t.dims()[dim];
}

inline int64_t ensure_nonempty_stride(const phi::DenseTensor& t, int64_t dim) {
  if (t.dims().size() == 0) {
    return 1;
  }
  auto strides = common::stride(t.dims());
  return strides[dim];
}

using IdxVec = std::vector<int64_t>;
inline IdxVec ensure_nonempty_vec(IdxVec vec) {
  if (vec.empty()) {
    vec.push_back(1);
  }
  return vec;
}

inline phi::DDim ensure_nonempty_ddim(phi::DDim dim) {
  if (dim.size() == 0) {
    return phi::make_ddim({1});
  }
  return dim;
}

inline DenseTensor as_strided(const DenseTensor& src,
                              const std::vector<int64_t>& shape,
                              const std::vector<int64_t>& strides) {
  phi::DenseTensor out;
  out.ShareDataWith(src);
  out.Resize(phi::make_ddim(shape));
  out.set_strides(phi::make_ddim(strides));
  return out;
}

inline DenseTensor restride_dim(const phi::DenseTensor& src,
                                int dim,
                                const std::vector<int64_t>& replacement_shape) {
  auto strides = ensure_nonempty_vec(common::vectorize(src.strides()));
  strides[dim] = 0;
  return as_strided(src, replacement_shape, strides);
}

template <int nt, int vt, typename func_t>
__global__ void scatter_gather_elementwise_kernel(int N, func_t f) {
  constexpr int nv = nt * vt;
  int idx = nv * blockIdx.x + threadIdx.x;

#pragma unroll
  for (int i = 0; i < vt; ++i) {
    if (idx < N) {
      f(idx);
      idx += nt;
    }
  }
}

template <typename T, typename IndexT = int>
void GPUScatterAdd(const phi::GPUContext& ctx,
                   const DenseTensor& src,
                   const DenseTensor& index,
                   DenseTensor* output,
                   int dim) {
  if (index.numel() == 0 || src.numel() == 0) return;

  auto index_dims = src.dims();
  auto index_sizes = ensure_nonempty_vec(common::vectorize(index_dims));
  auto self_strides = ensure_nonempty_vec(common::vectorize(output->strides()));
  auto src_strides = ensure_nonempty_vec(common::vectorize(src.strides()));

  auto self_restrided = restride_dim(*output, dim, index_sizes);
  auto src_restrided = as_strided(src, index_sizes, src_strides);

  int64_t numel = 0;
  std::vector<int64_t> desired_shape;
  std::array<int64_t*, 3> strides_array;
  std::array<std::vector<int64_t>, 3> strides_vec;

  std::vector<int64_t> new_strides(index_dims.size(), 0);
  if (!new_strides.empty()) {
    new_strides[0] = index.strides()[0];
  }

  ScatterAddStride<3>(common::vectorize(src_restrided.dims()),
                      common::vectorize(src_restrided.strides()),
                      phi::SizeOf(src_restrided.dtype()),
                      common::vectorize(self_restrided.dims()),
                      common::vectorize(self_restrided.strides()),
                      phi::SizeOf(self_restrided.dtype()),
                      index_sizes,
                      new_strides,
                      phi::SizeOf(index.dtype()),
                      &desired_shape,
                      &strides_array,
                      &numel,
                      strides_vec);

  auto self_dim_stride = ensure_nonempty_stride(*output, dim);
  auto self_dim_size = ensure_nonempty_size(*output, dim);
  auto index_stride = self_dim_stride;
  auto index_size = self_dim_size;

  char* self_ptr = reinterpret_cast<char*>(self_restrided.data<T>());
  const char* src_ptr = reinterpret_cast<const char*>(src_restrided.data<T>());
  const char* index_ptr = reinterpret_cast<const char*>(index.data<IndexT>());

  auto offset_calc =
      make_offset_calculator_put<3>(desired_shape, strides_array);

  auto reduce_add = [=] __device__(int i) {
    const auto offsets = offset_calc.get(i);
    int64_t idx_dim = *reinterpret_cast<const int64_t*>(index_ptr + offsets[2]);

    T* self_data = reinterpret_cast<T*>(self_ptr + offsets[0]);
    const T* src_data = reinterpret_cast<const T*>(src_ptr + offsets[1]);

    phi::fastAtomicAdd(self_data, idx_dim * index_stride, numel, *src_data);
  };  // NOLINT

  int64_t N;
  const auto output_dims = common::vectorize(output->dims());

  if (index.numel() == output_dims[dim]) {
    N = output->numel();
  } else {
    auto adjusted_dims = output_dims;
    adjusted_dims[dim] = index.numel();
    N = std::accumulate(adjusted_dims.begin(),
                        adjusted_dims.end(),
                        1LL,
                        std::multiplies<int64_t>());
  }

  constexpr int nt = 128;
  constexpr int vt = 8;
  const dim3 block(nt);
  const dim3 grid((N + block.x * vt - 1) / (block.x * vt));
  auto stream = ctx.stream();

  scatter_gather_elementwise_kernel<nt, vt>
      <<<grid, block, 0, stream>>>(N, reduce_add);
}

}  // namespace funcs
}  // namespace phi
