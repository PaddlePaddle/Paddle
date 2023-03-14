// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/argsort_kernel.h"

#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#ifdef __NVCC__
#include "cub/cub.cuh"
#endif
#ifdef __HIPCC__
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#endif

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/primitive/functor_primitives.h"
#include "paddle/phi/kernels/transpose_kernel.h"

#ifdef __HIPCC__
namespace rocprim {
namespace detail {
template <>
struct radix_key_codec_base<phi::dtype::float16>
    : radix_key_codec_integral<phi::dtype::float16, uint16_t> {};
}  // namespace detail
}  // namespace rocprim
#else
// set cub base traits in order to handle float16
namespace cub {
template <>
struct NumericTraits<phi::dtype::float16>
    : BaseTraits<FLOATING_POINT, true, false, uint16_t, phi::dtype::float16> {};
}  // namespace cub
#endif

namespace phi {

// Iter for move to next row
struct SegmentOffsetIter {
  EIGEN_DEVICE_FUNC
  explicit SegmentOffsetIter(int num_cols) : num_cols_(num_cols) {}

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE int operator()(int idx) const {
    return idx * num_cols_;
  }

  int num_cols_;
};

#define PADDLE_CUDA_NUM_THREADS 1024

template <typename T>
static __global__ void FillIndex(T *indices, T num_rows, T num_cols) {
  int col_id = threadIdx.x;
  int row_id = blockIdx.x;

  for (T j = row_id; j < num_rows; j += gridDim.x) {
    for (T i = col_id; i < num_cols; i += blockDim.x) {
      indices[j * num_cols + i] = i;
    }
  }
}

// Sort by flag descending, True: descending. False: Ascending.
// Default is false.
static __global__ void FillIndexAndSegmentKernel(int2 *data,
                                                 int numel,
                                                 int nsort) {
  CUDA_KERNEL_LOOP(idx, numel) {
    auto segment = static_cast<int>(idx / nsort);
    auto sort = static_cast<int>(idx % nsort);
    data[idx] = int2{segment, sort};
  }
}

#define CUB_WRAPPER(func, ctx, ...)                                            \
  do {                                                                         \
    size_t temp_storage_bytes = 0;                                             \
    gpuError_t err;                                                            \
    err = func(nullptr, temp_storage_bytes, __VA_ARGS__);                      \
    PADDLE_ENFORCE_GPU_SUCCESS(err);                                           \
    DenseTensor temp_storage;                                                  \
    int64_t temp_size = temp_storage_bytes;                                    \
    temp_storage.Resize({temp_size});                                          \
    ctx.template Alloc<uint8_t>(&temp_storage);                                \
    err = func(temp_storage.data<uint8_t>(), temp_storage_bytes, __VA_ARGS__); \
    PADDLE_ENFORCE_GPU_SUCCESS(err);                                           \
  } while (false)

template <typename KT, typename VT>
static void RadixSortPairs(const phi::GPUContext &ctx,
                           const KT *keys_in,
                           const VT *values_in,
                           KT *keys_out,
                           VT *values_out,
                           int64_t n,
                           bool descending = false,
                           int64_t begin_bit = 0,
                           int64_t end_bit = sizeof(KT) * 8) {
  if (keys_out == nullptr) {
    DenseTensor key_out_owner;
    key_out_owner.Resize({n});
    ctx.template Alloc<KT>(&key_out_owner);
    keys_out = key_out_owner.data<KT>();
  }

  if (descending) {
    CUB_WRAPPER(cub::DeviceRadixSort::SortPairsDescending,
                ctx,
                keys_in,
                keys_out,
                values_in,
                values_out,
                n,
                begin_bit,
                end_bit,
                ctx.stream());
  } else {
    CUB_WRAPPER(cub::DeviceRadixSort::SortPairs,
                ctx,
                keys_in,
                keys_out,
                values_in,
                values_out,
                n,
                begin_bit,
                end_bit,
                ctx.stream());
  }
}

template <typename KT>
static void RadixSortKeys(const phi::GPUContext &ctx,
                          const KT *keys_in,
                          KT *keys_out,
                          int64_t n,
                          bool descending,
                          int64_t begin_bit,
                          int64_t end_bit) {
  if (descending) {
    CUB_WRAPPER(cub::DeviceRadixSort::SortKeysDescending,
                ctx,
                keys_in,
                keys_out,
                n,
                begin_bit,
                end_bit,
                ctx.stream());
  } else {
    CUB_WRAPPER(cub::DeviceRadixSort::SortKeys,
                ctx,
                keys_in,
                keys_out,
                n,
                begin_bit,
                end_bit,
                ctx.stream());
  }
}

template <typename T>
static __global__ void SortPostprocessKernel(const T *in,
                                             const int2 *i_s_ptr,
                                             T *out,
                                             int64_t *index,
                                             int nsegments,
                                             int nsort) {
  CUDA_KERNEL_LOOP(i, nsegments * nsort) {
    int segment = i / nsort;  // segment_id
    int j = i % nsort;

    int offset = segment * nsort;
    const T *in_ = in + offset;
    T *out_ = out + offset;
    int64_t *index_ = index + offset;
    const int2 *i_s_ptr_ = i_s_ptr + offset;

    int idx = i_s_ptr_[j].y;
    index_[j] = idx;
    out_[j] = in_[idx];
  }
}

template <typename T>
inline void SegmentedSortPairsByFullSort(const phi::GPUContext &ctx,
                                         const T *const self_ptr,
                                         T *const values_ptr,
                                         int64_t *const indices_ptr,
                                         const int64_t nsegments,
                                         const int64_t nsort,
                                         const int64_t n,
                                         const bool descending) {
  int64_t segment_bits = std::max<int64_t>(
      1L, static_cast<int64_t>(std::ceil(std::log2(nsegments))));

  const auto numel = nsort * nsegments;

  DenseTensor indices_and_segment;
  int64_t indices_and_segment_size = numel;
  indices_and_segment.Resize({indices_and_segment_size * 2});
  ctx.template Alloc<int64_t>(&indices_and_segment);
  auto i_s_ptr_base = indices_and_segment.data<int64_t>();
  auto i_s_ptr = reinterpret_cast<int2 *>(i_s_ptr_base);

  dim3 block = PADDLE_CUDA_NUM_THREADS;
  auto block_num = (numel - 1) / PADDLE_CUDA_NUM_THREADS + 1;
  dim3 grid = static_cast<int>(block_num);

  auto cu_stream = ctx.stream();

  FillIndexAndSegmentKernel<<<grid, block, 0, cu_stream>>>(
      i_s_ptr, numel, nsort);

  DenseTensor indices_and_segment2;
  int64_t indices_and_segment2_size = numel;
  indices_and_segment2.Resize({indices_and_segment2_size * 2});
  ctx.template Alloc<int64_t>(&indices_and_segment2);
  auto i_s_ptr2_base = indices_and_segment2.data<int64_t>();
  auto i_s_ptr2 = reinterpret_cast<int2 *>(i_s_ptr2_base);

  RadixSortPairs<T, int2>(
      ctx, self_ptr, i_s_ptr, nullptr, i_s_ptr2, n, descending);

  RadixSortKeys<int64_t>(ctx,
                         reinterpret_cast<int64_t *>(i_s_ptr2),
                         reinterpret_cast<int64_t *>(i_s_ptr),
                         n,
                         false,
                         0,
                         segment_bits);

  SortPostprocessKernel<<<grid, block, 0, cu_stream>>>(
      self_ptr, i_s_ptr, values_ptr, indices_ptr, nsegments, nsort);
}

// The method is called when # of the rows of the input is less than or equal to
// 4
template <typename T, typename IndexType>
void ArgFullSortForTinyRows(const phi::GPUContext &ctx,
                            const DenseTensor *input,
                            DenseTensor *output,
                            DenseTensor *indices,
                            const IndexType num_rows,
                            const IndexType num_cols,
                            const bool descending) {
  auto gpu_stream = ctx.stream();
  size_t temp_storage_bytes = -1;

  IndexType numel = num_rows * num_cols;
  if (numel == 0) {
    return;
  }

  IndexType numel_or_intmax =
      std::min(numel, static_cast<int64_t>(std::numeric_limits<int>::max()));
  IndexType nsort = num_cols;
  IndexType nbatch = (numel_or_intmax / nsort) * nsort;

  T *sorted_out_ptr;
  IndexType *sorted_indices_ptr;
  const T *input_data = input->data<T>();
  T *out = ctx.template Alloc<T>(output);
  IndexType *ind = ctx.template Alloc<IndexType>(indices);
  sorted_out_ptr = out;
  sorted_indices_ptr = ind;

  int64_t remaining = numel;

  while (remaining > 0) {
    int64_t n = std::min(remaining, nbatch);
    IndexType nsegments = n / nsort;

    SegmentedSortPairsByFullSort(ctx,
                                 input_data,
                                 sorted_out_ptr,
                                 sorted_indices_ptr,
                                 nsegments,
                                 nsort,
                                 n,
                                 descending);

    remaining -= n;
    input_data += n;
    sorted_out_ptr += n;
    sorted_indices_ptr += n;
  }
}

template <typename T, typename IndexType>
void ArgFullSort(const phi::GPUContext &ctx,
                 const DenseTensor *input,
                 DenseTensor *output,
                 DenseTensor *indices,
                 const IndexType num_rows,
                 const IndexType num_cols,
                 const bool descending) {
  auto cu_stream = ctx.stream();
  DenseTensor input_indices;
  const std::vector<IndexType> dims = {num_rows, num_cols};
  auto dim = phi::make_ddim(dims);
  input_indices.Resize(dim);
  ctx.template Alloc<IndexType>(&input_indices);
  size_t temp_storage_bytes = -1;

  auto ComputeBlockSize = [](IndexType col) {
    if (col > 512)
      return 1024;
    else if (col > 256 && col <= 512)
      return 512;
    else if (col > 128 && col <= 256)
      return 256;
    else if (col > 64 && col <= 128)
      return 128;
    else
      return 64;
  };

  int block_size = ComputeBlockSize(num_cols);
  int maxGridDimX = ctx.GetCUDAMaxGridDimSize()[0];
  // actually, int num_rows < max_grid_size
  int grid_size = num_rows < maxGridDimX ? num_rows : maxGridDimX;
  // Init a index array
  FillIndex<<<grid_size, block_size, 0, cu_stream>>>(
      input_indices.data<IndexType>(), num_rows, num_cols);

  T *sorted_out_ptr;
  IndexType *sorted_indices_ptr;
  const T *inp = input->data<T>();
  T *out = ctx.template Alloc<T>(output);
  IndexType *ind = ctx.template Alloc<IndexType>(indices);
  sorted_out_ptr = out;
  sorted_indices_ptr = ind;

  // create iter for counting input
  cub::CountingInputIterator<IndexType> counting_iter(0);
  // segment_offset is used for move to next row
  cub::TransformInputIterator<IndexType,
                              SegmentOffsetIter,
                              cub::CountingInputIterator<IndexType>>
      segment_offsets_t(counting_iter, SegmentOffsetIter(num_cols));

  gpuError_t err;
  if (descending) {
    CUB_WRAPPER(cub::DeviceSegmentedRadixSort::SortPairsDescending,
                ctx,
                inp,
                sorted_out_ptr,
                input_indices.data<IndexType>(),
                sorted_indices_ptr,
                num_cols * num_rows,
                num_rows,
                segment_offsets_t,
                segment_offsets_t + 1,
                0,
                sizeof(T) * 8,
                ctx.stream());
  } else {
    CUB_WRAPPER(cub::DeviceSegmentedRadixSort::SortPairs,
                ctx,
                inp,
                sorted_out_ptr,
                input_indices.data<IndexType>(),
                sorted_indices_ptr,
                num_cols * num_rows,
                num_rows,
                segment_offsets_t,
                segment_offsets_t + 1,
                0,
                sizeof(T) * 8,
                ctx.stream());
  }
}

template <typename T, typename Context>
void ArgsortKernel(const Context &dev_ctx,
                   const DenseTensor &input,
                   int axis,
                   bool descending,
                   DenseTensor *output,
                   DenseTensor *indices) {
  auto in_dims = input.dims();
  auto rank = in_dims.size();
  axis = (axis < 0) ? (in_dims.size() + axis) : axis;

  const T *in_data = input.data<T>();
  auto size = input.numel();
  T *out_data = dev_ctx.template Alloc<T>(output);
  int64_t *ids_data = dev_ctx.template Alloc<int64_t>(indices);

  if (rank == 0) {
    phi::Copy<Context>(dev_ctx, input, dev_ctx.GetPlace(), false, output);
    phi::funcs::set_constant(dev_ctx, indices, 0);
    return;
  }

  // Use thrust for parallel acceleration when the input size is equal to the
  // length of the ‘axis’ dimension.
  // Compared to the following 'Special case for full sort', ascending sort is
  // 34 times faster and descending sort is 31 times faster.
  if (size == in_dims[axis]) {
    thrust::sequence(thrust::device, ids_data, ids_data + size);
    thrust::copy(thrust::device, in_data, in_data + size, out_data);
    thrust::sort_by_key(thrust::device, out_data, out_data + size, ids_data);
    if (descending) {
      thrust::reverse(thrust::device, out_data, out_data + size);
      thrust::reverse(thrust::device, ids_data, ids_data + size);
    }
    return;
  }

  // Special case for full sort, speedup ~190x.
  if (axis == -1 || axis + 1 == in_dims.size()) {
    const int64_t input_height =
        phi::product(phi::slice_ddim(in_dims, 0, in_dims.size() - 1));
    const int64_t input_width = in_dims[in_dims.size() - 1];
    if (input_height <= 4) {
      ArgFullSortForTinyRows<T, int64_t>(dev_ctx,
                                         &input,
                                         output,
                                         indices,
                                         input_height,
                                         input_width,
                                         descending);
    } else {
      ArgFullSort<T, int64_t>(dev_ctx,
                              &input,
                              output,
                              indices,
                              input_height,
                              input_width,
                              descending);
    }
  } else {
    // if not full sort, do transpose first
    std::vector<int> trans;
    for (int i = 0; i < axis; i++) {
      trans.push_back(i);
    }
    trans.push_back(in_dims.size() - 1);
    for (int i = axis + 1; i < in_dims.size() - 1; i++) {
      trans.push_back(i);
    }
    trans.push_back(axis);
    phi::DDim trans_dims(in_dims);
    for (int i = 0; i < trans.size(); i++) {
      trans_dims[i] = in_dims[trans[i]];
    }

    DenseTensor trans_inp;
    trans_inp.Resize(trans_dims);
    T *trans_inp_data = dev_ctx.template Alloc<T>(&trans_inp);
    // Do transpose
    TransposeKernel<T, Context>(dev_ctx, input, trans, &trans_inp);

    const int64_t input_height =
        phi::product(phi::slice_ddim(trans_dims, 0, trans_dims.size() - 1));
    const int64_t input_width = trans_dims[trans_dims.size() - 1];

    DenseTensor tmp_out;
    tmp_out.Resize(trans_dims);
    dev_ctx.template Alloc<T>(&tmp_out);

    DenseTensor tmp_indices;
    // temp indices for sorting
    tmp_indices.Resize(trans_dims);
    dev_ctx.template Alloc<int64_t>(&tmp_indices);
    dev_ctx.template Alloc<int64_t>(indices);

    if (input_height <= 4) {
      ArgFullSortForTinyRows<T, int64_t>(dev_ctx,
                                         &trans_inp,
                                         &tmp_out,
                                         &tmp_indices,
                                         input_height,
                                         input_width,
                                         descending);
    } else {
      ArgFullSort<T, int64_t>(dev_ctx,
                              &trans_inp,
                              &tmp_out,
                              &tmp_indices,
                              input_height,
                              input_width,
                              descending);
    }

    TransposeKernel<int64_t, Context>(dev_ctx, tmp_indices, trans, indices);
    // transpose back
    TransposeKernel<T, Context>(dev_ctx, tmp_out, trans, output);
    return;
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(argsort,
                   GPU,
                   ALL_LAYOUT,
                   phi::ArgsortKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::float16) {}
