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
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
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

template <>
struct radix_key_codec_base<phi::dtype::bfloat16>
    : radix_key_codec_integral<phi::dtype::bfloat16, uint16_t> {};

#if HIP_VERSION >= 50400000
template <>
struct float_bit_mask<phi::dtype::float16> : float_bit_mask<rocprim::half> {};

template <>
struct float_bit_mask<phi::dtype::bfloat16>
    : float_bit_mask<rocprim::bfloat16> {};
#endif
}  // namespace detail
}  // namespace rocprim
#else
// set cub base traits in order to handle float16
namespace cub {
template <>
struct NumericTraits<phi::dtype::float16>
    : BaseTraits<FLOATING_POINT, true, false, uint16_t, phi::dtype::float16> {};

template <>
struct NumericTraits<phi::dtype::bfloat16>
    : BaseTraits<FLOATING_POINT, true, false, uint16_t, phi::dtype::bfloat16> {
};
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

template <typename T, typename IndType>
__global__ void merge_kernel(const T* A,
                             size_t sizeA,
                             const T* B,
                             size_t sizeB,
                             const IndType* ids_A,
                             const IndType* ids_B,
                             T* out,
                             IndType* out_ids,
                             bool descending) {
  int64_t thread = blockDim.x * gridDim.x;
  int64_t num_per_thread = (sizeA + sizeB + thread) / thread;
  for (int offset = 0; offset < num_per_thread; offset++) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x + offset * thread;
    size_t total = sizeA + sizeB;
    if (idx >= total) return;
    size_t left = (idx > sizeB) ? idx - sizeB : 0;
    size_t right = (idx < sizeA) ? idx : sizeA;
    while (left < right) {
      size_t mid = (left + right) / 2;
      size_t b_idx = idx - mid;

      T A_mid, B_bidx;
      if (descending) {
        A_mid = (mid >= sizeA) ? std::numeric_limits<T>::lowest() : A[mid];
        B_bidx = (b_idx >= sizeB) ? std::numeric_limits<T>::lowest() : B[b_idx];
      } else {
        A_mid = (mid >= sizeA) ? std::numeric_limits<T>::max() : A[mid];
        B_bidx = (b_idx >= sizeB) ? std::numeric_limits<T>::max() : B[b_idx];
      }

      if (descending ? (A_mid >= B_bidx) : (A_mid <= B_bidx))
        left = mid + 1;
      else
        right = mid;
    }

    size_t a_idx = left;
    size_t b_idx = idx - a_idx;
    if (a_idx >= sizeA) {
      if (descending ? (A[sizeA - 1] < B[b_idx]) : (A[sizeA - 1] > B[b_idx])) {
        out[idx] = A[sizeA - 1];
        out_ids[idx] = ids_A[sizeA - 1];
      } else {
        out[idx] = B[b_idx];
        out_ids[idx] = ids_B[b_idx];
      }
    } else if (b_idx >= sizeB) {
      out[idx] = A[a_idx];
      out_ids[idx] = ids_A[a_idx];
    } else {
      if (descending ? (A[a_idx] >= B[b_idx]) : (A[a_idx] <= B[b_idx])) {
        out[idx] = A[a_idx];
        out_ids[idx] = ids_A[a_idx];
      } else if (descending ? (a_idx > 0 && (A[a_idx - 1] < B[b_idx]))
                            : (a_idx > 0 && (A[a_idx - 1] > B[b_idx]))) {
        out[idx] = A[a_idx - 1];
        out_ids[idx] = ids_A[a_idx - 1];
      } else {
        out[idx] = B[b_idx];
        out_ids[idx] = ids_B[b_idx];
      }
    }
  }
}

template <typename T>
static __global__ void FillIndex(T* indices, T num_rows, T num_cols) {
  int col_id = threadIdx.x;
  int row_id = blockIdx.x;

  for (T j = row_id; j < num_rows; j += gridDim.x) {
    for (T i = col_id; i < num_cols; i += blockDim.x) {
      indices[j * num_cols + i] = i;
    }
  }
}

#define CUB_ARGSORT_WRAPPER(func, ...)                                        \
  {                                                                           \
    size_t temp_storage_bytes = 0;                                            \
    PADDLE_ENFORCE_GPU_SUCCESS(                                               \
        func(nullptr, temp_storage_bytes, __VA_ARGS__));                      \
    DenseTensor temp_storage;                                                 \
    int64_t temp_size = static_cast<int64_t>(temp_storage_bytes);             \
    PADDLE_ENFORCE_GT(                                                        \
        temp_size,                                                            \
        0,                                                                    \
        common::errors::InvalidArgument(                                      \
            "Argsort temp storage size is %d, but should be greater than 0.", \
            temp_size));                                                      \
    temp_storage.Resize({temp_size});                                         \
    dev_ctx.template Alloc<uint8_t>(&temp_storage);                           \
    PADDLE_ENFORCE_GPU_SUCCESS(                                               \
        func(temp_storage.data<uint8_t>(), temp_storage_bytes, __VA_ARGS__)); \
  }

#define PREDICATE_CUB_ARGSORT(predicate, if_func, else_func, ...) \
  if (predicate)                                                  \
    CUB_ARGSORT_WRAPPER(if_func, __VA_ARGS__)                     \
  else                                                            \
    CUB_ARGSORT_WRAPPER(else_func, __VA_ARGS__)

// Sort by flag descending, True: descending. False: Ascending.
// Default is false.
template <typename T, typename IndType>
void ArgFullSort(const phi::GPUContext& dev_ctx,
                 const DenseTensor* input,
                 DenseTensor* output,
                 DenseTensor* indices,
                 const int64_t num_rows,
                 const int64_t num_cols,
                 const bool descending) {
  auto cu_stream = dev_ctx.stream();
  auto ComputeBlockSize = [](IndType col) {
    if (col > 512)
      return 1024;
    else if (col > 256 && col <= 512)
      return 512;
    else if (col > 128 && col <= 256)
      return 256;
    else
      return 128;
  };
  const int block_size = ComputeBlockSize(num_cols);
  const int64_t maxGridDimX = dev_ctx.GetCUDAMaxGridDimSize()[0];

  const T* inp = input->data<T>();
  IndType* sorted_indices_ptr = indices->data<IndType>();

  // create iter for counting input
  cub::CountingInputIterator<IndType> counting_iter(0);
  // segment_offset is used for move to next row
  cub::TransformInputIterator<IndType,
                              SegmentOffsetIter,
                              cub::CountingInputIterator<IndType>>
      segment_offsets_t(counting_iter, SegmentOffsetIter(num_cols));

  // num_rows is the total segments to be sorted
  constexpr int64_t max_elements = 1 << 30;
  const int64_t total_elements = num_cols * num_rows;
  const int64_t segment_size = num_cols;
  const int64_t element_per_call = std::min(max_elements, total_elements);
  // make sure batch size is the multiple of segment_size
  const int64_t batch_size = (element_per_call / segment_size) * segment_size;
  int64_t offset = 0;
  DenseTensor input_indices;

  T* sorted_out_ptr = sorted_out_ptr = output->data<T>();
  IndType* ind_ptr = nullptr;

  while (offset < total_elements) {
    const int64_t n_elements = std::min(batch_size, total_elements - offset);
    const int64_t n_segments = n_elements / segment_size;

    // allocate a temporary storage for input indices, with shape:
    // [num_segments = n_elements / segment_size, segment_size]
    // will be de-allocated once the sort is done, to save memory and
    // avoid repeated allocation and deallocation
    if (input_indices.initialized()) {
      ind_ptr = input_indices.data<IndType>();
    } else {
      input_indices.Resize({n_segments, segment_size});
      ind_ptr = dev_ctx.template Alloc<IndType>(&input_indices);
    }
    const int64_t grid_size = std::min(n_segments, maxGridDimX);
    // Init a index array
    FillIndex<<<grid_size, block_size, 0, cu_stream>>>(
        ind_ptr, n_segments, segment_size);

    PREDICATE_CUB_ARGSORT(descending,
                          cub::DeviceSegmentedRadixSort::SortPairsDescending,
                          cub::DeviceSegmentedRadixSort::SortPairs,
                          inp + offset,
                          sorted_out_ptr + offset,
                          ind_ptr,
                          sorted_indices_ptr + offset,
                          n_elements,
                          n_segments,
                          segment_offsets_t,
                          segment_offsets_t + 1,
                          0,
                          sizeof(T) * 8,
                          cu_stream);
    offset += n_elements;
  }
}
template <typename T, typename IndType>
void PerSort(const phi::GPUContext& dev_ctx,
             T* out_data,
             int64_t* ids_data,
             IndType start,
             IndType end,
             bool stable,
             bool descending) {
#ifdef PADDLE_WITH_CUDA
  const auto& exec_policy = thrust::cuda::par.on(dev_ctx.stream());
#else
  const auto& exec_policy = thrust::hip::par.on(dev_ctx.stream());
#endif
  if (stable) {
    if (descending) {
      thrust::stable_sort_by_key(exec_policy,
                                 out_data + start,
                                 out_data + end,
                                 ids_data + start,
                                 thrust::greater<T>());
    } else {
      thrust::stable_sort_by_key(
          exec_policy, out_data + start, out_data + end, ids_data + start);
    }
    return;
  } else {
    thrust::sort_by_key(
        exec_policy, out_data + start, out_data + end, ids_data + start);
    if (descending) {
      thrust::reverse(exec_policy, out_data + start, out_data + end);
      thrust::reverse(exec_policy, ids_data + start, ids_data + end);
    }
    return;
  }
}

template <typename T, typename Context>
void ArgsortKernel(const Context& dev_ctx,
                   const DenseTensor& input,
                   int axis,
                   bool descending,
                   bool stable,
                   DenseTensor* output,
                   DenseTensor* indices) {
  auto in_dims = input.dims();
  auto rank = in_dims.size();

  if (input.numel() == 0) {
    output->Resize(in_dims);
    indices->Resize(in_dims);
    dev_ctx.template Alloc<T>(output);
    dev_ctx.template Alloc<int64_t>(indices);
    return;
  }

  axis = (axis < 0) ? (in_dims.size() + axis) : axis;
  const T* in_data = input.data<T>();
  auto size = input.numel();

  if (rank == 0) {
    dev_ctx.template Alloc<T>(output);
    dev_ctx.template Alloc<int64_t>(indices);
    phi::Copy<Context>(dev_ctx, input, dev_ctx.GetPlace(), false, output);
    phi::funcs::set_constant(dev_ctx, indices, static_cast<int64_t>(0));
    return;
  }

  // Use thrust for parallel acceleration when the input size is equal to the
  // length of the 'axis' dimension.
  // Compared to the following 'Special case for full sort', ascending sort is
  // 34 times faster and descending sort is 31 times faster.
  if (size == in_dims[axis]) {
    T* out_data = dev_ctx.template Alloc<T>(output);
    int64_t* ids_data = dev_ctx.template Alloc<int64_t>(indices);
#ifdef PADDLE_WITH_CUDA
    const auto& exec_policy = thrust::cuda::par.on(dev_ctx.stream());
#else
    const auto& exec_policy = thrust::hip::par.on(dev_ctx.stream());
#endif
    auto cu_stream = dev_ctx.stream();
    thrust::sequence(exec_policy, ids_data, ids_data + size);
    thrust::copy(exec_policy, in_data, in_data + size, out_data);
    const int64_t per_number = (1LL << 31) - 1;
    int64_t start = 0;
    int64_t end = std::min(start + per_number, size);
    if (end == size) {
      PerSort<T, int64_t>(
          dev_ctx, out_data, ids_data, start, end, stable, descending);
    } else {
      // Sorting the segments and then merging them
      DenseTensor temp;
      DenseTensor ids;
      temp.Resize(in_dims);
      ids.Resize(in_dims);
      T* temp_data = dev_ctx.template Alloc<T>(&temp);
      int64_t* temp_ids = dev_ctx.template Alloc<int64_t>(&ids);

      while (start != size) {
        PerSort<T, int64_t>(
            dev_ctx, out_data, ids_data, start, end, stable, descending);
        if (start != 0) {
          auto config = phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, end);
          merge_kernel<<<config.block_per_grid.x,
                         config.thread_per_block.x,
                         0,
                         cu_stream>>>(out_data,
                                      start,
                                      out_data + start,
                                      end - start,
                                      ids_data,
                                      ids_data + start,
                                      temp_data,
                                      temp_ids,
                                      descending);
          thrust::copy(exec_policy, temp_ids, temp_ids + end, ids_data);
          thrust::copy(exec_policy, temp_data, temp_data + end, out_data);
        }
        start = end;
        end = std::min(start + per_number, size);
      }
    }
    return;
  }

  // Special case for full sort, speedup ~190x.
  if (axis == -1 || axis + 1 == in_dims.size()) {
    const int64_t input_height =
        common::product(common::slice_ddim(in_dims, 0, in_dims.size() - 1));
    const int64_t input_width = in_dims[in_dims.size() - 1];
    dev_ctx.template Alloc<int64_t>(indices);
    dev_ctx.template Alloc<T>(output);
    ArgFullSort<T, int64_t>(dev_ctx,
                            &input,
                            output,
                            indices,
                            input_height,
                            input_width,
                            descending);
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
    T* trans_inp_data = dev_ctx.template Alloc<T>(&trans_inp);
    // Do transpose
    TransposeKernel<T, Context>(dev_ctx, input, trans, &trans_inp);

    const int64_t input_height = common::product(
        common::slice_ddim(trans_dims, 0, trans_dims.size() - 1));
    const int64_t input_width = trans_dims[trans_dims.size() - 1];

    DenseTensor tmp_out;
    tmp_out.Resize(trans_dims);
    dev_ctx.template Alloc<T>(&tmp_out);

    DenseTensor tmp_indices;
    // temp indices for sorting
    tmp_indices.Resize(trans_dims);
    dev_ctx.template Alloc<int64_t>(&tmp_indices);

    ArgFullSort<T, int64_t>(dev_ctx,
                            &trans_inp,
                            &tmp_out,
                            &tmp_indices,
                            input_height,
                            input_width,
                            descending);
    // delay output allocation until after transpose, to avoid
    // allocating too much memory
    dev_ctx.template Alloc<T>(output);
    dev_ctx.template Alloc<int64_t>(indices);
    // transpose back
    TransposeKernel<T, Context>(dev_ctx, tmp_out, trans, output);
    TransposeKernel<int64_t, Context>(dev_ctx, tmp_indices, trans, indices);
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
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {
  kernel->OutputAt(1).SetDataType(phi::DataType::INT64);
}
