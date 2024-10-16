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

// Sort by flag descending, True: descending. False: Ascending.
// Default is false.
template <typename T, typename IndType>
void ArgFullSort(const phi::GPUContext& ctx,
                 const DenseTensor* input,
                 DenseTensor* output,
                 DenseTensor* indices,
                 const IndType num_rows,
                 const IndType num_cols,
                 const bool descending) {
  auto cu_stream = ctx.stream();
  DenseTensor input_indices;
  const std::vector<IndType> dims = {num_rows, num_cols};
  auto dim = common::make_ddim(dims);
  input_indices.Resize(dim);
  ctx.template Alloc<IndType>(&input_indices);
  size_t temp_storage_bytes = -1;

  auto ComputeBlockSize = [](IndType col) {
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
      input_indices.data<IndType>(), num_rows, num_cols);

  T* sorted_out_ptr;
  IndType* sorted_indices_ptr;
  const T* inp = input->data<T>();
  T* out = ctx.template Alloc<T>(output);
  IndType* ind = ctx.template Alloc<IndType>(indices);
  sorted_out_ptr = out;
  sorted_indices_ptr = ind;

  // create iter for counting input
  cub::CountingInputIterator<IndType> counting_iter(0);
  // segment_offset is used for move to next row
  cub::TransformInputIterator<IndType,
                              SegmentOffsetIter,
                              cub::CountingInputIterator<IndType>>
      segment_offsets_t(counting_iter, SegmentOffsetIter(num_cols));

  gpuError_t err;
  if (descending) {
    err = cub::DeviceSegmentedRadixSort::SortPairsDescending(
        nullptr,
        temp_storage_bytes,
        inp,
        sorted_out_ptr,
        input_indices.data<IndType>(),
        sorted_indices_ptr,
        num_cols * num_rows,
        num_rows,
        segment_offsets_t,
        segment_offsets_t + 1,
        0,
        sizeof(T) * 8,
        cu_stream);
  } else {
    err =
        cub::DeviceSegmentedRadixSort::SortPairs(nullptr,
                                                 temp_storage_bytes,
                                                 inp,
                                                 sorted_out_ptr,
                                                 input_indices.data<IndType>(),
                                                 sorted_indices_ptr,
                                                 num_cols * num_rows,
                                                 num_rows,
                                                 segment_offsets_t,
                                                 segment_offsets_t + 1,
                                                 0,
                                                 sizeof(T) * 8,
                                                 cu_stream);
  }
  PADDLE_ENFORCE_GPU_SUCCESS(err);

  DenseTensor temp_storage;
  int64_t temp_size = temp_storage_bytes;
  temp_storage.Resize({temp_size});
  ctx.template Alloc<uint8_t>(&temp_storage);

  if (descending) {
    err = cub::DeviceSegmentedRadixSort::SortPairsDescending(
        temp_storage.data<uint8_t>(),
        temp_storage_bytes,
        inp,
        sorted_out_ptr,
        input_indices.data<IndType>(),
        sorted_indices_ptr,
        num_cols * num_rows,
        num_rows,
        segment_offsets_t,
        segment_offsets_t + 1,
        0,
        sizeof(T) * 8,
        cu_stream);
  } else {
    err =
        cub::DeviceSegmentedRadixSort::SortPairs(temp_storage.data<uint8_t>(),
                                                 temp_storage_bytes,
                                                 inp,
                                                 sorted_out_ptr,
                                                 input_indices.data<IndType>(),
                                                 sorted_indices_ptr,
                                                 num_cols * num_rows,
                                                 num_rows,
                                                 segment_offsets_t,
                                                 segment_offsets_t + 1,
                                                 0,
                                                 sizeof(T) * 8,
                                                 cu_stream);
  }

  PADDLE_ENFORCE_GPU_SUCCESS(err);
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
  axis = (axis < 0) ? (in_dims.size() + axis) : axis;
  const T* in_data = input.data<T>();
  auto size = input.numel();
  T* out_data = dev_ctx.template Alloc<T>(output);
  int64_t* ids_data = dev_ctx.template Alloc<int64_t>(indices);

  if (rank == 0) {
    phi::Copy<Context>(dev_ctx, input, dev_ctx.GetPlace(), false, output);
    phi::funcs::set_constant(dev_ctx, indices, static_cast<int64_t>(0));
    return;
  }

  // Use thrust for parallel acceleration when the input size is equal to the
  // length of the ‘axis’ dimension.
  // Compared to the following 'Special case for full sort', ascending sort is
  // 34 times faster and descending sort is 31 times faster.
  if (size == in_dims[axis]) {
#ifdef PADDLE_WITH_CUDA
    const auto& exec_policy = thrust::cuda::par.on(dev_ctx.stream());
#else
    const auto& exec_policy = thrust::hip::par.on(dev_ctx.stream());
#endif
    if (stable) {
      thrust::sequence(exec_policy, ids_data, ids_data + size);
      thrust::copy(exec_policy, in_data, in_data + size, out_data);
      if (descending) {
        thrust::stable_sort_by_key(exec_policy,
                                   out_data,
                                   out_data + size,
                                   ids_data,
                                   thrust::greater<T>());
      } else {
        thrust::stable_sort_by_key(
            exec_policy, out_data, out_data + size, ids_data);
      }
      return;
    } else {
      thrust::sequence(exec_policy, ids_data, ids_data + size);
      thrust::copy(exec_policy, in_data, in_data + size, out_data);
      thrust::sort_by_key(exec_policy, out_data, out_data + size, ids_data);
      if (descending) {
        thrust::reverse(exec_policy, out_data, out_data + size);
        thrust::reverse(exec_policy, ids_data, ids_data + size);
      }
      return;
    }
  }

  // Special case for full sort, speedup ~190x.
  if (axis == -1 || axis + 1 == in_dims.size()) {
    const int64_t input_height =
        common::product(common::slice_ddim(in_dims, 0, in_dims.size() - 1));
    const int64_t input_width = in_dims[in_dims.size() - 1];
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
    dev_ctx.template Alloc<int64_t>(indices);

    ArgFullSort<T, int64_t>(dev_ctx,
                            &trans_inp,
                            &tmp_out,
                            &tmp_indices,
                            input_height,
                            input_width,
                            descending);

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
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {
  kernel->OutputAt(1).SetDataType(phi::DataType::INT64);
}
