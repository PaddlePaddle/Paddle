/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <thrust/execution_policy.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include "paddle/phi/api/lib/utils/allocator.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_meta.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/index_impl.cu.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/primitive/compute_primitives.h"
#include "paddle/phi/kernels/sparse/convolution_kernel.h"
#include "paddle/phi/kernels/sparse/gpu/convolution.cu.h"

namespace phi {
namespace sparse {

__global__ void SetFlagAndUpdateCounterKernel(const int* indexs,
                                              const int n,
                                              const int rulebook_len,
                                              const int kernel_size,
                                              int* rulebook_ptr,
                                              int* counter_ptr) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  extern __shared__ int cache_count[];  // kernel_size
  for (int i = threadIdx.x; i < kernel_size; i += blockDim.x) {
    cache_count[i] = 0;
  }
  __syncthreads();

  for (int i = tid; i < n; i += gridDim.x * blockDim.x) {
    int index = indexs[i];
    int kernel_index = rulebook_ptr[index];
    rulebook_ptr[index + rulebook_len] = -1;
    rulebook_ptr[index + 2 * rulebook_len] = -1;
    rulebook_ptr[index] = -1;
    atomicAdd(&cache_count[kernel_index], 1);
  }
  __syncthreads();

  for (int i = threadIdx.x; i < kernel_size; i += blockDim.x) {
    atomicSub(&counter_ptr[i], cache_count[i]);
  }
}

/**
 * @brief: update the out index and indices
 * unique_keys: save the index of the output feature list
 * unique_values: indiates the index of key before deduplication
 * out_indexs: indicates the position of the output index in the rulebook
 * rulebook_len: indicates the length of rulebook
 * out_dims: indicates the output dims
 * out_indices: the indices of output, out_indices = IndexToPoint(unique_keys)
 * rulebook_out_indexs: the output index in rulebook
**/
__global__ void UpdateIndexKernel(const int* unique_keys,
                                  const int* unique_values,
                                  const int* out_indexs,
                                  const int non_zero_num,
                                  const int rulebook_len,
                                  const Dims4D out_dims,
                                  int* out_indices,
                                  int* rulebook_out_indexs) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  for (int i = tid; i < non_zero_num; i += gridDim.x * blockDim.x) {
    const int index = unique_keys[i];
    int batch, x, y, z;
    IndexToPoint<Dims4D>(index, out_dims, &batch, &x, &y, &z);
    // get out indices
    out_indices[i] = batch;
    out_indices[i + non_zero_num] = z;
    out_indices[i + non_zero_num * 2] = y;
    out_indices[i + non_zero_num * 3] = x;

    // update rulebook
    int start = unique_values[i];
    int end = i == non_zero_num - 1 ? rulebook_len : unique_values[i + 1];
    // max(end-start) = kernel_size
    for (int j = start; j < end; j++) {
      rulebook_out_indexs[out_indexs[j]] = i;
    }
  }
}

/**
 * @brief product rulebook
 * for input_i in x_indices:
 *   if input_i participate in the convolution calculation:
 *       infer the output_i by input_i and kernel_i
 *       save output_i
 *
 * x_indices: the indices of input features
 * x_dims: the input dims
 * kernel_dims: the kernel dims
 * out_dims: the output dims
 * non_zero_num: the number of input features
 * rulebook: the rulebook to save the kernel index, input index and output index
 * counter: save the number of times each location in the kernel participates in
 *the caculation
**/
__global__ void ProductRuleBookKernel(const int* x_indices,
                                      const Dims4D x_dims,
                                      const Dims4D kernel_dims,
                                      const Dims4D out_dims,
                                      const int64_t non_zero_num,
                                      const Dims4D paddings,
                                      const Dims4D dilations,
                                      const Dims4D strides,
                                      const bool subm,
                                      int* rulebook,
                                      int* counter,
                                      int* in_indexs) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  extern __shared__ int counter_buf[];  // kernel_size
  const int kernel_size = kernel_dims[3] * kernel_dims[2] * kernel_dims[1];
  const int offset = kernel_size * non_zero_num;
  for (int i = threadIdx.x; i < kernel_size; i += blockDim.x) {
    counter_buf[i] = 0;
  }
  __syncthreads();

  for (int i = tid; i < non_zero_num; i += gridDim.x * blockDim.x) {
    int kernel_index = 0;
    int batch = x_indices[i];
    int in_z = x_indices[i + non_zero_num];
    int in_y = x_indices[i + 2 * non_zero_num];
    int in_x = x_indices[i + 3 * non_zero_num];
    if (subm) {
      in_indexs[i] = PointToIndex(batch, in_x, in_y, in_z, x_dims);
    }
    for (int kz = 0; kz < kernel_dims[1]; kz++) {
      for (int ky = 0; ky < kernel_dims[2]; ky++) {
        for (int kx = 0; kx < kernel_dims[3]; kx++) {
          int in_i = -1, out_index = -1, kernel_i = -1;
          if (Check(x_dims,
                    kernel_dims,
                    paddings,
                    dilations,
                    strides,
                    in_x,
                    in_y,
                    in_z,
                    kx,
                    ky,
                    kz)) {
            int out_z = (in_z + paddings[1] - kz * dilations[1]) / strides[1];
            int out_y = (in_y + paddings[2] - ky * dilations[2]) / strides[2];
            int out_x = (in_x + paddings[3] - kx * dilations[3]) / strides[3];
            in_i = i;
            out_index =
                PointToIndex<Dims4D>(batch, out_x, out_y, out_z, out_dims);
            atomicAdd(&counter_buf[kernel_index], 1);
            kernel_i = kernel_index;
          }
          rulebook[kernel_index * non_zero_num + i] = kernel_i;
          rulebook[kernel_index * non_zero_num + offset + i] = in_i;
          rulebook[kernel_index * non_zero_num + offset * 2 + i] = out_index;
          ++kernel_index;
        }
      }
    }
  }
  __syncthreads();
  for (int i = threadIdx.x; i < kernel_size; i += blockDim.x) {
    atomicAdd(&counter[i], counter_buf[i]);
  }
}

// brief: calculation the distance between start and end
__global__ void DistanceKernel(const int* start,
                               const int* end,
                               int* distance) {
  if (threadIdx.x == 0) {
    *distance = end - start;
  }
}

// the basic algorithm can refer to convolution_kernel.cc or
// the second paper
// example:
// 1. the rulebook:
//  the kernel_index:                       0, 0, 0, 1, 1, 1, 2, 2, ....
//  the out_index(key):                     20, 30, 33, 30, 33, 20, 25
// 2. mark the index of out_index(value):   0, 1, 2, 3, 4, 5, 6, ....
// 3. sorted the (key, value)
// 4. unique the (key, value):
//  unique_key:     20, 25, 30, 33
//  unique_values:  0, 2, 3, 5
//  the index of unique_values is: 0, 1, 2, 3
// 5. update the out_index by unique_key, uniqe_value and the index of
// unique_value:
//  the new out_index: 0, 2, 3, 2, 3, 0, 1
template <typename T, typename Context>
int ProductRuleBook(const Context& dev_ctx,
                    const SparseCooTensor& x,
                    const DenseTensor& kernel,
                    const std::vector<int>& paddings,
                    const std::vector<int>& dilations,
                    const std::vector<int>& strides,
                    const DDim& out_dims,
                    const bool subm,
                    DenseTensor* rulebook,
                    DenseTensor* counter_per_kernel,
                    DenseTensor* offsets_per_kernel,
                    DenseTensor* out_index,
                    DenseTensor* unique_key,
                    DenseTensor* unique_value,
                    SparseCooTensor* out,
                    std::vector<int>* h_counter,
                    std::vector<int>* h_offsets) {
  const auto& kernel_dims = kernel.dims();
  const int64_t non_zero_num = x.nnz();
  const auto& non_zero_indices = x.non_zero_indices();
  const int* indices_ptr = non_zero_indices.data<int>();
  DenseTensor in_indexs = phi::Empty<Context>(
      dev_ctx, DenseTensorMeta(DataType::INT32, {x.nnz()}, DataLayout::NCHW));
  int* counter_ptr = counter_per_kernel->data<int>();
  int* offsets_ptr = offsets_per_kernel->data<int>();
  int kernel_size = kernel_dims[0] * kernel_dims[1] * kernel_dims[2];
  const int rulebook_rows = 3;
  const int rulebook_cols = kernel_size * non_zero_num;
  rulebook->ResizeAndAllocate({rulebook_rows, rulebook_cols});
  int* rulebook_ptr = rulebook->data<int>();

  const auto x_dims = x.dims();
  Dims4D d_x_dims(x_dims[0], x_dims[3], x_dims[2], x_dims[1]);
  Dims4D d_kernel_dims(1, kernel_dims[2], kernel_dims[1], kernel_dims[0]);
  Dims4D d_out_dims(out_dims[0], out_dims[3], out_dims[2], out_dims[1]);
  Dims4D d_paddings(1, paddings[2], paddings[1], paddings[0]);
  Dims4D d_strides(1, strides[2], strides[1], strides[0]);
  Dims4D d_dilations(1, dilations[2], dilations[1], dilations[0]);

  // 1. product rule book
  phi::funcs::SetConstant<Context, int> set_zero;
  set_zero(dev_ctx, counter_per_kernel, 0);
  auto config =
      phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, non_zero_num, 1);

  ProductRuleBookKernel<<<config.block_per_grid.x,
                          config.thread_per_block.x,
                          kernel_size * sizeof(int),
                          dev_ctx.stream()>>>(indices_ptr,
                                              d_x_dims,
                                              d_kernel_dims,
                                              d_out_dims,
                                              non_zero_num,
                                              d_paddings,
                                              d_dilations,
                                              d_strides,
                                              subm,
                                              rulebook_ptr,
                                              counter_ptr,
                                              in_indexs.data<int>());

// 2. remove -1
#ifdef PADDLE_WITH_HIP
  int* last = thrust::remove(thrust::hip::par.on(dev_ctx.stream()),
#else
  int* last = thrust::remove(thrust::cuda::par.on(dev_ctx.stream()),
#endif
                             rulebook_ptr,
                             rulebook_ptr + rulebook_rows * rulebook_cols,
                             -1);

  DistanceKernel<<<1, 1, 0, dev_ctx.stream()>>>(
      rulebook_ptr, last, rulebook_ptr + 3 * kernel_size * non_zero_num - 1);
  int rulebook_len = 0;
  phi::backends::gpu::GpuMemcpyAsync(
      &rulebook_len,
      rulebook_ptr + 3 * kernel_size * non_zero_num - 1,
      sizeof(int),
#ifdef PADDLE_WITH_HIP
      hipMemcpyDeviceToHost,
#else
      cudaMemcpyDeviceToHost,
#endif
      dev_ctx.stream());
  rulebook_len /= 3;
  dev_ctx.Wait();

  if (subm) {
    // At present, hashtable is not used to map the input and output indexes.
    // At present, the intermediate output index is generated by normal
    // convolution,
    // and then the intermediate output index is subtracted from the input index
    // to obain the rulebook.
    // get difference
    int32_t* A_key_ptr = rulebook_ptr + 2 * rulebook_len;
    int32_t* B_key_ptr = in_indexs.data<int>();
    DenseTensor A_val = phi::Empty<Context>(
        dev_ctx,
        DenseTensorMeta(DataType::INT32, {rulebook_len}, DataLayout::NCHW));
    DenseTensor B_val = phi::Empty<Context>(
        dev_ctx, DenseTensorMeta(DataType::INT32, {x.nnz()}, DataLayout::NCHW));
    phi::IndexKernel<int, kps::IdentityFunctor<int>>(
        dev_ctx, &A_val, kps::IdentityFunctor<int>());
    phi::IndexKernel<int, kps::IdentityFunctor<int>>(
        dev_ctx, &B_val, kps::IdentityFunctor<int>());
    DenseTensor key_result = phi::Empty<Context>(
        dev_ctx,
        DenseTensorMeta(DataType::INT32, {rulebook_len + 1}, DataLayout::NCHW));
    DenseTensor val_result = phi::Empty<Context>(
        dev_ctx,
        DenseTensorMeta(DataType::INT32, {rulebook_len}, DataLayout::NCHW));

#ifdef PADDLE_WITH_HIP
    thrust::exclusive_scan(thrust::hip::par.on(dev_ctx.stream()),
#else
    thrust::exclusive_scan(thrust::cuda::par.on(dev_ctx.stream()),
#endif
                           counter_ptr,
                           counter_ptr + kernel_size,
                           offsets_ptr);
    std::vector<int> offsets(kernel_size, 0);
    // TODO(zhangkaihuo): used unified memcpy interface
    phi::backends::gpu::GpuMemcpyAsync(offsets.data(),
                                       offsets_ptr,
                                       kernel_size * sizeof(int),
#ifdef PADDLE_WITH_HIP
                                       hipMemcpyDeviceToHost,
#else
                                       cudaMemcpyDeviceToHost,
#endif
                                       dev_ctx.stream());
    dev_ctx.Wait();

    thrust::pair<int*, int*> end;
    // Because set_diff does not support duplicate data, set_diff is performed
    // separately for each segment of data.
    // TODO(zhangkaihuo): Using hashtable here may get better performance,
    // further tests ared needed.
    for (int i = 0; i < kernel_size; i++) {
      int start = offsets[i];
      int stop = i == kernel_size - 1 ? rulebook_len : offsets[i + 1];
      int* key_result_start = (i == 0 ? key_result.data<int>() : end.first);
      int* val_result_start = i == 0 ? val_result.data<int>() : end.second;
      end =
#ifdef PADDLE_WITH_HIP
          thrust::set_difference_by_key(thrust::hip::par.on(dev_ctx.stream()),
#else
          thrust::set_difference_by_key(thrust::cuda::par.on(dev_ctx.stream()),
#endif
                                        A_key_ptr + start,
                                        A_key_ptr + stop,
                                        B_key_ptr,
                                        B_key_ptr + x.nnz(),
                                        A_val.data<int>() + start,
                                        B_val.data<int>(),
                                        key_result_start,
                                        val_result_start);
    }

    DistanceKernel<<<1, 1, 0, dev_ctx.stream()>>>(
        key_result.data<int>(),
        end.first,
        key_result.data<int>() + rulebook_len);
    int len = 0;
    phi::backends::gpu::GpuMemcpyAsync(&len,
                                       key_result.data<int>() + rulebook_len,
                                       sizeof(int),
#ifdef PADDLE_WITH_HIP
                                       hipMemcpyDeviceToHost,
#else
                                       cudaMemcpyDeviceToHost,
#endif
                                       dev_ctx.stream());
    dev_ctx.Wait();
    // set the diff value = -1, and update counter
    auto config = phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, len, 1);
    SetFlagAndUpdateCounterKernel<<<config.block_per_grid.x,
                                    config.thread_per_block,
                                    kernel_size * sizeof(int),
                                    dev_ctx.stream()>>>(val_result.data<int>(),
                                                        len,
                                                        rulebook_len,
                                                        kernel_size,
                                                        rulebook_ptr,
                                                        counter_ptr);
// remove -1
#ifdef PADDLE_WITH_HIP
    int* last = thrust::remove(thrust::hip::par.on(dev_ctx.stream()),
#else
    int* last = thrust::remove(thrust::cuda::par.on(dev_ctx.stream()),
#endif
                               rulebook_ptr,
                               rulebook_ptr + 3 * rulebook_len,
                               -1);
    DistanceKernel<<<1, 1, 0, dev_ctx.stream()>>>(
        rulebook_ptr, last, key_result.data<int>() + rulebook_len);
    phi::backends::gpu::GpuMemcpyAsync(&rulebook_len,
                                       key_result.data<int>() + rulebook_len,
                                       sizeof(int),
#ifdef PADDLE_WITH_HIP
                                       hipMemcpyDeviceToHost,
#else
                                       cudaMemcpyDeviceToHost,
#endif
                                       dev_ctx.stream());
    dev_ctx.Wait();
    rulebook_len /= 3;
  }

#ifdef PADDLE_WITH_HIP
  thrust::exclusive_scan(thrust::hip::par.on(dev_ctx.stream()),
#else
  thrust::exclusive_scan(thrust::cuda::par.on(dev_ctx.stream()),
#endif
                         counter_ptr,
                         counter_ptr + kernel_size,
                         offsets_ptr);

#ifdef PADDLE_WITH_HIP
  phi::backends::gpu::GpuMemcpyAsync(&(*h_counter)[0],
                                     counter_ptr,
                                     kernel_size * sizeof(int),
                                     hipMemcpyDeviceToHost,
                                     dev_ctx.stream());
  phi::backends::gpu::GpuMemcpyAsync(&(*h_offsets)[0],
                                     offsets_ptr,
                                     kernel_size * sizeof(int),
                                     hipMemcpyDeviceToHost,
                                     dev_ctx.stream());
#else
  phi::backends::gpu::GpuMemcpyAsync(&(*h_counter)[0],
                                     counter_ptr,
                                     kernel_size * sizeof(int),
                                     cudaMemcpyDeviceToHost,
                                     dev_ctx.stream());
  phi::backends::gpu::GpuMemcpyAsync(&(*h_offsets)[0],
                                     offsets_ptr,
                                     kernel_size * sizeof(int),
                                     cudaMemcpyDeviceToHost,
                                     dev_ctx.stream());
#endif
  rulebook->Resize({rulebook_rows, rulebook_len});

  // 3. sorted or merge the out index
  out_index->ResizeAndAllocate({rulebook_len});
  unique_value->ResizeAndAllocate({rulebook_len});
  unique_key->ResizeAndAllocate({rulebook_len});
  int* out_index_ptr = out_index->data<int>();
  int* unique_value_ptr = unique_value->data<int>();
  int* unique_key_ptr = unique_key->data<int>();

  int* new_end = SortedAndUniqueIndex(dev_ctx,
                                      rulebook_ptr + 2 * rulebook_len,
                                      rulebook_len,
                                      out_index,
                                      unique_key,
                                      unique_value);
  // thrust::distance doesn't support stream parameters
  // const int out_non_zero_num = thrust::distance(unique_key_ptr,
  // new_end.first);
  DistanceKernel<<<1, 1>>>(unique_key_ptr,
                           new_end,
                           rulebook_ptr + rulebook_rows * rulebook_cols - 1);
  int out_non_zero_num = 0;
#ifdef PADDLE_WITH_HIP
  phi::backends::gpu::GpuMemcpyAsync(
      &out_non_zero_num,
      rulebook_ptr + rulebook_rows * rulebook_cols - 1,
      sizeof(int),
      hipMemcpyDeviceToHost,
      dev_ctx.stream());
#else
  phi::backends::gpu::GpuMemcpyAsync(
      &out_non_zero_num,
      rulebook_ptr + rulebook_rows * rulebook_cols - 1,
      sizeof(int),
      cudaMemcpyDeviceToHost,
      dev_ctx.stream());
#endif
  dev_ctx.Wait();

  // 5. update out_indices and rulebook by unique_value_ptr
  const int64_t sparse_dim = 4;
  DenseTensorMeta indices_meta(
      DataType::INT32, {sparse_dim, out_non_zero_num}, DataLayout::NCHW);
  DenseTensorMeta values_meta(
      x.dtype(), {out_non_zero_num, kernel_dims[4]}, x.layout());
  phi::DenseTensor out_indices = phi::Empty(dev_ctx, std::move(indices_meta));
  phi::DenseTensor out_values = phi::Empty(dev_ctx, std::move(values_meta));

  int* out_indices_ptr = out_indices.data<int>();

  config =
      phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, out_non_zero_num, 1);
  UpdateIndexKernel<<<config.block_per_grid.x,
                      config.thread_per_block.x,
                      0,
                      dev_ctx.stream()>>>(unique_key_ptr,
                                          unique_value_ptr,
                                          out_index_ptr,
                                          out_non_zero_num,
                                          rulebook_len,
                                          d_out_dims,
                                          out_indices_ptr,
                                          rulebook_ptr + 2 * rulebook_len);
  out->SetMember(out_indices, out_values, out_dims, true);
  return rulebook_len;
}

/**
 * x: (N, D, H, W, C)
 * kernel: (D, H, W, C, OC)
 * out: (N, D, H, W, OC)
**/
template <typename T, typename Context>
void Conv3dKernel(const Context& dev_ctx,
                  const SparseCooTensor& x,
                  const DenseTensor& kernel,
                  const std::vector<int>& paddings,
                  const std::vector<int>& dilations,
                  const std::vector<int>& strides,
                  const int groups,
                  const bool subm,
                  SparseCooTensor* out,
                  DenseTensor* rulebook) {
  // update padding and dilation
  // Currently, only support x.layout is NDHWC, groups = 1
  // if x.layout != NDHWC then transpose(x), transpose(weight)

  const auto& x_dims = x.dims();
  const auto& kernel_dims = kernel.dims();
  int kernel_size = kernel_dims[0] * kernel_dims[1] * kernel_dims[2];
  DDim out_dims = {1, 1, 1, 1, 1};
  GetOutShape(x_dims, kernel_dims, paddings, dilations, strides, &out_dims);
  out->set_dims(out_dims);
  const int in_channels = kernel_dims[3];
  const int out_channels = kernel_dims[4];
  std::vector<int> offsets(kernel_size + 1), h_counter(kernel_size);

  // Second algorithm:
  // https://pdfs.semanticscholar.org/5125/a16039cabc6320c908a4764f32596e018ad3.pdf
  // 1. product rulebook
  DenseTensorMeta counter_meta(
      DataType::INT32, {kernel_size}, DataLayout::NCHW);
  DenseTensorMeta offsets_meta(
      DataType::INT32, {kernel_size}, DataLayout::NCHW);
  DenseTensor counter_per_kernel = phi::Empty(dev_ctx, std::move(counter_meta));
  DenseTensor offsets_per_kernel = phi::Empty(dev_ctx, std::move(offsets_meta));
  DenseTensorMeta index_meta(DataType::INT32, {1}, DataLayout::NCHW);
  DenseTensor out_index = phi::Empty(dev_ctx, std::move(index_meta));
  DenseTensor unique_key = phi::Empty(dev_ctx, std::move(index_meta));
  DenseTensor unique_value = phi::Empty(dev_ctx, std::move(index_meta));

  std::vector<int> subm_paddings(paddings), subm_strides(strides);
  if (subm) {
    auto kernel_dims = kernel.dims();
    for (int i = 0; i < paddings.size(); i++) {
      subm_paddings[i] = kernel_dims[i] / 2;
      subm_strides[i] = 1;
    }
  }

  int n = ProductRuleBook<T, Context>(dev_ctx,
                                      x,
                                      kernel,
                                      subm_paddings,
                                      dilations,
                                      subm_strides,
                                      out_dims,
                                      subm,
                                      rulebook,
                                      &counter_per_kernel,
                                      &offsets_per_kernel,
                                      &out_index,
                                      &unique_key,
                                      &unique_value,
                                      out,
                                      &h_counter,
                                      &offsets);

  const int* counter_ptr = counter_per_kernel.data<int>();
  const int* offsets_ptr = counter_per_kernel.data<int>();
  const int* rulebook_ptr = rulebook->data<int>();

  // 2. gather
  DenseTensorMeta in_features_meta(
      x.dtype(), {n, in_channels}, DataLayout::NCHW);
  DenseTensorMeta out_features_meta(
      x.dtype(), {n, out_channels}, DataLayout::NCHW);
  phi::DenseTensor in_features =
      phi::Empty(dev_ctx, std::move(in_features_meta));
  phi::DenseTensor out_features =
      phi::Empty(dev_ctx, std::move(out_features_meta));
  T* in_features_ptr = in_features.data<T>();
  T* out_features_ptr = out_features.data<T>();
  phi::funcs::SetConstant<Context, T> set_zero;
  set_zero(dev_ctx, &out_features, static_cast<T>(0.0f));

  auto config =
      phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, n * in_channels, 1);
  GatherKernel<T, int><<<config.block_per_grid.x,
                         config.thread_per_block.x,
                         0,
                         dev_ctx.stream()>>>(x.non_zero_elements().data<T>(),
                                             rulebook_ptr + n,
                                             in_features_ptr,
                                             n,
                                             in_channels);

  // 3. call gemm for every werght
  auto blas = phi::funcs::GetBlas<Context, T>(dev_ctx);
  auto* out_values = out->mutable_non_zero_elements();
  T* out_values_ptr = out_values->data<T>();

  const T* kernel_ptr = kernel.data<T>();
  for (int i = 0; i < kernel_size; i++) {
    if (h_counter[i] <= 0) {
      continue;
    }

    // call gemm: (n, in_channels) * (in_channels, out_channels)
    const int M = h_counter[i];
    const int K = in_channels;
    const int N = out_channels;
    T* tmp_in_ptr = in_features_ptr + offsets[i] * in_channels;
    const T* tmp_kernel_ptr = kernel_ptr + i * K * N;
    T* tmp_out_ptr = out_features_ptr + offsets[i] * out_channels;

    blas.GEMM(CblasNoTrans,
              CblasNoTrans,
              M,
              N,
              K,
              static_cast<T>(1),
              tmp_in_ptr,
              tmp_kernel_ptr,
              static_cast<T>(0),
              tmp_out_ptr);
  }

  // 4. scatter
  config = phi::backends::gpu::GetGpuLaunchConfig1D(
      dev_ctx, out->nnz() * out_channels, 1);
  ScatterKernel<T><<<config.block_per_grid.x,
                     config.thread_per_block.x,
                     0,
                     dev_ctx.stream()>>>(out_features_ptr,
                                         unique_value.data<int>(),
                                         out_index.data<int>(),
                                         out->nnz(),
                                         n,
                                         out_channels,
                                         out_values_ptr);
}

}  // namespace sparse
}  // namespace phi

PD_REGISTER_KERNEL(sparse_conv3d,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::Conv3dKernel,
                   float,
                   double,
                   phi::dtype::float16) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
}
