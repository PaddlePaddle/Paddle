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

#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"
#include "paddle/pten/api/lib/utils/allocator.h"
#include "paddle/pten/backends/gpu/gpu_context.h"
#include "paddle/pten/core/kernel_registry.h"
#include "paddle/pten/core/tensor_meta.h"
#include "paddle/pten/kernels/sparse/convolution_kernel.h"
#include "paddle/pten/kernels/sparse/sparse_utils_kernel.h"

namespace pten {
namespace sparse {

struct Dims4D {
  int batch_, x_, y_, z_;
  Dims4D(const int batch, const int x, const int y, const int z) {
    batch_ = batch;
    x_ = x;
    y_ = y;
    z_ = z;
  }

  Dims4D(const int x, const int y, const int z) {
    batch_ = 1;
    x_ = x;
    y_ = y;
    z_ = z;
  }
};

inline __device__ bool Check(const int& x,
                             const int& y,
                             const int& z,
                             const Dims4D& dims) {
  if (x >= 0 && x < dims.x_ && y >= 0 && y < dims.y_ && z >= 0 && z < dims.z_) {
    return true;
  }
  return false;
}

inline __device__ int PointToIndex(const int& batch,
                                   const int& x,
                                   const int& y,
                                   const int& z,
                                   const Dims4D& dims) {
  return batch * dims.z_ * dims.y_ * dims.x_ + z * dims.y_ * dims.x_ +
         y * dims.x_ + x;
}

inline __device__ void IndexToPoint(
    const int index, const Dims4D& dims, int* batch, int* x, int* y, int* z) {
  int n = index;
  *x = n % dims.x_;
  n /= dims.x_;
  *y = n % dims.y_;
  n /= dims.y_;
  *z = n % dims.z_;
  n /= dims.z_;
  *batch = n;
}

__global__ void InitByIndexKernel(const int n, int* out1, int* out2) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  for (int i = tid; i < n; i += gridDim.x * blockDim.x) {
    out1[i] = i;
    out2[i] = i;
  }
}

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
    IndexToPoint(index, out_dims, &batch, &x, &y, &z);
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

__global__ void ProductRuleBookKernel(
    const int* x_indices,
    const Dims4D kernel_dims,
    const Dims4D out_dims,
    const int64_t non_zero_num,
    const Dims4D paddings,   // save to __constant__
    const Dims4D dilations,  // save to __constant__
    const Dims4D strides,    // save to __constant__
    int* rulebook,
    int* counter) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  extern __shared__ int counter_buf[];  // kernel_size
  const int kernel_size = kernel_dims.x_ * kernel_dims.y_ * kernel_dims.z_;
  const int offset = kernel_size * non_zero_num;
  for (int i = threadIdx.x; i < kernel_size; i += blockDim.x) {
    counter_buf[i] = 0;
  }
  __syncthreads();

  for (int i = tid; i < non_zero_num; i += gridDim.x * blockDim.x) {
    int kernel_index = 0;
    for (int kernel_z = 0; kernel_z < kernel_dims.z_; kernel_z++) {
      for (int kernel_y = 0; kernel_y < kernel_dims.y_; kernel_y++) {
        for (int kernel_x = 0; kernel_x < kernel_dims.x_; kernel_x++) {
          int batch = x_indices[i];
          int in_z = x_indices[i + non_zero_num];
          int in_y = x_indices[i + 2 * non_zero_num];
          int in_x = x_indices[i + 3 * non_zero_num];
          int out_z =
              (in_z + paddings.z_ - kernel_z * dilations.z_) / strides.z_;
          int out_y =
              (in_y + paddings.y_ - kernel_y * dilations.y_) / strides.y_;
          int out_x =
              (in_x + paddings.x_ - kernel_x * dilations.x_) / strides.x_;
          int in_i = -1, out_index = -1;
          if (Check(out_x, out_y, out_z, out_dims)) {
            in_i = i;
            out_index = PointToIndex(batch, out_x, out_y, out_z, out_dims);
            atomicAdd(&counter_buf[kernel_index], 1);
          }
          rulebook[kernel_index * non_zero_num + i] = in_i;
          rulebook[kernel_index * non_zero_num + offset + i] = out_index;
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

// TODO(zhangkaihuo): After the GatherCUDAKernel is migrated to phi, replace
// this kernel with
// phi::GatherCUDAKernel;
template <typename T, typename IndexT = int>
__global__ void GatherKernel(const T* params,
                             const IndexT* indices,
                             T* output,
                             size_t index_size,
                             size_t slice_size) {
  CUDA_KERNEL_LOOP_TYPE(i, index_size * slice_size, int64_t) {
    int64_t indices_i = i / slice_size;
    int64_t slice_i = i - indices_i * slice_size;  // offset inside the slice
    IndexT gather_i = indices[indices_i];
    int64_t params_i = gather_i * slice_size + slice_i;
    *(output + i) = *(params + params_i);
  }
}

// unique_value:
template <typename T>
__global__ void ScatterKernel(const T* input,
                              const int* unique_value,
                              const int* out_index,
                              const int non_zero_num,
                              const int rulebook_len,
                              const int channels,
                              T* out) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  for (int i = tid; i < non_zero_num * channels; i += gridDim.x * blockDim.x) {
    int indices_i = i / channels;
    int channels_i = i - indices_i * channels;

    int start = unique_value[indices_i];
    int end = indices_i == non_zero_num - 1 ? rulebook_len
                                            : unique_value[indices_i + 1];
    // max(end-start) = kernel_size
    for (int j = start; j < end; j++) {
      const int out_feature_i = out_index[j];
      out[indices_i * channels + channels_i] +=
          input[out_feature_i * channels + channels_i];
    }
  }
}

// the basic algorithm can refer to convolution_kernel.cc or
// the second paper
// example:
// 1. the rulebook:
//  the kernel_index:               0, 0, 0, 1, 1, 1, 2, 2, ....
//  the out_index(key):                  20, 30, 33, 30, 33, 20, 25
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
                    DenseTensor* rulebook,
                    DenseTensor* counter_per_kernel,
                    DenseTensor* offsets_per_kernel,
                    DenseTensor* out_index,
                    DenseTensor* unique_key,
                    DenseTensor* unique_value,
                    SparseCooTensor* out,
                    std::vector<int>* h_counter,
                    std::vector<int>* h_offsets) {
  const auto place = dev_ctx.GetPlace();
  const auto& kernel_dims = kernel.dims();
  const int64_t non_zero_num = x.nnz();
  const auto& non_zero_indices = x.non_zero_indices();
  const int* indices_ptr = non_zero_indices.data<int>();
  int* counter_ptr = counter_per_kernel->mutable_data<int>(place);
  int* offsets_ptr = offsets_per_kernel->mutable_data<int>(place);
  int kernel_size = kernel_dims[0] * kernel_dims[1] * kernel_dims[2];
  rulebook->ResizeAndAllocate({2 * kernel_size * non_zero_num});
  int* rulebook_ptr = rulebook->mutable_data<int>(place);

  Dims4D d_kernel_dims(kernel_dims[2], kernel_dims[1], kernel_dims[0]);
  Dims4D d_out_dims(out_dims[0], out_dims[3], out_dims[2], out_dims[1]);
  Dims4D d_paddings(paddings[2], paddings[1], paddings[0]);
  Dims4D d_strides(strides[2], strides[1], strides[0]);
  Dims4D d_dilations(dilations[2], dilations[1], dilations[0]);

  // 1. product rule book
  PADDLE_ENFORCE_GPU_SUCCESS(cudaMemsetAsync(
      counter_ptr, 0, sizeof(int) * kernel_size, dev_ctx.stream()));
  int grid_size = 1, block_size = 1;
  GetGpuLaunchConfig1D(dev_ctx, non_zero_num, &grid_size, &block_size);
  const size_t shared_size = kernel_size * sizeof(int);
  ProductRuleBookKernel<<<grid_size,
                          block_size,
                          shared_size,
                          dev_ctx.stream()>>>(indices_ptr,
                                              d_kernel_dims,
                                              d_out_dims,
                                              non_zero_num,
                                              d_paddings,
                                              d_dilations,
                                              d_strides,
                                              rulebook_ptr,
                                              counter_ptr);

  // 2. remove -1
  int* last = thrust::remove(thrust::cuda::par.on(dev_ctx.stream()),
                             rulebook_ptr,
                             rulebook_ptr + 2 * kernel_size * non_zero_num,
                             -1);
  thrust::exclusive_scan(thrust::cuda::par.on(dev_ctx.stream()),
                         counter_ptr,
                         counter_ptr + kernel_size,
                         offsets_ptr);

  PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyAsync(&(*h_counter)[0],
                                             counter_ptr,
                                             kernel_size * sizeof(int),
                                             cudaMemcpyDeviceToHost,
                                             dev_ctx.stream()));
  PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyAsync(&(*h_offsets)[0],
                                             offsets_ptr,
                                             kernel_size * sizeof(int),
                                             cudaMemcpyDeviceToHost,
                                             dev_ctx.stream()));
  dev_ctx.Wait();
  int rulebook_len =
      (*h_counter)[kernel_size - 1] + (*h_offsets)[kernel_size - 1];

  // 3. sorted or merge the out index
  out_index->ResizeAndAllocate({rulebook_len});
  unique_value->ResizeAndAllocate({rulebook_len});
  unique_key->ResizeAndAllocate({rulebook_len});
  int* out_index_ptr = out_index->mutable_data<int>(place);
  int* unique_value_ptr = unique_value->mutable_data<int>(place);
  int* unique_key_ptr = unique_key->mutable_data<int>(place);

  GetGpuLaunchConfig1D(dev_ctx, rulebook_len, &grid_size, &block_size);
  InitByIndexKernel<<<grid_size, block_size, 0, dev_ctx.stream()>>>(
      rulebook_len, out_index_ptr, unique_value_ptr);

  PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyAsync(unique_key_ptr,
                                             rulebook_ptr + rulebook_len,
                                             sizeof(int) * rulebook_len,
                                             cudaMemcpyDeviceToDevice,
                                             dev_ctx.stream()));

  // compared with thrust::sort_by_key, thrust::merge_by_key may achieved higher
  // performance, but thrust::merge_by_key limited by data size
  thrust::sort_by_key(thrust::cuda::par.on(dev_ctx.stream()),
                      unique_key_ptr,
                      unique_key_ptr + rulebook_len,
                      out_index_ptr);

  // 4. unique => tmp2_out_index
  thrust::pair<int*, int*> new_end =
      thrust::unique_by_key(thrust::cuda::par.on(dev_ctx.stream()),
                            unique_key_ptr,
                            unique_key_ptr + rulebook_len,
                            unique_value_ptr);
  dev_ctx.Wait();
  const int out_non_zero_num = thrust::distance(unique_key_ptr, new_end.first);

  // 5. update out_indices and rulebook by unique_value_ptr
  const int64_t sparse_dim = 4;
  DenseTensorMeta indices_meta(
      DataType::INT32, {sparse_dim, out_non_zero_num}, DataLayout::NCHW);
  DenseTensorMeta values_meta(
      x.dtype(), {out_non_zero_num, kernel_dims[4]}, x.layout());
  pten::DenseTensor out_indices = pten::Empty(dev_ctx, std::move(indices_meta));
  pten::DenseTensor out_values = pten::Empty(dev_ctx, std::move(values_meta));
  int* out_indices_ptr = out_indices.mutable_data<int>(dev_ctx.GetPlace());
  GetGpuLaunchConfig1D(dev_ctx, out_non_zero_num, &grid_size, &block_size);
  UpdateIndexKernel<<<grid_size, block_size, 0, dev_ctx.stream()>>>(
      unique_key_ptr,
      unique_value_ptr,
      out_index_ptr,
      out_non_zero_num,
      rulebook_len,
      d_out_dims,
      out_indices_ptr,
      rulebook_ptr + rulebook_len);
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
                  SparseCooTensor* out) {
  // update padding and dilation
  // Currently, only support x.layout is NDHWC, groups = 1
  // if x.layout != NDHWC then transpose(x), transpose(weight)

  const auto& place = dev_ctx.GetPlace();
  const auto& x_dims = x.dims();
  const auto& kernel_dims = kernel.dims();
  int kernel_size = kernel_dims[0] * kernel_dims[1] * kernel_dims[2];
  DDim out_dims = {1, 1, 1, 1, 1};
  GetOutShape(x_dims, kernel_dims, paddings, dilations, strides, &out_dims);
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
  DenseTensor rulebook = pten::Empty<int, Context>(dev_ctx);
  DenseTensor counter_per_kernel =
      pten::Empty(dev_ctx, std::move(counter_meta));
  DenseTensor offsets_per_kernel =
      pten::Empty(dev_ctx, std::move(offsets_meta));
  DenseTensor out_index = pten::Empty<int, Context>(dev_ctx);
  DenseTensor unique_key = pten::Empty<int, Context>(dev_ctx);
  DenseTensor unique_value = pten::Empty<int, Context>(dev_ctx);

  int n = ProductRuleBook<T, Context>(dev_ctx,
                                      x,
                                      kernel,
                                      paddings,
                                      dilations,
                                      strides,
                                      out_dims,
                                      &rulebook,
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

  // 2. gather
  DenseTensorMeta in_features_meta(
      x.dtype(), {n, in_channels}, DataLayout::NCHW);
  DenseTensorMeta out_features_meta(
      x.dtype(), {n, out_channels}, DataLayout::NCHW);
  pten::DenseTensor in_features =
      pten::Empty(dev_ctx, std::move(in_features_meta));
  pten::DenseTensor out_features =
      pten::Empty(dev_ctx, std::move(out_features_meta));
  T* in_features_ptr = in_features.mutable_data<T>(place);
  T* out_features_ptr = out_features.mutable_data<T>(place);

  int grid_size = 1, block_size = 1;
  GetGpuLaunchConfig1D(dev_ctx, n * in_channels, &grid_size, &block_size);
  GatherKernel<T, int><<<grid_size, block_size, 0, dev_ctx.stream()>>>(
      x.non_zero_elements().data<T>(),
      rulebook.data<int>(),
      in_features_ptr,
      n,
      in_channels);

  // 3. call gemm for every werght
  auto blas = paddle::operators::math::GetBlas<Context, T>(dev_ctx);
  T* out_values_ptr = out->mutable_non_zero_elements()->mutable_data<T>(place);
  PADDLE_ENFORCE_GPU_SUCCESS(
      cudaMemsetAsync(out_values_ptr,
                      0,
                      sizeof(T) * out->nnz() * out_channels,
                      dev_ctx.stream()));

  const T* kernel_ptr = kernel.data<T>();
  for (int i = 0; i < kernel_size; i++) {
    if (h_counter[i] <= 0) {
      continue;
    }

    // call gemm: (n, out_channels) = (n, in_channels) * (in_channels,
    // out_channels)
    const int M = h_counter[i];
    const int K = in_channels;   // in_channels
    const int N = out_channels;  // out_channels
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
  GetGpuLaunchConfig1D(
      dev_ctx, out->nnz() * out_channels, &grid_size, &block_size);
  ScatterKernel<T><<<grid_size, block_size, 0, dev_ctx.stream()>>>(
      out_features_ptr,
      unique_value.data<int>(),
      out_index.data<int>(),
      out->nnz(),
      n,
      out_channels,
      out_values_ptr);
}

}  // namespace sparse
}  // namespace pten

PT_REGISTER_KERNEL(conv3d,
                   GPU,
                   ALL_LAYOUT,
                   pten::sparse::Conv3dKernel,
                   float,
                   double,
                   pten::dtype::float16) {}
