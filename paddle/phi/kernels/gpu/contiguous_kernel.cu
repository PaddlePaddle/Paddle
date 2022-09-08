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

#include "paddle/phi/kernels/contiguous_kernel.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T>
__global__ void ContiguousFunc(const T* input_data,
                               T* out_data,
                               const int64_t* input_stride,
                               const int64_t* dims,
                               const int rank,
                               const int64_t numel) {
  int64_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  for (int64_t i = gid; i < numel; i += blockDim.x * gridDim.x) {
    int64_t input_offset = 0;
    int64_t index_tmp = i;
    for (int dim = 0; dim < rank; ++dim) {
      int64_t mod = index_tmp % dims[dim];
      index_tmp = index_tmp / dims[dim];
      input_offset += mod * input_stride[dim];
    }

    out_data[i] = input_data[input_offset];
  }
}

template <typename T, typename Context>
void ContiguousKernel(const Context& dev_ctx,
                      const DenseTensor& input,
                      DenseTensor* out) {
  const T* input_data = input.data<T>();
  T* output_data = dev_ctx.template Alloc<T>(out);
  int rank = input.dims().size();
  const int64_t* dims = input.dims().Get();
  const int64_t* strides = input.strides().Get();
  out->InitStrides();
  auto numel = input.numel();
  int64_t block = 512;
  int64_t grid = (numel + block - 1) / block;
  DenseTensor tmp;
  DenseTensor dims_strides;
  DenseTensorMeta meta;
  meta.dims = make_ddim({2, rank});
  meta.dtype = DataType::INT64;
  tmp.set_meta(meta);
  dims_strides.set_meta(meta);

  int64_t* tmp_data;
  cudaHostAlloc(&tmp_data, sizeof(int64_t) * rank * 2, cudaHostAllocPortable);

  int64_t* dims_strides_data =
      reinterpret_cast<int64_t*>(dev_ctx.Alloc(&dims_strides, DataType::INT64));

  std::memcpy(tmp_data, dims, sizeof(int64_t) * rank);
  std::memcpy(
      tmp_data + sizeof(int64_t) * rank, strides, sizeof(int64_t) * rank);

  cudaMemcpyAsync(dims_strides_data,
                  tmp_data,
                  sizeof(int64_t) * rank * 2,
                  cudaMemcpyHostToDevice,
                  dev_ctx.stream());

  cudaStreamCallback_t free_when_cb =
      [](cudaStream_t stream, cudaError_t status, void* userData) {
        cudaFreeHost(userData);
      };

  cudaStreamAddCallback(dev_ctx.stream(), free_when_cb, tmp_data, 0);

  ContiguousFunc<<<grid, block, 0, dev_ctx.stream()>>>(input_data,
                                                       output_data,
                                                       dims_strides_data + rank,
                                                       dims_strides_data,
                                                       rank,
                                                       numel);
}

}  // namespace phi

PD_REGISTER_KERNEL(contiguous,
                   GPU,
                   ALL_LAYOUT,
                   phi::ContiguousKernel,
                   bool,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int32_t,
                   int64_t,
                   float,
                   double,
                   ::phi::dtype::float16,
                   ::phi::dtype::bfloat16,
                   ::phi::dtype::complex<float>,
                   ::phi::dtype::complex<double>) {}
