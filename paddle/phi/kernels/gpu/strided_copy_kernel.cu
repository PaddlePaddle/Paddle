/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/kernels/strided_copy_kernel.h"
#include "paddle/fluid/memory/malloc.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

// TODO(wanghuancoder) delete output_rank and output_stride if they are same
// with input's
template <typename T>
__global__ void StridedCopyFunc(const T* input_data,
                                const int input_rank,
                                const int64_t* input_dims,
                                const int64_t* input_stride,
                                T* output_data,
                                const int output_rank,
                                const int64_t* output_dims,
                                const int64_t* output_stride,
                                const int64_t numel) {
  int64_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  for (int64_t i = gid; i < numel; i += blockDim.x * gridDim.x) {
    int64_t input_offset = 0;
    int64_t index_tmp = i;
    for (int dim = input_rank - 1; dim >= 0; --dim) {
      input_offset += (index_tmp % input_dims[dim]) * input_stride[dim];
      index_tmp = index_tmp / input_dims[dim];
    }
    int64_t output_offset = 0;
    index_tmp = i;
    for (int dim = output_rank - 1; dim >= 0; --dim) {
      output_offset += (index_tmp % output_dims[dim]) * output_stride[dim];
      index_tmp = index_tmp / output_dims[dim];
    }
    output_data[output_offset] = input_data[input_offset];
  }
}

template <typename T, typename Context>
void StridedCopyKernel(const Context& dev_ctx,
                       const DenseTensor& input,
                       const std::vector<int64_t>& dims,
                       const std::vector<int64_t>& out_stride,
                       DenseTensor* out) {
  phi::DenseTensorMeta meta = input.meta();
  meta.stride = phi::make_ddim(out_stride);
  meta.dims = phi::make_ddim(dims);
  meta.offset = out->offset();
  out->set_meta(meta);

  const T* input_data = input.data<T>();
  int input_rank = input.dims().size();
  const int64_t* input_dims = input.dims().Get();
  const int64_t* input_stride = input.stride().Get();

  T* output_data = dev_ctx.template Alloc<T>(out);
  int output_rank = meta.dims.size();
  const int64_t* output_dims = meta.dims.Get();
  const int64_t* output_stride = meta.stride.Get();

  PADDLE_ENFORCE_EQ(input.dims(),
                    out->dims(),
                    phi::errors::InvalidArgument(
                        "Input shape(%s) must be equal with out shape(%s).",
                        input.dims(),
                        out->dims()));

  PADDLE_ENFORCE_EQ(input.numel(),
                    out->numel(),
                    phi::errors::InvalidArgument(
                        "Input numel(%d) must be equal with out numel(%d).",
                        input.numel(),
                        out->numel()));

  auto numel = input.numel();
  int64_t block = 512;
  int64_t grid = (numel + block - 1) / block;

  int64_t* tmp_data = reinterpret_cast<int64_t*>(
      malloc(sizeof(int64_t) * (input_rank * 2 + output_rank * 2)));
  std::memcpy(tmp_data, input_dims, sizeof(int64_t) * input_rank);
  std::memcpy(
      tmp_data + input_rank, input_stride, sizeof(int64_t) * input_rank);
  std::memcpy(tmp_data + input_rank + input_rank,
              output_dims,
              sizeof(int64_t) * output_rank);
  std::memcpy(tmp_data + input_rank + input_rank + output_rank,
              output_stride,
              sizeof(int64_t) * output_rank);

  auto dims_stride = memory_utils::Alloc(
      dev_ctx.GetPlace(),
      sizeof(int64_t) * (input_rank * 2 + output_rank * 2),
      phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream())));
  int64_t* dims_stride_data = reinterpret_cast<int64_t*>(dims_stride->ptr());
  memory_utils::Copy(dev_ctx.GetPlace(),
                     dims_stride_data,
                     phi::CPUPlace(),
                     tmp_data,
                     sizeof(int64_t) * (input_rank * 2 + output_rank * 2),
                     dev_ctx.stream());

  cudaStreamCallback_t free_when_cb = [](cudaStream_t stream,
                                         cudaError_t status,
                                         void* userData) { free(userData); };
  cudaStreamAddCallback(dev_ctx.stream(), free_when_cb, tmp_data, 0);

  StridedCopyFunc<<<grid, block, 0, dev_ctx.stream()>>>(
      input_data,
      input_rank,
      dims_stride_data,
      dims_stride_data + input_rank,
      output_data,
      output_rank,
      dims_stride_data + input_rank + input_rank,
      dims_stride_data + input_rank + input_rank + output_rank,
      numel);
}

}  // namespace phi

PD_REGISTER_KERNEL(strided_copy,
                   GPU,
                   ALL_LAYOUT,
                   phi::StridedCopyKernel,
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
