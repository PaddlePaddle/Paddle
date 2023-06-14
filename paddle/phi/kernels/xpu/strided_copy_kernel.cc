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
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/xpu/xpu_api_wrapper.h"

namespace phi {

template <typename T>
__global__ void StridedCopyFunc(const T* input_data,
                                T* out_data,
                                const int64_t* input_stride,
                                const int64_t* output_stride,
                                const int64_t* dims,
                                const int rank,
                                const int64_t numel) {
  int64_t tid = core_id() * cluster_num() + cluster_id();
  int64_t nthreads = core_id() * cluster_num();

  for (int64_t i = tid; i < numel; i += nthreads) {
    int64_t input_offset = 0;
    int64_t output_offset = 0;
    int64_t index_tmp = i;
    for (int dim = rank - 1; dim >= 0; --dim) {
      int64_t mod = index_tmp % dims[dim];
      index_tmp = index_tmp / dims[dim];
      input_offset += mod * input_stride[dim];
      output_offset += mod * output_stride[dim];
    }

    out_data[output_offset] = input_data[input_offset];
  }
}

template <typename T, typename Context>
void StridedCopyKernel(const Context& dev_ctx,
                       const DenseTensor& input,
                       const std::vector<int64_t>& out_stride,
                       int64_t offset,
                       DenseTensor* out) {
  phi::DenseTensorMeta meta = input.meta();
  meta.stride = phi::make_ddim(out_stride);
  meta.dims = phi::make_ddim(dims);
  meta.offset = offset;
  out->set_meta(meta);

  const T* input_data = input.data<T>();
  T* output_data = dev_ctx.template Alloc<T>(out);
  int rank = input.dims().size();
  const int64_t* dims = input.dims().Get();
  const int64_t* input_stride = input.stride().Get();
  const int64_t* output_stride = meta.stride.Get();
  auto numel = input.numel();
  int64_t block = 512;
  int64_t grid = (numel + block - 1) / block;

  int64_t* tmp_data =
      reinterpret_cast<int64_t*>(malloc(sizeof(int64_t) * rank * 3));
  std::memcpy(tmp_data, dims, sizeof(int64_t) * rank);
  std::memcpy(tmp_data + rank, input_stride, sizeof(int64_t) * rank);
  std::memcpy(tmp_data + rank + rank, output_stride, sizeof(int64_t) * rank);

  auto dims_stride = paddle::memory::Alloc(
      dev_ctx.GetPlace(),
      sizeof(int64_t) * rank * 3,
      phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream())));
  int64_t* dims_stride_data = reinterpret_cast<int64_t*>(dims_stride->ptr());

  paddle::memory::Copy(dev_ctx.GetPlace(),
                       dims_stride_data,
                       phi::CPUPlace(),
                       tmp_data,
                       sizeof(int64_t) * rank * 3);

  int r = StridedCopyFunc<<<dev_ctx.x_context()->ncluster(),
                            64,
                            dev_ctx.stream()>>>(input_data,
                                                output_data,
                                                dims_stride_data + rank,
                                                dims_stride_data + rank + rank,
                                                dims_stride_data,
                                                rank,
                                                numel);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "StridedCopyFunc");
}

}  // namespace phi

PD_REGISTER_KERNEL(strided_copy,
                   XPU,
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
