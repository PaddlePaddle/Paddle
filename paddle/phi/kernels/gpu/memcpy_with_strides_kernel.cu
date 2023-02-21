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

#include "paddle/fluid/memory/malloc.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {
static inline void magic32(int64_t d, int64_t* magic, int64_t* shift) {
  int64_t tmpshift = 0;
  for (tmpshift = 0; tmpshift < 32; tmpshift++)
    if ((1U << tmpshift) >= d) break;

  uint64_t tmpmagic = ((1l << 32) * ((1l << tmpshift) - d)) / d + 1;
  *magic = tmpmagic;
  *shift = tmpshift;
}
static __device__ __forceinline__ int64_t fastU32Div(int64_t a,
                                                     int64_t magic,
                                                     int64_t shift) {
  return (__umulhi(a, magic) + a) >> shift;
}

template <typename T>
__global__ void cudaRelocKernel(int64_t size,
                                int64_t ndims,
                                const int64_t* dims,
                                const int64_t* dimMagic,
                                const int64_t* dimShift,
                                const T* src,
                                const int64_t* srcStrides,
                                T* dst,
                                const int64_t* dstStrides) {
  int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= size) return;

  int64_t srcOffset = 0;
  int64_t dstOffset = 0;
  for (int64_t i = 0; i < ndims; i++) {
    int64_t magicV = __ldg(dimMagic + i);
    int64_t shiftV = __ldg(dimShift + i);
    // tmp = tid / dim[i]
    int64_t tmp = fastU32Div(tid, magicV, shiftV);
    // k = tid % dim[i]
    int64_t k = tid - tmp * __ldg(dims + i);
    srcOffset += k * __ldg(srcStrides + i);
    dstOffset += k * __ldg(dstStrides + i);
    tid = tmp;
  }
  dst[dstOffset] = src[srcOffset];
}

template <typename T, typename Context>
void memcpyWithStridesKernel(const Context& dev_ctx,
                             const DenseTensor& input,
                             DenseTensor* out) {
  const T* src = input.data<T>();
  T* dst = out->data<T>();
  auto input_meta = input.meta();
  auto output_meta = out->meta();

  int64_t size = input.numel();
  int64_t ndims = input.dims().size();
  const int64_t* dims = input.dims().Get();
  const int64_t* srcStrides = input_meta.strides.Get();
  const int64_t* dstStrides = out->stride().Get();
  if (size == 0) return;
  std::vector<int64_t> hostBuffer(ndims * 5);
  int64_t* hSrcStrides = hostBuffer.data();
  int64_t* hDstStrides = hSrcStrides + ndims;
  int64_t* hDims = hDstStrides + ndims;
  int64_t* hDimMagic = hDims + ndims;
  int64_t* hDimShift = hDimMagic + ndims;
  for (size_t i = 0, j = ndims - 1; i < ndims; i++, j--) {
    hSrcStrides[i] = srcStrides[j];
    hDstStrides[i] = dstStrides[j];
    hDims[i] = dims[j];
    magic32(dims[j], hDimMagic + i, hDimShift + i);
  }
  int64_t bufferSize = ndims * 5 * sizeof(int64_t);
  auto deviceBuffer = paddle::memory::Alloc(
      dev_ctx.GetPlace(),
      bufferSize,
      phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream())));
  int64_t* device_buffer_ptr = reinterpret_cast<int64_t*>(deviceBuffer->ptr());
  paddle::memory::Copy(dev_ctx.GetPlace(),
                       device_buffer_ptr,
                       phi::CPUPlace(),
                       hostBuffer.data(),
                       bufferSize,
                       dev_ctx.stream());
  int64_t* dSrcStrides = reinterpret_cast<int64_t*>(deviceBuffer->ptr());
  int64_t* dDstStrides = dSrcStrides + ndims;
  int64_t* dDims = dDstStrides + ndims;
  int64_t* dDimMagic = dDims + ndims;
  int64_t* dDimShift = dDimMagic + ndims;
  cudaRelocKernel<<<(size + 255) / 256, 256, 0, dev_ctx.stream()>>>(
      size,
      ndims,
      dDims,
      dDimMagic,
      dDimShift,
      src,
      dSrcStrides,
      dst,
      dDstStrides);
}
}  // namespace phi

PD_REGISTER_KERNEL(copyWithStrides,
                   GPU,
                   ALL_LAYOUT,
                   phi::memcpyWithStridesKernel,
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
