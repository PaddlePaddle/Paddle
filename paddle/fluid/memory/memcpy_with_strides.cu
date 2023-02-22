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

#include "paddle/fluid/memory/memcpy.h"

#include "paddle/fluid/platform/device/device_wrapper.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/profiler/event_tracing.h"
#include "paddle/phi/common/place.h"

namespace paddle {
namespace memory {

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

template <typename T>
void memcpyWithStridesKernel(int64_t ndims,
                             const int64_t* dims,
                             T* dst,
                             const int64_t* dstStrides,
                             T* src,
                             const int64_t* srcStrides,
                             size_t num,
                             void* stream) {
  if (num == 0) return;
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
  int64_t* deviceBuffer{nullptr};
  PADDLE_ENFORCE_GPU_SUCCESS(cudaMalloc(&deviceBuffer, bufferSize));
  PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpy(
      deviceBuffer, hostBuffer.data(), bufferSize, cudaMemcpyHostToDevice));
  int64_t* dSrcStrides = deviceBuffer;
  int64_t* dDstStrides = dSrcStrides + ndims;
  int64_t* dDims = dDstStrides + ndims;
  int64_t* dDimMagic = dDims + ndims;
  int64_t* dDimShift = dDimMagic + ndims;
  LOG(WARNING) << "copy with stride cuda kernel";
  cudaRelocKernel<<<(num + 255) / 256,
                    256,
                    0,
                    reinterpret_cast<gpuStream_t>(stream)>>>(num,
                                                             ndims,
                                                             dDims,
                                                             dDimMagic,
                                                             dDimShift,
                                                             src,
                                                             dSrcStrides,
                                                             dst,
                                                             dDstStrides);
}

template <>
void CopywithStrides<float, phi::GPUPlace>(phi::GPUPlace place,
                                           int64_t ndims,
                                           const int64_t* dims,
                                           float* dst,
                                           const int64_t* dstStrides,
                                           float* src,
                                           const int64_t* srcStrides,
                                           size_t num,
                                           void* stream) {
  memcpyWithStridesKernel<float>(
      ndims, dims, dst, dstStrides, src, srcStrides, num, stream);
}

template <>
void CopywithStrides<double, phi::GPUPlace>(phi::GPUPlace place,
                                            int64_t ndims,
                                            const int64_t* dims,
                                            double* dst,
                                            const int64_t* dstStrides,
                                            double* src,
                                            const int64_t* srcStrides,
                                            size_t num,
                                            void* stream) {
  memcpyWithStridesKernel<double>(
      ndims, dims, dst, dstStrides, src, srcStrides, num, stream);
}

template <>
void CopywithStrides<int32_t, phi::GPUPlace>(phi::GPUPlace place,
                                             int64_t ndims,
                                             const int64_t* dims,
                                             int32_t* dst,
                                             const int64_t* dstStrides,
                                             int32_t* src,
                                             const int64_t* srcStrides,
                                             size_t num,
                                             void* stream) {
  memcpyWithStridesKernel<int32_t>(
      ndims, dims, dst, dstStrides, src, srcStrides, num, stream);
}

template <>
void CopywithStrides<int64_t, phi::GPUPlace>(phi::GPUPlace place,
                                             int64_t ndims,
                                             const int64_t* dims,
                                             int64_t* dst,
                                             const int64_t* dstStrides,
                                             int64_t* src,
                                             const int64_t* srcStrides,
                                             size_t num,
                                             void* stream) {
  memcpyWithStridesKernel<int64_t>(
      ndims, dims, dst, dstStrides, src, srcStrides, num, stream);
}
}  // namespace memory
}  // namespace paddle
