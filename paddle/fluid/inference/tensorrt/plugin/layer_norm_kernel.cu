// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/inference/tensorrt/plugin/layer_norm_kernel.h"
#include <cub/cub.cuh>
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/errors.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

template <typename T>
using kvp = cub::KeyValuePair<T, T>;

template <int VPT>
struct BytesToType;

template <>
struct BytesToType<2> {
  using type = uint16_t;
};
template <>
struct BytesToType<4> {
  using type = uint32_t;
};
template <>
struct BytesToType<8> {
  using type = uint64_t;
};
template <>
struct BytesToType<16> {
  using type = float4;
};

template <int Bytes>
__device__ inline void copy(const void* local, void* data) {
  using T = typename BytesToType<Bytes>::type;

  const T* in = static_cast<const T*>(local);
  T* out = static_cast<T*>(data);
  *out = *in;
}

template <typename T>
struct mySum {
  __host__ __device__ __forceinline__ kvp<T> operator()(kvp<T> const& a,
                                                        kvp<T> const& b) const {
    return kvp<T>(a.key + b.key, a.value + b.value);
  }
};

template <typename T, typename OP_T, int32_t TPB>
__global__ void LayerNormSmallKernel(int32_t const nHiddenDimension,
                                     T const* input,
                                     T const* gamma,
                                     T const* beta,
                                     T* output,
                                     float const epsilon) {
  int32_t const index = blockIdx.x * nHiddenDimension + threadIdx.x;
  T const denominator = T(1) / T(nHiddenDimension);
  OP_T val = 0;
  kvp<OP_T> threadData(0, 0);

  if (threadIdx.x < nHiddenDimension) {
    val = input[index];
    OP_T tmp0 = input[index] * denominator;
    OP_T tmp1 = tmp0 * static_cast<OP_T>(input[index]);
    threadData = mySum<OP_T>()(threadData, kvp<OP_T>(tmp0, tmp1));
  }

  using WarpReduce = cub::WarpReduce<kvp<OP_T>, TPB>;
  __shared__ typename WarpReduce::TempStorage temp;
  __shared__ OP_T mu, rsigma;

  auto const sumKV = WarpReduce(temp).Reduce(threadData, mySum<OP_T>());
  if (threadIdx.x == 0) {
    mu = sumKV.key;
    rsigma = rsqrt(sumKV.value - mu * mu + static_cast<OP_T>(epsilon));
  }
  __syncthreads();

  if (threadIdx.x < nHiddenDimension) {
    OP_T const g = gamma[threadIdx.x], b = beta[threadIdx.x];
    output[index] = (val - mu) * rsigma * g + b;
  }
}

template __global__ void LayerNormSmallKernel<float, float, 32>(int32_t const,
                                                                float const*,
                                                                float const*,
                                                                float const*,
                                                                float*,
                                                                float const);
template __global__ void LayerNormSmallKernel<__half, float, 32>(int32_t const,
                                                                 __half const*,
                                                                 __half const*,
                                                                 __half const*,
                                                                 __half*,
                                                                 float const);

template <typename T, typename OP_T, int32_t TPB, int32_t VPT>
__global__ void LayerNormMediumKernel(int32_t const nHiddenDimension,
                                      T const* input,
                                      T const* gamma,
                                      T const* beta,
                                      T* output,
                                      float const epsilon) {
  int32_t const index = blockIdx.x * nHiddenDimension + threadIdx.x * VPT;
  T localX[VPT], localGamma[VPT], localBeta[VPT];
  OP_T const denominator = OP_T(1) / OP_T(nHiddenDimension);
  kvp<OP_T> threadData(0, 0);

  copy<sizeof(T) * VPT>(&input[index], localX);
#pragma unroll
  for (int32_t it = 0; it < VPT; it++) {
    OP_T const tmp = static_cast<OP_T>(localX[it]) * denominator;
    threadData = mySum<OP_T>()(
        threadData, kvp<OP_T>(tmp, tmp * static_cast<OP_T>(localX[it])));
  }

  copy<sizeof(T) * VPT>(&beta[threadIdx.x * VPT], localBeta);
  copy<sizeof(T) * VPT>(&gamma[threadIdx.x * VPT], localGamma);

  using BlockReduce = cub::BlockReduce<kvp<OP_T>, TPB>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ OP_T mu, rsigma;

  auto const sumKV =
      BlockReduce(temp_storage).Reduce(threadData, mySum<OP_T>());
  if (threadIdx.x == 0) {
    mu = sumKV.key;
    rsigma = rsqrt(sumKV.value - mu * mu + static_cast<OP_T>(epsilon));
  }
  __syncthreads();

#pragma unroll
  for (int32_t it = 0; it < VPT; it++) {
    localX[it] = static_cast<OP_T>(localGamma[it]) *
                     (static_cast<OP_T>(localX[it]) - mu) * rsigma +
                 static_cast<OP_T>(localBeta[it]);
  }

  copy<sizeof(T) * VPT>(localX, &output[index]);
}

template __global__ void LayerNormMediumKernel<float, float, 64, 4>(
    int32_t const,
    float const*,
    float const*,
    float const*,
    float*,
    float const);
template __global__ void LayerNormMediumKernel<__half, float, 64, 4>(
    int32_t const,
    __half const*,
    __half const*,
    __half const*,
    __half*,
    float const);

template <typename T, typename OP_T, int32_t TPB>
__global__ void LayerNormLargeKernel(int32_t const nHiddenDimension,
                                     T const* input,
                                     T const* gamma,
                                     T const* beta,
                                     T* output,
                                     float const epsilon) {
  int32_t const offset = blockIdx.x * nHiddenDimension;
  OP_T const denominator = OP_T(1) / OP_T(nHiddenDimension);
  kvp<OP_T> threadData(0, 0);

  for (int32_t i = threadIdx.x; i < nHiddenDimension; i += TPB) {
    int32_t const index = offset + i;
    OP_T val = input[index];
    OP_T const tmp = val * denominator;
    threadData = mySum<OP_T>()(threadData, kvp<OP_T>(tmp, tmp * val));
    output[index] = val;
  }

  using BlockReduce = cub::BlockReduce<kvp<OP_T>, TPB>;
  __shared__ typename BlockReduce::TempStorage temp;
  __shared__ OP_T mu, rsigma;

  auto const sumKV = BlockReduce(temp).Reduce(threadData, mySum<OP_T>());

  if (threadIdx.x == 0) {
    mu = sumKV.key;
    rsigma = rsqrt(sumKV.value - mu * mu + static_cast<OP_T>(epsilon));
  }
  __syncthreads();

  for (int32_t i = threadIdx.x; i < nHiddenDimension; i += TPB) {
    int32_t const index = offset + i;
    output[index] = (static_cast<OP_T>(output[index]) - mu) * rsigma *
                        static_cast<OP_T>(gamma[i]) +
                    static_cast<OP_T>(beta[i]);
  }
}

template __global__ void LayerNormLargeKernel<float, float, 256>(int32_t const,
                                                                 float const*,
                                                                 float const*,
                                                                 float const*,
                                                                 float*,
                                                                 float const);
template __global__ void LayerNormLargeKernel<__half, float, 256>(int32_t const,
                                                                  __half const*,
                                                                  __half const*,
                                                                  __half const*,
                                                                  __half*,
                                                                  float const);

template <int32_t TPB, int32_t VPT>
__global__ void LayerNormQDQKernel(int32_t const nHiddenDimension,
                                   int8_t const* input,
                                   int8_t* output,
                                   __half const* gamma,
                                   __half const* beta,
                                   float const dqScaleIn,
                                   float const qScale,
                                   float const epsilon) {
  int32_t const index = nHiddenDimension * blockIdx.x + threadIdx.x * VPT;
  int8_t localX[VPT];
  __half localXDQ[VPT], localBeta[VPT], localGamma[VPT];

  copy<sizeof(int8_t) * VPT>(&input[index], localX);
  __half2 loc = __floats2half2_rn(0.f, 0.f);

  __half const denominator = __half(1) / __half(nHiddenDimension);
#pragma unroll
  for (int32_t it = 0; it < VPT; it++) {
    float const tmp_in = localX[it];
    localXDQ[it] = dqScaleIn * tmp_in;

    __half const tmp = localXDQ[it] * denominator;
    __half2 const tmp2 = __halves2half2(tmp, tmp * localXDQ[it]);
    loc = loc + tmp2;
  }

  copy<sizeof(__half) * VPT>(&beta[threadIdx.x * VPT], localBeta);
  copy<sizeof(__half) * VPT>(&gamma[threadIdx.x * VPT], localGamma);

  using BlockReduce = cub::BlockReduce<__half2, TPB>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ __half mu;      // mean
  __shared__ __half rsigma;  // 1 / std.dev.

  __half2 const sum2 = BlockReduce(temp_storage).Reduce(loc, cub::Sum());

  if (threadIdx.x == 0) {
    mu = __low2half(sum2);
    rsigma = rsqrt(__high2half(sum2) - mu * mu + (__half)epsilon);
  }
  __syncthreads();

#pragma unroll
  for (int32_t it = 0; it < VPT; it++) {
    float const tmp =
        localGamma[it] * (localXDQ[it] - mu) * rsigma + localBeta[it];
    int32_t tmpq = __float2int_rn(qScale * tmp);
    tmpq = max(-127, tmpq);
    tmpq = min(127, tmpq);
    localX[it] = tmpq;
  }

  copy<sizeof(int8_t) * VPT>(localX, &output[index]);
}

template __global__ void LayerNormQDQKernel<32, 8>(int32_t const,
                                                   int8_t const*,
                                                   int8_t*,
                                                   __half const*,
                                                   __half const*,
                                                   float const,
                                                   float const,
                                                   float const);
template __global__ void LayerNormQDQKernel<128, 8>(int32_t const,
                                                    int8_t const*,
                                                    int8_t*,
                                                    __half const*,
                                                    __half const*,
                                                    float const,
                                                    float const,
                                                    float const);

template <typename T>
cudaError_t computeLayerNorm(int32_t const gridSize,
                             int32_t const nHiddenDimension,
                             T const* input,
                             T const* gamma,
                             T const* beta,
                             T* output,
                             float const epsilon,
                             cudaStream_t stream) {
  constexpr int32_t VPT = 16 / sizeof(T);
  if (nHiddenDimension <= 32) {
    constexpr int32_t TPB = 32;
    (LayerNormSmallKernel<T, float, TPB>)<<<gridSize, TPB, 0, stream>>>(
        nHiddenDimension, input, gamma, beta, output, epsilon);
  } else if (nHiddenDimension == 320) {
    constexpr int32_t TPB = 320 / VPT;
    (LayerNormMediumKernel<T, float, TPB, VPT>)<<<gridSize, TPB, 0, stream>>>(
        nHiddenDimension, input, gamma, beta, output, epsilon);
  } else if (nHiddenDimension == 640) {
    constexpr int32_t TPB = 640 / VPT;
    (LayerNormMediumKernel<T, float, TPB, VPT>)<<<gridSize, TPB, 0, stream>>>(
        nHiddenDimension, input, gamma, beta, output, epsilon);
  } else if (nHiddenDimension == 768) {
    constexpr int32_t TPB = 768 / VPT;
    (LayerNormMediumKernel<T, float, TPB, VPT>)<<<gridSize, TPB, 0, stream>>>(
        nHiddenDimension, input, gamma, beta, output, epsilon);
  } else if (nHiddenDimension == 1280) {
    constexpr int32_t TPB = 1280 / VPT;
    (LayerNormMediumKernel<T, float, TPB, VPT>)<<<gridSize, TPB, 0, stream>>>(
        nHiddenDimension, input, gamma, beta, output, epsilon);
  } else {
    constexpr int32_t TPB = 256;
    (LayerNormLargeKernel<T, float, TPB>)<<<gridSize, TPB, 0, stream>>>(
        nHiddenDimension, input, gamma, beta, output, epsilon);
  }
  return cudaGetLastError();
}

template cudaError_t computeLayerNorm<float>(int const,
                                             int const,
                                             float const*,
                                             float const*,
                                             float const*,
                                             float*,
                                             float const,
                                             cudaStream_t);
template cudaError_t computeLayerNorm<half>(int const,
                                            int const,
                                            half const*,
                                            half const*,
                                            half const*,
                                            half*,
                                            float const,
                                            cudaStream_t);

cudaError_t computeLayerNormQDQ(int32_t const gridSize,
                                int32_t const nHiddenDimension,
                                int8_t const* input,
                                __half const* gamma,
                                __half const* beta,
                                int8_t* output,
                                float const dqScaleIn,
                                float const qScale,
                                float const epsilon,
                                cudaStream_t stream) {
  constexpr int32_t VPT = 16 / sizeof(__half);
  if (nHiddenDimension == 320) {
    constexpr int32_t TPB = 320 / VPT;
    (LayerNormQDQKernel<TPB, VPT>)<<<gridSize, TPB, 0, stream>>>(
        nHiddenDimension,
        input,
        output,
        gamma,
        beta,
        dqScaleIn,
        qScale,
        epsilon);
  } else if (nHiddenDimension == 640) {
    constexpr int32_t TPB = 640 / VPT;
    (LayerNormQDQKernel<TPB, VPT>)<<<gridSize, TPB, 0, stream>>>(
        nHiddenDimension,
        input,
        output,
        gamma,
        beta,
        dqScaleIn,
        qScale,
        epsilon);
  } else if (nHiddenDimension == 1280) {
    constexpr int32_t TPB = 1280 / VPT;
    (LayerNormQDQKernel<TPB, VPT>)<<<gridSize, TPB, 0, stream>>>(
        nHiddenDimension,
        input,
        output,
        gamma,
        beta,
        dqScaleIn,
        qScale,
        epsilon);
  } else {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "[computeLayerNormQDQ] Unsupport hidden dimension: %d",
        nHiddenDimension));
  }
  return cudaGetLastError();
}

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
