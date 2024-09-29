// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include <stdio.h>
#include "cutlass/numeric_conversion.h"
#include "cuda_gemm.h"
#include <cub/cub.cuh>
#define ENABLE_FP8
#define ENABLE_BF16

namespace phi
{
namespace fusion
{
namespace cutlass_internal
{


template <typename T>
struct ToCutlassTypeAdapter
{
    using type = T;
};

template <>
struct ToCutlassTypeAdapter<half>
{
    using type = cutlass::half_t;
};

#if defined(ENABLE_BF16)
template <>
struct ToCutlassTypeAdapter<__nv_bfloat16>
{
    using type = cutlass::bfloat16_t;
};
#endif

#if defined(ENABLE_FP8)
template <>
struct ToCutlassTypeAdapter<__nv_fp8_e4m3>
{
    using type = cutlass::float_e4m3_t;
};

template <>
struct ToCutlassTypeAdapter<__nv_fp8_e5m2>
{
    using type = cutlass::float_e5m2_t;
};
#endif

template <typename Type, int CtaM, int CtaN, int Threads>
__global__ void int8_gemm(int8_t const* act, int8_t const* weight, Type* output, int m, int n, int k)
{
    using VecType = int4;
    static constexpr int kStepK = 128 / (8 * sizeof(int8_t));
    static constexpr int CtaK = kStepK * Threads;
    int tile_id_m = blockIdx.x * CtaM;
    int tile_id_n = blockIdx.y * CtaN;
    int tid = threadIdx.x;
    int8_t tile_a[kStepK], tile_w[CtaN * kStepK];
    int acc[CtaM * CtaN];
#pragma unroll
    for (int i = 0; i < CtaM * CtaN; ++i)
    {
        acc[i] = 0;
    }
    act += tile_id_m * k;
    weight += tile_id_n * k;
    output += tile_id_m * n + tile_id_n;
    for (int idx_k = tid * kStepK; idx_k < k; idx_k += CtaK)
    {
#pragma unroll
        for (int i = 0; i < CtaN; ++i)
        {
            reinterpret_cast<VecType*>(tile_w)[i] = reinterpret_cast<VecType const*>(weight + i * k + idx_k)[0];
        }
#pragma unroll
        for (int i = 0; i < CtaM; ++i)
        {
            reinterpret_cast<VecType*>(tile_a)[0] = reinterpret_cast<VecType const*>(act + i * k + idx_k)[0];
#pragma unroll
            for (int j = 0; j < CtaN; ++j)
            {
#pragma unroll
                for (int l = 0; l < kStepK; l += 4)
                {
                    acc[i * CtaN + j] = __dp4a(reinterpret_cast<int*>(tile_a + l)[0],
                        reinterpret_cast<int*>(tile_w + j * kStepK + l)[0], acc[i * CtaN + j]);
                }
            }
        }
    }

    static constexpr int kWarpSize = 32;
    static constexpr int kWarpNum = Threads / kWarpSize;
    __shared__ int shmem[CtaM * CtaN * kWarpNum];
    int warp_id = tid / kWarpSize, lane_id = tid % kWarpSize;
#pragma unroll
    for (int i = 0; i < CtaM; ++i)
    {
#pragma unroll
        for (int j = 0; j < CtaN; ++j)
        {
            int val = acc[i * CtaN + j];
            val += __shfl_xor_sync(~0, val, 16);
            val += __shfl_xor_sync(~0, val, 8);
            val += __shfl_xor_sync(~0, val, 4);
            val += __shfl_xor_sync(~0, val, 2);
            val += __shfl_xor_sync(~0, val, 1);
            if (lane_id == 0)
            {
                shmem[i * CtaN + j + warp_id * CtaM * CtaN] = val;
            }
        }
    }
    __syncthreads();
#pragma unroll
    for (int ii = tid; ii < CtaM * CtaN; ii += Threads)
    {
        int mid = ii / CtaN, nid = ii % CtaN;
        int val = 0;
#pragma unroll
        for (int jj = 0; jj < kWarpNum; ++jj)
        {
            val += shmem[jj * CtaM * CtaN + ii];
        }
        output[mid * n + nid] = static_cast<Type>(static_cast<float>(val));
    }
}

template <typename InputType, typename OutputType, int32_t TILE_M, int32_t TILE_N, int32_t BLOCK_SIZE>
__global__ void cudaCoreGemm(InputType const* __restrict__ act, InputType const* __restrict__ weight,
    OutputType* __restrict__ output, int32_t m, int32_t n, int32_t k)
{
    using VecType = int4;
    static constexpr int32_t kStepK = static_cast<int32_t>(128 / (8 * sizeof(InputType)));
    static constexpr int32_t kTileK = kStepK * BLOCK_SIZE;
    auto tileIdM = static_cast<int32_t>(blockIdx.x * TILE_M);
    auto tileIdN = static_cast<int32_t>(blockIdx.y * TILE_N);
    auto tid = static_cast<int32_t>(threadIdx.x);
    float tile_a[kStepK], tile_w[TILE_N * kStepK];
    float acc[TILE_M * TILE_N];

    static_assert(kStepK % 4 == 0);
    using CvtInputType = typename ToCutlassTypeAdapter<InputType>::type;
    using Converter = cutlass::NumericArrayConverter<float, CvtInputType, 4>;
    using CvtSrcType = typename Converter::source_type;
    using CvtResType = typename Converter::result_type;

    static constexpr int32_t kCvtCount = static_cast<int32_t>(sizeof(VecType) / sizeof(CvtSrcType));

#pragma unroll
    for (int32_t i = 0; i < TILE_M * TILE_N; ++i)
    {
        acc[i] = 0;
    }
    act += tileIdM * k;
    weight += tileIdN * k;
    output += tileIdM * n + tileIdN;
    for (int32_t idxK = tid * kStepK; idxK < k; idxK += kTileK)
    {
        for (int32_t i = 0; i < TILE_N; ++i)
        {
            auto tile_w_quantized = reinterpret_cast<VecType const*>(weight + i * k + idxK)[0];
#pragma unroll
            for (int32_t cvtIdx = 0; cvtIdx < kCvtCount; ++cvtIdx)
            {
                reinterpret_cast<CvtResType*>(tile_w)[i * kCvtCount + cvtIdx]
                    = Converter::convert(reinterpret_cast<CvtSrcType*>(&tile_w_quantized)[cvtIdx]);
            }
        }
#pragma unroll
        for (int32_t i = 0; i < TILE_M; ++i)
        {
            auto tile_a_quantized = reinterpret_cast<VecType const*>(act + i * k + idxK)[0];
#pragma unroll
            for (int32_t cvtIdx = 0; cvtIdx < kCvtCount; ++cvtIdx)
            {
                reinterpret_cast<CvtResType*>(tile_a)[cvtIdx]
                    = Converter::convert(reinterpret_cast<CvtSrcType*>(&tile_a_quantized)[cvtIdx]);
            }
#pragma unroll
            for (int32_t j = 0; j < TILE_N; ++j)
            {
#pragma unroll
                for (int32_t l = 0; l < kStepK; ++l)
                {
                    acc[i * TILE_N + j] = fma(tile_a[l], tile_w[j * kStepK + l], acc[i * TILE_N + j]);
                }
            }
        }
    }

    typedef cub::WarpReduce<float> WarpReduce;

    static constexpr int32_t kWarpSize = 32;
    static constexpr int32_t kWarpNum = BLOCK_SIZE / kWarpSize;
    int32_t warpId = tid / kWarpSize, laneId = tid % kWarpSize;
    __shared__ float shmem[TILE_M * TILE_N * kWarpNum];
    __shared__ typename WarpReduce::TempStorage tempStorage[kWarpNum];
#pragma unroll
    for (int32_t mi = 0; mi < TILE_M; ++mi)
    {
#pragma unroll
        for (int32_t ni = 0; ni < TILE_N; ++ni)
        {
            float val = WarpReduce(tempStorage[warpId]).Sum(acc[mi * TILE_N + ni]);
            if (laneId == 0)
            {
                shmem[mi * TILE_N + ni + warpId * TILE_M * TILE_N] = val;
            }
        }
    }
    __syncthreads();
    for (int32_t ii = tid; ii < TILE_M * TILE_N; ii += BLOCK_SIZE)
    {
        int32_t mid = ii / TILE_N, nid = ii % TILE_N;
        float val = 0;
#pragma unroll
        for (int32_t jj = 0; jj < kWarpNum; ++jj)
        {
            val += shmem[jj * TILE_M * TILE_N + ii];
        }
        output[mid * n + nid] = static_cast<OutputType>(val);
    }
}

template <typename InputType, typename OutputType, int32_t TILE_M, int32_t TILE_N, int32_t BLOCK_SIZE>
void cudaCoreGemmKernel(GemmParams const& params)
{
    dim3 block(BLOCK_SIZE);
    dim3 grid(params.m / TILE_M, params.n / TILE_N);
    cudaCoreGemm<InputType, OutputType, TILE_M, TILE_N, BLOCK_SIZE><<<grid, block, 0, params.stream>>>(
        reinterpret_cast<InputType const*>(params.act), reinterpret_cast<InputType const*>(params.weight),
        reinterpret_cast<OutputType*>(params.output), params.m, params.n, params.k);
}

template <typename InputType, typename OutputType, int TILE_M, int TILE_N, int BLOCK_SIZE>
bool cudaCoreGemmTemplateCaller(GemmParams const& params)
{
    constexpr int cudaCoreGemmTemplateMaxM = 16;
    if (params.m == TILE_M)
    {
        cudaCoreGemmKernel<InputType, OutputType, TILE_M, TILE_N, BLOCK_SIZE>(params);
        return true;
    }
    if constexpr (TILE_M < cudaCoreGemmTemplateMaxM)
    {
        return cudaCoreGemmTemplateCaller<InputType, OutputType, TILE_M + 1, TILE_N, BLOCK_SIZE>(params);
    }
    return false;
}

template <typename InputType, typename OutputType>
bool cudaCoreGemmLauncher(GemmParams const& params)
{
    return cudaCoreGemmTemplateCaller<InputType, OutputType, 1, 2, 256>(params);
}

bool cudaGemmDispatcher(GemmParams params)
{
    // std::cout<<"cudaGemmDispatcher: inputType="<<params.inputType<<" outputType="<<params.outputType<<std::endl;
    bool dispatched = true;
    if (params.n % 2 != 0)
    {
        dispatched = false;
    }
    // for fp8
    else if (params.inputType == 0)
    {
        if (params.k % 16 != 0)
        {
            // Expect k % 16 == 0 for 128 bits alignment
            dispatched = false;
        }
        else if (params.outputType == 1)
        {
            dispatched = cudaCoreGemmLauncher<__nv_fp8_e4m3, half>(params);
        }
        else if (params.outputType == 2)
        {
            dispatched = cudaCoreGemmLauncher<__nv_fp8_e4m3, __nv_bfloat16>(params);
        }
        else
        {
            dispatched = false;
        }
    }
    // for int8
    else if (params.inputType == 4)
    {
        if (params.k % 16 != 0)
        {
            // Expect k % 16 == 0 for 128 bits alignment
            dispatched = false;
        }
        else if (params.outputType == 5)
        {
            dispatched = cudaCoreGemmLauncher<int8_t, int32_t>(params);
        }
        else
        {
            dispatched = false;
        }
    }
    else
    {
        dispatched = false;
    }

    // if (!dispatched)
    // {
    //     printf(
    //         "phi::fusion::cutlass_internal::cudaGemmDispatcher failed to dispatch: inputType=%d, "
    //         "outputType=%d, "
    //         "m=%d, "
    //         "n=%d, k=%d",
    //         params.inputType, params.outputType, params.m, params.n, params.k);
    // }
    // else {
    //     printf(
    //         "phi::fusion::cutlass_internal::cudaGemmDispatcher dispatched: inputType=%d, "
    //         "outputType=%d, "
    //         "m=%d, "
    //         "n=%d, k=%d",
    //         params.inputType, params.outputType, params.m, params.n, params.k);
    // }
    return dispatched;
}

} // namespace cutlass_internal
} // namespace fusion
} // namespace phi
