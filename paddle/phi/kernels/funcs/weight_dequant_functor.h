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

#pragma once

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/datatype_traits.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/aligned_vector.h"
#include "paddle/phi/kernels/fusion/cutlass/cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm_template.h"
#include "paddle/phi/kernels/matmul_kernel.h"

namespace phi {

template <typename T, int WeightBit>
struct FastWeightOnlyHalfConverter;

template <>
struct FastWeightOnlyHalfConverter<half, 8> {
  using Converter =
      cutlass::FastInterleavedAndBiasedNumericArrayConverter<cutlass::half_t,
                                                             uint8_t,
                                                             4>;
  static constexpr int kHalfLength = 4;
  static constexpr int kWeightOnlyLength = 4;

  __device__ static inline void convert(half halves[kHalfLength],
                                        uint8_t chars[kWeightOnlyLength],
                                        float scale) {
    *reinterpret_cast<Converter::result_type*>(halves) =
        Converter::convert(*reinterpret_cast<Converter::source_type*>(chars));
#pragma unroll
    for (int i = 0; i < kHalfLength; ++i) {
      float dequant_value = __half2float(halves[i]) * scale;
      halves[i] = __float2half_rn(dequant_value);
    }
  }
};

template <>
struct FastWeightOnlyHalfConverter<half, 4> {
  using Converter =
      cutlass::FastInterleavedAndBiasedNumericArrayConverter<cutlass::half_t,
                                                             cutlass::uint4b_t,
                                                             8>;
  static constexpr int kHalfLength = 8;
  static constexpr int kWeightOnlyLength = 4;

  __device__ static inline void convert(half halves[kHalfLength],
                                        uint8_t chars[kWeightOnlyLength],
                                        float scale) {
    *reinterpret_cast<Converter::result_type*>(halves) =
        Converter::convert(*reinterpret_cast<Converter::source_type*>(chars));
#pragma unroll
    for (int i = 0; i < kHalfLength; ++i) {
      float dequant_value = __half2float(halves[i]) * scale;
      halves[i] = __float2half_rn(dequant_value);
    }
  }
};

#if defined(PADDLE_CUDA_BF16)
template <>
struct FastWeightOnlyHalfConverter<__nv_bfloat16, 8> {
  using Converter = cutlass::FastInterleavedAndBiasedNumericArrayConverter<
      cutlass::bfloat16_t,
      uint8_t,
      4>;
  static constexpr int kHalfLength = 4;
  static constexpr int kWeightOnlyLength = 4;

  __device__ static inline void convert(__nv_bfloat16 halves[kHalfLength],
                                        uint8_t chars[kWeightOnlyLength],
                                        float scale) {
    *reinterpret_cast<Converter::result_type*>(halves) =
        Converter::convert(*reinterpret_cast<Converter::source_type*>(chars));
#pragma unroll
    for (int i = 0; i < kHalfLength; ++i) {
      float dequant_value = __bfloat162float(halves[i]) * scale;
      halves[i] = __float2bfloat16_rn(dequant_value);
    }
  }
};

template <>
struct FastWeightOnlyHalfConverter<__nv_bfloat16, 4> {
  using Converter = cutlass::FastInterleavedAndBiasedNumericArrayConverter<
      cutlass::bfloat16_t,
      cutlass::uint4b_t,
      8>;
  static constexpr int kHalfLength = 8;
  static constexpr int kWeightOnlyLength = 4;

  __device__ static inline void convert(__nv_bfloat16 halves[kHalfLength],
                                        uint8_t chars[kWeightOnlyLength],
                                        float scale) {
    *reinterpret_cast<Converter::result_type*>(halves) =
        Converter::convert(*reinterpret_cast<Converter::source_type*>(chars));
#pragma unroll
    for (int i = 0; i < kHalfLength; ++i) {
      float dequant_value = __bfloat162float(halves[i]) * scale;
      halves[i] = __float2bfloat16_rn(dequant_value);
    }
  }
};
#endif

template <typename T>
__global__ void int8_weight_only_dequant(const uint8_t* weight,
                                         const T* scale_list,
                                         T* output,
                                         const int n,
                                         const int k) {
  using Converter = FastWeightOnlyHalfConverter<T, 8>;
  AlignedVector<uint8_t, 16> vec_weight;
  T vec_weight_f16[16];
  AlignedVector<T, 16> vec_out;

  int warp_id = threadIdx.x / 32, lane_id = threadIdx.x % 32;
  int tile_id = blockIdx.x * blockDim.x / 32 + warp_id;
  // Every two rows of the original weights are interleaved into a row with
  // stride of 64, so if each thread processes 16 elements(for int8, we can use
  // ldg.128 to load weights), then every group of four adjacent threads will
  // alternately process two different row weights for example every 128
  // consecutive int8 elements [128*i, 128*(i+1)-1] of row N under interleave
  // layout, the first 64 are from [64*i, 64*(i+1)-1] of row 2N before
  // interleaving, and the last 64 are from [64*i, 64*(i+1)-1] of row 2N+1
  // before interleaving. So if each thread loads 16 int8 elements, then the
  // elements of the first four and last four threads of each 8 consecutive
  // threads will come from row 2N and row 2N+1 respectively before
  // interleaving.
  int row_id = tile_id * 2 + ((lane_id % 8) > 3 ? 1 : 0);
  weight += tile_id * k * 2;
  output += row_id * k;
  float scale = static_cast<float>(scale_list[row_id]);
#pragma unroll
  for (int i = lane_id * 16; i < k * 2; i += 16 * 32) {
    Load<uint8_t, 16>(&weight[i], &vec_weight);
#pragma unroll
    for (int p = 0; p < 16; p += Converter::kHalfLength) {
      // The rearrangement here counteracts the effect of
      // cutlass::add_bias_and_interleave_int8s_inplace Input int8 data layout
      //      [elt_3  elt_1  elt_2  elt_0] (each elt occupies 8 bits)
      //
      // Converted fp16 data layout
      //      [elt_3  elt_2  elt_1  elt_0] (each elt occupies 16 bits)
      // vec_weight_f16[p] = static_cast<T>(static_cast<float>(vec_weight[p]) *
      // scale);
      // fast_cvt_4_packed_signed_i8s_to_2_half2s<T>()
      Converter::convert(vec_weight_f16 + p, &vec_weight[p], scale);
    }
#pragma unroll
    for (int p = 0; p < 16; ++p) {
      // The index remapping here is to counteracts the effect of
      // cutlass::permute_B_rows_for_mixed_gemm input 0 1 2 3 4 5 6 7 8 9 10 11
      // 12 13 14 15 weight 0 1 8 9 2 3 10 11 4 5 12 13 6 7 14 15
      vec_out[p] = vec_weight_f16[4 * ((p % 8) / 2) + p % 2 + 2 * (p / 8)];
    }
    Store<T, 16>(vec_out, &output[i / 128 * 64 + (i % 64)]);
  }
}

template <typename T>
__global__ void int4_weight_only_dequant(const uint8_t* weight,
                                         const T* scale_list,
                                         T* output,
                                         const int n,
                                         const int k) {
  using Converter = FastWeightOnlyHalfConverter<T, 4>;

  AlignedVector<uint8_t, 16> vec_weight;
  T vec_weight_f16[32];
  AlignedVector<T, 32> vec_out;

  int warp_id = threadIdx.x / 32, lane_id = threadIdx.x % 32;
  int tile_id = blockIdx.x * blockDim.x / 32 + warp_id;
  // Every 4 rows of the original weights are interleaved into a row with
  // stride of 32, so if each thread processes 16 elements(for int8, we can use
  // ldg.128 to load weights), then every group of two adjacent threads will
  // alternately process four different row weights for example every 128
  // consecutive int8 elements [128*i, 128*(i+1)-1] of row N under interleave
  // layout, the first 64 are from [64*i, 64*(i+1)-1] of row 2N before
  // interleaving, and the last 64 are from [64*i, 64*(i+1)-1] of row 2N+1
  // before interleaving. So if each thread loads 16 int8 elements, then the
  // elements of the first four and last four threads of each 8 consecutive
  // threads will come from row 2N and row 2N+1 respectively before
  // interleaving.
  int row_id = tile_id * 4 + ((lane_id % 8) / 2);
  weight += tile_id * k / 2 * 4;
  output += row_id * k;
  float scale = static_cast<float>(scale_list[row_id]);
#pragma unroll
  for (int i = lane_id * 32; i < k * 4; i += 32 * 32) {
    Load<uint8_t, 16>(&weight[i / 2], &vec_weight);
#pragma unroll
    for (int p = 0; p < 32; p += Converter::kHalfLength) {
      // The rearrangement here counteracts the effect of
      // cutlass::add_bias_and_interleave_int4s_inplace Input int8 data layout
      //      [elt_7  elt_5  elt_3  elt_1  elt_6  elt_4  elt_2  elt_0] (each elt
      //      occupies 4 bits)
      //
      // Converted fp16 data layout
      //      [elt_7  elt_6  elt_5  elt_4  elt_3  elt_2  elt_1  elt_0] (each elt
      //      occupies 16 bits)
      // vec_weight_f16[p] =
      //     static_cast<T>(static_cast<float>(vec_weight[p]) * scale);
      Converter::convert(vec_weight_f16 + p, &vec_weight[p / 2], scale);
    }
#pragma unroll
    for (int p = 0; p < 32; ++p) {
      // The index remapping here is to counteracts the effect of
      // cutlass::permute_B_rows_for_mixed_gemm input 0 1 2 3 4 5 6 7 8 9 10 11
      // 12 13 14 15 ... 31 weight 0 1 8 9 16 17 24 25 2 3 10 11 18 19 26 27 4 5
      // 12 13 20 21 28 29 6 7 14 15 22 23 30 31
      vec_out[p] = vec_weight_f16[8 * ((p % 8) / 2) + p % 2 + 2 * (p / 8)];
    }
    Store<T, 32>(vec_out, &output[i / 256 * 64 + (i % 64)]);
  }
}

template <typename T>
__global__ void int8_weight_only_dequant(const uint8_t* weight,
                                         const T* scales,
                                         T* output,
                                         const int n,
                                         const int k,
                                         const int group_size) {
  using Converter = FastWeightOnlyHalfConverter<T, 8>;
  AlignedVector<uint8_t, 16> vec_weight;
  T vec_weight_f16[16];
  AlignedVector<T, 16> vec_out;

  int warp_id = threadIdx.x / 32, lane_id = threadIdx.x % 32;
  int tile_id = blockIdx.x * blockDim.x / 32 + warp_id;
  // Every two rows of the original weights are interleaved into a row with
  // stride of 64, so if each thread processes 16 elements(for int8, we can use
  // ldg.128 to load weights), then every group of four adjacent threads will
  // alternately process two different row weights for example every 128
  // consecutive int8 elements [128*i, 128*(i+1)-1] of row N under interleave
  // layout, the first 64 are from [64*i, 64*(i+1)-1] of row 2N before
  // interleaving, and the last 64 are from [64*i, 64*(i+1)-1] of row 2N+1
  // before interleaving. So if each thread loads 16 int8 elements, then the
  // elements of the first four and last four threads of each 8 consecutive
  // threads will come from row 2N and row 2N+1 respectively before
  // interleaving.
  int row_id = tile_id * 2 + ((lane_id % 8) > 3 ? 1 : 0);
  weight += tile_id * k * 2;
  output += row_id * k;

  scales += row_id;
#pragma unroll
  for (int i = lane_id * 16; i < k * 2; i += 16 * 32) {
    int scale_offset = i / 2 / group_size;
    float scale = static_cast<float>(scales[scale_offset * n]);
    Load<uint8_t, 16>(&weight[i], &vec_weight);
#pragma unroll
    for (int p = 0; p < 16; p += Converter::kHalfLength) {
      // The rearrangement here counteracts the effect of
      // cutlass::add_bias_and_interleave_int8s_inplace Input int8 data layout
      //      [elt_3  elt_1  elt_2  elt_0] (each elt occupies 8 bits)
      //
      // Converted fp16 data layout
      //      [elt_3  elt_2  elt_1  elt_0] (each elt occupies 16 bits)
      // vec_weight_f16[p] = static_cast<T>(static_cast<float>(vec_weight[p]) *
      // scale);
      // fast_cvt_4_packed_signed_i8s_to_2_half2s<T>()
      Converter::convert(vec_weight_f16 + p, &vec_weight[p], scale);
    }
#pragma unroll
    for (int p = 0; p < 16; ++p) {
      // The index remapping here is to counteracts the effect of
      // cutlass::permute_B_rows_for_mixed_gemm input 0 1 2 3 4 5 6 7 8 9 10 11
      // 12 13 14 15 weight 0 1 8 9 2 3 10 11 4 5 12 13 6 7 14 15
      vec_out[p] = vec_weight_f16[4 * ((p % 8) / 2) + p % 2 + 2 * (p / 8)];
    }
    Store<T, 16>(vec_out, &output[i / 128 * 64 + (i % 64)]);
  }
}

template <typename T>
__global__ void int4_weight_only_dequant(const uint8_t* weight,
                                         const T* scales,
                                         T* output,
                                         const int n,
                                         const int k,
                                         const int group_size) {
  using Converter = FastWeightOnlyHalfConverter<T, 4>;

  AlignedVector<uint8_t, 16> vec_weight;
  T vec_weight_f16[32];
  AlignedVector<T, 32> vec_out;

  int warp_id = threadIdx.x / 32, lane_id = threadIdx.x % 32;
  int tile_id = blockIdx.x * blockDim.x / 32 + warp_id;
  // Every two rows of the original weights are interleaved into a row with
  // stride of 64, so if each thread processes 16 elements(for int8, we can use
  // ldg.128 to load weights), then every group of four adjacent threads will
  // alternately process two different row weights for example every 128
  // consecutive int8 elements [128*i, 128*(i+1)-1] of row N under interleave
  // layout, the first 64 are from [64*i, 64*(i+1)-1] of row 2N before
  // interleaving, and the last 64 are from [64*i, 64*(i+1)-1] of row 2N+1
  // before interleaving. So if each thread loads 16 int8 elements, then the
  // elements of the first four and last four threads of each 8 consecutive
  // threads will come from row 2N and row 2N+1 respectively before
  // interleaving.
  int row_id = tile_id * 4 + ((lane_id % 8) / 2);
  weight += tile_id * k / 2 * 4;
  output += row_id * k;
  scales += row_id;
#pragma unroll
  for (int i = lane_id * 32; i < k * 4; i += 32 * 32) {
    Load<uint8_t, 16>(&weight[i / 2], &vec_weight);
    int scale_offset = i / 4 / group_size;
    float scale = static_cast<float>(scales[scale_offset * n]);
#pragma unroll
    for (int p = 0; p < 32; p += Converter::kHalfLength) {
      // The rearrangement here counteracts the effect of
      // cutlass::add_bias_and_interleave_int4s_inplace Input int8 data layout
      //      [elt_7  elt_5  elt_3  elt_1  elt_6  elt_4  elt_2  elt_0] (each elt
      //      occupies 4 bits)
      //
      // Converted fp16 data layout
      //      [elt_7  elt_6  elt_5  elt_4  elt_3  elt_2  elt_1  elt_0] (each elt
      //      occupies 16 bits)
      // vec_weight_f16[p] =
      //     static_cast<T>(static_cast<float>(vec_weight[p]) * scale);
      Converter::convert(vec_weight_f16 + p, &vec_weight[p / 2], scale);
    }
#pragma unroll
    for (int p = 0; p < 32; ++p) {
      // The index remapping here is to counteracts the effect of
      // cutlass::permute_B_rows_for_mixed_gemm input 0 1 2 3 4 5 6 7 8 9 10 11
      // 12 13 14 15 ... 31 weight 0 1 8 9 16 17 24 25 2 3 10 11 18 19 26 27 4 5
      // 12 13 20 21 28 29 6 7 14 15 22 23 30 31
      vec_out[p] = vec_weight_f16[8 * ((p % 8) / 2) + p % 2 + 2 * (p / 8)];
    }
    Store<T, 32>(vec_out, &output[i / 256 * 64 + (i % 64)]);
  }
}

template <typename T, typename Context>
void WeightDequantize(const Context& dev_ctx,
                      const DenseTensor& x,
                      const DenseTensor& scale,
                      const std::string& algo,
                      const bool transpose,
                      const int32_t group_size,
                      DenseTensor* out) {
  using DataType = typename PDDataTypeTraits<T>::DataType;

  int n = scale.dims()[0];
  int k = x.dims()[1];
  dim3 block(512);
  dim3 grid(n / 32);
  auto stream = dev_ctx.stream();

  if (algo == "weight_only_int8" && group_size == -1) {
    int8_weight_only_dequant<DataType><<<grid, block, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(x.data<int8_t>()),
        reinterpret_cast<const DataType*>(scale.data<T>()),
        reinterpret_cast<DataType*>(out->data<T>()),
        n,
        k);
  } else if (algo == "weight_only_int8" && group_size > 0) {
    int8_weight_only_dequant<DataType><<<grid, block, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(x.data<int8_t>()),
        reinterpret_cast<const DataType*>(scale.data<T>()),
        reinterpret_cast<DataType*>(out->data<T>()),
        n,
        k,
        group_size);
  } else if (algo == "weight_only_int4" && group_size == -1) {
    k *= 2;
    grid.x /= 2;
    int4_weight_only_dequant<DataType><<<grid, block, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(x.data<int8_t>()),
        reinterpret_cast<const DataType*>(scale.data<T>()),
        reinterpret_cast<DataType*>(out->data<T>()),
        n,
        k);
  } else if (algo == "weight_only_int4" && group_size > 0) {
    k *= 2;
    grid.x /= 2;
    int4_weight_only_dequant<DataType><<<grid, block, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(x.data<int8_t>()),
        reinterpret_cast<const DataType*>(scale.data<T>()),
        reinterpret_cast<DataType*>(out->data<T>()),
        n,
        k,
        group_size);
  }
}

}  // namespace phi
