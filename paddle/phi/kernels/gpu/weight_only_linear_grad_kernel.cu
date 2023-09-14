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
#include "paddle/phi/kernels/weight_only_linear_grad_kernel.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/datatype_traits.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/aligned_vector.h"
#include "paddle/phi/kernels/matmul_kernel.h"

#if defined(PADDLE_WITH_CUTLASS)
#include "paddle/phi/kernels/fusion/cutlass/cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm_template.h"
#endif

namespace phi {
#if defined(PADDLE_WITH_CUTLASS)
template <typename T, int WeightBit>
struct FastWeightOnlyHalfConverter;

template <>
struct FastWeightOnlyHalfConverter<dtype::float16, 8> {
  using Converter =
      cutlass::FastInterleavedAndBiasedNumericArrayConverter<cutlass::half_t,
                                                             uint8_t,
                                                             4>;
  static constexpr int kHalfLength = 4;
  static constexpr int kWeightOnlyLength = 4;

  __device__ static inline void convert(dtype::float16 halves[kHalfLength],
                                        uint8_t chars[kWeightOnlyLength],
                                        float scale) {
    *reinterpret_cast<Converter::result_type*>(halves) =
        Converter::convert(*reinterpret_cast<Converter::source_type*>(chars));
#pragma unroll
    for (int i = 0; i < kHalfLength; ++i) {
      halves[i] =
          static_cast<dtype::float16>(static_cast<float>(halves[i]) * scale);
    }
  }
};

template <>
struct FastWeightOnlyHalfConverter<dtype::float16, 4> {
  using Converter =
      cutlass::FastInterleavedAndBiasedNumericArrayConverter<cutlass::half_t,
                                                             cutlass::uint4b_t,
                                                             8>;
  static constexpr int kHalfLength = 8;
  static constexpr int kWeightOnlyLength = 4;

  __device__ static inline void convert(dtype::float16 halves[kHalfLength],
                                        uint8_t chars[kWeightOnlyLength],
                                        float scale) {
    *reinterpret_cast<Converter::result_type*>(halves) =
        Converter::convert(*reinterpret_cast<Converter::source_type*>(chars));
#pragma unroll
    for (int i = 0; i < kHalfLength; ++i) {
      halves[i] =
          static_cast<dtype::float16>(static_cast<float>(halves[i]) * scale);
    }
  }
};

#if defined(PADDLE_CUDA_BF16)
template <>
struct FastWeightOnlyHalfConverter<dtype::bfloat16, 8> {
  using Converter = cutlass::FastInterleavedAndBiasedNumericArrayConverter<
      cutlass::bfloat16_t,
      uint8_t,
      4>;
  static constexpr int kHalfLength = 4;
  static constexpr int kWeightOnlyLength = 4;

  __device__ static inline void convert(dtype::bfloat16 halves[kHalfLength],
                                        uint8_t chars[kWeightOnlyLength],
                                        float scale) {
    *reinterpret_cast<Converter::result_type*>(halves) =
        Converter::convert(*reinterpret_cast<Converter::source_type*>(chars));
#pragma unroll
    for (int i = 0; i < kHalfLength; ++i) {
      halves[i] =
          static_cast<dtype::bfloat16>(static_cast<float>(halves[i]) * scale);
    }
  }
};

template <>
struct FastWeightOnlyHalfConverter<dtype::bfloat16, 4> {
  using Converter = cutlass::FastInterleavedAndBiasedNumericArrayConverter<
      cutlass::bfloat16_t,
      cutlass::uint4b_t,
      8>;
  static constexpr int kHalfLength = 8;
  static constexpr int kWeightOnlyLength = 4;

  __device__ static inline void convert(dtype::bfloat16 halves[kHalfLength],
                                        uint8_t chars[kWeightOnlyLength],
                                        float scale) {
    *reinterpret_cast<Converter::result_type*>(halves) =
        Converter::convert(*reinterpret_cast<Converter::source_type*>(chars));
#pragma unroll
    for (int i = 0; i < kHalfLength; ++i) {
      halves[i] =
          static_cast<dtype::bfloat16>(static_cast<float>(halves[i]) * scale);
    }
  }
};
#endif

template <typename VecType, typename T0, typename T1>
__device__ __forceinline__ void load(T0* dst, T1* src, size_t offset = 0) {
  *reinterpret_cast<VecType*>(dst) =
      *(reinterpret_cast<const VecType*>(src) + offset);
}

template <typename VecType, typename T0, typename T1>
__device__ __forceinline__ void store(T0* src, T1* dst, size_t offset = 0) {
  *(reinterpret_cast<VecType*>(dst) + offset) =
      *reinterpret_cast<const VecType*>(src);
}

template <typename T, int K = 0>
__global__ void int8_weight_only_dequant(const int8_t* weight,
                                         const float* scale_list,
                                         T* output,
                                         const int n,
                                         const int k_) {
  int k = K != 0 ? K : k_;
  using Converter = FastWeightOnlyHalfConverter<T, 8>;

  uint8_t vec_weight[16];
  T vec_weight_f16[16];
  T vec_out[16];
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
  float scale = scale_list[row_id];
#pragma unroll
  for (int i = lane_id * 16; i < k * 2; i += 16 * 32) {
    load<uint4>(vec_weight, weight + i);
    int out_idx = i / 128 * 64 + (i % 64);
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
      Converter::convert(vec_weight_f16 + p, vec_weight + p, scale);
    }
#pragma unroll
    for (int p = 0; p < 16; ++p) {
      // The index remapping here is to counteracts the effect of
      // cutlass::permute_B_rows_for_mixed_gemm input 0 1 2 3 4 5 6 7 8 9 10 11
      // 12 13 14 15 weight 0 1 8 9 2 3 10 11 4 5 12 13 6 7 14 15
      vec_out[p] = static_cast<T>(
          vec_weight_f16[4 * ((p % 8) / 2) + p % 2 + 2 * (p / 8)]);
    }
    store<T>(vec_out, &output[out_idx]);
    // Store<OutType, VecSize>(vec_out, &output[out_idx]);
  }
}

template <typename T, int K = 0>
__global__ void int4_weight_only_dequant(const int8_t* weight,
                                         const float* scale_list,
                                         T* output,
                                         const int n,
                                         const int k_) {
  int k = K != 0 ? K : k_;
  using Converter = FastWeightOnlyHalfConverter<T, 4>;

  uint8_t vec_weight[16];
  T vec_weight_f16[32];
  T vec_out[32];
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
  float scale = scale_list[row_id];
#pragma unroll
  for (int i = lane_id * 32; i < k * 4; i += 32 * 32) {
    load<uint4>(vec_weight, weight + i / 2);
    int out_idx = i / 256 * 64 + (i % 64);
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
      Converter::convert(vec_weight_f16 + p, vec_weight + p, scale);
    }
#pragma unroll
    for (int p = 0; p < 32; ++p) {
      // The index remapping here is to counteracts the effect of
      // cutlass::permute_B_rows_for_mixed_gemm input 0 1 2 3 4 5 6 7 8 9 10 11
      // 12 13 14 15 ... 31 weight 0 1 8 9 16 17 24 25 2 3 10 11 18 19 26 27 4 5
      // 12 13 20 21 28 29 6 7 14 15 22 23 30 31
      vec_out[p] = static_cast<T>(
          vec_weight_f16[8 * ((p % 8) / 2) + p % 2 + 2 * (p / 8)]);
    }
    store<T>(vec_out, &output[out_idx]);
  }
}
#endif

template <typename T, typename Context>
void WeightOnlyLinearGradKernel(const Context& dev_ctx,
                                const DenseTensor& x,
                                const DenseTensor& weight,
                                const paddle::optional<DenseTensor>& bias,
                                const DenseTensor& weight_scale,
                                const DenseTensor& out_grad,
                                const std::string& weight_dtype,
                                DenseTensor* x_grad) {
#if defined(PADDLE_WITH_CUTLASS)
  int n = weight_scale.dims()[0];
  int k = weight.dims()[1];
  dim3 block(512);
  dim3 grid(n / 32);
  auto stream = dev_ctx.stream();

  dev_ctx.template Alloc<T>(x_grad);
  DenseTensor weight_dequantized;
  weight_dequantized.Resize({{n, k}});
  // int size = weight_dequantized.numel() * sizeof(T);
  dev_ctx.template Alloc<T>(&weight_dequantized);

  if (weight_dtype == "int8") {
    int8_weight_only_dequant<<<grid, block, 0, stream>>>(
        weight.data<int8_t>(),
        weight_scale.data<float>(),
        weight_dequantized.data<T>(),
        n,
        k);
  } else if (weight_dtype == "int4") {
    grid.x /= 2;
    int4_weight_only_dequant<<<grid, block, 0, stream>>>(
        weight.data<int8_t>(),
        weight_scale.data<float>(),
        weight_dequantized.data<T>(),
        n,
        k);
  }
  MatmulKernel<T, Context>(
      dev_ctx, out_grad, weight_dequantized, false, false, x_grad);
#endif
}
}  // namespace phi

PD_REGISTER_KERNEL(weight_only_linear_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::WeightOnlyLinearGradKernel,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
