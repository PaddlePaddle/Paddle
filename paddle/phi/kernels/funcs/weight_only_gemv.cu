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

#include "paddle/phi/kernels/funcs/weight_only_gemv.h"

#include <assert.h>
#include <stdint.h>
#include <cmath>
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/datatype_traits.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

namespace {

#ifdef PADDLE_WITH_CUDA
constexpr int kWarpSize = 32;
constexpr int kPerBlockWarpNum = 8;

/////////////////////////////////////////////////////////////////////

template <typename T>
struct CUDA_HALF_2_TYPE_TARIS {};

template <>
struct CUDA_HALF_2_TYPE_TARIS<half> {
  using type = half2;
};

#ifdef PADDLE_CUDA_BF16
template <>
struct CUDA_HALF_2_TYPE_TARIS<__nv_bfloat16> {
  using type = __nv_bfloat162;
};
#endif

template <typename T>
__device__ inline void fast_cvt_4_packed_signed_i8s_to_2_half2s(
    T halves[4], int8_t signed_chars[4]) {
  assert(false);
}

// Specialization for fast cast from FP16 -> int8
template <>
__device__ inline void fast_cvt_4_packed_signed_i8s_to_2_half2s(
    half halves[4], int8_t signed_chars[4]) {
#if CUDA_ARCH_FP16_SUPPORTED(__CUDA_ARCH__)
  uint32_t* h = reinterpret_cast<uint32_t*>(halves);
  uint32_t i8s = *reinterpret_cast<uint32_t*>(signed_chars);

  static constexpr uint32_t mask_for_elt_01 = 0x5150;
  static constexpr uint32_t mask_for_elt_23 = 0x5352;
  static constexpr uint32_t start_byte_for_fp16 = 0x64646464;
  asm volatile("prmt.b32 %0,%1,%2,%3;\n"
               : "=r"(h[0])
               : "r"(i8s), "n"(start_byte_for_fp16), "n"(mask_for_elt_01));
  asm volatile("prmt.b32 %0,%1,%2,%3;\n"
               : "=r"(h[1])
               : "r"(i8s), "n"(start_byte_for_fp16), "n"(mask_for_elt_23));

  static constexpr uint32_t I8s_TO_F16s_MAGIC_NUM = 0x64806480;
  asm volatile("sub.f16x2 %0, %1, %2;\n"
               : "=r"(h[0])
               : "r"(h[0]), "r"(I8s_TO_F16s_MAGIC_NUM));
  asm volatile("sub.f16x2 %0, %1, %2;\n"
               : "=r"(h[1])
               : "r"(h[1]), "r"(I8s_TO_F16s_MAGIC_NUM));
#endif
}

// Specialization for fast cast from BF16 -> int8
#ifdef PADDLE_CUDA_BF16
template <>
__device__ inline void fast_cvt_4_packed_signed_i8s_to_2_half2s(
    __nv_bfloat16 halves[4], int8_t signed_chars[4]) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
  uint32_t* bf16_result_ptr = reinterpret_cast<uint32_t*>(halves);
  uint32_t i8s = *reinterpret_cast<uint32_t*>(signed_chars);

  static constexpr uint32_t fp32_base = 0x4B000000;
  float fp32_intermediates[4];

  // Construct FP32s, bfloat does not have enough mantissa for IADD trick
  uint32_t* fp32_intermediates_casted =
      reinterpret_cast<uint32_t*>(fp32_intermediates);
  fp32_intermediates_casted[0] = __byte_perm(i8s, fp32_base, 0x7650);
  fp32_intermediates_casted[1] = __byte_perm(i8s, fp32_base, 0x7651);
  fp32_intermediates_casted[2] = __byte_perm(i8s, fp32_base, 0x7652);
  fp32_intermediates_casted[3] = __byte_perm(i8s, fp32_base, 0x7653);

// Subtract out fp32_base + 128 to make the unsigned integer signed.
#pragma unroll
  for (int ii = 0; ii < 4; ++ii) {
    fp32_intermediates[ii] -= 8388736.f;
  }

// Truncate the fp32 representation and pack up as bfloat16s.
#pragma unroll
  for (int ii = 0; ii < 2; ++ii) {
    bf16_result_ptr[ii] = __byte_perm(fp32_intermediates_casted[2 * ii + 0],
                                      fp32_intermediates_casted[2 * ii + 1],
                                      0x7632);
  }
#else
  // Disable this on architectures older than Ampere since they lack hardware
  // for bf16 mma. If one wishes to use HMMA on older hardware, they should
  // Convert directly to FP16 using FP16 converters.
  assert(false);
#endif
}
#endif

/* Gelu Activation */

__forceinline__ __device__ float copysignf_pos(float a, float b) {
  float r;
  r = __int_as_float(__float_as_int(a) | (__float_as_int(b) & 0x80000000));
  return r;
}

__inline__ __device__ float tanh_opt(float x) {
#if (__CUDA_ARCH__ >= 750 && CUDART_VERSION >= 11000)
  float r;
  asm("tanh.approx.f32 %0,%1; \n\t" : "=f"(r) : "f"(x));
  return r;
#else
  const float exp_val = -1.f * fabs(2 * x);
  return copysignf_pos((1.0f - __expf(exp_val)) / (__expf(exp_val) + 1.0f), x);
#endif
}

template <typename T, bool EnableFastGelu>
struct GeluActivation {
  using return_type = T;
  static __device__ __forceinline__ T apply(const T& val) {
    if (!EnableFastGelu) return val;
    const float cdf =
        0.5f * (1.0f + tanh_opt((0.7978845608028654f *
                                 (val + 0.044715f * val * val * val))));
    return val * cdf;
  }
};

template <typename T>
struct ConvertFloatFunc {
  ConvertFloatFunc() {}
  static __device__ __forceinline__ float apply(const T& val) {
    assert(false);
    return 0.0f;
  }
};

template <>
struct ConvertFloatFunc<half> {
  static __device__ __forceinline__ float apply(const half& val) {
    return __half2float(val);
  }
};

#ifdef PADDLE_CUDA_BF16
template <>
struct ConvertFloatFunc<__nv_bfloat16> {
  static __device__ __forceinline__ float apply(const __nv_bfloat16& val) {
    return __bfloat162float(val);
  }
};
#endif

template <typename T>
struct ConvertDstFunc_2 {
  static __device__ __forceinline__ T apply(const float& val) { assert(false); }
};

template <typename T>
struct ConvertDstFunc {
  static __device__ __forceinline__ T apply(const float& val) { assert(false); }
};

template <>
struct ConvertDstFunc<half> {
  static __device__ __forceinline__ half apply(const float& val) {
    return __float2half_rn(val);
  }
};

template <>
struct ConvertDstFunc<half2> {
  static __device__ __forceinline__ half2 apply(const float& val) {
    return __float2half2_rn(val);
  }
};

template <>
struct ConvertDstFunc_2<half2> {
  static __device__ __forceinline__ half2 apply(const half& val) {
    return __half2half2(val);
  }
};
#ifdef PADDLE_CUDA_BF16
template <>
struct ConvertDstFunc<__nv_bfloat16> {
  static __device__ __forceinline__ __nv_bfloat16 apply(const float& val) {
    return __float2bfloat16_rn(val);
  }
};

template <>
struct ConvertDstFunc<__nv_bfloat162> {
  static __device__ __forceinline__ __nv_bfloat162 apply(const float& val) {
    return __float2bfloat162_rn(val);
  }
};

template <>
struct ConvertDstFunc_2<__nv_bfloat162> {
  static __device__ __forceinline__ __nv_bfloat162
  apply(const __nv_bfloat16& val) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    __nv_bfloat162 val2;
    val2.x = val;
    val2.y = val;
    return val2;
#else
    return __bfloat162bfloat162(val);
#endif
  }
};
#endif

template <typename T>
struct HalfMul {
  static __device__ __forceinline__ T apply(const T& x, const T& y) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
    return __hmul(x, y);
#else
    float res = static_cast<float>(float16(x)) * static_cast<float>(float16(y));
    return float16(res).to_half();
#endif
  }
};

template <typename T>
struct HalfMulAdd {
  static __device__ __forceinline__ T apply(const T& x,
                                            const T& y,
                                            const T& z) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
    return __hfma2(x, y, z);
#else
    assert(0 && "HalfMulAdd cuda version error");
#endif
  }
};

#ifdef PADDLE_CUDA_BF16
template <>
struct HalfMul<__nv_bfloat16> {
  static __device__ __forceinline__ __nv_bfloat16
  apply(const __nv_bfloat16& x, const __nv_bfloat16& y) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
    return __hmul(x, y);
#else
    return __float2bfloat16_rn(__bfloat162float(x) * __bfloat162float(y));
#endif
  }
};

template <>
struct HalfMulAdd<__nv_bfloat162> {
  static __device__ __forceinline__ __nv_bfloat162
  apply(const __nv_bfloat162& x,
        const __nv_bfloat162& y,
        const __nv_bfloat162& z) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    return __hfma2(x, y, z);
#else
    assert(0 && "HalfMulAdd cuda version error");
#endif
  }
};
#endif

/*
Int8 Weightonly GEMV.
X: 1 x k
Weight(ColMajor): n x k
Each Warp Process: 1 x k matmul 1 x k
*/
template <typename T, bool Bias, bool Gelu>
__global__ void int8_weight_only_gemv(const T* input,
                                      const int8_t* weight,
                                      const T* scale_list,
                                      const T* bias,
                                      T* output,
                                      const int k,
                                      const int n) {
  constexpr int kWarpSize = 32;
  constexpr int kVecSize = 16;
  T vec_input[kVecSize];
  int8_t vec_weight[kVecSize];
  T vec_weight_f16[kVecSize];

  const int warp_id = threadIdx.x / kWarpSize;
  const int lane_id = threadIdx.x % kWarpSize;
  const int tile_id = blockIdx.x * blockDim.x / kWarpSize + warp_id;
  const int row_id = tile_id * 2 + ((lane_id % 8) > 3 ? 1 : 0);
  weight += tile_id * k * 2;

  float v = 0.f, scale = static_cast<float>(scale_list[row_id]), v_bias;

  if (Bias) {
    v_bias = ConvertFloatFunc<T>::apply(bias[row_id]);
  }

#pragma unroll
  for (int i = lane_id * kVecSize; i < k * 2; i += kVecSize * kWarpSize) {
    *reinterpret_cast<int4*>(vec_weight) =
        *reinterpret_cast<const int4*>(weight + i);  // NOLINT
    *reinterpret_cast<float4*>(vec_input) =          // NOLINT
        *reinterpret_cast<const float4*>(input + i / 128 * 64 +
                                         (i % 64));  // NOLINT
    *reinterpret_cast<float4*>(vec_input + 8) =      // NOLINT
        *reinterpret_cast<const float4*>(input + i / 128 * 64 + (i % 64) +
                                         8);  // NOLINT

#pragma unroll
    for (int p = 0; p < kVecSize; p += 4) {
      fast_cvt_4_packed_signed_i8s_to_2_half2s<T>(vec_weight_f16 + p,
                                                  vec_weight + p);
    }
#pragma unroll
    for (int p = 0; p < kVecSize; ++p) {
      v += ConvertFloatFunc<T>::apply(
          HalfMul<T>::apply(vec_input[p], vec_weight_f16[p / 8 + (p % 8) * 2]));
    }
  }
  // Do WarpReduceSum.
  v += __shfl_xor_sync(0xffffffff, v, 16);
  v += __shfl_xor_sync(0xffffffff, v, 8);
  v += __shfl_xor_sync(0xffffffff, v, 2);
  v += __shfl_xor_sync(0xffffffff, v, 1);
  if (lane_id == 0 || lane_id == 4) {
    if (Bias) {
      output[row_id] = ConvertDstFunc<T>::apply(
          GeluActivation<float, Gelu>::apply(v * scale + v_bias));
    } else {
      output[row_id] = ConvertDstFunc<T>::apply(
          GeluActivation<float, Gelu>::apply(v * scale));
    }
  }
}

enum class WeightOnlyQuantType { Int4b, Int8b };

enum class WeightOnlyType { PerChannel, GroupWise };

template <WeightOnlyQuantType QType>
struct WeightLayoutDetails;

template <>
struct WeightLayoutDetails<WeightOnlyQuantType::Int4b> {
  // Every four rows of the original weights are interleaved into a row with
  // stride of 64, so if each thread processes 32 elements(for int4, we can use
  // ldg.128 to load weights), then every group of two adjacent threads will
  // alternately process four different row weights for example every 256
  // consecutive int4 elements [256*i, 256*(i+1)-1] of row N under interleave
  // layout, the first 64 are from [64*i, 64*(i+1)-1] of row 4N before
  // interleaving, and the second 64 are from [64*i, 64*(i+1)-1] of row 4N+1
  // before interleaving, and so on. So if each thread loads 32 int4 elements,
  // then the elements of each 2 adjacent threads of each 8 consecutive threads
  // will come from row 4N ~ 4N+3 respectively before interleaving.
  static constexpr int kElemBits = 4;
  static constexpr int kInterleave = 4;
  static constexpr int kStride = 64;

  // The index remapping here is to counteracts the effect of
  // cutlass::permute_B_rows_for_mixed_gemm input 0 1 2 3 4 5 6 7 8 9 10 11 12
  // 13 14 15 ... 31 weight 0 1 8 9 16 17 24 25 2 3 10 11 18 19 26 27 4 5 12 13
  // 20 21 28 29 6 7 14 15 22 23 30 31
  static constexpr int kShuffleSize = 32;
  static constexpr int kShuffleBasicTile = 2;
  static constexpr int kShuffleContinuous = 4;
  static constexpr int kShuffleStrided = 4;

  // The rearrangement here counteracts the effect of
  // cutlass::add_bias_and_interleave_int4s_inplace Input int8 data layout
  //      [elt_7  elt_5  elt_3  elt_1  elt_6  elt_4  elt_2  elt_0] (each elt
  //      occupies 4 bits)
  //
  // Converted fp16 data layout
  //      [elt_7  elt_6  elt_5  elt_4  elt_3  elt_2  elt_1  elt_0] (each elt
  //      occupies 16 bits)
  static constexpr int kConvertCount = 8;
  // using Converter
  //     =
  //     cutlass::FastInterleavedAndBiasedNumericArrayConverter<cutlass::half_t,
  //     cutlass::uint4b_t, kConvertCount>;

  // Each warp completes the internal reduce and writes the [Batch * NPerBlock *
  // Interleave] results to the corresponding address in shared memory
  template <int Num, int WarpSize>
  __device__ __forceinline__ static void sync(float* res,
                                              float (*sm)[Num * kInterleave]) {
#pragma unroll
    for (int i = 0; i < Num; ++i) {
      res[i] += __shfl_xor_sync(~0, res[i], 16);
      res[i] += __shfl_xor_sync(~0, res[i], 8);
      res[i] += __shfl_xor_sync(~0, res[i], 1);
    }
    __syncthreads();
    int warp = threadIdx.x / WarpSize, lane = threadIdx.x % WarpSize;
    if (lane == 0 || lane == 2 || lane == 4 || lane == 6) {
#pragma unroll
      for (int i = 0; i < Num; ++i) {
        sm[warp][i * kInterleave + lane / 2] = res[i];
      }
    }
    __syncthreads();
  }
};

template <>
struct WeightLayoutDetails<WeightOnlyQuantType::Int8b> {
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
  static constexpr int kElemBits = 8;
  static constexpr int kInterleave = 2;
  static constexpr int kStride = 64;

  // The index remapping here is to counteracts the effect of
  // cutlass::permute_B_rows_for_mixed_gemm input 0 1 2 3 4 5 6 7 8 9 10 11 12
  // 13 14 15 weight 0 1 8 9 2 3 10 11 4 5 12 13 6 7 14 15
  static constexpr int kShuffleSize = 16;
  static constexpr int kShuffleBasicTile = 2;
  static constexpr int kShuffleContinuous = 2;
  static constexpr int kShuffleStrided = 4;

  // The rearrangement here counteracts the effect of
  // cutlass::add_bias_and_interleave_int8s_inplace Input int8 data layout
  //      [elt_3  elt_1  elt_2  elt_0] (each elt occupies 8 bits)
  //
  // Converted fp16 data layout
  //      [elt_3  elt_2  elt_1  elt_0] (each elt occupies 16 bits)
  static constexpr int kConvertCount = 4;
  // using Converter =
  // cutlass::FastInterleavedAndBiasedNumericArrayConverter<cutlass::half_t,
  // uint8_t, kConvertCount>;

  // Each warp completes the internal reduce and writes the [Batch * NPerBlock *
  // Interleave] results to the corresponding address in shared memory
  template <int Num, int WarpSize>
  __device__ __forceinline__ static void sync(float* res,
                                              float (*sm)[Num * kInterleave]) {
#pragma unroll
    for (int i = 0; i < Num; ++i) {
      res[i] += __shfl_xor_sync(~0, res[i], 16);
      res[i] += __shfl_xor_sync(~0, res[i], 8);
      res[i] += __shfl_xor_sync(~0, res[i], 2);
      res[i] += __shfl_xor_sync(~0, res[i], 1);
    }
    __syncthreads();
    int warp = threadIdx.x / WarpSize, lane = threadIdx.x % WarpSize;
    if (lane == 0 || lane == 4) {
#pragma unroll
      for (int i = 0; i < Num; ++i) {
        sm[warp][i * kInterleave + lane / 4] = res[i];
      }
    }
    __syncthreads();
  }
};

template <WeightOnlyQuantType QType>
struct WeightOnlyKernelDetails {
  using Layout = WeightLayoutDetails<QType>;

  static constexpr int kElemBits = Layout::kElemBits;
  static constexpr int kInterleave = Layout::kInterleave;
  static constexpr int kStride = Layout::kStride;

  static constexpr int kShuffleSize = Layout::kShuffleSize;
  static constexpr int kShuffleBasicTile = Layout::kShuffleBasicTile;
  static constexpr int kShuffleContinuous = Layout::kShuffleContinuous;
  static constexpr int kShuffleStrided = Layout::kShuffleStrided;

  // using Converter = typename Layout::Converter;
  static constexpr int kConvertCount = Layout::kConvertCount;

  // Use ldg128 load data from global memory
  static constexpr int kAccessSize = 128;
  using AccessType = uint4;

  static constexpr int kElemsPerByte = 8 / kElemBits;
  static constexpr int kElemsPerThread = kAccessSize / kElemBits;
  static constexpr int kBytePerThread = kElemsPerThread / kElemsPerByte;
  static constexpr int kThreadsNumPerTile = kStride / kElemsPerThread;
  static constexpr int kThreadsNumPerInterleave =
      kThreadsNumPerTile * kInterleave;

  static constexpr int kConvertIters = kElemsPerThread / kConvertCount;

  // Each thread loads 16(int8b)/32(int4b) quantized weight elements each time
  // through ldg128 So more times of ldg128 are needed to load the same number
  // of fp16 activation elements.
  static constexpr int kActivationElemNumPerAccess =
      kAccessSize / (sizeof(half) * 8);
  static constexpr int kActivationAccessNum =
      kElemsPerThread / kActivationElemNumPerAccess;
};

struct WeightOnlyPerChannel;
template <int GS>
struct WeightOnlyGroupWise;

template <typename WeightOnlyFlag>
struct WeightOnlyProperties;

template <>
struct WeightOnlyProperties<WeightOnlyPerChannel> {
  static constexpr bool kIsFineGrained = false;
  static constexpr int kGroupSize = 0;
};

template <int GS>
struct WeightOnlyProperties<WeightOnlyGroupWise<GS>> {
  static constexpr bool kIsFineGrained = true;
  static constexpr int kGroupSize = GS;
};

template <typename T,
          WeightOnlyQuantType QType,
          typename WeightOnlyFlag,
          bool Zero,
          int BlockSize>
struct WeightOnlyScaleLoader {
  using Details = WeightOnlyKernelDetails<QType>;
  static constexpr bool kIsFineGrained =
      WeightOnlyProperties<WeightOnlyFlag>::kIsFineGrained;
  static constexpr int kGroupSize =
      WeightOnlyProperties<WeightOnlyFlag>::kGroupSize;

 private:
  const T* _scales;
  const T* _zeros;
  int _stride;
  int _offset;

 public:
  __device__ __forceinline__ WeightOnlyScaleLoader(const T* scales,
                                                   const T* zeros,
                                                   int initial_offset,
                                                   int stride)
      : _scales(scales), _zeros(zeros), _stride(stride) {
    _scales += initial_offset;
    if (Zero) {
      _zeros += initial_offset;
    }
    // Calculate the k dimension index of the element processed by the current
    // thread of layout before interleave Used to load scales and zeros in
    // groupwise weight only quant
    _offset =
        threadIdx.x / Details::kThreadsNumPerInterleave * Details::kStride +
        (threadIdx.x % Details::kThreadsNumPerTile) * Details::kElemsPerThread;
  }

  __device__ __forceinline__ void load(T* scale, T* zero, int nid) {
    int offset = nid * Details::kInterleave;

// TODO(freeliuzc): cpplint has bug here
#ifndef WIN32
    if constexpr (kIsFineGrained) {
#else
    if (kIsFineGrained) {
#endif
      offset += _offset / kGroupSize * _stride;
    }
    *scale = _scales[offset];

// TODO(freeliuzc): cpplint has bug here
#ifndef WIN32
    if constexpr (Zero) {
#else
    if (Zero) {
#endif
      *zero = _zeros[offset];
    } else {
      *zero = static_cast<T>(0.f);
    }
  }

  __device__ __forceinline__ void advance() {
    _offset += BlockSize * Details::kElemsPerThread / Details::kInterleave;
  }

  __device__ __forceinline__ int offset() { return _offset; }
};  // NOLINT

template <typename T, WeightOnlyQuantType QType>
struct WeightOnlyConverter {};

template <>
struct WeightOnlyConverter<half, WeightOnlyQuantType::Int8b> {
  static __device__ inline void convert(half halves[4],
                                        int8_t signed_chars[4]) {
    uint32_t* h = reinterpret_cast<uint32_t*>(halves);
    uint32_t i8s = *reinterpret_cast<uint32_t*>(signed_chars);

    static constexpr uint32_t mask_for_elt_01 = 0x5150;
    static constexpr uint32_t mask_for_elt_23 = 0x5352;
    static constexpr uint32_t start_byte_for_fp16 = 0x64646464;
    asm volatile("prmt.b32 %0,%1,%2,%3;\n"
                 : "=r"(h[0])
                 : "r"(i8s), "n"(start_byte_for_fp16), "n"(mask_for_elt_01));
    asm volatile("prmt.b32 %0,%1,%2,%3;\n"
                 : "=r"(h[1])
                 : "r"(i8s), "n"(start_byte_for_fp16), "n"(mask_for_elt_23));

    static constexpr uint32_t I8s_TO_F16s_MAGIC_NUM = 0x64806480;
    asm volatile("sub.f16x2 %0, %1, %2;\n"
                 : "=r"(h[0])
                 : "r"(h[0]), "r"(I8s_TO_F16s_MAGIC_NUM));
    asm volatile("sub.f16x2 %0, %1, %2;\n"
                 : "=r"(h[1])
                 : "r"(h[1]), "r"(I8s_TO_F16s_MAGIC_NUM));
  }
};

#ifdef PADDLE_CUDA_BF16
template <>
struct WeightOnlyConverter<__nv_bfloat16, WeightOnlyQuantType::Int8b> {
  static __device__ inline void convert(__nv_bfloat16 halves[4],
                                        int8_t signed_chars[4]) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
    uint32_t* bf16_result_ptr = reinterpret_cast<uint32_t*>(halves);
    uint32_t i8s = *reinterpret_cast<uint32_t*>(signed_chars);

    static constexpr uint32_t fp32_base = 0x4B000000;
    float fp32_intermediates[4];

    // Construct FP32s, bfloat does not have enough mantissa for IADD trick
    uint32_t* fp32_intermediates_casted =
        reinterpret_cast<uint32_t*>(fp32_intermediates);
    fp32_intermediates_casted[0] = __byte_perm(i8s, fp32_base, 0x7650);
    fp32_intermediates_casted[1] = __byte_perm(i8s, fp32_base, 0x7651);
    fp32_intermediates_casted[2] = __byte_perm(i8s, fp32_base, 0x7652);
    fp32_intermediates_casted[3] = __byte_perm(i8s, fp32_base, 0x7653);

    // Subtract out fp32_base + 128 to make the unsigned integer signed.
#pragma unroll
    for (int ii = 0; ii < 4; ++ii) {
      fp32_intermediates[ii] -= 8388736.f;
    }

// Truncate the fp32 representation and pack up as bfloat16s.
#pragma unroll
    for (int ii = 0; ii < 2; ++ii) {
      bf16_result_ptr[ii] = __byte_perm(fp32_intermediates_casted[2 * ii + 0],
                                        fp32_intermediates_casted[2 * ii + 1],
                                        0x7632);
    }
#else
    // Disable this on architectures older than Ampere since they lack hardware
    // for bf16 mma. If one wishes to use HMMA on older hardware, they should
    // Convert directly to FP16 using FP16 converters.
    assert(false);
#endif
  }
};
#endif

template <>
struct WeightOnlyConverter<half, WeightOnlyQuantType::Int4b> {
  static __device__ inline void convert(half halves[8],
                                        int8_t signed_chars[4]) {
    uint32_t* h = reinterpret_cast<uint32_t*>(halves);
    uint32_t i4s = *reinterpret_cast<uint32_t*>(signed_chars);

    // First, we extract the i4s and construct an intermediate fp16 number.
    static constexpr uint32_t immLut = (0xf0 & 0xcc) | 0xaa;
    static constexpr uint32_t BOTTOM_MASK = 0x000f000f;
    static constexpr uint32_t TOP_MASK = 0x00f000f0;
    static constexpr uint32_t I4s_TO_F16s_MAGIC_NUM = 0x64006400;

    // Note that the entire sequence only requires 1 shift instruction. This is
    // thanks to the register packing format and the fact that we force our
    // integers to be unsigned, and account for this in the fp16 subtractions.
    // In addition, I exploit the fact that sub and fma have the same throughput
    // in order to convert elt_23 and elt_67 to fp16 without having to shift
    // them to the bottom bits before hand.

    // Shift right by 8 to now consider elt_45 and elt_67. Issue first to hide
    // RAW dependency if we issue immediately before required.
    const uint32_t top_i4s = i4s >> 8;
    // Extract elt_01 - (i4s & 0x000f000f) | 0x64006400
    asm volatile(
        "lop3.b32 %0, %1, %2, %3, %4;\n"
        : "=r"(h[0])
        : "r"(i4s), "n"(BOTTOM_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));
    // Extract elt_23 (i4s & 0x00f000f0) | 0x64006400
    asm volatile(
        "lop3.b32 %0, %1, %2, %3, %4;\n"
        : "=r"(h[1])
        : "r"(i4s), "n"(TOP_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));
    // Extract elt_45 (top_i4s & 0x000f000f) | 0x64006400
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                 : "=r"(h[2])
                 : "r"(top_i4s),
                   "n"(BOTTOM_MASK),
                   "n"(I4s_TO_F16s_MAGIC_NUM),
                   "n"(immLut));
    // Extract elt_67 (top_i4s & 0x00f000f0) | 0x64006400
    asm volatile(
        "lop3.b32 %0, %1, %2, %3, %4;\n"
        : "=r"(h[3])
        : "r"(top_i4s), "n"(TOP_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));

    // I use inline PTX below because I am not sure if the compiler will emit
    // float2half instructions if I use the half2 ctor. In this case, I chose
    // performance reliability over code readability.

    // This is the half2 {1032, 1032} represented as an integer.
    static constexpr uint32_t FP16_TOP_MAGIC_NUM = 0x64086408;
    // This is the half2 {1 / 16, 1 / 16} represented as an integer.
    static constexpr uint32_t ONE_SIXTEENTH = 0x2c002c00;
    // This is the half2 {-72, -72} represented as an integer.
    static constexpr uint32_t NEG_72 = 0xd480d480;
    // Finally, we construct the output numbers.
    // Convert elt_01
    asm volatile("sub.f16x2 %0, %1, %2;\n"
                 : "=r"(h[0])
                 : "r"(h[0]), "r"(FP16_TOP_MAGIC_NUM));
    // Convert elt_23
    asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n"
                 : "=r"(h[1])
                 : "r"(h[1]), "r"(ONE_SIXTEENTH), "r"(NEG_72));
    // Convert elt_45
    asm volatile("sub.f16x2 %0, %1, %2;\n"
                 : "=r"(h[2])
                 : "r"(h[2]), "r"(FP16_TOP_MAGIC_NUM));
    // Convert elt_67
    asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n"
                 : "=r"(h[3])
                 : "r"(h[3]), "r"(ONE_SIXTEENTH), "r"(NEG_72));
  }
};

#ifdef PADDLE_CUDA_BF16
template <>
struct WeightOnlyConverter<__nv_bfloat16, WeightOnlyQuantType::Int4b> {
  static __device__ inline void convert(__nv_bfloat16 halves[8],
                                        int8_t signed_chars[4]) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
    uint32_t* h = reinterpret_cast<uint32_t*>(halves);
    uint32_t const source_i4s = *reinterpret_cast<uint32_t*>(signed_chars);

    static constexpr uint32_t immLut = (0xf0 & 0xcc) | 0xaa;
    static constexpr uint32_t MASK = 0x000f000f;
    static constexpr uint32_t I4s_TO_BF16s_MAGIC_NUM = 0x43004300;

    // We don't have enough mantissa to remove as much shift overhead as FP16,
    // so we must loop. No shift needed for first item.
    uint32_t i4s = source_i4s;
    asm volatile(
        "lop3.b32 %0, %1, %2, %3, %4;\n"
        : "=r"(h[0])
        : "r"(i4s), "n"(MASK), "n"(I4s_TO_BF16s_MAGIC_NUM), "n"(immLut));
#pragma unroll
    for (int ii = 1; ii < 4; ++ii) {
      i4s >>= 4;
      // (i4s & 0x000f000f) | 0x43004300
      asm volatile(
          "lop3.b32 %0, %1, %2, %3, %4;\n"
          : "=r"(h[ii])
          : "r"(i4s), "n"(MASK), "n"(I4s_TO_BF16s_MAGIC_NUM), "n"(immLut));
    }

    // This is the BF16 {-136, -136} represented as an integer.
    static constexpr uint32_t BF16_BIAS = 0xC308C308;
    static constexpr uint32_t BF16_ONE = 0x3F803F80;

// Finally, we construct the output numbers.
#pragma unroll
    for (int ii = 0; ii < 4; ++ii) {
      // Since this section is for Ampere+, we use bf16 fma to do the bias
      // subtraction
      asm("fma.rn.bf16x2 %0, %1, %2, %3;\n"
          : "=r"(h[ii])
          : "r"(h[ii]), "r"(BF16_ONE), "r"(BF16_BIAS));
    }

#else
    // Disable this on architectures older than Ampere since they lack hardware
    // for bf16 mma. If one wishes to use HMMA on older hardware, they should
    // Convert directly to FP16 using FP16 converters.
    assert(false);
#endif
  }
};
#endif

template <typename VecType, typename T0, typename T1>
__device__ __forceinline__ void load(T0* dst, T1* src, size_t offset = 0) {
  *reinterpret_cast<VecType*>(dst) =
      *(reinterpret_cast<const VecType*>(src) + offset);
}

template <typename T, WeightOnlyQuantType QType, typename Details>
struct WeightPostProcessor {
  static __device__ __forceinline__ void run(T* weights_vec,
                                             T* weights_f16,
                                             T* scale,
                                             T* zero,
                                             int NPerBlock,
                                             int idx) {}
};

template <typename T, typename Details>
struct WeightPostProcessor<T, WeightOnlyQuantType::Int4b, Details> {
  static __device__ __forceinline__ void run(T* weights_vec,
                                             T* weights_f16,
                                             T* scale,
                                             T* zero,
                                             int NPerBlock,
                                             int idx) {
    using HALF_2_TYPE = typename CUDA_HALF_2_TYPE_TARIS<T>::type;
#pragma unroll
    for (int i = 0; i < Details::kShuffleContinuous; ++i) {
#pragma unroll
      for (int j = 0; j < Details::kShuffleStrided; ++j) {
        // Dequantize the weights and arrange the shuffled elements back to
        // the correct order in the register array
        HALF_2_TYPE v = *reinterpret_cast<HALF_2_TYPE*>(
            weights_vec + i * Details::kShuffleBasicTile +
            j * Details::kShuffleContinuous * Details::kShuffleBasicTile);
        v = HalfMulAdd<HALF_2_TYPE>::apply(
            v,
            ConvertDstFunc_2<HALF_2_TYPE>::apply(scale[idx]),
            ConvertDstFunc_2<HALF_2_TYPE>::apply(zero[idx]));
        weights_f16[(i * Details::kShuffleStrided * Details::kShuffleBasicTile +
                     j * Details::kShuffleBasicTile + 0) *
                        NPerBlock +
                    idx] = v.x;
        weights_f16[(i * Details::kShuffleStrided * Details::kShuffleBasicTile +
                     j * Details::kShuffleBasicTile + 1) *
                        NPerBlock +
                    idx] = v.y;
      }
    }
  }
};

template <typename T, typename Details>
struct WeightPostProcessor<T, WeightOnlyQuantType::Int8b, Details> {
  static __device__ __forceinline__ void run(T* weights_vec,
                                             T* weights_f16,
                                             T* scale,
                                             T* zero,
                                             int NPerBlock,
                                             int idx) {
#pragma unroll
    for (int p = 0; p < 16; ++p) {
      weights_f16[p * NPerBlock + idx] =
          weights_vec[p / 8 + (p % 8) * 2] * scale[idx];
    }
  }
};

template <typename T,
          WeightOnlyQuantType QType,
          typename WeightOnlyFlag,
          bool Gelu,
          bool Zero,
          bool Bias,
          int NPerBlock,
          int Batch,
          int BlockSize>
__global__ void weight_only_batched_gemv_multi_warp(const T* in,
                                                    const int8_t* qweight,
                                                    const T* bias,
                                                    const T* scales,
                                                    const T* zeros,
                                                    T* out,
                                                    const int n,
                                                    const int k) {
  static_assert(NPerBlock == 1 || (NPerBlock % 2 == 0),
                "NPerBlock must be 1 or even in gemv multi warp kernel. ");
  using Details = WeightOnlyKernelDetails<QType>;

  // using Converter = typename Details::Converter;
  using AccType = typename Details::AccessType;
  using CvtSrcType = int8_t;
  using CvtResType = T;
  using ScaleLoader =
      WeightOnlyScaleLoader<T, QType, WeightOnlyFlag, Zero, BlockSize>;
  using WeightProcessor = WeightPostProcessor<T, QType, Details>;

  extern __shared__ uint8_t shmem[];
  constexpr int Interleave = Details::kInterleave;
  constexpr int WarpSize = 32;
  constexpr int Num = Batch * NPerBlock;
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int n_start_id = bid * NPerBlock * Interleave;
  using HALF_2_TYPE = typename CUDA_HALF_2_TYPE_TARIS<T>::type;
  // Calculate the n-dimensional index of the data processed by the current
  // thread in the interleave tile
  const int interleave_n_id = (tid / Details::kThreadsNumPerTile) % Interleave;

  qweight += n_start_id * k / Details::kElemsPerByte;
  ScaleLoader scale_loader(scales, zeros, n_start_id + interleave_n_id, n);

  float(*sm)[Num * Interleave] =
      reinterpret_cast<float(*)[Num * Interleave]>(shmem);

  // In order to take advantage of hfma2, we use fp16 for accumulation within
  // threads and fp32 for accumulation between threads.
  T accumulator[Num];
  for (int i = 0; i < Num; ++i) {
    accumulator[i] = ConvertDstFunc<T>::apply(0.f);
  }

  // Iteration in k dimensions
  for (int local_k = tid * Details::kElemsPerThread; local_k < k * Interleave;
       local_k += BlockSize * Details::kElemsPerThread) {
    T weights_f16[Details::kElemsPerThread * NPerBlock];
    T scale[NPerBlock], zero[NPerBlock];
#pragma unroll
    for (int idx = 0; idx < NPerBlock; ++idx) {
      // Load quantized weight and scales/zeros
      int8_t weights_quantized[Details::kBytePerThread];
      load<AccType>(weights_quantized,
                    qweight + idx * Interleave * k / Details::kElemsPerByte +
                        local_k / Details::kElemsPerByte);
      scale_loader.load(scale + idx, zero + idx, idx);
      T weights_vec[Details::kElemsPerThread];

#pragma unroll
      for (int i = 0; i < Details::kConvertIters; ++i) {
        // Use cutlass::FastInterleavedAndBiasedNumericArrayConverter for I2F
        // type conversion
        WeightOnlyConverter<T, QType>::convert(
            weights_vec + i * Details::kConvertCount,
            weights_quantized +
                i * Details::kConvertCount / Details::kElemsPerByte);
      }
      // Assign weight and apply scales.
      // Currently not support zero.
      WeightProcessor::run(
          weights_vec, weights_f16, scale, zero, NPerBlock, idx);
    }
#pragma unroll
    for (int b = 0; b < Batch; ++b) {
      T in_v[Details::kElemsPerThread];
#pragma unroll
      for (int idx = 0; idx < Details::kActivationAccessNum; ++idx) {
        load<AccType>(in_v + idx * Details::kActivationElemNumPerAccess,
                      in + b * k + scale_loader.offset() +
                          idx * Details::kActivationElemNumPerAccess);
      }
      // Perform vector inner product and accumulate
#ifndef WIN32
      if constexpr (NPerBlock == 1) {
#else
      if (NPerBlock == 1) {
#endif
        HALF_2_TYPE v = ConvertDstFunc<HALF_2_TYPE>::apply(0.f);
#pragma unroll
        for (int y = 0; y < Details::kElemsPerThread; y += 2) {
          v = HalfMulAdd<HALF_2_TYPE>::apply(
              *reinterpret_cast<HALF_2_TYPE*>(weights_f16 + y),
              *reinterpret_cast<HALF_2_TYPE*>(in_v + y),
              v);
        }
        accumulator[b] = accumulator[b] + ConvertDstFunc<T>::apply(v.x + v.y);
      } else {
#pragma unroll
        for (int x = 0; x < NPerBlock / 2; ++x) {
#pragma unroll
          for (int y = 0; y < Details::kElemsPerThread; ++y) {
            *reinterpret_cast<HALF_2_TYPE*>(accumulator + b * NPerBlock +
                                            x * 2) =
                HalfMulAdd<HALF_2_TYPE>::apply(
                    *reinterpret_cast<HALF_2_TYPE*>(weights_f16 +
                                                    y * NPerBlock + x * 2),
                    ConvertDstFunc_2<HALF_2_TYPE>::apply(in_v[y]),
                    *reinterpret_cast<HALF_2_TYPE*>(accumulator +
                                                    b * NPerBlock + x * 2));
          }
        }
      }
    }
    scale_loader.advance();
  }
  float reses[Num];
#pragma unroll
  for (int i = 0; i < Num; ++i) {
    reses[i] = ConvertFloatFunc<T>::apply(accumulator[i]);
  }

  // Each warp completes the internal reduce and writes the [Batch * NPerBlock *
  // Interleave] results to the corresponding address in shared memory
  Details::Layout::sync<Num, WarpSize>(reses, sm);

  // Each thread is responsible for the accumulation and store to global memory
  // of one element
  for (int i = tid; i < Num * Interleave; i += BlockSize) {
    int nid = i % (NPerBlock * Interleave);
    float v = 0.f;
    for (int j = 0; j < BlockSize / WarpSize; ++j) {
      v += sm[j][i];
    }
    float bias_v = 0.f;
#ifndef WIN32
    if constexpr (Bias) {
#else
    if (Bias) {
#endif
      bias_v = ConvertFloatFunc<T>::apply(bias[n_start_id + nid]);
    }
    int b = i / NPerBlock / Interleave;
    out[b * n + n_start_id + nid] = ConvertDstFunc<T>::apply(
        GeluActivation<float, Gelu>::apply(v + bias_v));
  }
}
#endif

template <typename T,
          WeightOnlyQuantType QType,
          typename WeightOnlyFlag,
          int NPerBlock,
          int Batch,
          int BlockSize>
void select_activation_and_bias(const T* input,
                                const int8_t* weight,
                                const T* bias,
                                const T* scales,
                                const int m,
                                const int n,
                                const int k,
                                const std::string& act_method,
                                T* output,
                                cudaStream_t stream) {
#ifdef PADDLE_WITH_CUDA
  static constexpr int kInterleave = WeightLayoutDetails<QType>::kInterleave;
  dim3 grid(n / NPerBlock / kInterleave);
  dim3 block(BlockSize);
  int size = sizeof(float) * BlockSize / 32 * Batch * NPerBlock * kInterleave;
  if (bias) {
    if (act_method == "gelu") {
      weight_only_batched_gemv_multi_warp<T,
                                          QType,
                                          WeightOnlyFlag,
                                          true,
                                          false,
                                          true,
                                          NPerBlock,
                                          Batch,
                                          BlockSize>
          <<<grid, block, size, stream>>>(
              input, weight, bias, scales, /*zeros*/ nullptr, output, n, k);
    } else if (act_method == "None") {
      weight_only_batched_gemv_multi_warp<T,
                                          QType,
                                          WeightOnlyFlag,
                                          false,
                                          false,
                                          true,
                                          NPerBlock,
                                          Batch,
                                          BlockSize>
          <<<grid, block, size, stream>>>(
              input, weight, bias, scales, /*zeros*/ nullptr, output, n, k);
    } else {
      PADDLE_THROW(
          errors::InvalidArgument("Currently, weightonly GEMV act_method "
                                  "only support `gelu`, `None`. "));
    }
  } else {
    if (act_method == "gelu") {
      weight_only_batched_gemv_multi_warp<T,
                                          QType,
                                          WeightOnlyFlag,
                                          true,
                                          false,
                                          false,
                                          NPerBlock,
                                          Batch,
                                          BlockSize>
          <<<grid, block, size, stream>>>(
              input, weight, bias, scales, /*zeros*/ nullptr, output, n, k);
    } else if (act_method == "None") {
      weight_only_batched_gemv_multi_warp<T,
                                          QType,
                                          WeightOnlyFlag,
                                          false,
                                          false,
                                          false,
                                          NPerBlock,
                                          Batch,
                                          BlockSize>
          <<<grid, block, size, stream>>>(
              input, weight, bias, scales, /*zeros*/ nullptr, output, n, k);
    } else {
      PADDLE_THROW(
          errors::InvalidArgument("Currently, weightonly GEMV act_method "
                                  "only support `gelu`, `None`. "));
    }
  }
#endif
}

template <typename T, typename WeightOnlyFlag>
void weight_only_batched_gemv_launcher(
    const T* input,
    const int8_t* weight,
    const T* bias,
    const T* scales,
    int m,
    int n,
    int k,
    const std::string& weight_only_quant_type,
    const std::string& act_method,
    T* output,
    cudaStream_t stream) {
#ifdef PADDLE_WITH_CUDA
  if (weight_only_quant_type == "int4") {
    switch (m) {
      case 1: {
        select_activation_and_bias<T,
                                   WeightOnlyQuantType::Int4b,
                                   WeightOnlyFlag,
                                   1,
                                   1,
                                   192>(
            input, weight, bias, scales, m, n, k, act_method, output, stream);
        break;
      }
      case 2: {
        select_activation_and_bias<T,
                                   WeightOnlyQuantType::Int4b,
                                   WeightOnlyFlag,
                                   2,
                                   2,
                                   128>(
            input, weight, bias, scales, m, n, k, act_method, output, stream);
        break;
      }
      case 3: {
        select_activation_and_bias<T,
                                   WeightOnlyQuantType::Int4b,
                                   WeightOnlyFlag,
                                   2,
                                   3,
                                   256>(
            input, weight, bias, scales, m, n, k, act_method, output, stream);
        break;
      }
      case 4: {
        select_activation_and_bias<T,
                                   WeightOnlyQuantType::Int4b,
                                   WeightOnlyFlag,
                                   4,
                                   4,
                                   256>(
            input, weight, bias, scales, m, n, k, act_method, output, stream);
        break;
      }
      default: {
        throw std::runtime_error(
            "Weight only cuda kernel only supported bs <= 4");
        break;
      }
    }
  } else if (weight_only_quant_type == "int8") {
    switch (m) {
      case 1: {
        select_activation_and_bias<T,
                                   WeightOnlyQuantType::Int8b,
                                   WeightOnlyFlag,
                                   2,
                                   1,
                                   256>(
            input, weight, bias, scales, m, n, k, act_method, output, stream);
        break;
      }
      case 2: {
        select_activation_and_bias<T,
                                   WeightOnlyQuantType::Int8b,
                                   WeightOnlyFlag,
                                   2,
                                   2,
                                   256>(
            input, weight, bias, scales, m, n, k, act_method, output, stream);
        break;
      }
      case 3: {
        select_activation_and_bias<T,
                                   WeightOnlyQuantType::Int8b,
                                   WeightOnlyFlag,
                                   2,
                                   3,
                                   256>(
            input, weight, bias, scales, m, n, k, act_method, output, stream);
        break;
      }
      case 4: {
        select_activation_and_bias<T,
                                   WeightOnlyQuantType::Int8b,
                                   WeightOnlyFlag,
                                   2,
                                   4,
                                   256>(
            input, weight, bias, scales, m, n, k, act_method, output, stream);
        break;
      }
      default: {
        throw std::runtime_error(
            "Weight only cuda kernel only supported bs <= 4");
        break;
      }
    }
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "WeightOnlyGemvKernel quant_type only support 'int4' or 'int8'."));
  }
#endif
}

}  // namespace

template <typename T, typename Context>
void WeightOnlyGemvWrapper(const Context& dev_ctx,
                           const T* input,
                           const int8_t* weight,
                           const T* bias,
                           const T* scales,
                           int m,
                           int n,
                           int k,
                           int group_size,
                           const std::string& weight_only_quant_type,
                           const std::string& weight_only_type,
                           const std::string& act_method,
                           T* output) {
  using DataType = typename PDDataTypeTraits<T>::DataType;
  if (weight_only_type == "per_channel") {
    PADDLE_ENFORCE_EQ(group_size,
                      -1,
                      common::errors::InvalidArgument(
                          "group size must be -1 in per-channel mode."));

    weight_only_batched_gemv_launcher<DataType, WeightOnlyPerChannel>(
        reinterpret_cast<const DataType*>(input),
        reinterpret_cast<const int8_t*>(weight),
        reinterpret_cast<const DataType*>(bias),
        reinterpret_cast<const DataType*>(scales),
        m,
        n,
        k,
        weight_only_quant_type,
        act_method,
        reinterpret_cast<DataType*>(output),
        dev_ctx.stream());
  } else if (weight_only_type == "group_wise") {
    if (group_size == 64) {
      weight_only_batched_gemv_launcher<DataType, WeightOnlyGroupWise<64>>(
          reinterpret_cast<const DataType*>(input),
          reinterpret_cast<const int8_t*>(weight),
          reinterpret_cast<const DataType*>(bias),
          reinterpret_cast<const DataType*>(scales),
          m,
          n,
          k,
          weight_only_quant_type,
          act_method,
          reinterpret_cast<DataType*>(output),
          dev_ctx.stream());
    } else if (group_size == 128) {
      weight_only_batched_gemv_launcher<DataType, WeightOnlyGroupWise<128>>(
          reinterpret_cast<const DataType*>(input),
          reinterpret_cast<const int8_t*>(weight),
          reinterpret_cast<const DataType*>(bias),
          reinterpret_cast<const DataType*>(scales),
          m,
          n,
          k,
          weight_only_quant_type,
          act_method,
          reinterpret_cast<DataType*>(output),
          dev_ctx.stream());
    } else {
      PADDLE_THROW(common::errors::InvalidArgument(
          "WeightOnlyGemvKernel group_size only support 64 or 128."));
    }
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "WeightOnlyGemvKernel type only support "
        "'per_channel' or 'group_wise'."));
  }
}

template <>
void WeightOnlyGemvWrapper(const phi::GPUContext& dev_ctx,
                           const float* input,
                           const int8_t* weight,
                           const float* bias,
                           const float* scales,
                           int m,
                           int n,
                           int k,
                           int group_size,
                           const std::string& weight_only_quant_type,
                           const std::string& weight_only_type,
                           const std::string& act_method,
                           float* output) {
  PADDLE_THROW(common::errors::Unimplemented(
      "WeightOnlyGemvKernel type only support 'float16' and 'bfloa16."
      "Not support float32."));
}

template <typename T, typename Context>
void WeightOnlyGemvKernel(const Context& dev_ctx,
                          const DenseTensor& x,
                          const DenseTensor& weight,
                          const paddle::optional<DenseTensor>& bias,
                          const DenseTensor& weight_scale,
                          int group_size,
                          const std::string& weight_only_quant_type,
                          const std::string& weight_only_type,
                          const std::string& act_method,
                          DenseTensor* out) {
  const T* x_data = x.data<T>();
  const int8_t* weight_data = weight.data<int8_t>();
  // Actually, we pass the weight datatype is uint8_t type.
  const T* bias_data = bias ? bias.get().data<T>() : nullptr;
  const T* weight_scale_data = weight_scale.data<T>();
  T* out_data = dev_ctx.template Alloc<T>(out);
  int m = x.dims()[0];
  int k = x.dims()[1];
  int n = weight.dims()[0];

  WeightOnlyGemvWrapper<T>(dev_ctx,
                           x_data,
                           weight_data,
                           bias_data,
                           weight_scale_data,
                           m,
                           n,
                           k,
                           group_size,
                           weight_only_quant_type,
                           weight_only_type,
                           act_method,
                           out_data);
}

template void WeightOnlyGemvWrapper(const phi::GPUContext& ctx,
                                    const float* input,
                                    const int8_t* weight,
                                    const float* bias,
                                    const float* scales,
                                    int m,
                                    int n,
                                    int k,
                                    int group_size,
                                    const std::string& weight_only_quant_type,
                                    const std::string& weight_only_type,
                                    const std::string& act_method,
                                    float* output);

template void WeightOnlyGemvWrapper(const phi::GPUContext& ctx,
                                    const phi::dtype::float16* input,
                                    const int8_t* weight,
                                    const phi::dtype::float16* bias,
                                    const phi::dtype::float16* scales,
                                    int m,
                                    int n,
                                    int k,
                                    int group_size,
                                    const std::string& weight_only_quant_type,
                                    const std::string& weight_only_type,
                                    const std::string& act_method,
                                    phi::dtype::float16* output);
#ifdef PADDLE_CUDA_BF16
template void WeightOnlyGemvWrapper(const phi::GPUContext& ctx,
                                    const phi::dtype::bfloat16* input,
                                    const int8_t* weight,
                                    const phi::dtype::bfloat16* bias,
                                    const phi::dtype::bfloat16* scales,
                                    int m,
                                    int n,
                                    int k,
                                    int group_size,
                                    const std::string& weight_only_quant_type,
                                    const std::string& weight_only_type,
                                    const std::string& act_method,
                                    phi::dtype::bfloat16* output);
#endif

}  // namespace phi
