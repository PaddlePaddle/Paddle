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
constexpr int kPerBlockWarpNum = 16;

/////////////////////////////////////////////////////////////////////
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
struct ConvertDstFunc {
  static __device__ __forceinline__ T apply(const float& val) { assert(false); }
};

template <>
struct ConvertDstFunc<half> {
  static __device__ __forceinline__ half apply(const float& val) {
    return __float2half_rn(val);
  }
};

#ifdef PADDLE_CUDA_BF16
template <>
struct ConvertDstFunc<__nv_bfloat16> {
  static __device__ __forceinline__ __nv_bfloat16 apply(const float& val) {
    return __float2bfloat16_rn(val);
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
                                      const float* scale_list,
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

  float v = 0.f, scale = scale_list[row_id], v_bias;

  if (Bias) {
    v_bias = ConvertFloatFunc<T>::apply(bias[row_id]);
  }

#pragma unroll
  for (int i = lane_id * kVecSize; i < k * 2; i += kVecSize * kWarpSize) {
    *(int4*)vec_weight = *(int4*)(weight + i);            // NOLINT
    *(float4*)vec_input =                                 // NOLINT
        *(float4*)(input + i / 128 * 64 + (i % 64));      // NOLINT
    *(float4*)(vec_input + 8) =                           // NOLINT
        *(float4*)(input + i / 128 * 64 + (i % 64) + 8);  // NOLINT
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
#endif

template <typename T>
void int8_weight_only_gemv_launcher(const T* input,
                                    const int8_t* weight,
                                    const float* scale_list,
                                    const T* bias,
                                    T* output,
                                    const int k,
                                    const int n,
                                    const bool gelu,
                                    gpuStream_t stream) {
#ifdef PADDLE_WITH_CUDA
  dim3 block(kWarpSize * kPerBlockWarpNum);  // equal to 512;
  dim3 grid(n / kPerBlockWarpNum /
            2);  // Note(zhengzekang): Since each warp process 2 rows of matrix.
  if (bias) {
    if (gelu) {
      int8_weight_only_gemv<T, true, true><<<grid, block, 0, stream>>>(
          input, weight, scale_list, bias, output, k, n);
    } else {
      int8_weight_only_gemv<T, true, false><<<grid, block, 0, stream>>>(
          input, weight, scale_list, bias, output, k, n);
    }
  } else {
    if (gelu) {
      int8_weight_only_gemv<T, false, true><<<grid, block, 0, stream>>>(
          input, weight, scale_list, bias, output, k, n);
    } else {
      int8_weight_only_gemv<T, false, false><<<grid, block, 0, stream>>>(
          input, weight, scale_list, bias, output, k, n);
    }
  }
#endif
}

template <>
void int8_weight_only_gemv_launcher(const float* input,
                                    const int8_t* weight,
                                    const float* scale_list,
                                    const float* bias,
                                    float* output,
                                    const int k,
                                    const int n,
                                    const bool gelu,
                                    gpuStream_t stream) {
  // Weightonly GEMV do not support float.
  assert(false);
}

template <>
void int8_weight_only_gemv_launcher(const phi::dtype::bfloat16* input,
                                    const int8_t* weight,
                                    const float* scale_list,
                                    const phi::dtype::bfloat16* bias,
                                    phi::dtype::bfloat16* output,
                                    const int k,
                                    const int n,
                                    const bool gelu,
                                    gpuStream_t stream) {
  // Environment do not support bf16.
  assert(false);
}

}  // namespace

template <typename T, typename Context>
void GemvWeightonlyInt8Wrapper(const Context& ctx,
                               const T* x,
                               const int8_t* weight,
                               const T* bias,
                               const float* weight_scale,
                               const int n,
                               const int k,
                               const std::string& act_method,
                               T* output) {
  using DataType = typename PDDataTypeTraits<T>::DataType;

  bool gelu = false;
  if (act_method == "gelu") {
    gelu = true;
  } else if (act_method == "None") {
    gelu = false;
  } else {
    PADDLE_THROW(
        errors::InvalidArgument("Currently, Int8 weightonly GEMV act_method "
                                "only support `gelu`, `None`. "));
  }

  int8_weight_only_gemv_launcher<DataType>(
      reinterpret_cast<const DataType*>(x),
      weight,
      weight_scale,
      reinterpret_cast<const DataType*>(bias),
      reinterpret_cast<DataType*>(output),
      k,
      n,
      gelu,
      ctx.stream());
}

template <typename T, typename Context>
void GemvWeightonlyInt8Kernel(const Context& dev_ctx,
                              const DenseTensor& x,
                              const DenseTensor& weight,
                              const paddle::optional<DenseTensor>& bias,
                              const DenseTensor& weight_scale,
                              const std::string& act_method,
                              DenseTensor* out) {
  const T* x_data = x.data<T>();
  const int8_t* weight_data =
      weight.data<int8_t>();  // Actually, we pass the weight datatype is
                              // uint8_t type.
  const T* bias_data = bias ? bias.get().data<T>() : nullptr;
  const float* weight_scale_data = weight_scale.data<float>();
  T* out_data = dev_ctx.template Alloc<T>(out);

  int k = x.dims()[1];
  int n = weight.dims()[0];
  GemvWeightonlyInt8Wrapper<T, Context>(dev_ctx,
                                        x_data,
                                        weight_data,
                                        bias_data,
                                        weight_scale_data,
                                        n,
                                        k,
                                        act_method,
                                        out_data);
}

template void GemvWeightonlyInt8Wrapper(const phi::GPUContext& ctx,
                                        const phi::dtype::float16* x,
                                        const int8_t* weight,
                                        const phi::dtype::float16* bias,
                                        const float* weight_scale,
                                        const int n,
                                        const int k,
                                        const std::string& act_method,
                                        phi::dtype::float16* output);

template void GemvWeightonlyInt8Wrapper(const phi::GPUContext& ctx,
                                        const phi::dtype::bfloat16* x,
                                        const int8_t* weight,
                                        const phi::dtype::bfloat16* bias,
                                        const float* weight_scale,
                                        const int n,
                                        const int k,
                                        const std::string& act_method,
                                        phi::dtype::bfloat16* output);

template void GemvWeightonlyInt8Wrapper(const phi::GPUContext& ctx,
                                        const float* x,
                                        const int8_t* weight,
                                        const float* bias,
                                        const float* weight_scale,
                                        const int n,
                                        const int k,
                                        const std::string& act_method,
                                        float* output);

}  // namespace phi
