/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>

#include <cuda_fp16.h>

#if defined(__CUDACC__) && CUDA_VERSION >= 11000
#define CUDA_BFLOAT16_AVALIABLE
#include <cuda_bf16.h>
#endif

#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>
#include <cub/cub.cuh>

#include "cute/algorithm/copy.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/tensor.hpp"

using namespace cute;

namespace phi {
namespace fusion {

template <typename T>
struct CuteDataType;

#ifdef CUDA_BFLOAT16_AVALIABLE
template <>
struct CuteDataType<phi::dtype::float16> {
  using Type = cutlass::half_t;
};

template <>
struct CuteDataType<phi::dtype::bfloat16> {
  using Type = cutlass::bfloat16_t;
};
#endif

template <typename T, typename _out_type, int cache_nbits = 16>
struct Block_attn_params {
  using value_type = T;
  constexpr static int kCacheBits = cache_nbits;
  using cache_type =
      typename std::conditional<kCacheBits == 16, value_type, uint8_t>::type;
  using out_type = _out_type;
  static_assert(kCacheBits == 4 || kCacheBits == 8 || kCacheBits == 16);
  T *__restrict__ qkv_input;
  T *__restrict__ qkv_bias;
  cache_type *__restrict__ cache_k;
  cache_type *__restrict__ cache_v;

  out_type *__restrict__ attn_out;
  T *__restrict__ partition_attn_out;
  T *__restrict__ cache_k_dequant_scale;
  T *__restrict__ cache_v_dequant_scale;
  T *__restrict__ cache_k_quant_scale;
  T *__restrict__ cache_v_quant_scale;
  T *__restrict__ cache_k_zp;
  T *__restrict__ cache_v_zp;
  T *__restrict__ smooth_weight;
  T *__restrict__ shift_bias;
  float out_linear_in_scale;
  float *sums;
  float *maxs;
  int *seq_lens;
  int *block_table;
  float *rotary_emb;
  int *cum_offsets;
  float inv_compression_ratio;
  float rope_theta;
  int rotary_emb_dims;
  int max_input_length;
  int max_seq_len;
  int head_num;
  int kv_head_num;
  int max_num_blocks_per_seq;
  float scale_softmax;
  int batch_size;
  int max_num_partitions;
  float inv_sqrt_dh;

  const int *w_offsets = nullptr;
  const float *out_linear_in_scales = nullptr;
  const T *lora_scales = nullptr;
  const T *lora_smooths = nullptr;
  int8_t *lora_out;
  int num_layers;
  int layer_id;
};

template <typename elem_type_,
          int DataBits_,
          bool KHasZp_ = true,
          bool VHasZp_ = true,
          bool KChannelWise_ = true,
          bool VChannelWise_ = false>
struct CacheKV_quant_traits {
  using cuteType = typename CuteDataType<elem_type_>::Type;
  // c4 就是4，c16就是16
  static constexpr int kDataBits = DataBits_;
  static constexpr bool KHasZp = KHasZp_;
  static constexpr bool VHasZp = VHasZp_;
  static constexpr bool KChannelWise = KChannelWise_;
  static constexpr bool VChannelWise = VChannelWise_;
  static_assert(VHasZp && KHasZp && !VChannelWise && KChannelWise);
  using cache_index = uint32_t;
  using cache_type = uint8_t;
  static constexpr int kBlockSize = 64;
  static constexpr int kHeadDim = 128;
  // 这个值不能超64，否则会有bank flict
  static constexpr int kBlockKSmem = 64;
  using SmemLayoutAtomQ = decltype(composition(
      Swizzle<3, 3, 3>{},
      Layout<Shape<Int<8>, Int<kBlockKSmem>>, Stride<Int<kBlockKSmem>, _1>>{}));

  using SmemLayoutKV = decltype(tile_to_shape(
      SmemLayoutAtomQ{}, Shape<Int<kBlockSize>, Int<kHeadDim>>{}));

  static constexpr int kNWarps = 4;
  static constexpr int kNThreads = kNWarps * 32;

  static constexpr int kThreadPerValue = 16 / sizeof(cuteType);
  static constexpr int kThreadsPerRow = kHeadDim / kThreadPerValue;

  using GmemLayoutAtom =
      Layout<Shape<Int<kNThreads / kThreadsPerRow>, Int<kThreadsPerRow>>,
             Stride<Int<kThreadsPerRow>, _1>>;

  using GmemTiledCopyQ = decltype(make_tiled_copy(
      Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, cuteType>{},
      GmemLayoutAtom{},
      Layout<Shape<_1, Int<kThreadPerValue>>>{}));

  using MMA_Atom_Arch =
      std::conditional_t<std::is_same_v<cuteType, cutlass::half_t>,
                         MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
                         MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>>;

  using TiledMma = TiledMMA<MMA_Atom_Arch,
                            Layout<Shape<_1, Int<kNWarps>, _1>>,
                            Layout<Shape<_1, _2, _1>>>;

  using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, cuteType>;

  using SmemLayoutAtomVtransposed =
      decltype(composition(Swizzle<3, 3, 3>{},
                           Layout<Shape<Int<kBlockKSmem>, Int<kBlockSize>>,
                                  Stride<_1, Int<kBlockKSmem>>>{}));

  using SmemLayoutVtransposed = decltype(tile_to_shape(
      SmemLayoutAtomVtransposed{}, Shape<Int<kHeadDim>, Int<kBlockSize>>{}));

  using SmemLayoutVtransposedNoSwizzle =
      decltype(SmemLayoutVtransposed{}.layout_fn());

  using SmemCopyAtomTransposed = Copy_Atom<SM75_U16x8_LDSM_T, cuteType>;

  static constexpr int kShareMemSize =
      size(SmemLayoutKV{}) * 2 * sizeof(cuteType);
};

template <int kGqaGroupSize_, int kTileN_, typename CacheKV_traits_>
struct Block_attn_kernel_traits {
  using ElementAccum = float;
  using CacheKV_traits = CacheKV_traits_;
  using cuteType = typename CacheKV_traits::cuteType;
  using index_t = typename CacheKV_traits::cache_index;
  static constexpr int kDataBits = CacheKV_traits::kDataBits;
  static constexpr int kTileN = kTileN_;
  static constexpr int kGqaGroupSize = kGqaGroupSize_;
  static constexpr int kHeadDim = CacheKV_traits::kHeadDim;
  static constexpr int kHeadDimKV = kHeadDim / (16 / kDataBits);
  static constexpr int kMinGemmM = 16;
  static constexpr int kBlockM =
      (kGqaGroupSize + kMinGemmM - 1) / kMinGemmM * kMinGemmM;
  static constexpr int kBlockSize = CacheKV_traits::kBlockSize;
  static_assert(kGqaGroupSize <= 16);
  static constexpr int32_t kNWarps = CacheKV_traits::kNWarps;

  // 这个值不能超64，否则会有bank flict
  static constexpr int kBlockKSmem = CacheKV_traits::kBlockKSmem;
  static constexpr int kBlockKVSmem = kHeadDimKV <= 64 ? kHeadDimKV : 64;
  static_assert(kHeadDim % kBlockKSmem == 0);
  static constexpr int kNReduceWarps = 4;
  static constexpr int kNReduceThreads = kNReduceWarps * 32;

  using SmemLayoutAtomQ = typename CacheKV_traits::SmemLayoutAtomQ;

  using SmemLayoutQ = decltype(tile_to_shape(
      SmemLayoutAtomQ{}, Shape<Int<kBlockM>, Int<kHeadDim>>{}));

  using SmemLayoutQK = decltype(tile_to_shape(
      SmemLayoutAtomQ{}, Shape<Int<kBlockM>, Int<kBlockSize>>{}));

  using SmemLayoutAtomKV =
      decltype(composition(Swizzle<3, 3, 3>{},
                           Layout<Shape<Int<8>, Int<kBlockKVSmem>>,
                                  Stride<Int<kBlockKVSmem>, _1>>{}));

  using SmemLayoutKV_ = decltype(tile_to_shape(
      SmemLayoutAtomKV{}, Shape<Int<kBlockSize>, Int<kHeadDimKV>>{}));

  using SmemLayoutKV =
      std::conditional_t<kDataBits == 16,
                         SmemLayoutKV_,
                         decltype(SmemLayoutKV_{}.layout_fn())>;

  constexpr static int kBlockKVSize = kDataBits == 4 ? 32 : kBlockSize;
  using SmemLayoutAtomVtransposed =
      decltype(composition(Swizzle<3, 3, 3>{},
                           Layout<Shape<Int<kBlockKSmem>, Int<kBlockKVSize>>,
                                  Stride<_1, Int<kBlockKSmem>>>{}));

  using SmemLayoutVtransposed = decltype(tile_to_shape(
      SmemLayoutAtomVtransposed{}, Shape<Int<kHeadDim>, Int<kBlockKVSize>>{}));

  using SmemLayoutVtransposedNoSwizzle =
      decltype(SmemLayoutVtransposed{}.layout_fn());

  static constexpr int kThreadsPerRow = CacheKV_traits::kThreadsPerRow;
  static constexpr int kThreadsKVPerRow = kThreadsPerRow / (16 / kDataBits);
  static constexpr int kNThreads = CacheKV_traits::kNThreads;

  using GmemKVLayoutAtom =
      Layout<Shape<Int<kNThreads / kThreadsKVPerRow>, Int<kThreadsKVPerRow>>,
             Stride<Int<kThreadsKVPerRow>, _1>>;

  using SmemCopyAtom = typename CacheKV_traits::SmemCopyAtom;
  using TiledMma = typename CacheKV_traits::TiledMma;

  static constexpr int kThreadPerValue = CacheKV_traits::kThreadPerValue;

  using GmemTiledCopyQ = typename CacheKV_traits::GmemTiledCopyQ;
  using GmemLayoutAtom = typename CacheKV_traits::GmemLayoutAtom;
  using GmemTiledCopyKV = decltype(make_tiled_copy(
      Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, cuteType>{},
      GmemKVLayoutAtom{},
      Layout<Shape<_1, Int<kThreadPerValue>>>{}));

  using SmemCopyAtomTransposed =
      typename CacheKV_traits::SmemCopyAtomTransposed;

  using GmemTiledCopyO =
      decltype(make_tiled_copy(Copy_Atom<DefaultCopy, cuteType>{},
                               GmemLayoutAtom{},
                               Layout<Shape<_1, Int<kThreadPerValue>>>{}));
  using SmemCopyAtomO = Copy_Atom<DefaultCopy, cuteType>;

  using SmemLayoutAtomO = decltype(composition(
      Swizzle<3, 3, 3>{},
      Layout<Shape<Int<8>, Int<kBlockKSmem>>, Stride<Int<kBlockKSmem>, _1>>{}));

  using SmemLayoutO = decltype(tile_to_shape(
      SmemLayoutAtomO{}, Shape<Int<kBlockM>, Int<kHeadDim>>{}));

  static constexpr int kShareMemSize =
      (size(SmemLayoutQ{}) * 2 + size(SmemLayoutQK{}) +
       size(SmemLayoutKV{}) * 2) *
      sizeof(cuteType);
};

namespace block_attn {

template <typename T>
struct PackedHalf;

template <>
struct PackedHalf<cutlass::half_t> {
  using Type = __half2;
};

template <>
struct PackedHalf<cutlass::bfloat16_t> {
  using Type = nv_bfloat162;
};

template <>
struct PackedHalf<phi::dtype::float16> {
  using Type = __half2;
};

template <>
struct PackedHalf<phi::dtype::bfloat16> {
  using Type = nv_bfloat162;
};

template <typename T>
struct HalfSub;

template <>
struct HalfSub<cutlass::half_t> {
  inline __device__ void operator()(uint32_t *result_ptr,
                                    const uint32_t magic_num) {
    asm volatile("sub.f16x2 %0, %1, %2;\n"
                 : "=r"(*result_ptr)
                 : "r"(*result_ptr), "r"(magic_num));
  }
};

#ifdef CUDA_BFLOAT16_AVALIABLE
template <>
struct HalfSub<cutlass::bfloat16_t> {
  inline __device__ void operator()(uint32_t *result_ptr,
                                    const uint32_t magic_num) {
    *reinterpret_cast<nv_bfloat162 *>(result_ptr) -=
        *reinterpret_cast<const nv_bfloat162 *>(&magic_num);
  }
};
#endif

template <typename T>
struct HalfMul;

template <>
struct HalfMul<cutlass::half_t> {
  inline __device__ void operator()(uint32_t *result_ptr,
                                    const uint32_t magic_num) {
    asm volatile("mul.f16x2 %0, %1, %2;\n"
                 : "=r"(*result_ptr)
                 : "r"(*result_ptr), "r"(magic_num));
  }
};

template <>
struct HalfMul<cutlass::bfloat16_t> {
  inline __device__ void operator()(uint32_t *result_ptr,
                                    const uint32_t magic_num) {
    *reinterpret_cast<nv_bfloat162 *>(result_ptr) *=
        *reinterpret_cast<const nv_bfloat162 *>(&magic_num);
  }
};

template <typename T>
struct HalfMax;
template <>
struct HalfMax<cutlass::half_t> {
  inline __device__ __half2 operator()(const __half2 x, const __half2 y) {
    __half2 res;
    asm volatile("max.f16x2 %0, %1, %2;\n"
                 : "=r"(*reinterpret_cast<uint32_t *>(&res))
                 : "r"(*reinterpret_cast<const uint32_t *>(&x)),
                   "r"(*reinterpret_cast<const uint32_t *>(&y)));
    return res;
  }
};

template <>
struct HalfMax<cutlass::bfloat16_t> {
  inline __device__ nv_bfloat162 operator()(const nv_bfloat162 x,
                                            const nv_bfloat162 y) {
    nv_bfloat162 res;
    asm volatile("max.bf16x2 %0, %1, %2;\n"
                 : "=r"(*reinterpret_cast<uint32_t *>(&res))
                 : "r"(*reinterpret_cast<const uint32_t *>(&x)),
                   "r"(*reinterpret_cast<const uint32_t *>(&y)));
    return res;
  }
};

template <typename T>
struct HalfMin;
template <>
struct HalfMin<cutlass::half_t> {
  inline __device__ __half2 operator()(const __half2 x, const __half2 y) {
    __half2 res;
    asm volatile("min.f16x2 %0, %1, %2;\n"
                 : "=r"(*reinterpret_cast<uint32_t *>(&res))
                 : "r"(*reinterpret_cast<const uint32_t *>(&x)),
                   "r"(*reinterpret_cast<const uint32_t *>(&y)));
    return res;
  }
};

template <>
struct HalfMin<cutlass::bfloat16_t> {
  inline __device__ nv_bfloat162 operator()(const nv_bfloat162 x,
                                            const nv_bfloat162 y) {
    nv_bfloat162 res;
    asm volatile("min.bf16x2 %0, %1, %2;\n"
                 : "=r"(*reinterpret_cast<uint32_t *>(&res))
                 : "r"(*reinterpret_cast<const uint32_t *>(&x)),
                   "r"(*reinterpret_cast<const uint32_t *>(&y)));
    return res;
  }
};

struct uint16 {
  uint4 u;
  uint4 v;
  uint4 s;
  uint4 t;
};

struct uint8 {
  uint4 u;
  uint4 v;
};

template <int BYTES>
struct BytesToType {};

template <>
struct BytesToType<64> {
  using Type = uint16;
  static_assert(sizeof(Type) == 64);
};

template <>
struct BytesToType<32> {
  using Type = uint8;
  static_assert(sizeof(Type) == 32);
};

template <>
struct BytesToType<16> {
  using Type = uint4;
  static_assert(sizeof(Type) == 16);
};

template <>
struct BytesToType<8> {
  using Type = uint64_t;
  static_assert(sizeof(Type) == 8);
};

template <>
struct BytesToType<4> {
  using Type = uint32_t;
  static_assert(sizeof(Type) == 4);
};

template <>
struct BytesToType<2> {
  using Type = uint16_t;
  static_assert(sizeof(Type) == 2);
};

template <>
struct BytesToType<1> {
  using Type = uint8_t;
  static_assert(sizeof(Type) == 1);
};

template <typename Elt_type, uint32_t NUM_ELT>
struct Vec {
  enum { BYTES = NUM_ELT * sizeof(Elt_type) };

  using Vec_type = typename BytesToType<BYTES>::Type;

  using Alias_type = union {
    Vec_type vec;
    Elt_type elt[NUM_ELT];
  };

  Alias_type data;

  inline __device__ Vec() {}

  inline __device__ Vec(const Elt_type value) {
#pragma unroll
    for (int it = 0; it < NUM_ELT; it++) {
      this->data.elt[it] = value;
    }
  }

  template <typename S>
  inline __device__ void to(Vec<S, NUM_ELT> &other) {
#pragma unroll
    for (int it = 0; it < NUM_ELT; it++) {
      other.data.elt[it] = S(this->data.elt[it]);
    }
  }

  template <typename Op>
  inline __device__ void assign(const Op &op) {
#pragma unroll
    for (int it = 0; it < NUM_ELT; it++) {
      this->data.elt[it] = op(it);
    }
  }

  inline __device__ void load_from(const void *base_ptr) {
    this->data.vec = *reinterpret_cast<const Vec_type *>(base_ptr);
  }

  inline __device__ void store_to(void *base_ptr) {
    *reinterpret_cast<Vec_type *>(base_ptr) = this->data.vec;
  }

  inline __device__ void add(const Vec<Elt_type, NUM_ELT> &other) {
    static_assert(NUM_ELT % 2 == 0);
    using type = typename PackedHalf<Elt_type>::Type;
#pragma unroll
    for (int it = 0; it < NUM_ELT / 2; it++) {
      type b = *reinterpret_cast<const type *>(other.data.elt + it * 2);
      *reinterpret_cast<type *>(this->data.elt + it * 2) += b;
    }
  }

  inline __device__ float operator*(const Vec<Elt_type, NUM_ELT> &other) {
    static_assert(NUM_ELT % 2 == 0);
    using type = typename PackedHalf<Elt_type>::Type;
    type c(0.0f, 0.0f);
#pragma unroll
    for (int it = 0; it < NUM_ELT / 2; it++) {
      type a = *reinterpret_cast<type *>(this->data.elt + it * 2);
      type b = *reinterpret_cast<const type *>(other.data.elt + it * 2);
      c = __hfma2(a, b, c);
    }
    return static_cast<float>(c.x + c.y);
  }
};

template <typename T, int PackSize>
inline __device__ void quant_to_int8(const T *src,
                                     const T *smooth_weight,
                                     const T *shift_bias,
                                     int8_t *dst,
                                     const float out_linear_in_scale) {
#pragma unroll
  for (int i = 0; i < PackSize; i++) {
    float src_data =
        round((static_cast<float>(src[i]) + static_cast<float>(shift_bias[i])) *
              static_cast<float>(smooth_weight[i]) * out_linear_in_scale);
    src_data =
        src_data > 127.0f ? 127.0f : (src_data < -128.0f ? -128.0f : src_data);
    dst[i] = static_cast<int8_t>(src_data);
  }
}

template <typename T, int PackSize>
inline __device__ void quant_to_int8_lora(const T *src,
                                          const T *smooth_weight,
                                          const T *shift_bias,
                                          int8_t *dst,
                                          const float out_linear_in_scale) {
#pragma unroll
  for (int i = 0; i < PackSize; i++) {
    float src_data = rint(
        (static_cast<float>(src[i]) + static_cast<float>(shift_bias[i])) *
        static_cast<float>(smooth_weight[i]) * out_linear_in_scale * 127.0f);
    src_data =
        src_data > 127.0f ? 127.0f : (src_data < -128.0f ? -128.0f : src_data);
    dst[i] = static_cast<int8_t>(src_data);
  }
}

template <typename T, int PackSize>
inline __device__ void quant_to_int8_lora(const T *src,
                                          const T *smooth_weight,
                                          int8_t *dst,
                                          const float out_linear_in_scale) {
#pragma unroll
  for (int i = 0; i < PackSize; i++) {
    float src_data = rint((static_cast<float>(src[i])) *
                          static_cast<float>(smooth_weight[i]) *
                          out_linear_in_scale * 127.0f);
    src_data =
        src_data > 127.0f ? 127.0f : (src_data < -128.0f ? -128.0f : src_data);
    dst[i] = static_cast<int8_t>(src_data);
  }
}

template <bool Is_even_MN = true,
          typename TiledCopy,
          typename Engine0,
          typename Layout0,
          typename Engine1,
          typename Layout1,
          typename Engine2,
          typename Layout2>
inline __device__ void copy(TiledCopy thr_copy,
                            Tensor<Engine0, Layout0> &S,
                            Tensor<Engine1, Layout1> &D,
                            Tensor<Engine2, Layout2> const &identity_MN,
                            const int max_MN = 0) {
  CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
  CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
  CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));  // MMA
  CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));  // MMA_M
  CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));  // MMA_K
  auto src_ptr = S.data();
#pragma unroll
  for (int m = 0; m < size<1>(S); ++m) {
    if (Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN) {
#pragma unroll
      for (int k = 0; k < size<2>(S); ++k) {
        copy(thr_copy, S(_, m, k), D(_, m, k));
      }
    } else {
      clear(D(_, m, _));
    }
  }
}

template <int kMiLen, typename Engine, typename Layout>
inline __device__ void apply_mask(Tensor<Engine, Layout> &scores,
                                  const uint32_t warp_id,
                                  const uint32_t col,
                                  const uint32_t reamin_seq_len) {
  const int cols = size<1>(scores) / 2;
#pragma unroll
  for (int mi = 0; mi < kMiLen; ++mi) {
#pragma unroll
    for (int ni = 0; ni < cols; ++ni) {
      const int col_index = warp_id * 8 + ni * 32 + col * 2;
      if (col_index >= reamin_seq_len) {
        scores(mi, ni * 2) = -INFINITY;
      }
      if (col_index + 1 >= reamin_seq_len) {
        scores(mi, ni * 2 + 1) = -INFINITY;
      }
    }
  }
}

template <typename T, int PackSize, int kHeadDim>
inline __device__ void apply_rotary_embedding(Vec<T, PackSize> &vec,
                                              const int tid,
                                              const float *rope_cos_sin) {
  static_assert(PackSize % 4 == 0);
#pragma unroll
  for (int i = 0; i < PackSize; i += 2) {
    const float cos_inv_freq = rope_cos_sin[tid * 4 + i];
    const float sin_inv_freq = rope_cos_sin[tid * 4 + i + 1];
    const float v1 = static_cast<float>(vec.data.elt[i]);
    const float v2 = static_cast<float>(vec.data.elt[i + 1]);
    vec.data.elt[i] = static_cast<T>(cos_inv_freq * v1 - sin_inv_freq * v2);
    vec.data.elt[i + 1] = static_cast<T>(sin_inv_freq * v1 + cos_inv_freq * v2);
  }
}

template <bool A_in_regs = false,
          bool B_in_regs = false,
          typename Tensor0,
          typename Tensor1,
          typename Tensor2,
          typename Tensor3,
          typename Tensor4,
          typename TiledMma,
          typename TiledCopy0,
          typename TiledCopy1>
inline __device__ void gemm(Tensor0 &acc,
                            Tensor1 &tCrA,
                            Tensor2 &tCrB,
                            Tensor3 const &tCsA,
                            Tensor4 const &tCsB,
                            TiledMma tiled_mma,
                            TiledCopy0 smem_thr_copy_A,
                            TiledCopy1 smem_thr_copy_B) {
  CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));   // MMA_M
  CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));   // MMA_N
  CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));  // MMA_K
  Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);
  CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));  // M
  Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
  CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));  // N
  if (!A_in_regs) {
    copy(smem_thr_copy_A, tCsA(_, _, _0{}), tCrA_copy_view(_, _, _0{}));
  }
  if (!B_in_regs) {
    copy(smem_thr_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
  }
#pragma unroll
  for (int i = 0; i < size<2>(tCrA); ++i) {
    if (i < size<2>(tCrA) - 1) {
      if (!A_in_regs) {
        copy(smem_thr_copy_A, tCsA(_, _, i + 1), tCrA_copy_view(_, _, i + 1));
      }
      if (!B_in_regs) {
        copy(smem_thr_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
      }
    }
    cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
  }
}

template <typename T, bool IsChannel, int ValuePerTidx>
inline __device__ static uint32_t convert_half_2_c8(T *src,
                                                    const T *cache_scale) {
  using uint8_4_vec = typename block_attn::Vec<uint8_t, ValuePerTidx>;
  uint8_4_vec value;
  static_assert(ValuePerTidx == 4);
  const float magic_num =
      std::is_same_v<T, cutlass::bfloat16_t> ? 0.0f : 1024.0f;
#pragma unroll
  for (int i = 0; i < ValuePerTidx; i++) {
    const int ith_col = IsChannel ? i : 0;
    float x = static_cast<float>(reinterpret_cast<T *>(src)[i]);

    float scale =
        static_cast<float>(reinterpret_cast<const T *>(cache_scale)[ith_col]);

    x = round(x * scale);
    x = x > 255.0f ? 255.0f : x;
    x = x < 0.0f ? 0.0f : x;
    value.data.elt[i] = static_cast<uint8_t>(x);
  }
  return *reinterpret_cast<uint32_t *>(value.data.elt);
}

template <typename T, bool Is_K>
inline __device__ static void convert_c8_2_half(uint32_t *src,
                                                T *dst,
                                                const T *cache_scale) {
  uint32_t *half_result_ptr = reinterpret_cast<uint32_t *>(dst);
  if constexpr (std::is_same_v<T, cutlass::bfloat16_t>) {
    static constexpr uint32_t fp32_base = 0x4B000000;
    float fp32_intermediates[4];

    uint32_t *fp32_intermediates_casted =
        reinterpret_cast<uint32_t *>(fp32_intermediates);
    fp32_intermediates_casted[0] = __byte_perm(*src, fp32_base, 0x7650);
    fp32_intermediates_casted[1] = __byte_perm(*src, fp32_base, 0x7651);
    fp32_intermediates_casted[2] = __byte_perm(*src, fp32_base, 0x7652);
    fp32_intermediates_casted[3] = __byte_perm(*src, fp32_base, 0x7653);

#pragma unroll
    for (int ii = 0; ii < 4; ++ii) {
      fp32_intermediates[ii] -= 8388608.f;
    }

#pragma unroll
    for (int ii = 0; ii < 2; ++ii) {
      half_result_ptr[ii] = __byte_perm(fp32_intermediates_casted[2 * ii + 0],
                                        fp32_intermediates_casted[2 * ii + 1],
                                        0x7632);
    }
  } else {
    static constexpr uint32_t head_for_fp16 = 0x64006400;
    half_result_ptr[0] = __byte_perm(*src, head_for_fp16, 0x7150);
    half_result_ptr[1] = __byte_perm(*src, head_for_fp16, 0x7352);
  }

  using pack_half = typename PackedHalf<T>::Type;
#pragma unroll
  for (int i = 0; i < 2; i++) {
    if constexpr (Is_K) {
      HalfMul<T>()(half_result_ptr + i,
                   *reinterpret_cast<const uint32_t *>(cache_scale + i * 2));
    } else {
      pack_half scale;
      scale.x = cache_scale[0];
      scale.y = cache_scale[0];
      HalfMul<T>()(half_result_ptr + i,
                   *reinterpret_cast<const uint32_t *>(&scale));
    }
  }
}

template <typename CacheKV_traits,
          typename T,
          int kHeadDim,
          int kDataNumPer2Byte,
          bool A_in_regs = false,
          typename Tensor0,
          typename Tensor1,
          typename Tensor2,
          typename Tensor3,
          typename Tensor4,
          typename TiledMma,
          typename TiledCopy0>
inline __device__ void gemm_qk_quant(Tensor0 &acc,
                                     Tensor1 &tCrA,
                                     Tensor2 &tCsA,
                                     Tensor3 &tCrB,
                                     Tensor4 const &sB,
                                     TiledMma tiled_mma,
                                     TiledCopy0 smem_thr_copy_A,
                                     const int32_t tidx,
                                     const T *cache_scale) {
  CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));
  CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));
  Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);
  CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));
  if (!A_in_regs) {
    copy(smem_thr_copy_A, tCsA(_, _, _0{}), tCrA_copy_view(_, _, _0{}));
  }
  uint32_t *sBdata = reinterpret_cast<uint32_t *>(sB.data().get()) +
                     tidx * (kDataNumPer2Byte / 4);

#pragma unroll
  for (int i = 0; i < size<2>(tCrA); ++i) {
    if (i < size<2>(tCrA) - 1) {
      if (!A_in_regs) {
        copy(smem_thr_copy_A, tCsA(_, _, i + 1), tCrA_copy_view(_, _, i + 1));
      }
    }
    if constexpr (kDataNumPer2Byte == 8) {
      convert_c8_2_half<T, true>(
          sBdata + i * (kHeadDim * 2), tCrB.data(), cache_scale + i * 4);
      convert_c8_2_half<T, true>(sBdata + i * (kHeadDim * 2) + 1,
                                 tCrB.data() + 4,
                                 cache_scale + i * 4);
    }

    cute::gemm(tiled_mma, tCrA(_, _, i), tCrB, acc);
  }
}

template <typename CacheKV_traits,
          typename T,
          int kHeadDim,
          int kDataNumPer2Byte,
          bool A_in_regs = false,
          typename Tensor0,
          typename Tensor1,
          typename Tensor2,
          typename Tensor3,
          typename Tensor4,
          typename TiledMma,
          typename TiledCopy0>
inline __device__ void gemm_value_quant(Tensor0 &acc,
                                        Tensor1 &tCrA,
                                        Tensor2 &tCsA,
                                        Tensor3 &tCrB,
                                        Tensor4 const &sB,
                                        TiledMma tiled_mma,
                                        TiledCopy0 smem_thr_copy_A,
                                        int32_t tidx,
                                        const T *cache_scale) {
  CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));
  CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));
  Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);
  CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));
  if (!A_in_regs) {
    copy(smem_thr_copy_A, tCsA(_, _, _0{}), tCrA_copy_view(_, _, _0{}));
  }
  uint32_t *sBdata = reinterpret_cast<uint32_t *>(sB.data().get()) +
                     tidx * (2 * kDataNumPer2Byte / 4);

#pragma unroll
  for (int i = 0; i < size<2>(tCrA); ++i) {
    const int cur_idx = i * kHeadDim * (2 * kDataNumPer2Byte / 4);

    if (i < size<2>(tCrA) - 1) {
      if (!A_in_regs) {
        copy(smem_thr_copy_A, tCsA(_, _, i + 1), tCrA_copy_view(_, _, i + 1));
      }
    }
    if constexpr (kDataNumPer2Byte == 8) {
      convert_c8_2_half<T, false>(sBdata + cur_idx, tCrB.data(), cache_scale);
      convert_c8_2_half<T, false>(
          sBdata + cur_idx + 1, tCrB.data() + 4, cache_scale + 1);
      convert_c8_2_half<T, false>(
          sBdata + cur_idx + 2, tCrB.data() + 8, cache_scale + 2);
      convert_c8_2_half<T, false>(
          sBdata + cur_idx + 3, tCrB.data() + 12, cache_scale + 3);
    }
    cute::gemm(tiled_mma, tCrA(_, _, i), tCrB, acc);
  }
}

template <typename Layout>
inline __device__ auto convert_layout_acc_rowcol(Layout acc_layout) {
  static_assert(decltype(size<0>(acc_layout))::value == 4);
  static_assert(decltype(rank(acc_layout))::value == 3);
  auto l = logical_divide(acc_layout, Shape<_2>{});  // ((2, 2), MMA_M, MMA_N)
  return make_layout(make_layout(get<0, 1>(l), get<1>(l)),
                     make_layout(get<0, 0>(l), get<2>(l)));
};

template <typename cuteType, int kDataNumPer2Byte, int kNThreads>
inline __device__ void write_cache_k(cuteType *fragment_data1,
                                     cuteType *fragment_data2,
                                     uint32_t *smem,
                                     const cuteType *scale,
                                     const int tidx,
                                     const int idx) {
  using uint_vec_2 = typename block_attn::Vec<uint32_t, 2>;
  if constexpr (kDataNumPer2Byte == 8) {
    uint32_t value1 =
        block_attn::convert_half_2_c8<cuteType, true, 4>(fragment_data1, scale);
    uint32_t value2 =
        block_attn::convert_half_2_c8<cuteType, true, 4>(fragment_data2, scale);
    uint_vec_2 value;
    value.data.elt[0] = value1;
    value.data.elt[1] = value2;
    value.store_to(smem + idx * kNThreads * 2 + tidx * 2);
  }
};

template <typename cuteType, int kDataNumPer2Byte>
inline __device__ void write_cache_v(cuteType *fragment_data1,
                                     cuteType *fragment_data2,
                                     uint32_t *fragment_dst,
                                     const cuteType *scale,
                                     const int tidx,
                                     int idx) {
  if constexpr (kDataNumPer2Byte == 8) {
    uint32_t value1 = block_attn::convert_half_2_c8<cuteType, false, 4>(
        fragment_data1, scale);
    uint32_t value2 = block_attn::convert_half_2_c8<cuteType, false, 4>(
        fragment_data2, scale + 1);
    fragment_dst[0] = value1;
    fragment_dst[1] = value2;
  }
};

template <int N>
CUTE_HOST_DEVICE void cp_async_wait() {
#if defined(CUTE_ARCH_CP_ASYNC_SM80_ENABLED)
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
#endif
}

template <typename T>
struct MaxOp {
  __device__ inline T operator()(T const &x, T const &y) {
    return x > y ? x : y;
  }
};

template <>
struct MaxOp<float> {
  __device__ inline float operator()(float const &x, float const &y) {
    return max(x, y);
  }
};

template <typename T>
struct SumOp {
  __device__ inline T operator()(T const &x, T const &y) { return x + y; }
};

template <int THREADS>
struct Allreduce {
  static_assert(THREADS == 32 || THREADS == 16 || THREADS == 8 || THREADS == 4);
  template <typename T, typename Operator>
  static __device__ inline T run(T x, Operator &op) {
    constexpr int OFFSET = THREADS / 2;
    x = op(x, __shfl_xor_sync(uint32_t(-1), x, OFFSET));
    return Allreduce<OFFSET>::run(x, op);
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
struct Allreduce<2> {
  template <typename T, typename Operator>
  static __device__ inline T run(T x, Operator &op) {
    x = op(x, __shfl_xor_sync(uint32_t(-1), x, 1));
    return x;
  }
};

template <int kMiLen, typename Engine0, typename Layout0, typename T>
__device__ inline void reduce_max(Tensor<Engine0, Layout0> const &tensor,
                                  T *scores_max) {
  static_assert(Layout0::rank == 2, "Only support 2D Tensor");
  MaxOp<T> max_op;
#pragma unroll
  for (int mi = 0; mi < kMiLen; ++mi) {
#pragma unroll
    for (int ni = 0; ni < size<1>(tensor); ni++) {
      scores_max[mi] = max_op(scores_max[mi], tensor(mi, ni));
    }
    scores_max[mi] = block_attn::Allreduce<4>::run(scores_max[mi], max_op);
  }
}

// Apply the exp to all the elements.
template <int kMiLen, typename Engine0, typename Layout0, typename T>
inline __device__ void scale_apply_exp2(Tensor<Engine0, Layout0> &tensor,
                                        T const *max,
                                        T *sum,
                                        const float scale) {
  static_assert(Layout0::rank == 2, "Only support 2D Tensor");
#pragma unroll
  for (int mi = 0; mi < kMiLen; ++mi) {
    const float max_scaled = max[mi] * scale;
#pragma unroll
    for (int ni = 0; ni < size<1>(tensor); ++ni) {
      tensor(mi, ni) = expf(tensor(mi, ni) * scale - max_scaled);
      sum[mi] += tensor(mi, ni);
    }
  }
}

template <typename To_type, typename Engine, typename Layout>
inline __device__ auto convert_type(Tensor<Engine, Layout> const &tensor) {
  using From_type = typename Engine::value_type;
  constexpr int numel = decltype(size(tensor))::value;
  cutlass::NumericArrayConverter<To_type, From_type, numel> convert_op;
  // HACK: this requires tensor to be "contiguous"
  auto frag =
      convert_op(*reinterpret_cast<const cutlass::Array<From_type, numel> *>(
          tensor.data()));
  return make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout());
}

template <typename T, typename ReductionOp, int thread_group_width = 32>
__inline__ __device__ T WarpAllReduce(T val) {
  ReductionOp op;
#pragma unroll
  for (int mask = thread_group_width / 2; mask > 0; mask /= 2) {
    val = op(val, __shfl_xor_sync(0xffffffff, val, mask));
  }
  return val;
}

template <typename T, typename ReductionOp, int block_size>
__inline__ __device__ T BlockAllReduce(T val) {
  typedef cub::BlockReduce<T, block_size> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ T result_broadcast;
  T result = BlockReduce(temp_storage).Reduce(val, ReductionOp());
  if (threadIdx.x == 0) {
    result_broadcast = result;
  }
  __syncthreads();
  return result_broadcast;
}

template <typename Src, typename Dst>
struct Converter {
  static inline __device__ __host__ Dst convert(const Src &from) {
    return Dst(from);
  }
};

template <>
struct Converter<float2, half2> {
  static inline __device__ __host__ half2 convert(const float2 &x) {
    return __float22half2_rn(x);
  }
};

template <>
struct Converter<float, half> {
  static inline __device__ __host__ half convert(const float &x) {
    return __float2half_rn(x);
  }
};

template <>
struct Converter<float2, nv_bfloat162> {
  static inline __device__ __host__ nv_bfloat162 convert(const float2 &x) {
#if __CUDA_ARCH__ >= 800
    return __float22bfloat162_rn(x);
#else
    union {
      nv_bfloat162 raw;
      nv_bfloat16 x;
      nv_bfloat16 y;
    } tmp;
    tmp.x = __float2bfloat16_rn(x.x);
    tmp.y = __float2bfloat16_rn(x.y);
    return tmp.raw;
#endif
  }
};

template <>
struct Converter<float, nv_bfloat16> {
  static inline __device__ __host__ nv_bfloat16 convert(const float &x) {
#if __CUDA_ARCH__ >= 800
    return __float2bfloat16_rn(x);
#else
    union {
      nv_bfloat16 raw;
      nv_bfloat16 x;
    } tmp;
    tmp.x = __float2bfloat16_rn(x);
    return tmp.raw;
#endif
  }
};

}  // namespace block_attn
}  // namespace fusion
}  // namespace phi
