#pragma once
#include <immintrin.h>
#include "cpu_patch/epilogue.h"
#include <type_traits>

namespace gops {
namespace kernel {

static int8_t mask[32]
    __attribute__((aligned(64))) = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0};

struct Vec256Pair {
    __m256* v0;
    __m256* v1;
};

template<int E, typename... Ts>
inline void fma_inner(float* blockA_packed, float* blockB_packed,
                      __m256* a0_packFloat8, __m256* a1_packFloat8,
                      __m256* b_packFloat8, int kc,
                      Vec256Pair C_accum, Ts... rest) {
    *b_packFloat8 = _mm256_broadcast_ss(blockB_packed + E);
    *C_accum.v0 = _mm256_fmadd_ps(*a0_packFloat8, *b_packFloat8, *C_accum.v0);
    *C_accum.v1 = _mm256_fmadd_ps(*a1_packFloat8, *b_packFloat8, *C_accum.v1);

    if constexpr (sizeof...(rest) > 0) {
        fma_inner<E + 1>(blockA_packed, blockB_packed, 
                         a0_packFloat8, a1_packFloat8, 
                         b_packFloat8, kc,
                         rest...);
    }
}

template<typename... Ts>
inline void fma_loop(float* blockA_packed, float* blockB_packed,
                     __m256* a0_packFloat8, __m256* a1_packFloat8,
                     __m256* b_packFloat8, int kc,
                     Ts... rest) {

    for (int p = 0; p < kc; p++) {
        *a0_packFloat8 = _mm256_loadu_ps(blockA_packed);
        *a1_packFloat8 = _mm256_loadu_ps(blockA_packed + 8);
        fma_inner<0>(blockA_packed, blockB_packed, a0_packFloat8, a1_packFloat8, b_packFloat8, kc, rest...);
        blockA_packed += 16;
        blockB_packed += 6;
    }
}

inline static void build_masks(__m256i* packed_mask_0, __m256i* packed_mask_1, int mr) {
#if defined(__GNUC__) && (__GNUC__ < 9)
    *packed_mask_0 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i const*)&mask[16 - mr]));
    *packed_mask_1 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i const*)&mask[16 - mr + 8]));
#else
    *packed_mask_0 = _mm256_cvtepi8_epi32(_mm_loadu_si64(&mask[16 - mr]));
    *packed_mask_1 = _mm256_cvtepi8_epi32(_mm_loadu_si64(&mask[16 - mr + 8]));
#endif
}

template<int E, typename... Ts>
inline void maskload_accum_inner(float* C, int M,
                                __m256i packed_mask_0, __m256i packed_mask_1,
                                Vec256Pair C_accum, Ts... rest) {
    *C_accum.v0 = _mm256_maskload_ps(&C[E*M], packed_mask_0);
    *C_accum.v1 = _mm256_maskload_ps(&C[E*M+8], packed_mask_1);
    if constexpr(sizeof...(rest) > 0) {
        maskload_accum_inner<E+1>(C, M, packed_mask_0, packed_mask_1, rest...);
    }
}

template<typename... Ts>
inline void maskload_accum(float* C, int M, 
                           __m256i packed_mask_0, __m256i packed_mask_1, 
                           Ts... rest) {
    maskload_accum_inner<0>(C, M, packed_mask_0, packed_mask_1, rest...);
}

template<int E, typename... Ts>
inline void load_accum_inner(float* C, int M, Vec256Pair C_accum, Ts... rest) {
    *C_accum.v0 = _mm256_loadu_ps(&C[E*M]);
    *C_accum.v1 = _mm256_loadu_ps(&C[E*M+8]);
    if constexpr(sizeof...(rest) > 0) {
        load_accum_inner<E+1>(C, M, rest...);
    }
}

template<typename... Ts>
inline void load_accum(float* C, int M, Ts... rest) {
    load_accum_inner<0>(C, M, rest...);
}

template<typename Epilogue>
inline void _mm256_store(const Epilogue& epilogue, 
                         float* C, 
                         int batch, int row, int col, 
                         __m256* C_accum) {
    if constexpr (std::is_same_v<Epilogue, gops::epilogue::Passthrough>) {
        _mm256_storeu_ps(C, *C_accum); 
    } else {
        epilogue(*C_accum, batch, row, col, true);
        _mm256_storeu_ps(C, *C_accum);
    }
}

template<int E, typename Epilogue, typename... Ts>
inline void store_accum_inner(const Epilogue& epilogue, 
                              float* C, int M, 
                              int batch, int row, int col, 
                              Vec256Pair C_accum, Ts... rest) {
    _mm256_store(epilogue, &C[E*M], batch, row, col + E, C_accum.v0);
    _mm256_store(epilogue, &C[E*M+8], batch, row + 8, col + E, C_accum.v1);
    if constexpr (sizeof...(rest) > 0) {
        store_accum_inner<E+1>(epilogue, C, M, batch, row, col, rest...);
    }
}

template<typename Epilogue, typename... Ts>
inline void store_accum(const Epilogue& epilogue, 
                        float* C, int M, 
                        int batch, int row, int col, 
                        Ts... rest) {
    store_accum_inner<0>(epilogue, C, M, batch, row, col, rest...);
}

template<typename Epilogue>
inline void _mm256_maskstore(const Epilogue& epilogue, 
                             float* C, 
                             int batch, int row, int col, 
                             __m256i packed_mask, __m256* C_accum) {
    if constexpr (std::is_same_v<Epilogue, gops::epilogue::Passthrough>) {
        _mm256_maskstore_ps(C, packed_mask, *C_accum);
    } else {
        epilogue(*C_accum, batch, row, col, true); // not pass the mask, cause epilogue with condition will affect the performance, ignore the mask wont cause crash.
        _mm256_maskstore_ps(C, packed_mask, *C_accum);
    }
}

template<int E, typename Epilogue, typename... Ts>
inline void maskstore_accum_inner(const Epilogue& epilogue, 
                                  float* C, int M, 
                                  int batch, int row, int col, 
                                  __m256i packed_mask_0, __m256i packed_mask_1, 
                                  Vec256Pair C_accum, Ts... rest) {
    _mm256_maskstore(epilogue, &C[E*M], batch, row, col + E, packed_mask_0, C_accum.v0);
    _mm256_maskstore(epilogue, &C[E*M+8], batch, row + 8, col + E, packed_mask_1, C_accum.v1);
    if constexpr (sizeof...(rest) > 0) {
        maskstore_accum_inner<E+1>(epilogue, C, M, batch, row, col, packed_mask_0, packed_mask_1, rest...);
    }   
} 

template<typename Epilogue, typename... Ts>
inline void maskstore_accum(const Epilogue& epilogue, 
                            float* C, int M, 
                            int batch, int row, int col, 
                            __m256i packed_mask_0, __m256i packed_mask_1, 
                            Ts... rest) {
    maskstore_accum_inner<0>(epilogue, C, M, batch, row, col, packed_mask_0, packed_mask_1, rest...);
}

struct LoadAccumInit {};
struct ZeroAccumInit {};

template<
    int NR,
    bool Masked,
    typename InitPolicy,
    typename Epilogue,
    size_t... Is
>
inline void kernel_16x6_impl_inner(std::index_sequence<Is...>,
                                   float* blockA_packed, float* blockB_packed,
                                   float* C, int mr, int M, int kc,
                                   const Epilogue& epilogue,
                                   int batch, int row, int col) {
    __m256 C_accum[NR][2];
    __m256 b_packFloat8 = {};
    __m256 a0_packFloat8 = {};
    __m256 a1_packFloat8 = {};

    __m256i packed_mask_0 = {};
    __m256i packed_mask_1 = {};

    if constexpr (Masked) {
        build_masks(&packed_mask_0, &packed_mask_1, mr);
    }

    if constexpr (std::is_same_v<InitPolicy, ZeroAccumInit>) {
        // 全零初始化
        ((C_accum[Is][0] = _mm256_setzero_ps(),
          C_accum[Is][1] = _mm256_setzero_ps()), ...);
    } else {
        if constexpr (Masked) {
            maskload_accum(
                C, M,
                packed_mask_0, packed_mask_1,
                Vec256Pair{&C_accum[Is][0], &C_accum[Is][1]}...
            );
        } else {
            load_accum(
                C, M,
                Vec256Pair{&C_accum[Is][0], &C_accum[Is][1]}...
            );
        }
    }

    fma_loop(
        blockA_packed, blockB_packed,
        &a0_packFloat8, &a1_packFloat8,
        &b_packFloat8, kc,
        Vec256Pair{&C_accum[Is][0], &C_accum[Is][1]}...
    );

    if constexpr (Masked) {
        maskstore_accum(
            epilogue, C, M, batch, row, col,
            packed_mask_0, packed_mask_1,
            Vec256Pair{&C_accum[Is][0], &C_accum[Is][1]}...
        );
    } else {
        store_accum(
            epilogue, C, M, batch, row, col,
            Vec256Pair{&C_accum[Is][0], &C_accum[Is][1]}...
        );
    }
}

template<
    int NR,
    typename InitPolicy,
    typename Epilogue
>
inline void kernel_16x6_dispatch(float* blockA_packed, float* blockB_packed,
                                 float* C, int mr, int M, int kc,
                                 const Epilogue& epilogue,
                                 int batch, int row, int col) {
    if (mr != 16) {
        kernel_16x6_impl_inner<
            NR, true, InitPolicy
        >(
            std::make_index_sequence<NR>{},
            blockA_packed, blockB_packed,
            C, mr, M, kc, epilogue,
            batch, row, col
        );
    } else {
        kernel_16x6_impl_inner<
            NR, false, InitPolicy
        >(
            std::make_index_sequence<NR>{},
            blockA_packed, blockB_packed,
            C, mr, M, kc, epilogue,
            batch, row, col
        );
    }
}

#define KERNEL_16X6_ARGS \
    blockA_packed, blockB_packed, C, \
    mr, M, kc, epilogue, batch, row, col

template<typename InitPolicy, typename Epilogue>
void kernel_16x6(float* blockA_packed, float* blockB_packed,
                 float* C, int mr, int nr, int kc, int M,
                 const Epilogue& epilogue,
                 int batch, int row, int col) {
    switch (nr) {
    case 1: kernel_16x6_dispatch<1, InitPolicy>(KERNEL_16X6_ARGS); break;
    case 2: kernel_16x6_dispatch<2, InitPolicy>(KERNEL_16X6_ARGS); break;
    case 3: kernel_16x6_dispatch<3, InitPolicy>(KERNEL_16X6_ARGS); break;
    case 4: kernel_16x6_dispatch<4, InitPolicy>(KERNEL_16X6_ARGS); break;
    case 5: kernel_16x6_dispatch<5, InitPolicy>(KERNEL_16X6_ARGS); break;
    case 6: kernel_16x6_dispatch<6, InitPolicy>(KERNEL_16X6_ARGS); break;
    }
}

} // namespace kernel
} // namespace gops