#pragma once

#include "cpu_patch/kernel.h"
#include "cpu_patch/epilogue.h"
#include <algorithm>
#include <type_traits>

#ifndef NTHREADS
    #define NTHREADS 16
#endif

#define MC (16 * (40 / NTHREADS) * NTHREADS)
#define NC (6 * (800 / NTHREADS) * NTHREADS)
#define KC 500

#ifndef OMP_SCHEDULE
    #define OMP_SCHEDULE auto
#endif

#define PRAGMA_OMP_PARALLEL_FOR _Pragma("omp parallel for schedule(OMP_SCHEDULE) num_threads(NTHREADS)")

namespace gops {

// currently support column-major
template <typename ElementT, 
          typename ElementComputeT,
          typename Epilogue>
struct GemmWithVariadic {
    static_assert(std::is_same_v<ElementT, float> && std::is_same_v<ElementComputeT, float>, "only support fp32 type");

    inline static ElementT blockA_packed[MC * KC] __attribute__((aligned(64))) = {};
    inline static ElementT blockB_packed[NC * KC] __attribute__((aligned(64))) = {};

    struct Argument {
        Argument(const void* p_a, 
                 const void* p_b, 
                 void* p_c,
                 int batch,
                 int M, 
                 int N,
                 int K,
                 int batch_stride_a,
                 int batch_stride_b,
                 int batch_stride_c,
                 Epilogue epilogue)
                : p_a_{static_cast<const ElementT*>(p_b)}, // adapt to column-major kernel
                  p_b_{static_cast<const ElementT*>(p_a)}, // adapt to column-major kernel
                  p_c_{static_cast<ElementT*>(p_c)},
                  batch_{batch},
                  M_{N}, // adapt to column-major kernel
                  N_{M}, // adapt to column-major kernel
                  K_{K},
                  batch_stride_a_{batch_stride_b}, // adapt to column-major kernel
                  batch_stride_b_{batch_stride_a}, // adapt to column-major kernel
                  batch_stride_c_{batch_stride_c},
                  epilogue_{epilogue} {}
        
        const ElementT* p_a_;
        const ElementT* p_b_;
        ElementT* p_c_;
        int batch_;
        int M_;
        int N_;
        int K_;
        int batch_stride_a_;
        int batch_stride_b_;
        int batch_stride_c_;
        Epilogue epilogue_;
    };

private:
    void pack_panelB(const ElementT* B, ElementT* blockB_packed, int nr, int kc, int K) {
        for (int p = 0; p < kc; p++) {
            for (int j = 0; j < nr; j++) {
                *blockB_packed++ = B[j * K + p];
            }
            for (int j = nr; j < 6; j++) {
                *blockB_packed++ = 0;
            }
        }
    }

    void pack_blockB(const ElementT* B, ElementT* blockB_packed, int nc, int kc, int K) {
        PRAGMA_OMP_PARALLEL_FOR
        for (int j = 0; j < nc; j += 6) {
            int nr = std::min(6, nc - j);
            pack_panelB(&B[j * K], &blockB_packed[j * kc], nr, kc, K);
        }
    }

    void pack_panelA(const ElementT* A, ElementT* blockA_packed, int mr, int kc, int M) {
        for (int p = 0; p < kc; p++) {
            for (int i = 0; i < mr; i++) {
                *blockA_packed++ = A[p * M + i];
            }
            for (int i = mr; i < 16; i++) {
                *blockA_packed++ = 0;
            }
        }
    }

    void pack_blockA(const ElementT* A, ElementT* blockA_packed, int mc, int kc, int M) {
        PRAGMA_OMP_PARALLEL_FOR
        for (int i = 0; i < mc; i += 16) {
            int mr = std::min(16, mc - i);
            pack_panelA(&A[i], &blockA_packed[i * kc], mr, kc, M);
        }
    }

public:
    #define COMPUTE_MNBLOCK(kernel, epilogue)              \
        PRAGMA_OMP_PARALLEL_FOR                            \
        for (int jr = 0; jr < nc; jr += 6) {               \
            int nr = std::min(6, nc - jr);                 \
            for (int ir = 0; ir < mc; ir += 16) {          \
                int mr = std::min(16, mc - ir);            \
                kernel(&blockA_packed[ir * kc],            \
                    &blockB_packed[jr * kc],               \
                    &C[(j + jr) * M + (i + ir)],           \
                    mr,                                    \
                    nr,                                    \
                    kc,                                    \
                    M,                                     \
                    epilogue,                              \
                    b,                                     \
                    i + ir,                                \
                    j + jr);                               \
            }                                              \
        }

    void run(const Argument& arg) {

        // The function computes C[M x N] = A[M x K] @ B[K x N]
        // All operands are stored in column-major format, with lda=M, ldb=K, ldc=M

        gops::epilogue::Passthrough pass;

        int batch = arg.batch_;
        int N = arg.N_;
        int M = arg.M_;
        int K = arg.K_;
        const ElementT* A = arg.p_a_;
        const ElementT* B = arg.p_b_;
        ElementT* C = arg.p_c_;

        using LoadAccumInit = gops::kernel::LoadAccumInit;
        using ZeroAccumInit = gops::kernel::ZeroAccumInit;

        for(int b = 0; b < batch; b++) {
            for (int j = 0; j < N; j += NC) {
                int nc = std::min(NC, N - j);
                int kc = std::min(KC, K);
                pack_blockB(&B[j * K], blockB_packed, nc, kc, K);
                for (int i = 0; i < M; i += MC) {
                    int mc = std::min(MC, M - i);
                    pack_blockA(&A[i], blockA_packed, mc, kc, M);
                    if(K <= KC) {
                        COMPUTE_MNBLOCK(gops::kernel::kernel_16x6<ZeroAccumInit>, arg.epilogue_);
                    } else {
                        COMPUTE_MNBLOCK(gops::kernel::kernel_16x6<ZeroAccumInit>, pass);
                    }
                }
                for (int p = kc; p < K; p += KC) {
                    int kc = std::min(KC, K - p);
                    pack_blockB(&B[j * K + p], blockB_packed, nc, kc, K);
                    for (int i = 0; i < M; i += MC) {
                        int mc = std::min(MC, M - i);
                        pack_blockA(&A[p * M + i], blockA_packed, mc, kc, M);
                        if(p + KC >= K) {
                            COMPUTE_MNBLOCK(gops::kernel::kernel_16x6<LoadAccumInit>, arg.epilogue_);
                        } else {
                            COMPUTE_MNBLOCK(gops::kernel::kernel_16x6<LoadAccumInit>, pass);
                        }
                    }
                }
            }
            A += arg.batch_stride_a_;
            B += arg.batch_stride_b_;
            C += arg.batch_stride_c_;
        }
    }
};

}; // namespace gops

