// Copyright (c) 2023 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ============================================================================
#pragma once
#ifndef MATMUL_HELPER_H
#define MATMUL_HELPER_H
#include "../common/bfloat16.h"
#include "../common/float16.h"
#include "../common/my_types.h"
#include "../common/normal_float4x2.h"
#include "../common/uint4x2.h"
// #include "../common/transformer_ctx.h"
#include "split_util.h"
#include "third_party/xdnn/xdnn.h"

#include <cstring>
#include <map>
#include <tuple>
#include "glog/logging.h"

#define USE_AMX_M 8
#ifdef AVX512_FP16_WEIGHT_ONLY_INT8
#undef AVX512_FP16_WEIGHT_ONLY_INT8
#endif
#ifndef AVX512_FP32_WEIGHT_ONLY_INT8
#define AVX512_FP32_WEIGHT_ONLY_INT8
#endif

class MMHelper {
public:
    // Pack the MatMul weight from 'src(rows, cols)' to 'weight'
    // trans: 'src' is transposed or not
    // verticalSplit: vertical split or horizontal split, vertical vs. horizontal:
    //  _________________________            _________________________
    // |            |            |          |                         |
    // |            |            |          |_________________________|
    // |            |            |          |                         |
    // |____________|____________|          |_________________________|
    //           vertical                            horizontal
    //
    // ****************************************************************************
    //
    // Vertical split like the left one, but if transposed, like the right one
    //      |<-------- cols ----------|           |<-------- rows ----------|
    //  _    _________________________        _    _________________________
    //  ↑   |            |            |       ↑   |                         |
    //  |   |            |            |       |   |_________________________|
    // rows |            |            |      cols |                         |
    //  ↓   |____________|____________|       ↓   |_________________________|
    //             not_transposed                          transposed
    //
    // ****************************************************************************
    //
    // Horizontal split like the right one, but if transposed, like the left one
    //      |<-------- rows ----------|           |<-------- cols ----------|
    //  _    _________________________        _    _________________________
    //  ↑   |            |            |       ↑   |                         |
    //  |   |            |            |       |   |_________________________|
    // cols |            |            |      rows |                         |
    //  ↓   |____________|____________|       ↓   |_________________________|
    //               transposed                          not_transposed
    //
    template <typename WeiT>
    static void convertWeight(bool trans, int rows, int cols, const float *src, int splitOffset, int splitSize,
            bool verticalSplit, hpj::Matrix<WeiT> &quantizedWeight, hpj::Vector<float> &scaleWeight,
            hpj::Vector<float> &zeroWeight, bool unused) {
        // FP32 transpose
        if constexpr (std::is_same_v<WeiT, float>) {
            if (verticalSplit) {
                int colsPerSplit = splitSize;
                if (trans) {
                    quantizedWeight.Resize(colsPerSplit, rows);
                    const float *base = src + splitOffset * quantizedWeight.Stride();
#pragma omp parallel for
                    for (int i = 0; i < quantizedWeight.Rows(); ++i) {
                        memcpy(quantizedWeight.Data() + i * quantizedWeight.Stride(), base + i * rows,
                                quantizedWeight.Cols());
                    }
                } else {
                    quantizedWeight.Resize(rows, colsPerSplit);
#pragma omp parallel for
                    for (int i = 0; i < quantizedWeight.Rows(); ++i) {
                        memcpy(quantizedWeight.Data() + i * quantizedWeight.Stride(), src + i * cols + splitOffset,
                                quantizedWeight.Cols());
                    }
                }
            } else {
                int rowsPerSplit = splitSize;
                if (trans) {
                    quantizedWeight.Resize(cols, rowsPerSplit);
#pragma omp parallel for
                    for (int i = 0; i < quantizedWeight.Rows(); ++i) {
                        memcpy(quantizedWeight.Data() + i * quantizedWeight.Stride(), src + i * rows + splitOffset,
                                quantizedWeight.Cols());
                    }
                } else {
                    quantizedWeight.Resize(rowsPerSplit, cols);
                    const float *base = src + splitOffset * quantizedWeight.Stride();
#pragma omp parallel for
                    for (int i = 0; i < quantizedWeight.Rows(); ++i) {
                        memcpy(quantizedWeight.Data() + i * quantizedWeight.Stride(), base + i * cols,
                                quantizedWeight.Cols());
                    }
                }
            }
        }

        // FP32 -> FP16
        else if constexpr (std::is_same_v<WeiT, float16_t>) {
            if (verticalSplit) {
                int colsPerSplit = splitSize;
                if (trans) { // right
                    quantizedWeight.Resize(colsPerSplit, rows);
                    const float *base = src + splitOffset * quantizedWeight.Stride();
#pragma omp parallel for
                    for (int i = 0; i < quantizedWeight.Rows(); ++i) {
                        float16_t::cvt_float_to_float16(base + i * rows,
                                quantizedWeight.Data() + i * quantizedWeight.Stride(), quantizedWeight.Cols());
                    }
                } else {
                    quantizedWeight.Resize(rows, colsPerSplit);
#pragma omp parallel for
                    for (int i = 0; i < quantizedWeight.Rows(); ++i) {
                        float16_t::cvt_float_to_float16(src + i * cols + splitOffset,
                                quantizedWeight.Data() + i * quantizedWeight.Stride(), quantizedWeight.Cols());
                    }
                }
            } else {
                int rowsPerSplit = splitSize;
                if (trans) {
                    quantizedWeight.Resize(cols, rowsPerSplit);
#pragma omp parallel for
                    for (int i = 0; i < quantizedWeight.Rows(); ++i) {
                        float16_t::cvt_float_to_float16(src + i * rows + splitOffset,
                                quantizedWeight.Data() + i * quantizedWeight.Stride(), quantizedWeight.Cols());
                    }
                } else {
                    quantizedWeight.Resize(rowsPerSplit, cols);
                    const float *base = src + splitOffset * quantizedWeight.Stride();
#pragma omp parallel for
                    for (int i = 0; i < quantizedWeight.Rows(); ++i) {
                        float16_t::cvt_float_to_float16(base + i * cols,
                                quantizedWeight.Data() + i * quantizedWeight.Stride(), quantizedWeight.Cols());
                    }
                }
            }
        }

        // FP32 -> BF16
        else if constexpr (std::is_same_v<WeiT, bfloat16_t>) {
            if (verticalSplit) {
                int colsPerSplit = splitSize;
                if (trans) { // right
                    quantizedWeight.Resize(colsPerSplit, rows);
                    const float *base = src + splitOffset * quantizedWeight.Stride();
#pragma omp parallel for
                    for (int i = 0; i < quantizedWeight.Rows(); ++i) {
                        for (int j = 0; j < quantizedWeight.Cols(); ++j) {
                            quantizedWeight.Data()[i * quantizedWeight.Stride() + j] = bfloat16_t(base[i * rows + j]);
                        }
                    }
                } else {
                    quantizedWeight.Resize(rows, colsPerSplit);
#pragma omp parallel for
                    for (int i = 0; i < quantizedWeight.Rows(); ++i) {
                        for (int j = 0; j < quantizedWeight.Cols(); ++j) {
                            quantizedWeight.Data()[i * quantizedWeight.Stride() + j]
                                    = bfloat16_t(src[i * cols + splitOffset + j]);
                        }
                    }
                }
            } else {
                int rowsPerSplit = splitSize;
                if (trans) {
                    quantizedWeight.Resize(cols, rowsPerSplit);
#pragma omp parallel for
                    for (int i = 0; i < quantizedWeight.Rows(); ++i) {
                        for (int j = 0; j < quantizedWeight.Cols(); ++j) {
                            quantizedWeight.Data()[i * quantizedWeight.Stride() + j]
                                    = bfloat16_t(src[i * rows + splitOffset + j]);
                        }
                    }
                } else {
                    quantizedWeight.Resize(rowsPerSplit, cols);
                    const float *base = src + splitOffset * quantizedWeight.Stride();
#pragma omp parallel for
                    for (int i = 0; i < quantizedWeight.Rows(); ++i) {
                        for (int j = 0; j < quantizedWeight.Cols(); ++j) {
                            quantizedWeight.Data()[i * quantizedWeight.Stride() + j] = bfloat16_t(base[i * cols + j]);
                        }
                    }
                }
            }
        }

        // FP32 -> INT8
        else if constexpr (std::is_same_v<WeiT, int8_t>) {
            if (verticalSplit) {
                int colsPerSplit = splitSize;
                if (trans) {
                    quantizedWeight.Resize(colsPerSplit, rows);
                    scaleWeight.Resize(colsPerSplit);
                    zeroWeight.Resize(colsPerSplit);
                } else {
                    quantizedWeight.Resize(rows, colsPerSplit);
                    scaleWeight.Resize(colsPerSplit);
                    zeroWeight.Resize(colsPerSplit);
                }
#ifdef AVX512_FP32_WEIGHT_ONLY_INT8
                xdnn_sgemm_f32s8f32_quantize(trans, colsPerSplit, rows,
                        trans ? (src + rows * splitOffset) : (src + splitOffset), trans ? rows : cols, 0.9999f,
                        quantizedWeight.Data(), trans ? rows : colsPerSplit, scaleWeight.Data(), zeroWeight.Data());
#elif defined(AVX512_FP16_WEIGHT_ONLY_INT8)
                xdnn_hgemm_f32s8f32_quantize(trans, colsPerSplit, rows,
                        trans ? (src + rows * splitOffset) : (src + splitOffset), trans ? rows : cols, 0.9999f,
                        quantizedWeight.Data(), trans ? rows : colsPerSplit, scaleWeight.Data(), zeroWeight.Data());
#else
                printf("%s:%d: Need to define WEIGHT_ONLY_INT8 kernel data type.\n", __FILE__, __LINE__);
                exit(-1);
#endif
            } else {
                int rowsPerSplit = splitSize;
                if (trans) {
                    quantizedWeight.Resize(cols, rowsPerSplit);
                    scaleWeight.Resize(cols);
                    zeroWeight.Resize(cols);
                } else {
                    quantizedWeight.Resize(rowsPerSplit, cols);
                    scaleWeight.Resize(cols);
                    zeroWeight.Resize(cols);
                }
#ifdef AVX512_FP32_WEIGHT_ONLY_INT8
                xdnn_sgemm_f32s8f32_quantize(trans, cols, rowsPerSplit,
                        trans ? (src + splitOffset) : (src + splitOffset * cols), trans ? rows : cols, 0.9999f,
                        quantizedWeight.Data(), trans ? rowsPerSplit : cols, scaleWeight.Data(), zeroWeight.Data());
#elif defined(AVX512_FP16_WEIGHT_ONLY_INT8)
                xdnn_hgemm_f32s8f32_quantize(trans, cols, rowsPerSplit,
                        trans ? (src + splitOffset) : (src + splitOffset * cols), trans ? rows : cols, 0.9999f,
                        quantizedWeight.Data(), trans ? rowsPerSplit : cols, scaleWeight.Data(), zeroWeight.Data());
#else
                printf("%s:%d: Need to define WEIGHT_ONLY_INT8 kernel data type.\n", __FILE__, __LINE__);
                exit(-1);
#endif
            }
        }

        // FP32 -> UINT4
        else if constexpr (std::is_same_v<WeiT, uint4x2_t>) {
            if (verticalSplit) {
                int colsPerSplit = splitSize;
                if (trans) {
                    quantizedWeight.Resize(colsPerSplit, rows);
                    scaleWeight.Resize(colsPerSplit);
                    zeroWeight.Resize(colsPerSplit);
                } else {
                    quantizedWeight.Resize(rows, colsPerSplit);
                    scaleWeight.Resize(colsPerSplit);
                    zeroWeight.Resize(colsPerSplit);
                }
#ifdef AVX512_FP32_WEIGHT_ONLY_INT4
                xdnn_sgemm_f32u4f32_quantize(trans, colsPerSplit, rows,
                        trans ? (src + rows * splitOffset) : (src + splitOffset), trans ? rows : cols, 0.9999f,
                        (XDNN_UINT4x2 *)quantizedWeight.Data(), trans ? rows : colsPerSplit, scaleWeight.Data(),
                        zeroWeight.Data());
#elif defined(AVX512_FP16_WEIGHT_ONLY_INT4)
                xdnn_hgemm_f32u4f32_quantize(trans, colsPerSplit, rows,
                        trans ? (src + rows * splitOffset) : (src + splitOffset), trans ? rows : cols, 0.9999f,
                        (XDNN_UINT4x2 *)quantizedWeight.Data(), trans ? rows : colsPerSplit, scaleWeight.Data(),
                        zeroWeight.Data());
#else
                printf("%s:%d: Need to define WEIGHT_ONLY_INT4 kernel data type.\n", __FILE__, __LINE__);
                exit(-1);
#endif
            } else {
                int rowsPerSplit = splitSize;
                if (trans) {
                    quantizedWeight.Resize(cols, rowsPerSplit);
                    scaleWeight.Resize(cols);
                    zeroWeight.Resize(cols);
                } else {
                    quantizedWeight.Resize(rowsPerSplit, cols);
                    scaleWeight.Resize(cols);
                    zeroWeight.Resize(cols);
                }
#ifdef AVX512_FP32_WEIGHT_ONLY_INT4
                xdnn_sgemm_f32u4f32_quantize(trans, cols, rowsPerSplit,
                        trans ? (src + splitOffset) : (src + splitOffset * cols), trans ? rows : cols, 0.9999f,
                        (XDNN_UINT4x2 *)quantizedWeight.Data(), trans ? rowsPerSplit : cols, scaleWeight.Data(),
                        zeroWeight.Data());
#elif defined(AVX512_FP16_WEIGHT_ONLY_INT4)
                xdnn_hgemm_f32u4f32_quantize(trans, cols, rowsPerSplit,
                        trans ? (src + splitOffset) : (src + splitOffset * cols), trans ? rows : cols, 0.9999f,
                        (XDNN_UINT4x2 *)quantizedWeight.Data(), trans ? rowsPerSplit : cols, scaleWeight.Data(),
                        zeroWeight.Data());
#else
                printf("%s:%d: Need to define WEIGHT_ONLY_INT4 kernel data type.\n", __FILE__, __LINE__);
                exit(-1);
#endif
            }
        }

        // FP32 -> NF4
        else if constexpr (std::is_same_v<WeiT, nf4x2_t>) {
            if (verticalSplit) {
                int colsPerSplit = splitSize;
                if (trans) {
                    quantizedWeight.Resize(colsPerSplit, rows);
                    scaleWeight.Resize(colsPerSplit);
                    zeroWeight.Resize(colsPerSplit);
                } else {
                    quantizedWeight.Resize(rows, colsPerSplit);
                    scaleWeight.Resize(colsPerSplit);
                    zeroWeight.Resize(colsPerSplit);
                }
#ifdef AVX512_FP32_WEIGHT_ONLY_NF4
                xdnn_sgemm_f32nf4f32_quantize(trans, colsPerSplit, rows,
                        trans ? (src + rows * splitOffset) : (src + splitOffset), trans ? rows : cols, 0.9999f,
                        (XDNN_NF4x2 *)quantizedWeight.Data(), trans ? rows : colsPerSplit, scaleWeight.Data(),
                        zeroWeight.Data());
#elif defined(AVX512_FP16_WEIGHT_ONLY_NF4)
                xdnn_hgemm_f32nf4f32_quantize(trans, colsPerSplit, rows,
                        trans ? (src + rows * splitOffset) : (src + splitOffset), trans ? rows : cols, 0.9999f,
                        (XDNN_NF4x2 *)quantizedWeight.Data(), trans ? rows : colsPerSplit, scaleWeight.Data(),
                        zeroWeight.Data());
#else
                printf("%s:%d: Need to define WEIGHT_ONLY_NF4 kernel data type.\n", __FILE__, __LINE__);
                exit(-1);
#endif
            } else {
                int rowsPerSplit = splitSize;
                if (trans) {
                    quantizedWeight.Resize(cols, rowsPerSplit);
                    scaleWeight.Resize(cols);
                    zeroWeight.Resize(cols);
                } else {
                    quantizedWeight.Resize(rowsPerSplit, cols);
                    scaleWeight.Resize(cols);
                    zeroWeight.Resize(cols);
                }
#ifdef AVX512_FP32_WEIGHT_ONLY_NF4
                xdnn_sgemm_f32nf4f32_quantize(trans, cols, rowsPerSplit,
                        trans ? (src + splitOffset) : (src + splitOffset * cols), trans ? rows : cols, 0.9999f,
                        (XDNN_NF4x2 *)quantizedWeight.Data(), trans ? rowsPerSplit : cols, scaleWeight.Data(),
                        zeroWeight.Data());
#elif defined(AVX512_FP16_WEIGHT_ONLY_NF4)
                xdnn_hgemm_f32nf4f32_quantize(trans, cols, rowsPerSplit,
                        trans ? (src + splitOffset) : (src + splitOffset * cols), trans ? rows : cols, 0.9999f,
                        (XDNN_NF4x2 *)quantizedWeight.Data(), trans ? rowsPerSplit : cols, scaleWeight.Data(),
                        zeroWeight.Data());
#else
                printf("%s:%d: Need to define WEIGHT_ONLY_NF4 kernel data type.\n", __FILE__, __LINE__);
                exit(-1);
#endif
            }
        }
    }

    template <typename WeiT>
    static void convertWeight(bool trans, int rows, int cols, const float *src, int numSplit, int splitIdx,
            bool verticalSplit, hpj::Matrix<WeiT> &quantizedWeight, hpj::Vector<float> &scaleWeight,
            hpj::Vector<float> &zeroWeight) {
        int totalSize = verticalSplit ? cols : rows;
        std::pair<int, int> range = SplitUtil::getTaskRange(totalSize, numSplit, splitIdx);

        int splitSize = range.second - range.first;
        int splitOffset = range.first;

        convertWeight(trans, rows, cols, src, splitOffset, splitSize, verticalSplit, quantizedWeight, scaleWeight,
                zeroWeight, true);
    }

    template <typename WeiT>
    static void convertWeight(bool trans, int rows, int cols, const float *src, hpj::Matrix<WeiT> &quantizedWeight,
            hpj::Vector<float> &scaleWeight, hpj::Vector<float> &zeroWeight) {
        convertWeight(trans, rows, cols, src, 1, 0, true, quantizedWeight, scaleWeight, zeroWeight);
    }

//     template <typename WeiT>
//     static void convertWeight(DecoderContext *ctx, bool trans, int rows, int cols, const float *src, bool verticalSplit,
//             hpj::Matrix<WeiT> &quantizedWeight, hpj::Vector<float> &scaleWeight, hpj::Vector<float> &zeroWeight) {
//         convertWeight(trans, rows, cols, src, ctx->numSplit, ctx->splitIdx, verticalSplit, quantizedWeight, scaleWeight,
//                 zeroWeight);
//     }

    template <typename WeiT>
    static void packWeight(bool trans, hpj::Matrix<WeiT> &src, hpj::Matrix<WeiT> &weight) {
        int K = trans ? src.Cols() : src.Rows();
        int N = trans ? src.Rows() : src.Cols();

        // FP32
        if constexpr (std::is_same_v<WeiT, float>) {
            weight.Resize(K, N);
            xdnn_sgemm_packb(trans, N, K, src.Data(), src.Stride(), weight.Data());
        }

        // FP16
        else if constexpr (std::is_same_v<WeiT, float16_t>) {
            weight.Resize(K, N);
#ifdef AVX512_FP32_WEIGHT_ONLY_FP16
            xdnn_sgemm_f32f16f32_packb(
                    trans, N, K, (const XDNN_FP16 *)src.Data(), src.Stride(), (XDNN_FP16 *)weight.Data());
#elif defined(AVX512_FP16_WEIGHT_ONLY_FP16)
            xdnn_hgemm_f32f16f32_packb(
                    trans, N, K, (const XDNN_FP16 *)src.Data(), src.Stride(), (XDNN_FP16 *)weight.Data());
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_FP16 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // BF16
        else if constexpr (std::is_same_v<WeiT, bfloat16_t>) {
        //     set_amx_data_type(dnnl::memory::format_tag::BA16a64b2a);
            int amx_rows = (int)((K + 15) / 16) * 16;
            int amx_cols = (int)((N + 63) / 64) * 64;
            weight.Resize(amx_rows, amx_cols);
            memset(weight.Data(), 0, amx_rows * amx_cols * sizeof(bfloat16_t));
#ifdef AVX512_FP32_WEIGHT_ONLY_BF16
            xdnn_sgemm_f32bf16f32_packb(
                    trans, N, K, (const XDNN_BF16 *)src.Data(), src.Stride(), (XDNN_BF16 *)weight.Data(), 16, 64);
#elif defined(AVX512_BF16_WEIGHT_ONLY_BF16)
            xdnn_bgemm_f32bf16f32_packb(
                    trans, N, K, (const XDNN_BF16 *)src.Data(), src.Stride(), (XDNN_BF16 *)weight.Data(), 16, 64);
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_BF16 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // INT8
        else if constexpr (std::is_same_v<WeiT, int8_t>) {
            weight.Resize(K, N);
#ifdef AVX512_FP32_WEIGHT_ONLY_INT8
            xdnn_sgemm_f32s8f32_packb(trans, N, K, src.Data(), src.Stride(), weight.Data());
#elif defined(AVX512_FP16_WEIGHT_ONLY_INT8)
            xdnn_hgemm_f32s8f32_packb(trans, N, K, src.Data(), src.Stride(), weight.Data());
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_INT8 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // INT4
        else if constexpr (std::is_same_v<WeiT, uint4x2_t>) {
            weight.Resize(K, N);
#ifdef AVX512_FP32_WEIGHT_ONLY_INT4
            xdnn_sgemm_f32u4f32_packb(
                    trans, N, K, (const XDNN_UINT4x2 *)src.Data(), src.Stride(), (XDNN_UINT4x2 *)weight.Data());
#elif defined(AVX512_FP16_WEIGHT_ONLY_INT4)
            xdnn_hgemm_f32u4f32_packb(
                    trans, N, K, (const XDNN_UINT4x2 *)src.Data(), src.Stride(), (XDNN_UINT4x2 *)weight.Data());
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_INT4 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // NF4
        else if constexpr (std::is_same_v<WeiT, nf4x2_t>) {
            weight.Resize(K, N);
#ifdef AVX512_FP32_WEIGHT_ONLY_NF4
            xdnn_sgemm_f32nf4f32_packb(
                    trans, N, K, (const XDNN_NF4x2 *)src.Data(), src.Stride(), (XDNN_NF4x2 *)weight.Data());
#elif defined(AVX512_FP16_WEIGHT_ONLY_NF4)
            xdnn_hgemm_f32nf4f32_packb(
                    trans, N, K, (const XDNN_NF4x2 *)src.Data(), src.Stride(), (XDNN_NF4x2 *)weight.Data());
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_NF4 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }
    }

    template <typename InT, typename WeiT, typename OutT>
    static void compute(bool transA, int M, int N, int K, float alpha, const InT *A, int lda, const WeiT *packedB,
            const float *scaleB, const float *zeroB, float beta, OutT *C, int ldc) {
        // FP32
        if constexpr (std::is_same_v<WeiT, float>) {
            // Timline t("xdnn_sgemm_compute");
            xdnn_sgemm_compute(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc);
        }

        // FP16
        else if constexpr (std::is_same_v<WeiT, float16_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_FP16
            // Timline t("xdnn_sgemm_f32f16f32_compute");
            xdnn_sgemm_f32f16f32_compute(transA, M, N, K, alpha, A, lda, (const XDNN_FP16 *)packedB, beta, C, ldc);
#elif defined(AVX512_FP16_WEIGHT_ONLY_FP16)
            // Timline t("xdnn_hgemm_f32f16f32_compute");
            xdnn_hgemm_f32f16f32_compute(transA, M, N, K, alpha, A, lda, (const XDNN_FP16 *)packedB, beta, C, ldc);
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_FP16 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // BF16
        else if constexpr (std::is_same_v<WeiT, bfloat16_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_BF16
            // Timline t("xdnn_sgemm_f32bf16f32_compute");
            xdnn_sgemm_f32bf16f32_compute(transA, M, N, K, alpha, A, lda, (const XDNN_UINT4x2 *)packedB, beta, C, ldc);
#elif defined(AVX512_BF16_WEIGHT_ONLY_BF16)
            if (M > USE_AMX_M) {
                // Timline t("onednn_amx_sgemm_f32bf16f32_compute");
                onednn_amx_sgemm_f32bf16f32_compute(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc);
            } else {
                // Timline t("xdnn_bgemm_f32bf16f32_compute");
                xdnn_bgemm_f32bf16f32_compute(transA, M, N, K, alpha, A, lda, (const XDNN_BF16 *)packedB, beta, C, ldc);
            }
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_BF16 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // INT8
        else if constexpr (std::is_same_v<WeiT, int8_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_INT8
            VLOG(1) << "【LOG】xdnn_sgemm_f32s8f32_compute: \n";
            // Timline t("xdnn_sgemm_f32s8f32_compute");
            xdnn_sgemm_f32s8f32_compute(transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc);
#elif defined(AVX512_FP16_WEIGHT_ONLY_INT8)
            VLOG(1) << "【LOG】xdnn_hgemm_f32s8f32_compute: \n";
            // Timline t("xdnn_hgemm_f32s8f32_compute");
            xdnn_hgemm_f32s8f32_compute(transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc);
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_INT8 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // INT4
        else if constexpr (std::is_same_v<WeiT, uint4x2_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_INT4
            VLOG(1) << "【LOG】xdnn_sgemm_f32u4f32_compute: \n";
            // Timline t("xdnn_sgemm_f32u4f32_compute");
            xdnn_sgemm_f32u4f32_compute(
                    transA, M, N, K, alpha, A, lda, (const XDNN_UINT4x2 *)packedB, scaleB, zeroB, beta, C, ldc);
#elif defined(AVX512_FP16_WEIGHT_ONLY_INT4)
            VLOG(1) << "【LOG】xdnn_hgemm_f32u4f32_compute: \n";
            // Timline t("xdnn_hgemm_f32u4f32_compute");
            xdnn_hgemm_f32u4f32_compute(
                    transA, M, N, K, alpha, A, lda, (const XDNN_UINT4x2 *)packedB, scaleB, zeroB, beta, C, ldc);
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_INT4 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // NF4
        else if constexpr (std::is_same_v<WeiT, nf4x2_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_NF4
            // Timline t("xdnn_sgemm_f32nf4f32_compute");
            xdnn_sgemm_f32nf4f32_compute(
                    transA, M, N, K, alpha, A, lda, (const XDNN_NF4x2 *)packedB, scaleB, zeroB, beta, C, ldc);
#elif defined(AVX512_FP16_WEIGHT_ONLY_NF4)
            // Timline t("xdnn_hgemm_f32nf4f32_compute");
            xdnn_hgemm_f32nf4f32_compute(
                    transA, M, N, K, alpha, A, lda, (const XDNN_NF4x2 *)packedB, scaleB, zeroB, beta, C, ldc);
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_NF4 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }
    }

    template <typename InT, typename WeiT, typename OutT>
    static void compute_bias(bool transA, int M, int N, int K, float alpha, const InT *A, int lda, const WeiT *packedB,
            const float *scaleB, const float *zeroB, float beta, OutT *C, int ldc, const float *bias) {
        // FP32
        if constexpr (std::is_same_v<WeiT, float>) {
            // Timline t("xdnn_sgemm_compute_biasadd");
            xdnn_sgemm_compute_biasadd(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias);
        }

        // FP16
        else if constexpr (std::is_same_v<WeiT, float16_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_FP16
            // Timline t("xdnn_sgemm_f32f16f32_compute_biasadd");
            xdnn_sgemm_f32f16f32_compute_biasadd(
                    transA, M, N, K, alpha, A, lda, (const XDNN_FP16 *)packedB, beta, C, ldc, bias);
#elif defined(AVX512_FP16_WEIGHT_ONLY_FP16)
            // Timline t("xdnn_hgemm_f32f16f32_compute_biasadd");
            xdnn_hgemm_f32f16f32_compute_biasadd(
                    transA, M, N, K, alpha, A, lda, (const XDNN_FP16 *)packedB, beta, C, ldc, bias);
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_FP16 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // BF16
        else if constexpr (std::is_same_v<WeiT, bfloat16_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_BF16
            // Timline t("xdnn_sgemm_f32bf16f32_compute_biasadd");
            xdnn_sgemm_f32bf16f32_compute_biasadd(
                    transA, M, N, K, alpha, A, lda, (const XDNN_UINT4x2 *)packedB, beta, C, ldc, bias);
#elif defined(AVX512_BF16_WEIGHT_ONLY_BF16)
            if (M > USE_AMX_M) {
                // Timline t("onednn_amx_sgemm_f32bf16f32_compute_biasadd");
                onednn_amx_sgemm_f32bf16f32_compute_biasadd(
                        transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias);
            } else {
                // Timline t("xdnn_bgemm_f32bf16f32_compute_biasadd");
                xdnn_bgemm_f32bf16f32_compute_biasadd(
                        transA, M, N, K, alpha, A, lda, (const XDNN_BF16 *)packedB, beta, C, ldc, bias);
            }
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_BF16 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // INT8
        else if constexpr (std::is_same_v<WeiT, int8_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_INT8
            // Timline t("xdnn_sgemm_f32s8f32_compute_biasadd");
            xdnn_sgemm_f32s8f32_compute_biasadd(
                    transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc, bias);
#elif defined(AVX512_FP16_WEIGHT_ONLY_INT8)
            // Timline t("xdnn_hgemm_f32s8f32_compute_biasadd");
            xdnn_hgemm_f32s8f32_compute_biasadd(
                    transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc, bias);
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_INT8 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // INT4
        else if constexpr (std::is_same_v<WeiT, uint4x2_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_INT4
            // Timline t("xdnn_sgemm_f32u4f32_compute_biasadd");
            xdnn_sgemm_f32u4f32_compute_biasadd(
                    transA, M, N, K, alpha, A, lda, (const XDNN_UINT4x2 *)packedB, scaleB, zeroB, beta, C, ldc, bias);
#elif defined(AVX512_FP16_WEIGHT_ONLY_INT4)
            // Timline t("xdnn_hgemm_f32u4f32_compute_biasadd");
            xdnn_hgemm_f32u4f32_compute_biasadd(
                    transA, M, N, K, alpha, A, lda, (const XDNN_UINT4x2 *)packedB, scaleB, zeroB, beta, C, ldc, bias);
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_INT4 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // NF4
        else if constexpr (std::is_same_v<WeiT, nf4x2_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_NF4
            // Timline t("xdnn_sgemm_f32nf4f32_compute_biasadd");
            xdnn_sgemm_f32nf4f32_compute_biasadd(
                    transA, M, N, K, alpha, A, lda, (const XDNN_NF4x2 *)packedB, scaleB, zeroB, beta, C, ldc, bias);
#elif defined(AVX512_FP16_WEIGHT_ONLY_NF4)
            // Timline t("xdnn_hgemm_f32nf4f32_compute_biasadd");
            xdnn_hgemm_f32nf4f32_compute_biasadd(
                    transA, M, N, K, alpha, A, lda, (const XDNN_NF4x2 *)packedB, scaleB, zeroB, beta, C, ldc, bias);
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_NF4 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }
    }

    template <typename InT, typename WeiT, typename OutT>
    static void compute_biasadd_relu(bool transA, int M, int N, int K, float alpha, const InT *A, int lda,
            const WeiT *packedB, const float *scaleB, const float *zeroB, float beta, OutT *C, int ldc,
            const float *bias) {
        // FP32
        if constexpr (std::is_same_v<WeiT, float>) {
            // Timline t("xdnn_sgemm_compute_biasadd_relu");
            xdnn_sgemm_compute_biasadd_relu(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias);
        }

        // FP16
        else if constexpr (std::is_same_v<WeiT, float16_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_FP16
            // Timline t("xdnn_sgemm_f32f16f32_compute_biasadd_relu");
            xdnn_sgemm_f32f16f32_compute_biasadd_relu(
                    transA, M, N, K, alpha, A, lda, (const XDNN_FP16 *)packedB, beta, C, ldc, bias);
#elif defined(AVX512_FP16_WEIGHT_ONLY_FP16)
            // Timline t("xdnn_hgemm_f32f16f32_compute_biasadd_relu");
            xdnn_hgemm_f32f16f32_compute_biasadd_relu(
                    transA, M, N, K, alpha, A, lda, (const XDNN_FP16 *)packedB, beta, C, ldc, bias);
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_FP16 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // BF16
        else if constexpr (std::is_same_v<WeiT, bfloat16_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_BF16
            // Timline t("xdnn_sgemm_f32bf16f32_compute_biasadd_relu");
            xdnn_sgemm_f32bf16f32_compute_biasadd_relu(
                    transA, M, N, K, alpha, A, lda, (const XDNN_UINT4x2 *)packedB, beta, C, ldc, bias);
#elif defined(AVX512_BF16_WEIGHT_ONLY_BF16)
            if (M > USE_AMX_M) {
                // Timline t("onednn_amx_sgemm_f32bf16f32_compute_biasadd_relu");
                onednn_amx_sgemm_f32bf16f32_compute_biasadd_relu(
                        transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias);
            } else {
                // Timline t("xdnn_bgemm_f32bf16f32_compute_biasadd_relu");
                xdnn_bgemm_f32bf16f32_compute_biasadd_relu(
                        transA, M, N, K, alpha, A, lda, (const XDNN_BF16 *)packedB, beta, C, ldc, bias);
            }
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_BF16 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // INT8
        else if constexpr (std::is_same_v<WeiT, int8_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_INT8
            // Timline t("xdnn_sgemm_f32s8f32_compute_biasadd_relu");
            xdnn_sgemm_f32s8f32_compute_biasadd_relu(
                    transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc, bias);
#elif defined(AVX512_FP16_WEIGHT_ONLY_INT8)
            // Timline t("xdnn_hgemm_f32s8f32_compute_biasadd_relu");
            xdnn_hgemm_f32s8f32_compute_biasadd_relu(
                    transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc, bias);
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_INT8 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // INT4
        else if constexpr (std::is_same_v<WeiT, uint4x2_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_INT4
            // Timline t("xdnn_sgemm_f32u4f32_compute_biasadd_relu");
            xdnn_sgemm_f32u4f32_compute_biasadd_relu(
                    transA, M, N, K, alpha, A, lda, (const XDNN_UINT4x2 *)packedB, scaleB, zeroB, beta, C, ldc, bias);
#elif defined(AVX512_FP16_WEIGHT_ONLY_INT4)
            // Timline t("xdnn_hgemm_f32u4f32_compute_biasadd_relu");
            xdnn_hgemm_f32u4f32_compute_biasadd_relu(
                    transA, M, N, K, alpha, A, lda, (const XDNN_UINT4x2 *)packedB, scaleB, zeroB, beta, C, ldc, bias);
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_INT4 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // NF4
        else if constexpr (std::is_same_v<WeiT, nf4x2_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_NF4
            // Timline t("xdnn_sgemm_f32nf4f32_compute_biasadd_relu");
            xdnn_sgemm_f32nf4f32_compute_biasadd_relu(
                    transA, M, N, K, alpha, A, lda, (const XDNN_NF4x2 *)packedB, scaleB, zeroB, beta, C, ldc, bias);
#elif defined(AVX512_FP16_WEIGHT_ONLY_NF4)
            // Timline t("xdnn_hgemm_f32nf4f32_compute_biasadd_relu");
            xdnn_hgemm_f32nf4f32_compute_biasadd_relu(
                    transA, M, N, K, alpha, A, lda, (const XDNN_NF4x2 *)packedB, scaleB, zeroB, beta, C, ldc, bias);
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_NF4 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }
    }

    template <typename InT, typename WeiT, typename OutT>
    static void compute_silu(bool transA, int M, int N, int K, float alpha, const InT *A, int lda, const WeiT *packedB,
            const float *scaleB, const float *zeroB, float beta, OutT *C, int ldc) {
        // FP32
        if constexpr (std::is_same_v<WeiT, float>) {
            // Timline t("xdnn_sgemm_compute_silu");
            xdnn_sgemm_compute_silu(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc);
        }

        // FP16
        else if constexpr (std::is_same_v<WeiT, float16_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_FP16
            // Timline t("xdnn_sgemm_f32f16f32_compute_silu");
            xdnn_sgemm_f32f16f32_compute_silu(transA, M, N, K, alpha, A, lda, (const XDNN_FP16 *)packedB, beta, C, ldc);
#elif defined(AVX512_FP16_WEIGHT_ONLY_FP16)
            // Timline t("xdnn_hgemm_f32f16f32_compute_silu");
            xdnn_hgemm_f32f16f32_compute_silu(transA, M, N, K, alpha, A, lda, (const XDNN_FP16 *)packedB, beta, C, ldc);
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_FP16 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // BF16
        else if constexpr (std::is_same_v<WeiT, bfloat16_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_BF16
            // Timline t("xdnn_sgemm_f32bf16f32_compute_silu");
            xdnn_sgemm_f32bf16f32_compute_silu(
                    transA, M, N, K, alpha, A, lda, (const XDNN_UINT4x2 *)packedB, beta, C, ldc);
#elif defined(AVX512_BF16_WEIGHT_ONLY_BF16)
            if (M > USE_AMX_M) {
                // Timline t("onednn_amx_sgemm_f32bf16f32_compute_silu");
                onednn_amx_sgemm_f32bf16f32_compute_silu(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc);
            } else {
                // Timline t("xdnn_bgemm_f32bf16f32_compute_silu");
                xdnn_bgemm_f32bf16f32_compute_silu(
                        transA, M, N, K, alpha, A, lda, (const XDNN_BF16 *)packedB, beta, C, ldc);
            }
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_BF16 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // INT8
        else if constexpr (std::is_same_v<WeiT, int8_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_INT8
            // Timline t("xdnn_sgemm_f32s8f32_compute_silu");
            xdnn_sgemm_f32s8f32_compute_silu(transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc);
#elif defined(AVX512_FP16_WEIGHT_ONLY_INT8)
            // Timline t("xdnn_hgemm_f32s8f32_compute_silu");
            xdnn_hgemm_f32s8f32_compute_silu(transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc);
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_INT8 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // INT4
        else if constexpr (std::is_same_v<WeiT, uint4x2_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_INT4
            // Timline t("xdnn_sgemm_f32u4f32_compute_silu");
            xdnn_sgemm_f32u4f32_compute_silu(
                    transA, M, N, K, alpha, A, lda, (const XDNN_UINT4x2 *)packedB, scaleB, zeroB, beta, C, ldc);
#elif defined(AVX512_FP16_WEIGHT_ONLY_INT4)
            // Timline t("xdnn_hgemm_f32u4f32_compute_silu");
            xdnn_hgemm_f32u4f32_compute_silu(
                    transA, M, N, K, alpha, A, lda, (const XDNN_UINT4x2 *)packedB, scaleB, zeroB, beta, C, ldc);
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_INT4 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // NF4
        else if constexpr (std::is_same_v<WeiT, nf4x2_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_NF4
            // Timline t("xdnn_sgemm_f32nf4f32_compute_silu");
            xdnn_sgemm_f32nf4f32_compute_silu(
                    transA, M, N, K, alpha, A, lda, (const XDNN_NF4x2 *)packedB, scaleB, zeroB, beta, C, ldc);
#elif defined(AVX512_FP16_WEIGHT_ONLY_NF4)
            // Timline t("xdnn_hgemm_f32nf4f32_compute_silu");
            xdnn_hgemm_f32nf4f32_compute_silu(
                    transA, M, N, K, alpha, A, lda, (const XDNN_NF4x2 *)packedB, scaleB, zeroB, beta, C, ldc);
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_NF4 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }
    }

    template <typename InT, typename WeiT, typename OutT>
    static void compute_resmul(bool transA, int M, int N, int K, float alpha, const InT *A, int lda,
            const WeiT *packedB, const float *scaleB, const float *zeroB, float beta, OutT *C, int ldc,
            const float *res, int ldres) {
        // FP32
        if constexpr (std::is_same_v<WeiT, float>) {
            // Timline t("xdnn_sgemm_compute_resmul");
            xdnn_sgemm_compute_resmul(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, res, ldres);
        }

        // FP16
        else if constexpr (std::is_same_v<WeiT, float16_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_FP16
            // Timline t("xdnn_sgemm_f32f16f32_compute_resmul");
            xdnn_sgemm_f32f16f32_compute_resmul(
                    transA, M, N, K, alpha, A, lda, (const XDNN_FP16 *)packedB, beta, C, ldc, res, ldres);
#elif defined(AVX512_FP16_WEIGHT_ONLY_FP16)
            // Timline t("xdnn_hgemm_f32f16f32_compute_resmul");
            xdnn_hgemm_f32f16f32_compute_resmul(
                    transA, M, N, K, alpha, A, lda, (const XDNN_FP16 *)packedB, beta, C, ldc, res, ldres);
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_FP16 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // BF16
        else if constexpr (std::is_same_v<WeiT, bfloat16_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_BF16
            // Timline t("xdnn_sgemm_f32bf16f32_compute_resmul");
            xdnn_sgemm_f32bf16f32_compute_resmul(
                    transA, M, N, K, alpha, A, lda, (const XDNN_UINT4x2 *)packedB, beta, C, ldc, res, ldres);
#elif defined(AVX512_BF16_WEIGHT_ONLY_BF16)
            if (M > USE_AMX_M) {
                // Timline t("onednn_amx_sgemm_f32bf16f32_compute_resmul");
                onednn_amx_sgemm_f32bf16f32_compute_resmul(
                        transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, res, ldres);
            } else {
                // Timline t("xdnn_bgemm_f32bf16f32_compute_resmul");
                xdnn_bgemm_f32bf16f32_compute_resmul(
                        transA, M, N, K, alpha, A, lda, (const XDNN_BF16 *)packedB, beta, C, ldc, res, ldres);
            }
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_BF16 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // INT8
        else if constexpr (std::is_same_v<WeiT, int8_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_INT8
            // Timline t("xdnn_sgemm_f32s8f32_compute_resmul");
            xdnn_sgemm_f32s8f32_compute_resmul(
                    transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc, res, ldres);
#elif defined(AVX512_FP16_WEIGHT_ONLY_INT8)
            // Timline t("xdnn_hgemm_f32s8f32_compute_resmul");
            xdnn_hgemm_f32s8f32_compute_resmul(
                    transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc, res, ldres);
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_INT8 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // INT4
        else if constexpr (std::is_same_v<WeiT, uint4x2_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_INT4
            // Timline t("xdnn_sgemm_f32u4f32_compute_resmul");
            xdnn_sgemm_f32u4f32_compute_resmul(transA, M, N, K, alpha, A, lda, (const XDNN_UINT4x2 *)packedB, scaleB,
                    zeroB, beta, C, ldc, res, ldres);
#elif defined(AVX512_FP16_WEIGHT_ONLY_INT4)
            // Timline t("xdnn_hgemm_f32u4f32_compute_resmul");
            xdnn_hgemm_f32u4f32_compute_resmul(transA, M, N, K, alpha, A, lda, (const XDNN_UINT4x2 *)packedB, scaleB,
                    zeroB, beta, C, ldc, res, ldres);
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_INT4 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // NF4
        else if constexpr (std::is_same_v<WeiT, nf4x2_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_NF4
            // Timline t("xdnn_sgemm_f32nf4f32_compute_resmul");
            xdnn_sgemm_f32nf4f32_compute_resmul(transA, M, N, K, alpha, A, lda, (const XDNN_NF4x2 *)packedB, scaleB,
                    zeroB, beta, C, ldc, res, ldres);
#elif defined(AVX512_FP16_WEIGHT_ONLY_NF4)
            // Timline t("xdnn_hgemm_f32nf4f32_compute_resmul");
            xdnn_hgemm_f32nf4f32_compute_resmul(transA, M, N, K, alpha, A, lda, (const XDNN_NF4x2 *)packedB, scaleB,
                    zeroB, beta, C, ldc, res, ldres);
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_NF4 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }
    }

    template <typename InT, typename WeiT, typename OutT>
    static void compute_residential(bool transA, int M, int N, int K, float alpha, const InT *A, int lda,
            const WeiT *packedB, const float *scaleB, const float *zeroB, float beta, OutT *C, int ldc,
            const float *bias, const float *res, int ldres) {
        // FP32
        if constexpr (std::is_same_v<WeiT, float>) {
            // Timline t("xdnn_sgemm_compute_residential");
            xdnn_sgemm_compute_residential(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias, res, ldres);
        }

        // FP16
        else if constexpr (std::is_same_v<WeiT, float16_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_FP16
            // Timline t("xdnn_sgemm_f32f16f32_compute_residential");
            xdnn_sgemm_f32f16f32_compute_residential(
                    transA, M, N, K, alpha, A, lda, (const XDNN_FP16 *)packedB, beta, C, ldc, bias, res, ldres);
#elif defined(AVX512_FP16_WEIGHT_ONLY_FP16)
            // Timline t("xdnn_hgemm_f32f16f32_compute_residential");
            xdnn_hgemm_f32f16f32_compute_residential(
                    transA, M, N, K, alpha, A, lda, (const XDNN_FP16 *)packedB, beta, C, ldc, bias, res, ldres);
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_FP16 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // BF16
        else if constexpr (std::is_same_v<WeiT, bfloat16_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_BF16
            // Timline t("xdnn_sgemm_f32bf16f32_compute_residential");
            xdnn_sgemm_f32bf16f32_compute_residential(
                    transA, M, N, K, alpha, A, lda, (const XDNN_UINT4x2 *)packedB, beta, C, ldc, bias, res, ldres);
#elif defined(AVX512_BF16_WEIGHT_ONLY_BF16)
            if (M > USE_AMX_M) {
                // Timline t("onednn_amx_sgemm_f32bf16f32_compute_residential");
                onednn_amx_sgemm_f32bf16f32_compute_residential(
                        transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias, res, ldres);
            } else {
                // Timline t("xdnn_bgemm_f32bf16f32_compute_residential");
                xdnn_bgemm_f32bf16f32_compute_residential(
                        transA, M, N, K, alpha, A, lda, (const XDNN_BF16 *)packedB, beta, C, ldc, bias, res, ldres);
            }
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_BF16 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // INT8
        else if constexpr (std::is_same_v<WeiT, int8_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_INT8
            // Timline t("xdnn_sgemm_f32s8f32_compute_residential");
            xdnn_sgemm_f32s8f32_compute_residential(
                    transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc, bias, res, ldres);
#elif defined(AVX512_FP16_WEIGHT_ONLY_INT8)
            // Timline t("xdnn_hgemm_f32s8f32_compute_residential");
            xdnn_hgemm_f32s8f32_compute_residential(
                    transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc, bias, res, ldres);
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_INT8 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // INT4
        else if constexpr (std::is_same_v<WeiT, uint4x2_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_INT4
            // Timline t("xdnn_sgemm_f32u4f32_compute_residential");
            xdnn_sgemm_f32u4f32_compute_residential(transA, M, N, K, alpha, A, lda, (const XDNN_UINT4x2 *)packedB,
                    scaleB, zeroB, beta, C, ldc, bias, res, ldres);
#elif defined(AVX512_FP16_WEIGHT_ONLY_INT4)
            // Timline t("xdnn_hgemm_f32u4f32_compute_residential");
            xdnn_hgemm_f32u4f32_compute_residential(transA, M, N, K, alpha, A, lda, (const XDNN_UINT4x2 *)packedB,
                    scaleB, zeroB, beta, C, ldc, bias, res, ldres);
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_INT4 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // NF4
        else if constexpr (std::is_same_v<WeiT, nf4x2_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_NF4
            // Timline t("xdnn_sgemm_f32nf4f32_compute_residential");
            xdnn_sgemm_f32nf4f32_compute_residential(transA, M, N, K, alpha, A, lda, (const XDNN_NF4x2 *)packedB,
                    scaleB, zeroB, beta, C, ldc, bias, res, ldres);
#elif defined(AVX512_FP16_WEIGHT_ONLY_NF4)
            // Timline t("xdnn_hgemm_f32nf4f32_compute_residential");
            xdnn_hgemm_f32nf4f32_compute_residential(transA, M, N, K, alpha, A, lda, (const XDNN_NF4x2 *)packedB,
                    scaleB, zeroB, beta, C, ldc, bias, res, ldres);
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_NF4 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }
    }

    template <typename InT, typename WeiT, typename OutT>
    static void compute_resext(bool transA, int M, int N, int K, float alpha, const InT *A, int lda,
            const WeiT *packedB, const float *scaleB, const float *zeroB, float beta, OutT *C, int ldc,
            const float *bias, float gamma, float *res, int ldres) {
        // FP32
        if constexpr (std::is_same_v<WeiT, float>) {
            // Timline t("xdnn_sgemm_compute_resext");
            xdnn_sgemm_compute_resext(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias, gamma, res, ldres);
        }

        // FP16
        else if constexpr (std::is_same_v<WeiT, float16_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_FP16
            // Timline t("xdnn_sgemm_f32f16f32_compute_resext");
            xdnn_sgemm_f32f16f32_compute_resext(
                    transA, M, N, K, alpha, A, lda, (const XDNN_FP16 *)packedB, beta, C, ldc, bias, gamma, res, ldres);
#elif defined(AVX512_FP16_WEIGHT_ONLY_FP16)
            // Timline t("xdnn_hgemm_f32f16f32_compute_resext");
            xdnn_hgemm_f32f16f32_compute_resext(
                    transA, M, N, K, alpha, A, lda, (const XDNN_FP16 *)packedB, beta, C, ldc, bias, gamma, res, ldres);
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_FP16 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // BF16
        else if constexpr (std::is_same_v<WeiT, bfloat16_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_BF16
            // Timline t("xdnn_sgemm_f32bf16f32_compute_resext");
            xdnn_sgemm_f32bf16f32_compute_resext(transA, M, N, K, alpha, A, lda, (const XDNN_UINT4x2 *)packedB, beta, C,
                    ldc, bias, gamma, res, ldres);
#elif defined(AVX512_BF16_WEIGHT_ONLY_BF16)
            if (M > USE_AMX_M) {
                // Timline t("onednn_amx_sgemm_f32bf16f32_compute_residential");
#pragma omp parallel for collapse(2)
                for (int i = 0; i < M; ++i) {
                    for (int j = 0; j < N; ++j) {
                        res[i * ldres + j] = res[i * ldres + j] * gamma;
                    }
                }
                onednn_amx_sgemm_f32bf16f32_compute_residential(
                        transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias, res, ldres);
            } else {
                // Timline t("xdnn_bgemm_f32bf16f32_compute_resext");
                xdnn_bgemm_f32bf16f32_compute_resext(transA, M, N, K, alpha, A, lda, (const XDNN_BF16 *)packedB, beta,
                        C, ldc, bias, gamma, res, ldres);
            }
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_BF16 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // INT8
        else if constexpr (std::is_same_v<WeiT, int8_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_INT8
            // Timline t("xdnn_sgemm_f32s8f32_compute_resext");
            xdnn_sgemm_f32s8f32_compute_resext(
                    transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc, bias, gamma, res, ldres);
#elif defined(AVX512_FP16_WEIGHT_ONLY_INT8)
            // Timline t("xdnn_hgemm_f32s8f32_compute_resext");
            xdnn_hgemm_f32s8f32_compute_resext(
                    transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc, bias, gamma, res, ldres);
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_INT8 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // INT4
        else if constexpr (std::is_same_v<WeiT, uint4x2_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_INT4
            // Timline t("xdnn_sgemm_f32u4f32_compute_resext");
            xdnn_sgemm_f32u4f32_compute_resext(transA, M, N, K, alpha, A, lda, (const XDNN_UINT4x2 *)packedB, scaleB,
                    zeroB, beta, C, ldc, bias, gamma, res, ldres);
#elif defined(AVX512_FP16_WEIGHT_ONLY_INT4)
            // Timline t("xdnn_hgemm_f32u4f32_compute_resext");
            xdnn_hgemm_f32u4f32_compute_resext(transA, M, N, K, alpha, A, lda, (const XDNN_UINT4x2 *)packedB, scaleB,
                    zeroB, beta, C, ldc, bias, gamma, res, ldres);
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_INT4 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // NF4
        else if constexpr (std::is_same_v<WeiT, nf4x2_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_NF4
            // Timline t("xdnn_sgemm_f32nf4f32_compute_resext");
            xdnn_sgemm_f32nf4f32_compute_resext(transA, M, N, K, alpha, A, lda, (const XDNN_NF4x2 *)packedB, scaleB,
                    zeroB, beta, C, ldc, bias, gamma, res, ldres);
#elif defined(AVX512_FP16_WEIGHT_ONLY_NF4)
            // Timline t("xdnn_hgemm_f32nf4f32_compute_resext");
            xdnn_hgemm_f32nf4f32_compute_resext(transA, M, N, K, alpha, A, lda, (const XDNN_NF4x2 *)packedB, scaleB,
                    zeroB, beta, C, ldc, bias, gamma, res, ldres);
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_INT4 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }
    }

    enum matmul_kinds {
        Basic = 0,
        BiasAdd = 1,
        BiasAdd_Relu = 2,
        Silu = 3,
        Resmul = 4,
        Residential = 5,
        Resext = 6,
    };

    static std::string create_key(bool transA, int M, int N, int K, int matmul_kind) {
        std::string key = std::to_string(transA) + "_" + std::to_string(M) + "_" + std::to_string(N) + "_"
                + std::to_string(K) + "_" + std::to_string(matmul_kind);
        return key;
    }
    
private:
};

#endif //MATMUL_HELPER_H
