/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */


#ifndef HL_MATRIX_APPLY_H_
#define HL_MATRIX_APPLY_H_

#include "hl_base.h"
#include "hl_cpu_matrix_kernel.cuh"
#include "hl_gpu_matrix_kernel.cuh"

/**
 * @brief   CPU element wise unary operator.
 *
 *  element wise op(a) for 0 <= i < dimM & for 0 <= j < dimN.
 *
 * @param[in]       op          unary op. see namespace unary
 * @param[in,out]   A_h         matrix.
 * @param[in]       dimM        matrix height.
 * @param[in]       dimN        matrix width.
 * @param[in]       lda         leading dimension of A.
 *
 */
template <class T, class Op>
extern void hl_cpu_apply_unary_op(Op op,
                                  T* A_h,
                                  int dimM,
                                  int dimN,
                                  int lda);

/**
 * @brief   CPU element wise binary operator.
 *
 * element wise op(a, b) for 0 <= i < dimM & for 0 <= j < dimN.
 *
 * if (BAsRowVector == 0 && BAsColVector == 0)
 *   op(A[i * lda + j], B[i * ldb + j])
 *
 * if (BAsRowVector == 1 && BAsColVector == 0)
 *   op(A[i * lda + j], B[j])
 *
 * if (BAsRowVector == 0 && BAsColVector == 1)
 *   op(A[i * lda + j], B[i * ldb])
 *
 * if (BAsRowVector == 1 && BAsColVector == 1)
 *   op(A[i * lda + j], B[0])
 *
 * @param[in]       op          binary op. see namespace binary.
 * @param[in,out]   A_h         matrix.
 * @param[in,out]   B_h         matrix.
 * @param[in]       dimM        matrix height.
 * @param[in]       dimN        matrix width.
 * @param[in]       lda         leading dimension of A.
 * @param[in]       ldb         leading dimension of B.
 *
 */
template <class T, class Op, bool BAsRowVector, bool BAsColVector>
extern void hl_cpu_apply_binary_op(Op op,
                                   T* A_h,
                                   T* B_h,
                                   int dimM,
                                   int dimN,
                                   int lda,
                                   int ldb);

/**
 * @brief   CPU element wise ternary operator.
 *
 * element wise op(a, b, c) for 0 <= i < dimM & for 0 <= j < dimN.
 *
 * if (CAsRowVector == 0 && CAsColVector == 0)
 *   op(A[i*lda + j], B[i*ldb + j], C[i*ldc + j])
 *
 * if (CAsRowVector == 1 && CAsColVector == 0)
 *   op(A[i*lda + j], B[i*ldb + j], C[j])
 *
 * if (CAsRowVector == 0 && CAsColVector == 1)
 *   op(A[i*lda + j], B[i*ldb + j], C[i*ldc])
 *
 * if (CAsRowVector == 1 && CAsColVector == 1)
 *   op(A[i*lda + j], B[i*ldb + j], C[0])
 *
 * @param[in]       op          ternary op. see namespace ternary.
 * @param[in,out]   A_h         matrix.
 * @param[in,out]   B_h         matrix.
 * @param[in,out]   C_h         matrix.
 * @param[in]       dimM        matrix height.
 * @param[in]       dimN        matrix width.
 * @param[in]       lda         leading dimension of A.
 * @param[in]       ldb         leading dimension of B.
 * @param[in]       ldc         leading dimension of C.
 *
 */
template <class T, class Op, bool CAsRowVector, bool CAsColVector>
extern void hl_cpu_apply_ternary_op(Op op,
                                    T* A_h,
                                    T* B_h,
                                    T* C_h,
                                    int dimM,
                                    int dimN,
                                    int lda,
                                    int ldb,
                                    int ldc);

/**
 * @brief   CPU element wise quaternary operator.
 *          element wise op(a, b, c, d) for 0 <= i < dimM & for 0 <= j < dimN.
 *
 * @param[in]       op          quaternary op. see namespace ternary.
 * @param[in,out]   A_h         matrix.
 * @param[in,out]   B_h         matrix.
 * @param[in,out]   C_h         matrix.
 * @param[in,out]   D_h         matrix.
 * @param[in]       dimM        matrix height.
 * @param[in]       dimN        matrix width.
 * @param[in]       lda         leading dimension of A.
 * @param[in]       ldb         leading dimension of B.
 * @param[in]       ldc         leading dimension of C.
 * @param[in]       ldd         leading dimension of D.
 *
 */
template <class T, class Op>
extern void hl_cpu_apply_quaternary_op(Op op,
                                       T* A_h,
                                       T* B_h,
                                       T* C_h,
                                       T* D_h,
                                       int dimM,
                                       int dimN,
                                       int lda,
                                       int ldb,
                                       int ldc,
                                       int ldd);

/**
 * @brief   GPU element wise unary operator.
 *          element wise op(a) for 0 <= i < dimM & for 0 <= j < dimN.
 *
 * @param[in]       op          unary op. see namespace unary.
 * @param[in,out]   A_d         matrix.
 * @param[in]       dimM        matrix height.
 * @param[in]       dimN        matrix width.
 * @param[in]       lda         leading dimension of A.
 *
 */
template <class T, class Op>
extern void hl_gpu_apply_unary_op(Op op,
                                  T* A_d,
                                  int dimM,
                                  int dimN,
                                  int lda);

/**
 * @brief   GPU element wise binary operator.
 *
 * element wise op(a, b) for 0 <= i < dimM & for 0 <= j < dimN
 *
 * if (BAsRowVector == 0 && BAsColVector == 0)
 *   op(A[i * lda + j], B[i * ldb + j])
 *
 * if (BAsRowVector == 1 && BAsColVector == 0)
 *   op(A[i * lda + j], B[j])
 *
 * if (BAsRowVector == 0 && BAsColVector == 1)
 *   op(A[i * lda + j], B[i * ldb])
 *
 * if (BAsRowVector == 1 && BAsColVector == 1)
 *   op(A[i * lda + j], B[0])
 *
 * @param[in]       op          binary op. see namespace binary.
 * @param[in,out]   A_d         matrix.
 * @param[in,out]   B_d         matrix.
 * @param[in]       dimM        matrix height.
 * @param[in]       dimN        matrix width.
 * @param[in]       lda         leading dimension of A.
 * @param[in]       ldb         leading dimension of B.
 *
 */
template <class T, class Op, bool BAsRowVector, bool BAsColVector>
extern void hl_gpu_apply_binary_op(Op op,
                                   T* A_d,
                                   T* B_d,
                                   int dimM,
                                   int dimN,
                                   int lda,
                                   int ldb);
/**
 * @brief   GPU element wise ternary operator.
 *
 * element wise op(a, b, c) for 0 <= i < dimM & for 0 <= j < dimN.
 *
 * if (CAsRowVector == 0 && CAsColVector == 0)
 *   op(A[i*lda + j], B[i*ldb + j], C[i*ldc + j])
 *
 * if (CAsRowVector == 1 && CAsColVector == 0)
 *   op(A[i*lda + j], B[i*ldb + j], C[j])
 *
 * if (CAsRowVector == 0 && CAsColVector == 1)
 *   op(A[i*lda + j], B[i*ldb + j], C[i*ldc])
 *
 * if (CAsRowVector == 1 && CAsColVector == 1)
 *   op(A[i*lda + j], B[i*ldb + j], C[0])
 *
 * @param[in]       op          ternary op. see namespace ternary.
 * @param[in,out]   A_d         matrix.
 * @param[in,out]   B_d         matrix.
 * @param[in,out]   C_d         matrix.
 * @param[in]       dimM        matrix height.
 * @param[in]       dimN        matrix width.
 * @param[in]       lda         leading dimension of A.
 * @param[in]       ldb         leading dimension of B.
 * @param[in]       ldc         leading dimension of C.
 *
 */
template <class T, class Op, bool CAsRowVector, bool CAsColVector>
extern void hl_gpu_apply_ternary_op(Op op,
                                    T* A_d,
                                    T* B_d,
                                    T* C_d,
                                    int dimM,
                                    int dimN,
                                    int lda,
                                    int ldb,
                                    int ldc);


/**
 * @brief   GPU element wise quaternary operator.
 *          element wise op(a, b, c, d) for 0 <= i < dimM & for 0 <= j < dimN.
 *
 * @param[in]       op          quaternary op. see namespace ternary.
 * @param[in,out]   A_d         matrix.
 * @param[in,out]   B_d         matrix.
 * @param[in,out]   C_d         matrix.
 * @param[in,out]   D_d         matrix.
 * @param[in]       dimM        matrix height.
 * @param[in]       dimN        matrix width.
 * @param[in]       lda         leading dimension of A.
 * @param[in]       ldb         leading dimension of B.
 * @param[in]       ldc         leading dimension of C.
 * @param[in]       ldd         leading dimension of D.
 *
 */
template <class T, class Op>
extern void hl_gpu_apply_quaternary_op(Op op,
                                       T* A_d,
                                       T* B_d,
                                       T* C_d,
                                       T* D_d,
                                       int dimM,
                                       int dimN,
                                       int lda,
                                       int ldb,
                                       int ldc,
                                       int ldd);

/**
 * @brief  CPU matrix row operator.
 */
template <class Agg, class Op, class Saver>
extern void hl_cpu_matrix_row_op(Agg agg, Op op, Saver sv,
                                 int dimM, int dimN,
                                 real *dst, int ld,
                                 real *A, int lda);

/**
 * @brief  CPU matrix row operator.
 *
 * @param[in]  agg    aggregate operator expression.
 * @param[in]  op     operator expression.
 * @param[in]  dimM   matrix height.
 * @param[in]  dimN   matrix width.
 * @param[out] dst    destination matrix.
 * @param[in]  ld     leading dimension of dst matrix.
 * @param[in]  *A     matrix A.
 * @param[in]  lda    leading dimension of matrix A.
 * @param[in]  *B     matrix B.
 * @param[in]  ldb    leading dimension of matrix B.
 *
 */
template <class Saver, class Agg, class Op>
extern void hl_cpu_matrix_row_op(Agg agg, Op op,
                                 int dimM, int dimN,
                                 real *dst, int ld,
                                 real *A, int lda,
                                 real *B, int ldb);

/**
 * @brief  CPU matrix column operator.
 *
 * @param[in]  agg    aggregate operator expression.
 * @param[in]  op     operator expression.
 * @param[in]  sv     assignment operator expression.
 * @param[in]  dimM   matrix height.
 * @param[in]  dimN   matrix width.
 * @param[out] dst    destination matrix.
 * @param[in]  *A     matrix A.
 * @param[in]  lda    leading dimension of matrix A.
 *
 */
template <class Agg, class Op, class Saver>
extern void hl_cpu_matrix_column_op(Agg agg, Op op, Saver sv,
                                    int dimM, int dimN,
                                    real *dst,
                                    real *A, int lda);

/**
 * @brief  CPU matrix column operator.
 *
 * @param[in]  agg    aggregate operator expression.
 * @param[in]  op     operator expression.
 * @param[in]  sv     assignment operator expression.
 * @param[in]  dimM   matrix height.
 * @param[in]  dimN   matrix width.
 * @param[out] dst    destination matrix.
 * @param[in]  *A     matrix A.
 * @param[in]  lda    leading dimension of matrix A.
 * @param[in]  *B     matrix B.
 * @param[in]  ldb    leading dimension of matrix B.
 *
 */
template <class Agg, class Op, class Saver>
extern void hl_cpu_matrix_column_op(Agg agg, Op op, Saver sv,
                                    int dimM, int dimN,
                                    real *dst,
                                    real *A, int lda,
                                    real *B, int ldb);

/**
 * @brief  GPU matrix row operator.
 *
 * @param[in]  agg    aggregate operator expression.
 * @param[in]  op     operator expression.
 * @param[in]  sv     assignment operator expression.
 * @param[in]  dimM   matrix height.
 * @param[in]  dimN   matrix width.
 * @param[out] dst    destination matrix.
 * @param[in]  ld     leading dimension of dst.
 * @param[in]  *A     matrix A.
 * @param[in]  lda    leading dimension of matrix A.
 *
 */
template <class Agg, class Op, class Saver>
extern void hl_gpu_matrix_row_op(Agg agg, Op op, Saver sv,
                                 int dimM, int dimN,
                                 real *dst, int ld,
                                 real *A, int lda);

/**
 * @brief  GPU matrix row operator.
 *
 * @param[in]  agg    aggregate operator expression.
 * @param[in]  op     operator expression.
 * @param[in]  dimM   matrix height.
 * @param[in]  dimN   matrix width.
 * @param[out] dst    destination matrix.
 * @param[in]  ld     leading dimension of dst matrix.
 * @param[in]  *A     matrix A.
 * @param[in]  lda    leading dimension of matrix A.
 * @param[in]  *B     matrix B.
 * @param[in]  ldb    leading dimension of matrix B.
 *
 */
template <class Saver, class Agg, class Op>
extern void hl_gpu_matrix_row_op(Agg agg, Op op,
                                 int dimM, int dimN,
                                 real *dst, int ld,
                                 real *A, int lda,
                                 real *B, int ldb);

/**
 * @brief  GPU matrix column operator.
 *
 * @param[in]  agg    aggregate operator expression.
 * @param[in]  op     operator expression.
 * @param[in]  sv     assignment operator expression.
 * @param[in]  dimM   matrix height.
 * @param[in]  dimN   matrix width.
 * @param[out] dst    destination matrix.
 * @param[in]  *A     matrix A.
 * @param[in]  lda    leading dimension of matrix A.
 *
 */
template <class Agg, class Op, class Saver>
extern void hl_gpu_matrix_column_op(Agg agg, Op op, Saver sv,
                                    int dimM, int dimN,
                                    real *dst,
                                    real *A, int lda);

/**
 * @brief  GPU matrix column operator.
 *
 * @param[in]  agg    aggregate operator expression.
 * @param[in]  op     operator expression.
 * @param[in]  sv     assignment operator expression.
 * @param[in]  dimM   matrix height.
 * @param[in]  dimN   matrix width.
 * @param[out] dst    destination matrix.
 * @param[in]  *A     matrix A.
 * @param[in]  lda    leading dimension of matrix A.
 * @param[in]  *B     matrix B.
 * @param[in]  ldb    leading dimension of matrix B.
 *
 */
template <class Agg, class Op, class Saver>
extern void hl_gpu_matrix_column_op(Agg agg, Op op, Saver sv,
                                    int dimM, int dimN,
                                    real *dst,
                                    real *A, int lda,
                                    real *B, int ldb);

#endif /* HL_MATRIX_APPLY_H_ */
