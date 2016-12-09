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

#ifndef HL_CUDA_CUBLAS_H_
#define HL_CUDA_CUBLAS_H_

#include "hl_base.h"

/**
 * @brief   Matrix transpose: C_d = T(A_d)
 *
 * @param[in]   A_d     input matrix (dimM x dimN).
 * @param[out]  C_d     output matrix (dimN x dimM).
 * @param[in]   dimM    matrix height.
 * @param[in]   dimN    matrix width.
 * @param[in]   lda     the first dimension of A_d.
 * @param[in]   ldc     the first dimension of C_d.
 *
 */
extern void hl_matrix_transpose(
    real *A_d, real *C_d, int dimM, int dimN, int lda, int ldc);

/*
 * @brief Matrix transpose, while lda = dimN, ldc = dimM.
 *
 * @param[in]   A_d     input matrix (dimM x dimN).
 * @param[out]  C_d     output matrix (dimN x dimM).
 * @param[in]   dimM    matrix height.
 * @param[in]   dimN    matrix width.
 *
 */
extern void hl_matrix_transpose(real *A_d, real *C_d, int dimM, int dimN);

/*
 * @brief Matrix inverse
 *
 * @param[in]   A_d    input matrix (dimN x dimN).
 * @param[out]  C_d    output matrix (dimN x dimN).
 * @param[in]   dimN   matrix height = matrix width
 * @param[in]   lda    the first dimension of A_d
 * @param[in]   ldc    the first dimension of C_d
 *
 */
extern void hl_matrix_inverse(real *A_d, real *C_d, int dimN, int lda, int ldc);

/**
 * @brief   C_d = alpha*(op(A_d) * op(B_d)) + beta*C_d
 *
 * @param[in]   A_d     input.
 * @param[in]   transa  operation op(A) that is non-or transpose.
 * @param[in]   B_d     input.
 * @param[in]   transb  operation op(B) that is non-or transpose.
 * @param[out]  C_d     output.
 * @param[in]   dimM    matrix height of op(A) & C
 * @param[in]   dimN    matrix width of op(B) & C
 * @param[in]   dimK    width of op(A) & height of op(B)
 * @param[in]   alpha   scalar used for multiplication.
 * @param[in]   beta    scalar used for multiplication.
 * @param[in]   lda     the first dimension of A_d.
 * @param[in]   ldb     the first dimension of B_d.
 * @param[in]   ldc     the first dimension of C_d.
 *
 */
extern void hl_matrix_mul(real *A_d,
                          hl_trans_op_t transa,
                          real *B_d,
                          hl_trans_op_t transb,
                          real *C_d,
                          int dimM,
                          int dimN,
                          int dimK,
                          real alpha,
                          real beta,
                          int lda,
                          int ldb,
                          int ldc);

/**
 * @brief   C_d = alpha*(op(A_d) * op(B_d)) + beta*C_d
 *
 * @param[in]   A_d     input.
 * @param[in]   transa  operation op(A) that is non-or transpose.
 * @param[in]   B_d     input.
 * @param[in]   transb  operation op(B) that is non-or transpose.
 * @param[out]  C_d     output.
 * @param[in]   dimM    matrix height of op(A) & C
 * @param[in]   dimN    matrix width of op(B) & C
 * @param[in]   dimK    width of op(A) & height of op(B)
 * @param[in]   alpha   scalar used for multiplication.
 * @param[in]   beta    scalar used for multiplication.
 *
 */
extern void hl_matrix_mul(real *A_d,
                          hl_trans_op_t transa,
                          real *B_d,
                          hl_trans_op_t transb,
                          real *C_d,
                          int dimM,
                          int dimN,
                          int dimK,
                          real alpha,
                          real beta);

/**
 * @brief   This function performs the matrix-vector multiplication.
 *          C_d = alpha*op(A_d)*B_d + beta*C_d
 *
 * @param[in]     A_d    matrix.
 * @param[in]     trans  operation op(A) that is non-or transpose.
 * @param[in]     B_d    vector with dimN(dimM) elements
 *                       if trans==HPPL_OP_N(HPPL_OP_T).
 * @param[in,out] C_d    vector with dimM(dimN) elements
 *                       if trans==HPPL_OP_N(HPPL_OP_T).
 * @param[in]     dimM   number of rows of matrix A_d.
 * @param[in]     dimN   number of columns of matrix A_d.
 * @param[in]     alpha  scalar used for multiplication.
 * @param[in]     beta   scalar used for multiplication.
 * @param[in]     lda    the first dimension of A_d.
 * @param[in]     incb   increase B_d size for compaction.
 * @param[in]     incc   increase C_d size for compaction.
 *
 */

extern void hl_matrix_mul_vector(real *A_d,
                                 hl_trans_op_t trans,
                                 real *B_d,
                                 real *C_d,
                                 int dimM,
                                 int dimN,
                                 real alpha,
                                 real beta,
                                 int lda,
                                 int incb,
                                 int incc);

/**
 * @brief   This function performs the matrix-vector multiplication.
 *          C_d = alpha*op(A_d)*B_d + beta*C_d
 *
 * @param[in]     A_d    matrix.
 * @param[in]     trans  operation op(A) that is non-or transpose.
 * @param[in]     B_d    vector with dimN(dimM) elements
 *                       if trans==HPPL_OP_N(HPPL_OP_T).
 * @param[in,out] C_d    vector with dimM(dimN) elements
 *                       if trans==HPPL_OP_N(HPPL_OP_T).
 * @param[in]     dimM   number of rows of matrix A_d.
 * @param[in]     dimN   number of columns of matrix A_d.
 * @param[in]     alpha  scalar used for multiplication.
 * @param[in]     beta   scalar used for multiplication.
 *
 */
extern void hl_matrix_mul_vector(real *A_d,
                                 hl_trans_op_t trans,
                                 real *B_d,
                                 real *C_d,
                                 int dimM,
                                 int dimN,
                                 real alpha,
                                 real beta);

#endif /* HL_CUDA_CUBLAS_H_ */
