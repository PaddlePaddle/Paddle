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

#ifndef HL_SPARSE_H_
#define HL_SPARSE_H_

#include "hl_base.h"

/**
 * @brief   Malloc a sparse matrix.
 *
 * @param[out]  A_d        sparse matrix.
 * @param[in]   format     format.
 * @param[in]   value_type valueType.
 * @param[in]   dimM       height.
 * @param[in]   dimN       width.
 * @param[in]   nnz        number of none zero element.
 *
 */
extern void hl_malloc_sparse_matrix(hl_sparse_matrix_s *A_d,
                                    hl_matrix_format_t format,
                                    hl_matrix_value_t value_type,
                                    int dimM,
                                    int dimN,
                                    int nnz);

/**
 * @brief   Free a sparse matrix.
 *
 * @param[in]  A_d  GPU sparse matrix.
 *
 */
extern void hl_free_sparse_matrix(hl_sparse_matrix_s A_d);

/**
 * @brief   Construct a sparse matrix use input gpu memory.
 *
 * @param[out]  A_d         sparse matrix.
 * @param[in]   dest_d      gpu memory.
 * @param[in]   size        size of dest_d.
 * @param[in]   format      format.
 * @param[in]   value_type  valueType.
 * @param[in]   dimM        height.
 * @param[in]   dimN        width.
 * @param[in]   nnz         number of none zero element.
 *
 * @note    Destruct api is hl_destruct_sparse_matrix.
 *
 */
extern void hl_construct_sparse_matrix(hl_sparse_matrix_s *A_d,
                                       void *dest_d,
                                       size_t size,
                                       hl_matrix_format_t format,
                                       hl_matrix_value_t value_type,
                                       int dimM,
                                       int dimN,
                                       int nnz);

/**
 * @brief   Use three arrays to construct sparse matrix.
 *
 * if format is HL_SPARSE_CSR, size of rows_d is dimM + 1,
 * and size of cols_d is nnz;
 *
 * if format is HL_SPARSE_CSC, size of rows_d is nnz, and size of
 * cols_d is dimN + 1.
 *
 * if valueType is HL_NO_VALUE, size of value_d is zero,
 * else size of value_d is nnz.
 *
 * @param[out]  A_d        sparse matrix.
 * @param[in]   value_d    value.
 * @param[in]   rows_d     row.
 * @param[in]   cols_d     col.
 * @param[in]   format     format.
 * @param[in]   value_type valueType.
 * @param[in]   dimM       height.
 * @param[in]   dimN       width.
 * @param[in]   nnz        number of none zero element.
 *
 * @note    The corresponding destructor interface is hl_destruct_sparse_matrix.
 *
 */
extern void hl_construct_sparse_matrix(hl_sparse_matrix_s *A_d,
                                       real *value_d,
                                       int *rows_d,
                                       int *cols_d,
                                       hl_matrix_format_t format,
                                       hl_matrix_value_t value_type,
                                       int dimM,
                                       int dimN,
                                       int nnz);

/**
 * @brief   Destruct sparse matrix.
 *
 * @param[in] A_d  sparse matrix.
 *
 */
extern void hl_destruct_sparse_matrix(hl_sparse_matrix_s A_d);

/**
 * @brief   Copy value & index to sparse matrix.
 *
 * if csr_matrix is HL_FLOAT_VALUE.
 *
 *  1. csr_val, csr_row, csr_col three pointers are not null.
 *
 *  2. csr_val is not null, csr_row adn csr_col are null.
 *
 * if csr_matrix is HL_NO_VALUE.
 *
 *  1. csr_val will be ignore, csr_row and csr_col are not null.
 *
 *
 * @param[in,out]   csr_matrix sparse matrix.
 * @param[in]       csr_val    point to csr value array(nnz).
 * @param[in]       csr_row    point to csr row indices array(dimM+1).
 * @param[in]       csr_col    point to csr col indices array(nnz).
 * @param[in]       stream     hl_stream_t type.
 *
 */
extern void hl_memcpy_csr_matrix(hl_sparse_matrix_s csr_matrix,
                                 real *csr_val,
                                 int *csr_row,
                                 int *csr_col,
                                 hl_stream_t stream);

/**
 * @brief   Copy value & index to sparse matrix.
 *
 * if csr_matrix is HL_FLOAT_VALUE.
 *
 *   1. csc_val, csc_row, csc_col three pointers are not null.
 *
 *   2. csc_val is not null, csc_row and csc_col are null.
 *
 * if csr_matrix is HL_NO_VALUE.
 *
 *   1. csc_val will be ignore, csc_row and csc_col are not null.
 *
 * @param[in,out]   csc_matrix sparse matrix.
 * @param[in]       csc_val    point to csc value array(nnz).
 * @param[in]       csc_row    point to csc row indices array(nnz).
 * @param[in]       csc_col    point to csc col indices array(dimN+1).
 * @param[in]       stream     hl_stream_t type.
 *
 *
 */
extern void hl_memcpy_csc_matrix(hl_sparse_matrix_s csc_matrix,
                                 real *csc_val,
                                 int *csc_row,
                                 int *csc_col,
                                 hl_stream_t stream);

/**
 * @brief   Copy sparse matrix to sparse matrix.
 *
 * @param[out]  dst     sparse matrix.
 * @param[in]   src     sparse matrix.
 * @param[in]   stream  hl_stream_t type.
 *
 *
 * @note    1. Format of the src matrix and dst matrix needs to be consistent.
 *          2. Source matrix has value, the destination matrix has value or
 *             no value can be; the source matrix is no value, then the
 *             destination matrix must also be no value;
 */
extern void hl_memcpy_sparse_matrix(hl_sparse_matrix_s dst,
                                    hl_sparse_matrix_s src,
                                    hl_stream_t stream);

/**
 * @brief   csr matrix to dense matrix.
 *
 * @param[in]   A_d     csr matrix.
 * @param[out]  C_d     dense matrix.
 * @param[in]   dimM    height.
 * @param[in]   dimN    width.
 *
 */
extern void hl_matrix_csr2dense(hl_sparse_matrix_s A_d,
                                real *C_d,
                                int dimM,
                                int dimN);

/**
 * @brief   csc matrix to dense matrix.
 *
 * @param[in]   A_d     csc matrix.
 * @param[out]  C_d     dense matrix.
 * @param[in]   dimM    height.
 * @param[in]   dimN    width.
 *
 */
extern void hl_matrix_csc2dense(hl_sparse_matrix_s A_d,
                                real *C_d,
                                int dimM,
                                int dimN);

/**
 * @brief   C_d = alpha*(op(A_d) * op(B_d)) + beta*C_d.
 *
 * @param[in]   A_d     csr sparse matrix.
 * @param[in]   transa  operation op(A) that is non-or transpose.
 * @param[in]   B_d     dense matrix.
 * @param[in]   transb  operation op(B) that is non-or transpose.
 * @param[out]  C_d     dense matrix.
 * @param[in]   dimM    matrix height of op(A) & C
 * @param[in]   dimN    matrix width of op(B) & C
 * @param[in]   dimK    width of op(A) & height of op(B)
 * @param[in]   alpha   scalar used for multiplication.
 * @param[in]   beta    scalar used for multiplication.
 *                      If beta is zero, C does not have to be a valid input.
 *
 * @note    transb is not support HPPL_OP_T.
 *
 */
extern void hl_matrix_csr_mul_dense(hl_sparse_matrix_s A_d,
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
 * @brief   C_d = alpha*(op(A_d) * op(B_d)) + beta*C_d.
 *
 * @param[in]   A_d     sparse matrix.
 * @param[in]   transa  operation op(A) that is non-or transpose.
 * @param[in]   B_d     dense matrix.
 * @param[in]   transb  operation op(B) that is non-or transpose.
 * @param[out]  C_d     dense matrix.
 * @param[in]   dimM    matrix height of op(A) & C
 * @param[in]   dimN    matrix width of op(B) & C
 * @param[in]   dimK    width of op(A) & height of op(B)
 * @param[in]   alpha   scalar used for multiplication.
 * @param[in]   beta    scalar used for multiplication.
 *                      If beta is zero, C does not have to be a valid input.
 *
 * @note    transb is not support HPPL_OP_T.
 *
 */
extern void hl_matrix_csc_mul_dense(hl_sparse_matrix_s A_d,
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
 * @brief   C_d = alpha*(op(A_d) * op(B_d)) + beta*C_d.
 *
 * @param[in]   A_d     dense matrix.
 * @param[in]   transa  operation op(A) that is non-or transpose.
 * @param[in]   B_d     csc sparse matrix.
 * @param[in]   transb  operation op(B) that is non-or transpose.
 * @param[out]  C_d     dense matrix.
 * @param[in]   dimM    matrix height of op(A) & C
 * @param[in]   dimN    matrix width of op(B) & C
 * @param[in]   dimK    width of op(A) & height of op(B)
 * @param[in]   alpha   scalar used for multiplication.
 * @param[in]   beta    scalar used for multiplication.
 *                      If beta is zero, C does not have to be a valid input.
 *
 * @note    transa is not support HPPL_OP_T.
 *
 */
extern void hl_matrix_dense_mul_csc(real *A_d,
                                    hl_trans_op_t transa,
                                    hl_sparse_matrix_s B_d,
                                    hl_trans_op_t transb,
                                    real *C_d,
                                    int dimM,
                                    int dimN,
                                    int dimK,
                                    real alpha,
                                    real beta);

/**
 * @brief   C_d = alpha*(op(A_d) * op(B_d)) + beta*C_d.
 *          Calculated based on the non-zero elements of the matrix C.
 *
 * @param[in]     A_d     dense matrix.
 * @param[in]     transa  operation op(A) that is non-or transpose.
 * @param[in]     B_d     dense matrix.
 * @param[in]     transb  operation op(B) that is non-or transpose.
 * @param[in,out] C_d     sparse matrix.
 * @param[in]     dimM    matrix height of op(A) & C
 * @param[in]     dimN    matrix width of op(B) & C
 * @param[in]     dimK    width of op(A) & height of op(B)
 * @param[in]     alpha   scalar used for multiplication.
 * @param[in]     beta    scalar used for multiplication.
 *
 * @note    transb is not support HPPL_OP_T.
 *
 */
extern void hl_sparse_matrix_mul(real *A_d,
                                 hl_trans_op_t transa,
                                 real *B_d,
                                 hl_trans_op_t transb,
                                 hl_sparse_matrix_s C_d,
                                 int dimM,
                                 int dimN,
                                 int dimK,
                                 real alpha,
                                 real beta);

/**
 * @brief   C_d = alpha*(op(A_d) * op(B_d)) + beta*C_d
 *
 * @param[in]   A_d     dense matrix.
 * @param[in]   transa  operation op(A) that is non-or transpose.
 * @param[in]   B_d     sparse matrix.
 * @param[in]   transb  operation op(B) that is non-or transpose.
 * @param[out]  C_d     dense matrix.
 * @param[in]   dimM    matrix height of op(A) & C
 * @param[in]   dimN    matrix width of op(B) & C
 * @param[in]   dimK    width of op(A) & height of op(B)
 * @param[in]   alpha   scalar used for multiplication.
 * @param[in]   beta    scalar used for multiplication.
 *                      If beta is zero, C does not have to be a valid input.
 *
 *
 * @note    transa is not support HPPL_OP_T.
 *
 */
extern void hl_matrix_dense_mul_csr(real *A_d,
                                    hl_trans_op_t transa,
                                    hl_sparse_matrix_s B_d,
                                    hl_trans_op_t transb,
                                    real *C_d,
                                    int dimM,
                                    int dimN,
                                    int dimK,
                                    real alpha,
                                    real beta);

/**
 * @brief   Memcpy csc_matrix to host.
 *
 * a. according to csc_matrix, update three arrays
 *
 *  1. csc_val, csc_row, csc_col are dest Address.
 *
 *  2. if type of csc_matrix is HL_NO_VALUE, update csc_row and csc_col
 *
 *  3. if type of csc_matrix is HL_FLOAT_VALUE, update csc_row,
 *     csc_col and csc_value.
 *
 * b. The interface is asynchronous copy. To ensure that the data is copied
 *     please call the synchronous interface;
 *
 *
 * @param[out]  csc_val     point to csc value array(nnz).
 * @param[in]   val_size    csc value size.
 * @param[out]  csc_row     point to csc row indices array(nnz).
 * @param[in]   row_size    csc row size.
 * @param[out]  csc_col     point to csc col indices array(dimN + 1).
 * @param[in]   col_size    csc column size.
 * @param[in]   csc_matrix  sparse matrix.
 * @param[in]   stream      hl_stream_t type.
 *
 */
extern void hl_memcpy_from_csc_matrix(real *csc_val,
                                      size_t val_size,
                                      int *csc_row,
                                      size_t row_size,
                                      int *csc_col,
                                      size_t col_size,
                                      hl_sparse_matrix_s csc_matrix,
                                      hl_stream_t stream);

/**
 * @brief   Memcpy sparse matrix to host.
 *
 * a. according to csr_matrix, update three arrays
 *
 *  1. csr_val, csr_row, csr_col are dest Address.
 *
 *  2. if type of csr_matrix is HL_NO_VALUE, update csr_row and csr_col
 *
 *  3. if type of csr_matrix is HL_FLOAT_VALUE, update csr_row,
 *     csr_col and csr_value
 *
 * b. The interface is asynchronous copy. To ensure that the data is copied
 *     please call the synchronous interface;
 *
 * @param[out]  csr_val     point to csr value array(nnz).
 * @param[in]   val_size    csr value size.
 * @param[out]  csr_row     point to csr row indices array(nnz).
 * @param[in]   row_size    csr row size.
 * @param[out]  csr_col     point to csr col indices array(dimN + 1).
 * @param[in]   col_size    csr column size.
 * @param[in]   csr_matrix  sparse matrix.
 * @param[in]   stream      hl_stream_t type.
 *
 */
extern void hl_memcpy_from_csr_matrix(real *csr_val,
                                      size_t val_size,
                                      int *csr_row,
                                      size_t row_size,
                                      int *csr_col,
                                      size_t col_size,
                                      hl_sparse_matrix_s csr_matrix,
                                      hl_stream_t stream);

/**
 * @brief   A_d[j] += B_d[i,j] for i in range(height)
 *
 * @param[in,out]   A_d    vector, size = width.
 * @param[in]       B_d    sparse matrix.
 * @param[in]       dimM   height.
 * @param[in]       dimN   width.
 * @param[in]       scale  scale of B_d
 *
 */
extern void hl_sparse_matrix_column_sum(
    real *A_d, hl_sparse_matrix_s B_d, int dimM, int dimN, real scale);
/**
 * @brief implementation of csr sparse matrix in hl_sparse_matirx_column_sum
 */
extern void hl_matrix_csr_column_sum(
    real *A_d, hl_sparse_matrix_s B_d, int dimM, int dimN, real scale);

/**
 * @brief   A_d[i,j] += B_d[j]
 *
 * @param[in,out]   A_d    sprare matrix.
 * @param[in]       B_d    vector, size = A_d.width.
 * @param[in]       scale  scale of B_d.
 *
 */
extern void hl_sparse_matrix_add_bias(hl_sparse_matrix_s A_d,
                                      real *B_d,
                                      real scale);
/**
 * @brief implementation of csr sparse matrix in hl_sparse_matrix_add_bias
 */
extern void hl_matrix_csr_add_bias(hl_sparse_matrix_s A_d,
                                   real *B_d,
                                   real scale);

/**
 * @brief   sparseMatrix = alpha * denseMatrix + beta *sparseMatrix
 *          A_d[i,j] = alpha * B_d[i,j] + beta * A_d[i,j]
 *          Only add value of same (row, col) index in dense matrix and
 *          do not use others values whoes postions are not in sparse matirx.
 *
 * @param[in,out]   A_d    sprare matrix.
 * @param[in]       B_d    dense matrix.
 * @param[in]       dimM   height of B_d.
 * @param[in]       dimN   width of B_d.
 * @param[in]       alpha  scale of B_d.
 * @param[in]       beta   scale of A_d.
 *
 */
extern void hl_sparse_matrix_add_dense(hl_sparse_matrix_s A_d,
                                       real *B_d,
                                       int dimM,
                                       int dimN,
                                       real alpha,
                                       real beta);
/**
 * @brief implementation of csr sparse matrix in hl_sparse_matrix_add_dense
 */
extern void hl_matrix_csr_add_dense(hl_sparse_matrix_s A_d,
                                    real *B_d,
                                    int dimM,
                                    int dimN,
                                    real alpha,
                                    real beta);

/**
 * @brief get rows pionter of GpuSparseMatrix
 *
 * @param[in]    sMat  sparse matrix
 *
 * @return   return rows pointer, which is gpu address
 *
 */
extern int *hl_sparse_matrix_get_rows(hl_sparse_matrix_s sMat);

/**
 * @brief get cols pionter of GpuSparseMatrix
 *
 * @param[in]    sMat  sparse matrix
 *
 * @return   return cols pointer, which is gpu address
 *
 */
extern int *hl_sparse_matrix_get_cols(hl_sparse_matrix_s sMat);

/**
 * @brief get value pionter of GpuSparseMatrix
 *
 * @param[in]    sMat  sparse matrix
 *
 * @return   return value pointer, which is gpu address
 *
 */
extern real *hl_sparse_matrix_get_value(hl_sparse_matrix_s sMat);

#endif /* HL_SPARSE_H_ */
