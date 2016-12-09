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


#include "hl_cuda.h"
#include "hl_sparse.h"
#include "hl_sparse.ph"
#include "hl_matrix_ops.cuh"
#include "hl_matrix_apply.cuh"
#include "hl_cuda_sparse.cuh"
#include "paddle/utils/Logging.h"

DEFINE_MATRIX_UNARY_PARAMETER_OP(mul_scalar, ONE_PARAMETER, a = a * p);
DEFINE_MATRIX_UNARY_OP(Zero, a = 0);

void hl_matrix_csr2dense(hl_sparse_matrix_s A_d,
                         real *C_d,
                         int dimM,
                         int dimN) {
  CHECK_NOTNULL(A_d);
  CHECK_NOTNULL(C_d);
  CHECK(dimM > 0 && dimN > 0 && A_d->rows == dimM && A_d->cols == dimN);
  CHECK(A_d->format == HL_SPARSE_CSR) << "matrix format error!";

  if (A_d->nnz == 0) {
    hl_gpu_apply_unary_op(
        unary::Zero<real>(), C_d, dimM, dimN, dimN);
    return;
  }

  /* nnz != 0 */
  hl_csr_matrix A_d2 = (hl_csr_matrix)(A_d->matrix);
  CHECK((A_d2->csr_val || A_d->type == HL_NO_VALUE) &&
        A_d2->csr_row && A_d2->csr_col) << "parameter transa error!";

  int blocksX = (dimN + CU_CSR2DENSE_THREAD_X - 1) / CU_CSR2DENSE_THREAD_X;
  int blocksY = (dimM + CU_CSR2DENSE_THREAD_X - 1) / CU_CSR2DENSE_THREAD_X;
  dim3 threads(CU_CSR2DENSE_THREAD_X, CU_CSR2DENSE_THREAD_X);
  dim3 grid(blocksX, blocksY);

  if (A_d->type == HL_NO_VALUE) {
    KeSMatrixCsr2Dense<0>
      <<<grid, threads, 0, STREAM_DEFAULT>>>(A_d2->csr_val,
                                             A_d2->csr_row,
                                             A_d2->csr_col,
                                             C_d,
                                             dimM,
                                             dimN);
  } else if (A_d->type == HL_FLOAT_VALUE) {
    KeSMatrixCsr2Dense<1>
      <<<grid, threads, 0, STREAM_DEFAULT>>>(A_d2->csr_val,
                                             A_d2->csr_row,
                                             A_d2->csr_col,
                                             C_d,
                                             dimM,
                                             dimN);
  } else {
  }
  CHECK_SYNC("hl_matrix_csr2dense failed");
}

void hl_matrix_csc2dense(hl_sparse_matrix_s A_d,
                         real *C_d,
                         int dimM,
                         int dimN) {
  CHECK_NOTNULL(A_d);
  CHECK_NOTNULL(C_d);
  CHECK(dimM > 0 && dimN > 0 && A_d->rows == dimM && A_d->cols == dimN);
  CHECK(A_d->format == HL_SPARSE_CSC) << "matrix format error!";

  if (A_d->nnz == 0) {
    hl_gpu_apply_unary_op(
        unary::Zero<real>(), C_d, dimM, dimN, dimN);
    return;
  }

  /* nnz != 0 */
  hl_csc_matrix A_d2 = (hl_csc_matrix)(A_d->matrix);
  CHECK((A_d2->csc_val || A_d->type == HL_NO_VALUE) &&
        A_d2->csc_row && A_d2->csc_col) << "parameter transa error!";

  int blocksX = (dimN + CU_CSR2DENSE_THREAD_X - 1) / CU_CSR2DENSE_THREAD_X;
  int blocksY = (dimM + CU_CSR2DENSE_THREAD_X - 1) / CU_CSR2DENSE_THREAD_X;
  dim3 threads(CU_CSR2DENSE_THREAD_X, CU_CSR2DENSE_THREAD_X);
  dim3 grid(blocksX, blocksY);

  if (A_d->type == HL_NO_VALUE) {
    KeSMatrixCsc2Dense<0>
      <<<grid, threads, 0, STREAM_DEFAULT>>>(A_d2->csc_val,
                                             A_d2->csc_row,
                                             A_d2->csc_col,
                                             C_d,
                                             dimM,
                                             dimN);
  } else if (A_d->type == HL_FLOAT_VALUE) {
    KeSMatrixCsc2Dense<1>
      <<<grid, threads, 0, STREAM_DEFAULT>>>(A_d2->csc_val,
                                             A_d2->csc_row,
                                             A_d2->csc_col,
                                             C_d,
                                             dimM,
                                             dimN);
  } else {
  }
  CHECK_SYNC("hl_matrix_csc2dense failed");
}

void hl_malloc_sparse_matrix(hl_sparse_matrix_s *A_d,
                             hl_matrix_format_t format,
                             hl_matrix_value_t  value_type,
                             int dimM,
                             int dimN,
                             int nnz) {
  CHECK_NOTNULL(A_d);
  CHECK(format == HL_SPARSE_CSR || format == HL_SPARSE_CSC)
    << "sparse matrix format error!";
  CHECK(value_type == HL_FLOAT_VALUE || value_type == HL_NO_VALUE)
    << "sparse matrix value type error!";
  /* avoid malloc 0 bytes */
  int nnz_s = (nnz == 0 ? 1 : nnz);

  if (format == HL_SPARSE_CSR) {
    CHECK(dimM > 0 && nnz >= 0) << "sparse matrix size error!";

    char* tmp = (char*)malloc(sizeof(_hl_sparse_matrix_s)
                              + sizeof(_hl_csr_matrix));
    CHECK_NOTNULL(tmp);

    hl_csr_matrix csr = (hl_csr_matrix)(tmp+sizeof(_hl_sparse_matrix_s));
    csr->sparsity = -1.0;

    if (value_type == HL_NO_VALUE) {
      csr->csr_val = NULL;
      csr->nnz_s = nnz_s;
      csr->row_s = dimM+1;
      csr->csr_row = (int*)hl_malloc_device((dimM+1)*sizeof(int));
      csr->csr_col = (int*)hl_malloc_device((nnz_s)*sizeof(int));

      *A_d = (hl_sparse_matrix_s)tmp;
      (*A_d)->matrix = (hl_matrix_s)csr;
    } else if (value_type == HL_FLOAT_VALUE) {
      csr->nnz_s = nnz_s;
      csr->row_s = dimM+1;
      csr->csr_val = (real*)hl_malloc_device((nnz_s)*sizeof(real));
      csr->csr_row = (int*)hl_malloc_device((dimM+1)*sizeof(int));
      csr->csr_col = (int*)hl_malloc_device((nnz_s)*sizeof(int));

      *A_d = (hl_sparse_matrix_s)tmp;
      (*A_d)->matrix = (hl_matrix_s)csr;
    }
  } else if (format == HL_SPARSE_CSC) {
    CHECK(dimM > 0 && nnz >= 0) << "sparse matrix size error!";

    char* tmp = (char*)malloc(sizeof(_hl_sparse_matrix_s)
                              + sizeof(_hl_csc_matrix));
    CHECK_NOTNULL(tmp);

    hl_csc_matrix csc = (hl_csc_matrix)(tmp+sizeof(_hl_sparse_matrix_s));
    csc->sparsity = -1.0f;

    if (value_type == HL_NO_VALUE) {
      csc->csc_val = NULL;
      csc->nnz_s = nnz_s;
      csc->col_s = dimN+1;
      csc->csc_row = (int*)hl_malloc_device((nnz_s)*sizeof(int));
      csc->csc_col = (int*)hl_malloc_device((dimN+1)*sizeof(int));

      *A_d = (hl_sparse_matrix_s)tmp;
      (*A_d)->matrix = (hl_matrix_s)csc;
    } else if (value_type == HL_FLOAT_VALUE) {
      csc->nnz_s = nnz_s;
      csc->col_s = dimN+1;
      csc->csc_val = (real*)hl_malloc_device((nnz_s)*sizeof(real));
      csc->csc_row = (int*)hl_malloc_device((nnz_s)*sizeof(int));
      csc->csc_col = (int*)hl_malloc_device((dimN+1)*sizeof(int));

      *A_d = (hl_sparse_matrix_s)tmp;
      (*A_d)->matrix = (hl_matrix_s)csc;
    }
  }

  (*A_d)->format = format;
  (*A_d)->type = value_type;
  (*A_d)->rows = dimM;
  (*A_d)->cols = dimN;
  (*A_d)->nnz = nnz;
}

void hl_free_sparse_matrix(hl_sparse_matrix_s A_d) {
  CHECK_NOTNULL(A_d);
  CHECK(A_d->format == HL_SPARSE_CSR || A_d->format == HL_SPARSE_CSC)
    << "sparse matrix format error!";

  if (A_d->matrix == NULL) {
    free(A_d);
    return;
  }

  if (A_d->format == HL_SPARSE_CSR) {
    hl_csr_matrix csr = (hl_csr_matrix)A_d->matrix;
    if (csr->csr_val != NULL) {
      hl_free_mem_device(csr->csr_val);
      csr->csr_val = NULL;
    }

    if (csr->csr_row != NULL) {
      hl_free_mem_device(csr->csr_row);
      csr->csr_row = NULL;
    }

    if (csr->csr_col != NULL) {
      hl_free_mem_device(csr->csr_col);
      csr->csr_col = NULL;
    }

    A_d->matrix = NULL;
    free(A_d);
  } else if (A_d->format == HL_SPARSE_CSC) {
    hl_csc_matrix csc = (hl_csc_matrix)A_d->matrix;
    if (csc->csc_val != NULL) {
      hl_free_mem_device(csc->csc_val);
      csc->csc_val = NULL;
    }

    if (csc->csc_row != NULL) {
      hl_free_mem_device(csc->csc_row);
      csc->csc_row = NULL;
    }

    if (csc->csc_col != NULL) {
      hl_free_mem_device(csc->csc_col);
      csc->csc_col = NULL;
    }

    A_d->matrix = NULL;
    free(A_d);
  }
}

void hl_construct_sparse_matrix(hl_sparse_matrix_s *A_d,
                                void * dest_d,
                                size_t size,
                                hl_matrix_format_t format,
                                hl_matrix_value_t  value_type,
                                int dimM,
                                int dimN,
                                int nnz) {
  CHECK_NOTNULL(A_d);
  CHECK(format == HL_SPARSE_CSR || format == HL_SPARSE_CSC)
    << "sparse matrix format error!";

  if (format == HL_SPARSE_CSR) {
    CHECK(dimM > 0 && nnz >= 0) << "sparse matrix size error!";

    size_t size_ = (dimM+1)*sizeof(int) + nnz*sizeof(int);
    if (value_type != HL_NO_VALUE) {
      size_ += nnz*sizeof(real);
    }
    CHECK_LE(size_, size) << "dest_d size(" << size
      << ") too small, should bigger than(" << size_ << ")!";

    char* tmp = (char*)malloc(sizeof(_hl_sparse_matrix_s)
                              + sizeof(_hl_csr_matrix));
    CHECK_NOTNULL(tmp);

    hl_csr_matrix csr = (hl_csr_matrix)(tmp+sizeof(_hl_sparse_matrix_s));

    if (value_type == HL_NO_VALUE) {
      csr->csr_val = NULL;
      csr->csr_row = (int*)dest_d;
      csr->csr_col = (int*)((char*)dest_d + (dimM+1)*sizeof(int));
    } else {
      csr->csr_val = (real*)dest_d;
      csr->csr_row = (int*)((char*)dest_d + nnz*sizeof(real));
      csr->csr_col = (int*)((char*)dest_d +
                            nnz*sizeof(real) +
                            (dimM+1)*sizeof(int));
    }
    csr->nnz_s = nnz;
    csr->row_s = dimM+1;
    csr->sparsity = -1.0;
    *A_d = (hl_sparse_matrix_s)tmp;
    (*A_d)->matrix = (hl_matrix_s)csr;
  } else if (format == HL_SPARSE_CSC) {
    CHECK(dimM > 0 && nnz >= 0) << "sparse matrix size error!";

    size_t size_ = (dimN+1)*sizeof(int) + nnz*sizeof(int);
    if (value_type != HL_NO_VALUE) {
      size_ += nnz*sizeof(real);
    }
    CHECK_LE(size_, size) << "dest_d size(" << size
      << ") too small, should bigger than(" << size_ << ")!";

    char* tmp = (char*)malloc(sizeof(_hl_sparse_matrix_s)
                              + sizeof(_hl_csc_matrix));
    CHECK_NOTNULL(tmp);

    hl_csc_matrix csc = (hl_csc_matrix)(tmp+sizeof(_hl_sparse_matrix_s));
    if (value_type == HL_NO_VALUE) {
      csc->csc_val = NULL;
      csc->csc_col = (int*)dest_d;
      csc->csc_row = (int*)((char*)dest_d + (dimN+1)*sizeof(int));
    } else {
      csc->csc_val = (real*)dest_d;
      csc->csc_col = (int*)((char*)dest_d + nnz*sizeof(real));
      csc->csc_row = (int*)((char*)dest_d +
                            nnz*sizeof(real) +
                            (dimN+1)*sizeof(int));
    }
    csc->nnz_s = nnz;
    csc->col_s = dimN+1;
    csc->sparsity = -1.0f;
    *A_d = (hl_sparse_matrix_s)tmp;
    (*A_d)->matrix = (hl_matrix_s)csc;
  }

  (*A_d)->format = format;
  (*A_d)->type = value_type;
  (*A_d)->rows = dimM;
  (*A_d)->cols = dimN;
  (*A_d)->nnz = nnz;
}

void hl_construct_sparse_matrix(hl_sparse_matrix_s *A_d,
                                real* value_d,
                                int* rows_d,
                                int* cols_d,
                                hl_matrix_format_t format,
                                hl_matrix_value_t  value_type,
                                int dimM,
                                int dimN,
                                int nnz) {
  CHECK_NOTNULL(A_d);
  CHECK(dimM > 0 && nnz >= 0) << "sparse matrix size error!";

  CHECK(format == HL_SPARSE_CSR || format == HL_SPARSE_CSC)
    << "sparse matrix format error!";

  if (format == HL_SPARSE_CSR) {
    char* tmp = (char*)malloc(sizeof(_hl_sparse_matrix_s)
                              + sizeof(_hl_csr_matrix));
    CHECK_NOTNULL(tmp);

    hl_csr_matrix csr = (hl_csr_matrix)(tmp + sizeof(_hl_sparse_matrix_s));
    csr->csr_row = rows_d;
    csr->csr_col = cols_d;
    csr->csr_val = value_d;
    csr->nnz_s = nnz;
    csr->row_s = dimM + 1;
    csr->sparsity = -1.0;
    *A_d = (hl_sparse_matrix_s)tmp;
    (*A_d)->matrix = (hl_matrix_s)csr;
  } else if (format == HL_SPARSE_CSC) {
    char* tmp = (char*)malloc(sizeof(_hl_sparse_matrix_s)
                              + sizeof(_hl_csc_matrix));
    CHECK_NOTNULL(tmp);

    hl_csc_matrix csc = (hl_csc_matrix)(tmp + sizeof(_hl_sparse_matrix_s));
    csc->csc_row = rows_d;
    csc->csc_col = cols_d;
    csc->csc_val = value_d;
    csc->nnz_s = nnz;
    csc->col_s = dimN + 1;
    csc->sparsity = -1.0f;
    *A_d = (hl_sparse_matrix_s)tmp;
    (*A_d)->matrix = (hl_matrix_s)csc;
  }

  (*A_d)->format = format;
  (*A_d)->type = value_type;
  (*A_d)->rows = dimM;
  (*A_d)->cols = dimN;
  (*A_d)->nnz = nnz;
}

void hl_destruct_sparse_matrix(hl_sparse_matrix_s A_d) {
  CHECK_NOTNULL(A_d);
  free(A_d);
}

void hl_memcpy_csr_matrix(hl_sparse_matrix_s csr_matrix,
                          real *csr_val,
                          int *csr_row,
                          int *csr_col,
                          hl_stream_t stream) {
  CHECK_NOTNULL(csr_matrix);
  CHECK_EQ(csr_matrix->format, HL_SPARSE_CSR)
    << "csr_matrix is not csr format!";
  CHECK_NOTNULL(csr_matrix->matrix);

  hl_csr_matrix csr = (hl_csr_matrix)(csr_matrix->matrix);
  CHECK_LE(csr_matrix->nnz, csr->nnz_s)
    << "copy size " << csr_matrix->nnz
    << " is big than alloc size " << csr->nnz_s;

  CHECK_LE((csr_matrix->rows+1), csr->row_s)
    << "copy size " << (csr_matrix->rows + 1)
    << " is big than alloc size " << csr->row_s;

  CHECK(csr_matrix->type == HL_FLOAT_VALUE ||
        csr_matrix->type == HL_NO_VALUE)
        << "sparse matrix value type error!";

  if (csr_matrix->type == HL_NO_VALUE) {
    if (csr_row == NULL && csr_col == NULL) {
      return;
    } else if (csr_row != NULL && csr_col != NULL) {
      hl_memcpy_async(csr->csr_row,
                      csr_row,
                      (csr_matrix->rows+1)*sizeof(int),
                      stream);

      hl_memcpy_async(csr->csr_col,
                      csr_col,
                      (csr_matrix->nnz)*sizeof(int),
                      stream);
    } else {
      LOG(FATAL) << "parameter csr_row or csr_col is null pointer!";
    }
  } else if (csr_matrix->type == HL_FLOAT_VALUE) {
    if (csr_val == NULL && csr_row == NULL && csr_col == NULL) {
      return;
    } else if (csr_val != NULL && csr_row == NULL && csr_col == NULL) {
      hl_memcpy_async(csr->csr_val,
                      csr_val,
                      (csr_matrix->nnz)*sizeof(real),
                      stream);
    } else if (csr_val != NULL && csr_row != NULL && csr_col != NULL) {
      hl_memcpy_async(csr->csr_val,
                      csr_val,
                      (csr_matrix->nnz)*sizeof(real),
                      stream);
      hl_memcpy_async(csr->csr_row,
                      csr_row,
                      (csr_matrix->rows+1)*sizeof(int),
                      stream);
      hl_memcpy_async(csr->csr_col,
                      csr_col,
                      (csr_matrix->nnz)*sizeof(int),
                      stream);
    } else {
      LOG(FATAL) << "parameter csr_row or csr_col is null pointer!";
    }
  }

  csr->sparsity = ((float)csr_matrix->nnz) /
                  ((float)csr_matrix->rows) /
                  ((float)csr_matrix->cols);
}

void hl_memcpy_csc_matrix(hl_sparse_matrix_s csc_matrix,
                          real *csc_val,
                          int *csc_row,
                          int *csc_col,
                          hl_stream_t stream) {
  CHECK_NOTNULL(csc_matrix);
  CHECK_EQ(csc_matrix->format, HL_SPARSE_CSC)
    << "csc_matrix is not csc format error!";

  hl_csc_matrix csc = (hl_csc_matrix)(csc_matrix->matrix);
  CHECK_LE(csc_matrix->nnz, csc->nnz_s)
    << "copy size " << csc_matrix->nnz
    << " is big than alloc size " << csc->nnz_s;

  CHECK_LE((csc_matrix->cols+1), csc->col_s)
    << "copy size " <<(csc_matrix->cols + 1)
    << " is big than alloc size " << csc->col_s;

  CHECK(csc_matrix->type == HL_FLOAT_VALUE ||
        csc_matrix->type == HL_NO_VALUE)
        << "sparse matrix value type error!";

  if (csc_matrix->type == HL_NO_VALUE) {
    if (csc_row == NULL && csc_col == NULL) {
      return;
    } else if (csc_row != NULL && csc_col != NULL) {
      hl_memcpy_async(csc->csc_row,
                      csc_row,
                      (csc_matrix->nnz)*sizeof(int),
                      stream);
      hl_memcpy_async(csc->csc_col,
                      csc_col,
                      (csc_matrix->cols+1)*sizeof(int),
                      stream);
    } else {
      LOG(FATAL) << "parameter csc_row or csc_col is null pointer!";
    }
  } else if (csc_matrix->type == HL_FLOAT_VALUE) {
    if (csc_val == NULL && csc_row == NULL && csc_col == NULL) {
      return;
    } else if (csc_val != NULL && csc_row == NULL && csc_col == NULL) {
      hl_memcpy_async(csc->csc_val,
                      csc_val,
                      (csc_matrix->nnz)*sizeof(real),
                      stream);
    } else if (csc_val != NULL && csc_row != NULL && csc_col != NULL) {
      hl_memcpy_async(csc->csc_val,
                      csc_val,
                      (csc_matrix->nnz)*sizeof(real),
                      stream);
      hl_memcpy_async(csc->csc_row,
                      csc_row,
                      (csc_matrix->nnz)*sizeof(int),
                      stream);
      hl_memcpy_async(csc->csc_col,
                      csc_col,
                      (csc_matrix->cols+1)*sizeof(int),
                      stream);
    } else {
      LOG(FATAL) << "parameter csc_row or csc_col is null pointer!";
    }
  }

  csc->sparsity = ((float)csc_matrix->nnz) /
                  ((float)csc_matrix->rows) /
                  ((float)csc_matrix->cols);
}

void hl_memcpy_sparse_matrix(hl_sparse_matrix_s dst,
                             hl_sparse_matrix_s src,
                             hl_stream_t stream) {
  CHECK(dst && src && dst->matrix && src->matrix)
    << "parameter dst or src is null pointer!";
  CHECK_EQ(dst->format, src->format)
    << "sparse matrix format does not match!";
  CHECK(dst->type != HL_FLOAT_VALUE || src->type != HL_NO_VALUE)
    << "src sparse matrix is no value, dst sparse matrix has value!";

  if (dst->format == HL_SPARSE_CSR) {
    dst->rows = src->rows;
    dst->cols = src->cols;
    dst->nnz  = src->nnz;
    hl_csr_matrix csr = (hl_csr_matrix)src->matrix;
    hl_memcpy_csr_matrix(dst,
                         csr->csr_val,
                         csr->csr_row,
                         csr->csr_col,
                         stream);
  } else if (dst->format == HL_SPARSE_CSC) {
    dst->rows = src->rows;
    dst->cols = src->cols;
    dst->nnz  = src->nnz;
    hl_csc_matrix csc = (hl_csc_matrix)src->matrix;
    hl_memcpy_csc_matrix(dst,
                         csc->csc_val,
                         csc->csc_row,
                         csc->csc_col,
                         stream);
  } else {
    LOG(FATAL) << "sparse matrix format error!";
  }
}

/**
 * Calculate beta * C, if beta is zero, C does not have to be a valid input.
 */
static void _beta_mul_c(real *c, int dimM, int dimN, real beta) {
  if (beta == 0.0) {
    hl_gpu_apply_unary_op(unary::Zero<real>(), c, dimM, dimN, dimN);
  } else {
    if (beta != 1.0){
      hl_gpu_apply_unary_op(
        unary::mul_scalar<real>(beta), c, dimM, dimN, dimN);
    }
  }

  return;
}

void hl_matrix_csr_mul_dense(hl_sparse_matrix_s A_d, hl_trans_op_t transa,
                             real *B_d, hl_trans_op_t transb,
                             real *C_d,
                             int dimM, int dimN, int dimK,
                             real alpha, real beta) {
  CHECK_EQ(transb, HPPL_OP_N);
  CHECK_NOTNULL(A_d);
  CHECK_NOTNULL(B_d);
  CHECK_NOTNULL(C_d);
  CHECK(dimM > 0 && dimN > 0 && dimK > 0);
  CHECK_EQ(A_d->format, HL_SPARSE_CSR) << "matrix format error!";

  if ((HPPL_OP_N == transa && (A_d->rows != dimM || A_d->cols != dimK)) ||
      (HPPL_OP_T == transa && (A_d->rows != dimK || A_d->cols != dimM))) {
      LOG(FATAL) << "parameter error!";
  }

  if (A_d->nnz == 0) {
    _beta_mul_c(C_d, dimM, dimN, beta);
    return;
  }

  /* nnz != 0 */
  hl_csr_matrix A_d2 = (hl_csr_matrix)(A_d->matrix);
  if ((A_d2->csr_val == NULL && A_d->type != HL_NO_VALUE) ||
       A_d2->csr_row == NULL ||
       A_d2->csr_col == NULL) {
    LOG(FATAL) << "parameter error!";
  }

  if (HPPL_OP_N == transa) {
    int blocksX = (dimN + CU_CSRMM_BLOCK_N - 1) / CU_CSRMM_BLOCK_N;
    int blocksY = (dimM + CU_CSRMM_THREAD_Y - 1) / CU_CSRMM_THREAD_Y;
    dim3 threads(CU_CSRMM_THREAD_X, CU_CSRMM_THREAD_Y);
    dim3 grid(blocksX, blocksY);

    /* sparsity pattern */
    // A_d->sparsity;
    if (A_d->type == HL_NO_VALUE) {
      KeSMatrixCsrMulDense<0>
        <<<grid, threads, 0, STREAM_DEFAULT>>>(C_d,
                                               A_d2->csr_val,
                                               A_d2->csr_col,
                                               A_d2->csr_row,
                                               B_d,
                                               dimM,
                                               dimN,
                                               dimK,
                                               alpha,
                                               beta);
    } else {
      KeSMatrixCsrMulDense<1>
        <<<grid, threads, 0, STREAM_DEFAULT>>>(C_d,
                                               A_d2->csr_val,
                                               A_d2->csr_col,
                                               A_d2->csr_row,
                                               B_d,
                                               dimM,
                                               dimN,
                                               dimK,
                                               alpha,
                                               beta);
    }
  } else if (HPPL_OP_T == transa) {
    _beta_mul_c(C_d, dimM, dimN, beta);

    int blocksX = (dimN + CU_CSC_MUL_DENSE_BLOCK_N - 1) /
                  CU_CSC_MUL_DENSE_BLOCK_N;
    int blocksY = (dimK + CU_CSC_MUL_DENSE_BLOCK_K - 1) /
                  CU_CSC_MUL_DENSE_BLOCK_K;
    dim3 threads(CU_CSC_MUL_DENSE_THREAD_X, CU_CSC_MUL_DENSE_THREAD_Y);
    dim3 grid(blocksX, blocksY);
    if (A_d->type == HL_NO_VALUE) {
      KeSMatrixCscMulDense<0>
        <<<grid, threads, 0, STREAM_DEFAULT>>>(C_d,
                                               A_d2->csr_val,
                                               A_d2->csr_col,
                                               A_d2->csr_row,
                                               B_d,
                                               dimM,
                                               dimN,
                                               dimK,
                                               alpha,
                                               beta);
    } else {
      KeSMatrixCscMulDense<1>
        <<<grid, threads, 0, STREAM_DEFAULT>>>(C_d,
                                               A_d2->csr_val,
                                               A_d2->csr_col,
                                               A_d2->csr_row,
                                               B_d,
                                               dimM,
                                               dimN,
                                               dimK,
                                               alpha,
                                               beta);
    }
  } else {
    LOG(FATAL) << "parameter transa error!";
  }

  CHECK_SYNC("hl_matrix_csr_mul_dense failed");
}

void hl_matrix_dense_mul_csc(real *A_d, hl_trans_op_t transa,
                             hl_sparse_matrix_s B_d, hl_trans_op_t transb,
                             real *C_d,
                             int dimM, int dimN, int dimK,
                             real alpha, real beta) {
  CHECK_EQ(transa, HPPL_OP_N);
  CHECK_NOTNULL(A_d);
  CHECK_NOTNULL(B_d);
  CHECK_NOTNULL(C_d);

  if (dimM <= 0 || dimN <= 0 || dimK <= 0 ||
      ((transb == HPPL_OP_N) && (B_d->rows != dimK || B_d->cols != dimN)) ||
      ((transb == HPPL_OP_T) && (B_d->rows != dimN || B_d->cols != dimK))) {
    LOG(FATAL) << "parameter dims error!";
  }

  CHECK_EQ(B_d->format, HL_SPARSE_CSC)
    << "matrix format error!";

  if (B_d->nnz == 0) {
    _beta_mul_c(C_d, dimM, dimN, beta);
    return;
  }

  /* nnz != 0 */
  hl_csc_matrix B_d2 = (hl_csc_matrix)(B_d->matrix);
  if ((B_d2->csc_val == NULL && B_d->type != HL_NO_VALUE) ||
       B_d2->csc_row == NULL ||
       B_d2->csc_col == NULL) {
    LOG(FATAL) << "parameter B is null!";
  }

  if (transb == HPPL_OP_N) {
    int blocksX = (dimM + CU_CSCMM_BLOCK_M_BEST - 1) / CU_CSCMM_BLOCK_M_BEST;
    int blocksY = (dimN + CU_CSCMM_BLOCK_N_BEST - 1) / CU_CSCMM_BLOCK_N_BEST;
    dim3 threads(CU_CSCMM_THREAD_X_BEST, CU_CSCMM_THREAD_Y_BEST);
    dim3 grid(blocksX, blocksY);

    if (B_d->type == HL_NO_VALUE) {
      KeSMatrixDenseMulCsc<0>
        <<<grid, threads, 0, STREAM_DEFAULT>>>(C_d,
                                               A_d,
                                               B_d2->csc_val,
                                               B_d2->csc_row,
                                               B_d2->csc_col,
                                               dimM,
                                               dimN,
                                               dimK,
                                               alpha,
                                               beta);
    } else {
      KeSMatrixDenseMulCsc<1>
        <<<grid, threads, 0, STREAM_DEFAULT>>>(C_d,
                                               A_d,
                                               B_d2->csc_val,
                                               B_d2->csc_row,
                                               B_d2->csc_col,
                                               dimM,
                                               dimN,
                                               dimK,
                                               alpha,
                                               beta);
    }
  } else if (transb == HPPL_OP_T) {
    _beta_mul_c(C_d, dimM, dimN, beta);
    int blocksX = 1 + (dimK-1)/CU_DM_CSR_THREAD_X;
    int blocksY = 1 + (dimM-1)/CU_DM_CSR_BLOCK_M;
    dim3 threads(CU_DM_CSR_THREAD_X, CU_DM_CSR_THREAD_Y);
    dim3 grid(blocksX, blocksY);
    if (B_d->type == HL_NO_VALUE) {
      KeSMatrixDenseMulCsr<0>
        <<<grid, threads, 0, STREAM_DEFAULT>>>(C_d,
                                               A_d,
                                               B_d2->csc_val,
                                               B_d2->csc_col,
                                               B_d2->csc_row,
                                               dimM,
                                               dimN,
                                               dimK,
                                               alpha,
                                               beta);
    } else {
      KeSMatrixDenseMulCsr<1>
        <<<grid, threads, 0, STREAM_DEFAULT>>>(C_d,
                                               A_d,
                                               B_d2->csc_val,
                                               B_d2->csc_col,
                                               B_d2->csc_row,
                                               dimM,
                                               dimN,
                                               dimK,
                                               alpha,
                                               beta);
    }
  } else {
    LOG(FATAL) << "parameter transb error!";
  }

  CHECK_SYNC("hl_matrix_dense_mul_csc failed");
}

void hl_matrix_dense_mul_csr(real *A_d, hl_trans_op_t transa,
                             hl_sparse_matrix_s B_d, hl_trans_op_t transb,
                             real *C_d,
                             int dimM, int dimN, int dimK,
                             real alpha, real beta) {
  CHECK_EQ(transa, HPPL_OP_N);
  CHECK_NOTNULL(A_d);
  CHECK_NOTNULL(B_d);
  CHECK_NOTNULL(C_d);

  if (dimM <= 0 || dimN <= 0 || dimK <= 0
      || (transb == HPPL_OP_N && (B_d->rows != dimK || B_d->cols != dimN))
      || (transb == HPPL_OP_T && (B_d->rows != dimN || B_d->cols != dimK))) {
    LOG(FATAL) << "parameter dims error!";
  }

  CHECK_EQ(B_d->format, HL_SPARSE_CSR)
    << "matrix format error!";

  if (B_d->nnz == 0) {
    _beta_mul_c(C_d, dimM, dimN, beta);
    return;
  }

  /* nnz != 0 */
  hl_csr_matrix B_d2 = (hl_csr_matrix)(B_d->matrix);
  if ((B_d2->csr_val == NULL && B_d->type != HL_NO_VALUE) ||
       B_d2->csr_row == NULL ||
       B_d2->csr_col == NULL) {
    LOG(FATAL) << "parameter transa error!";
  }

  if (transb == HPPL_OP_N) {
    _beta_mul_c(C_d, dimM, dimN, beta);
    int blocksX = 1 + (dimK-1)/CU_DM_CSR_THREAD_X;
    int blocksY = 1 + (dimM-1)/CU_DM_CSR_BLOCK_M;
    dim3 threads(CU_DM_CSR_THREAD_X, CU_DM_CSR_THREAD_Y);
    dim3 grid(blocksX, blocksY);
    if (B_d->type == HL_NO_VALUE) {
      KeSMatrixDenseMulCsr<0>
        <<<grid, threads, 0, STREAM_DEFAULT>>>(C_d,
                                               A_d,
                                               B_d2->csr_val,
                                               B_d2->csr_row,
                                               B_d2->csr_col,
                                               dimM,
                                               dimN,
                                               dimK,
                                               alpha,
                                               beta);
    } else {
      KeSMatrixDenseMulCsr<1>
        <<<grid, threads, 0, STREAM_DEFAULT>>>(C_d,
                                               A_d,
                                               B_d2->csr_val,
                                               B_d2->csr_row,
                                               B_d2->csr_col,
                                               dimM,
                                               dimN,
                                               dimK,
                                               alpha,
                                               beta);
    }
  } else if (transb == HPPL_OP_T) {
    int blocksX = (dimM + CU_CSCMM_BLOCK_M_BEST - 1) / CU_CSCMM_BLOCK_M_BEST;
    int blocksY = (dimN + CU_CSCMM_BLOCK_N_BEST - 1) / CU_CSCMM_BLOCK_N_BEST;
    dim3 threads(CU_CSCMM_THREAD_X_BEST, CU_CSCMM_THREAD_Y_BEST);
    dim3 grid(blocksX, blocksY);
    if (B_d->type == HL_NO_VALUE) {
      KeSMatrixDenseMulCsc<0>
        <<<grid, threads, 0, STREAM_DEFAULT>>>(C_d,
                                               A_d,
                                               B_d2->csr_val,
                                               B_d2->csr_col,
                                               B_d2->csr_row,
                                               dimM,
                                               dimN,
                                               dimK,
                                               alpha,
                                               beta);
    } else {
      KeSMatrixDenseMulCsc<1>
        <<<grid, threads, 0, STREAM_DEFAULT>>>(C_d,
                                               A_d,
                                               B_d2->csr_val,
                                               B_d2->csr_col,
                                               B_d2->csr_row,
                                               dimM,
                                               dimN,
                                               dimK,
                                               alpha,
                                               beta);
    }
  } else {
    LOG(FATAL) << "parameter transb error!";
  }

  CHECK_SYNC("hl_matrix_dense_mul_csr failed");
}

void hl_matrix_csc_mul_dense(hl_sparse_matrix_s A_d, hl_trans_op_t transa,
                             real *B_d, hl_trans_op_t transb,
                             real *C_d,
                             int dimM, int dimN, int dimK,
                             real alpha, real beta) {
  CHECK_EQ(transb, HPPL_OP_N);
  CHECK_NOTNULL(A_d);
  CHECK_NOTNULL(B_d);
  CHECK_NOTNULL(C_d);
  CHECK(dimM > 0 && dimN > 0 && dimK > 0) << "parameter error!";
  CHECK_EQ(A_d->format, HL_SPARSE_CSC) << "matrix format error!";

  if ((HPPL_OP_N == transa && (A_d->rows != dimM || A_d->cols != dimK)) ||
      (HPPL_OP_T == transa && (A_d->rows != dimK || A_d->cols != dimM))) {
    LOG(FATAL) << "parameter error!";
  }

  if (A_d->nnz == 0) {
    _beta_mul_c(C_d, dimM, dimN, beta);
    return;
  }

  /* nnz != 0 */
  hl_csc_matrix A_d2 = (hl_csc_matrix)(A_d->matrix);
  if ((A_d2->csc_val == NULL && A_d->type != HL_NO_VALUE) ||
       A_d2->csc_row == NULL ||
       A_d2->csc_col == NULL) {
    LOG(FATAL) << "parameter error!";
  }

  if (HPPL_OP_N == transa) {
    _beta_mul_c(C_d, dimM, dimN, beta);

    int blocksX = (dimN + CU_CSC_MUL_DENSE_BLOCK_N -1)/CU_CSC_MUL_DENSE_BLOCK_N;
    int blocksY = (dimK + CU_CSC_MUL_DENSE_BLOCK_K -1)/CU_CSC_MUL_DENSE_BLOCK_K;
    dim3 threads(CU_CSC_MUL_DENSE_THREAD_X, CU_CSC_MUL_DENSE_THREAD_Y);
    dim3 grid(blocksX, blocksY);
    if (A_d->type == HL_NO_VALUE) {
      KeSMatrixCscMulDense<0>
        <<<grid, threads, 0, STREAM_DEFAULT>>>(C_d,
                                               A_d2->csc_val,
                                               A_d2->csc_row,
                                               A_d2->csc_col,
                                               B_d,
                                               dimM,
                                               dimN,
                                               dimK,
                                               alpha,
                                               beta);
    } else {
      KeSMatrixCscMulDense<1>
        <<<grid, threads, 0, STREAM_DEFAULT>>>(C_d,
                                               A_d2->csc_val,
                                               A_d2->csc_row,
                                               A_d2->csc_col,
                                               B_d,
                                               dimM,
                                               dimN,
                                               dimK,
                                               alpha,
                                               beta);
    }
  } else if (HPPL_OP_T == transa) {
    int blocksX = (dimN + CU_CSRMM_BLOCK_N - 1) / CU_CSRMM_BLOCK_N;
    int blocksY = (dimM + CU_CSRMM_THREAD_Y - 1) / CU_CSRMM_THREAD_Y;
    dim3 threads(CU_CSRMM_THREAD_X, CU_CSRMM_THREAD_Y);
    dim3 grid(blocksX, blocksY);

    /* sparsity pattern */
    // A_d->sparsity;
    if (A_d->type == HL_NO_VALUE) {
      KeSMatrixCsrMulDense<0>
        <<<grid, threads, 0, STREAM_DEFAULT>>>(C_d,
                                               A_d2->csc_val,
                                               A_d2->csc_row,
                                               A_d2->csc_col,
                                               B_d,
                                               dimM,
                                               dimN,
                                               dimK,
                                               alpha,
                                               beta);
    } else {
      KeSMatrixCsrMulDense<1>
        <<<grid, threads, 0, STREAM_DEFAULT>>>(C_d,
                                               A_d2->csc_val,
                                               A_d2->csc_row,
                                               A_d2->csc_col,
                                               B_d,
                                               dimM,
                                               dimN,
                                               dimK,
                                               alpha,
                                               beta);
    }
  } else {
    LOG(FATAL) << "parameter transa error!";
  }

  CHECK_SYNC("hl_matrix_csc_mul_dense failed");
}

void hl_sparse_matrix_mul(real *A_d, hl_trans_op_t transa,
                          real *B_d, hl_trans_op_t transb,
                          hl_sparse_matrix_s  C_d,
                          int dimM, int dimN, int dimK,
                          real alpha, real beta) {
  CHECK_NOTNULL(A_d);
  CHECK_NOTNULL(B_d);
  CHECK_NOTNULL(C_d);
  CHECK(dimM > 0 && dimN > 0 && dimK > 0) << "parameter error!";
  CHECK_NE(C_d->type, HL_NO_VALUE) << "C value type error!";

  if (C_d->nnz == 0) return;

  if (C_d->format == HL_SPARSE_CSC) {
    hl_csc_matrix C_d2 = (hl_csc_matrix)(C_d->matrix);
    if (C_d2->csc_val == NULL ||
        C_d2->csc_row == NULL ||
        C_d2->csc_col == NULL) {
      LOG(FATAL) << "parameter error!";
    }

    if (beta != 1.0) {
      hl_gpu_apply_unary_op(unary::mul_scalar<real>(beta),
                            C_d2->csc_val,
                            1,
                            C_d->nnz,
                            C_d->nnz);
    }

    int blocksX = dimN;
    int blocksY = 1;
    dim3 threads(CU_CSCMM_DMD2CSC_THREAD_X, 1);
    dim3 grid(blocksX, blocksY);
    bool transA = transa == HPPL_OP_T ? 1 : 0;
    bool transB = transb == HPPL_OP_T ? 1 : 0;
    KeSMatrixDenseMulDense2CSC
      <<<grid, threads, 0, STREAM_DEFAULT>>>(C_d2->csc_val,
                                             C_d2->csc_row,
                                             C_d2->csc_col,
                                             A_d,
                                             B_d,
                                             transA,
                                             transB,
                                             dimM,
                                             dimN,
                                             dimK,
                                             alpha,
                                             beta);
    CHECK_SYNC("hl_sparse_matrix_mul failed");
  } else {
    hl_csr_matrix C_d2 = (hl_csr_matrix)(C_d->matrix);
    if ((C_d2->csr_val == NULL && C_d->type != HL_NO_VALUE) ||
         C_d2->csr_row == NULL ||
         C_d2->csr_col == NULL) {
      LOG(FATAL) << "parameter error!";
    }

    if (beta != 1.0) {
      hl_gpu_apply_unary_op(unary::mul_scalar<real>(beta),
                            C_d2->csr_val,
                            1,
                            C_d->nnz,
                            C_d->nnz);
    }

    bool transA = transa == HPPL_OP_T ? 1 : 0;
    bool transB = transb == HPPL_OP_T ? 1 : 0;
    if (!transB) {
      int blocksX = dimM;
      int blocksY = 1;
      dim3 threads(CU_CSCMM_DMD2CSR_THREAD_X, 1);
      dim3 grid(blocksX, blocksY);

      KeSMatrixDenseMulDense2CSR
        <<<grid, threads, 0, STREAM_DEFAULT>>>(C_d2->csr_val,
                                               C_d2->csr_row,
                                               C_d2->csr_col,
                                               A_d,
                                               B_d,
                                               transA,
                                               transB,
                                               dimM,
                                               dimN,
                                               dimK,
                                               alpha,
                                               beta);
     CHECK_SYNC("hl_sparse_matrix_mul failed");
    } else {
      CHECK(!transA) << "Not supported A is trans and B is not trans!";

      dim3 block(CU_BLOCK_SIZE, 1);
      int avgNnzPerRow = C_d->nnz / dimM;
      avgNnzPerRow = avgNnzPerRow > 0 ? avgNnzPerRow : 1;
      int gridx = DIVUP(avgNnzPerRow, CU_BLOCK_SIZE);
      dim3 grid(gridx, dimM);
      KeSMatrixDenseMulDenseTrans2CSR
         <<<grid, block, 0, STREAM_DEFAULT>>>(C_d2->csr_val,
                                               C_d2->csr_row,
                                               C_d2->csr_col,
                                               A_d,
                                               B_d,
                                               transA,
                                               transB,
                                               dimM,
                                               dimN,
                                               dimK,
                                               alpha,
                                               beta);
     CHECK_SYNC("hl_sparse_matrix_mul failed");
   }
  }
}

void hl_memcpy_from_csc_matrix(real *csc_val,
                               size_t val_size,
                               int *csc_row,
                               size_t row_size,
                               int *csc_col,
                               size_t col_size,
                               hl_sparse_matrix_s csc_matrix,
                               hl_stream_t stream) {
  CHECK_NOTNULL(csc_matrix);
  CHECK_NOTNULL(csc_row);
  CHECK_NOTNULL(csc_col);

  CHECK_EQ(csc_matrix->format, HL_SPARSE_CSC)
     << "csc_matrix is not csc format error!";

  if (csc_matrix->nnz > row_size ||
      csc_matrix->cols + 1 > static_cast<int>(col_size)) {
    LOG(FATAL) << "size not match!";
  }

  hl_csc_matrix csc = (hl_csc_matrix)(csc_matrix->matrix);
  hl_memcpy_async((void*)csc_row,
                  (void*)csc->csc_row,
                  (csc_matrix->nnz) * sizeof(int),
                  stream);
  hl_memcpy_async((void*)csc_col,
                  (void*)csc->csc_col,
                  (csc_matrix->cols + 1) * sizeof(int),
                  stream);
  if (csc_matrix->type == HL_FLOAT_VALUE) {
    if (csc_val != NULL) {
      CHECK_LE(csc_matrix->nnz, val_size) << "size not match!";
      hl_memcpy_async((void*)csc_val,
                      (void*)csc->csc_val,
                      (csc_matrix->nnz)*sizeof(real),
                      stream);
    } else {
      LOG(FATAL) << "parameter csr_val is null pointer!";
    }
  }
}

void hl_memcpy_from_csr_matrix(real *csr_val,
                               size_t val_size,
                               int *csr_row,
                               size_t row_size,
                               int *csr_col,
                               size_t col_size,
                               hl_sparse_matrix_s csr_matrix,
                               hl_stream_t stream) {
  CHECK_NOTNULL(csr_matrix);
  CHECK_NOTNULL(csr_row);
  CHECK_NOTNULL(csr_col);
  CHECK_EQ(csr_matrix->format, HL_SPARSE_CSR)
    << "csr_matrix is not csr format error!";

  if (csr_matrix->nnz > col_size ||
      csr_matrix->rows + 1 > static_cast<int>(row_size)) {
    LOG(FATAL) << "size not match!";
  }

  hl_csr_matrix csr = (hl_csr_matrix)(csr_matrix->matrix);
  hl_memcpy_async((void*)csr_row,
                  (void*)csr->csr_row,
                  (csr_matrix->rows+1)*sizeof(int),
                  stream);
  hl_memcpy_async((void*)csr_col,
                  (void*)csr->csr_col,
                  (csr_matrix->nnz)*sizeof(int),
                  stream);
  if (csr_matrix->type == HL_FLOAT_VALUE) {
    if (csr_val != NULL) {
      CHECK_LE(csr_matrix->nnz, val_size) << "size not match!";
      hl_memcpy_async((void*)csr_val,
                      (void*)csr->csr_val,
                      (csr_matrix->nnz)*sizeof(real),
                      stream);
    } else {
      LOG(FATAL) << "parameter csr_val is null pointer!";
    }
  }
}

void hl_sparse_matrix_column_sum(real* A_d, hl_sparse_matrix_s B_d, int dimM,
                                 int dimN, real scale) {
  if (B_d->format == HL_SPARSE_CSR) {
    hl_matrix_csr_column_sum(A_d, B_d, dimM, dimN, scale);
  } else {
    LOG(FATAL) << "Not support CSC format error!";
  }
}

void hl_matrix_csr_column_sum(real* A_d, hl_sparse_matrix_s B_d,
                              int dimM, int dimN, real scale) {
  CHECK_NOTNULL(A_d);
  CHECK_NOTNULL(B_d);

  if (dimM <= 0 || dimN <= 0 || (B_d->rows != dimM || B_d->cols != dimN)) {
    LOG(FATAL) << "parameter dims error!";
  }

  hl_csr_matrix B_d2 = (hl_csr_matrix)(B_d->matrix);
  if ((B_d2->csr_val == NULL && B_d->type != HL_NO_VALUE) ||
      B_d2->csr_row == NULL || B_d2->csr_col == NULL) {
    LOG(FATAL) << "parameter B is null!";
  }

  if (B_d->nnz == 0) return;

  int nnz = B_d->nnz;
  int block = 512;
  int grid = DIVUP(nnz, 512);
  KeSMatrixCsrColumnSum<<<grid, block, 0, STREAM_DEFAULT>>>(
      A_d, B_d2->csr_val, B_d2->csr_col, nnz);

  CHECK_SYNC("hl_matrix_csr_column_sum failed");
}

void hl_sparse_matrix_add_bias(hl_sparse_matrix_s A_d,
                               real* B_d, real scale) {
  if (A_d->format == HL_SPARSE_CSR) {
    hl_matrix_csr_add_bias(A_d, B_d, scale);
  } else {
    LOG(FATAL) << "Not support CSC format error!";
  }
}

void hl_matrix_csr_add_bias(hl_sparse_matrix_s A_d, real* B_d,
                            real scale) {
  CHECK_NOTNULL(A_d);
  CHECK_NOTNULL(B_d);

  hl_csr_matrix A_d2 = (hl_csr_matrix)(A_d->matrix);
  if ((A_d2->csr_val == NULL && A_d->type != HL_NO_VALUE) ||
      A_d2->csr_row == NULL || A_d2->csr_col == NULL) {
    LOG(FATAL) << "parameter A_d is null!";
  }

  if (A_d->nnz == 0) return;

  int nnz = A_d->nnz;
  int block = 512;
  int grid = DIVUP(nnz, 512);
  KeSMatrixCsrAddBias<<<grid, block, 0, STREAM_DEFAULT>>>(
      A_d2->csr_val, A_d2->csr_col, B_d, scale, nnz);

  CHECK_SYNC("hl_sparse_matrix_add_bias failed");
}

void hl_sparse_matrix_add_dense(hl_sparse_matrix_s A_d, real *B_d, int dimM,
                                int dimN, real alpha, real beta) {
  if (A_d->format == HL_SPARSE_CSR) {
    hl_matrix_csr_add_dense(A_d, B_d, dimM, dimN, alpha, beta);
  } else {
    LOG(FATAL) << "Not support CSC format error!";
  }
}

void hl_matrix_csr_add_dense(hl_sparse_matrix_s A_d, real* B_d, int dimM,
                             int dimN, real alpha, real beta) {
  CHECK_NOTNULL(A_d);
  CHECK_NOTNULL(B_d);

  if (dimM <= 0 || dimN <= 0 || A_d->rows != dimM || A_d->cols != dimN) {
    LOG(FATAL) << "parameter dim error!";
  }

  hl_csr_matrix A_d2 = (hl_csr_matrix)(A_d->matrix);
  if ((A_d2->csr_val == NULL && A_d->type != HL_NO_VALUE) ||
      A_d2->csr_row == NULL || A_d2->csr_col == NULL) {
    LOG(FATAL) << "parameter A_d is null!";
  }

  if (A_d->nnz == 0) return;

  int gridX = DIVUP((A_d->nnz / dimM), 512);
  gridX = gridX > 0 ? gridX : 1;
  dim3 block(512, 1);
  dim3 grid(gridX, dimM);
  KeSMatrixCsrAddDense<<<grid, block, 0, STREAM_DEFAULT>>>(
    A_d2->csr_val, A_d2->csr_row, A_d2->csr_col, B_d, alpha, beta, dimM, dimN);

  CHECK_SYNC("hl_sparse_matrix_add_dense failed");
}

int* hl_sparse_matrix_get_rows(hl_sparse_matrix_s sMat) {
  __sparse_get_return__(sMat, row);
}

int* hl_sparse_matrix_get_cols(hl_sparse_matrix_s sMat) {
  __sparse_get_return__(sMat, col);
}

real* hl_sparse_matrix_get_value(hl_sparse_matrix_s sMat) {
  __sparse_get_return__(sMat, val);
}
