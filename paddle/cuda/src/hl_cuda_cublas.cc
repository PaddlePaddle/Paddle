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

#include "hl_cuda_cublas.h"
#include <sys/time.h>
#include "hl_cuda.h"
#include "hl_thread.ph"
#include "paddle/utils/DynamicLoader.h"
#include "paddle/utils/Logging.h"

namespace dynload {

std::once_flag cublas_dso_flag;
void *cublas_dso_handle = nullptr;

/**
 * The following macro definition can generate structs
 * (for each function) to dynamic load cublas routine
 * via operator overloading.
 *
 * note: default dynamic linked libs
 */
#ifdef PADDLE_USE_DSO
#define DYNAMIC_LOAD_CUBLAS_WRAP(__name)                                       \
  struct DynLoad__##__name {                                                   \
    template <typename... Args>                                                \
    cublasStatus_t operator()(Args... args) {                                  \
      typedef cublasStatus_t (*cublasFunc)(Args...);                           \
      std::call_once(cublas_dso_flag, GetCublasDsoHandle, &cublas_dso_handle); \
      void *p_##__name = dlsym(cublas_dso_handle, #__name);                    \
      return reinterpret_cast<cublasFunc>(p_##__name)(args...);                \
    }                                                                          \
  } __name;  // struct DynLoad__##__name
#else
#define DYNAMIC_LOAD_CUBLAS_WRAP(__name)      \
  struct DynLoad__##__name {                  \
    template <typename... Args>               \
    cublasStatus_t operator()(Args... args) { \
      return __name(args...);                 \
    }                                         \
  } __name;  // struct DynLoad__##__name
#endif

#define DYNAMIC_LOAD_CUBLAS_V2_WRAP(__name) DYNAMIC_LOAD_CUBLAS_WRAP(__name)

// include all needed cublas functions in HPPL
// clang-format off
#define CUBLAS_BLAS_ROUTINE_EACH(__macro) \
  __macro(cublasSgemv)                    \
  __macro(cublasDgemv)                    \
  __macro(cublasSgemm)                    \
  __macro(cublasDgemm)                    \
  __macro(cublasSgeam)                    \
  __macro(cublasDgeam)                    \

DYNAMIC_LOAD_CUBLAS_V2_WRAP(cublasCreate)
DYNAMIC_LOAD_CUBLAS_V2_WRAP(cublasDestroy)
DYNAMIC_LOAD_CUBLAS_V2_WRAP(cublasSetStream)
DYNAMIC_LOAD_CUBLAS_V2_WRAP(cublasSetPointerMode)
DYNAMIC_LOAD_CUBLAS_V2_WRAP(cublasGetPointerMode)
DYNAMIC_LOAD_CUBLAS_WRAP(cublasSgemmBatched)
DYNAMIC_LOAD_CUBLAS_WRAP(cublasDgemmBatched)
DYNAMIC_LOAD_CUBLAS_WRAP(cublasCgemmBatched)
DYNAMIC_LOAD_CUBLAS_WRAP(cublasZgemmBatched)
DYNAMIC_LOAD_CUBLAS_WRAP(cublasSgetrfBatched)
DYNAMIC_LOAD_CUBLAS_WRAP(cublasSgetriBatched)
DYNAMIC_LOAD_CUBLAS_WRAP(cublasDgetrfBatched)
DYNAMIC_LOAD_CUBLAS_WRAP(cublasDgetriBatched)
CUBLAS_BLAS_ROUTINE_EACH(DYNAMIC_LOAD_CUBLAS_V2_WRAP)

#undef DYNAMIC_LOAD_CUBLAS_WRAP
#undef DYNAMIC_LOAD_CUBLAS_V2_WRAP
#undef CUBLAS_BLAS_ROUTINE_EACH

} /* namespace dynload */

// clang-format on
#ifndef PADDLE_TYPE_DOUBLE
#define CUBLAS_GEAM dynload::cublasSgeam
#define CUBLAS_GEMV dynload::cublasSgemv
#define CUBLAS_GEMM dynload::cublasSgemm
#define CUBLAS_GETRF dynload::cublasSgetrfBatched
#define CUBLAS_GETRI dynload::cublasSgetriBatched
#else
#define CUBLAS_GEAM dynload::cublasDgeam
#define CUBLAS_GEMV dynload::cublasDgemv
#define CUBLAS_GEMM dynload::cublasDgemm
#define CUBLAS_GETRF dynload::cublasDgetrfBatched
#define CUBLAS_GETRI dynload::cublasDgetriBatched
#endif

const char *hl_cublas_get_error_string(cublasStatus_t status) {
  switch (status) {
    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "[cublas status]: not initialized";
    case CUBLAS_STATUS_ALLOC_FAILED:
      return "[cublas status]: allocate failed";
    case CUBLAS_STATUS_INVALID_VALUE:
      return "[cublas status]: invalid value";
    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "[cublas status]: arch mismatch";
    case CUBLAS_STATUS_MAPPING_ERROR:
      return "[cublas status]: mapping error";
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "[cublas status]: execution failed";
    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "[cublas status]: internal error";
    case CUBLAS_STATUS_SUCCESS:
      return "[cublas status]: success";
    default:
      return "[cublas status]: unknown error";
  }
}

/**
 * Check build-in cublas function using glog and it also
 * support << operator for more details error info.
 */
cublasStatus_t g_cublasStat;
#define CHECK_CUBLAS(cublas_func)               \
  g_cublasStat = cublas_func;                   \
  CHECK_EQ(CUBLAS_STATUS_SUCCESS, g_cublasStat) \
      << "Cublas Error: " << hl_cublas_get_error_string(g_cublasStat) << " "

void hl_cublas_init(cublasHandle_t *cublas_handle, cudaStream_t stream) {
  CHECK_CUBLAS(dynload::cublasCreate(cublas_handle))
      << "[cublas init] Cublas create handle faild!";

  CHECK_CUBLAS(dynload::cublasSetStream(*cublas_handle, stream))
      << "[cublas init] Cublas set stream faild!";
}

void hl_matrix_transpose(
    real *A_d, real *C_d, int dimM, int dimN, int lda, int ldc) {
  real alpha = 1.0;
  real beta = 0.0;

  CHECK_NOTNULL(A_d);
  CHECK_NOTNULL(C_d);

  CHECK_CUBLAS(CUBLAS_GEAM(t_resource.handle,
                           CUBLAS_OP_T,
                           CUBLAS_OP_N,
                           dimM,
                           dimN,
                           &alpha,
                           A_d,
                           lda,
                           &beta,
                           nullptr,
                           dimM,
                           C_d,
                           ldc));
  CHECK_SYNC("hl_matrix_transpose failed");
}

void hl_matrix_transpose(real *A_d, real *C_d, int dimM, int dimN) {
  hl_matrix_transpose(A_d, C_d, dimM, dimN, dimN, dimM);
}

void hl_matrix_inverse(real *A_d, real *C_d, int dimN, int lda, int ldc) {
  /* Solve Ax = I */
  CHECK_NOTNULL(A_d);
  CHECK_NOTNULL(C_d);

  /* Step 1: Compute the LU decomposition of matrix A */
  real **inout_h = &A_d;
  real **inout_d = (real **)hl_malloc_device(sizeof(real *));
  hl_memcpy(inout_d, inout_h, sizeof(real *));

  int *pivot_d = (int *)hl_malloc_device(dimN * sizeof(int));
  int *info_d = (int *)t_resource.gpu_mem;

  /* Note: cublasSgetrfBatched is used to calculate a number of
     small-sized matrices. There may be a better way to reconstruct
     the API for better performance.
   */
  CHECK_CUBLAS(
      CUBLAS_GETRF(t_resource.handle, dimN, inout_d, lda, pivot_d, info_d, 1));

  int info_h;
  hl_memcpy(&info_h, info_d, sizeof(int));
  if (info_h != 0) {
    LOG(FATAL) << "Factorization of matrix failed: matrix may be singular.\n";
  }

  /* Step 2: Compute the inverse of the matrix given its LU decomposition */
  real **out_h = &C_d;
  real **out_d = (real **)hl_malloc_device(sizeof(real *));
  hl_memcpy(out_d, out_h, sizeof(real *));

  CHECK_CUBLAS(CUBLAS_GETRI(t_resource.handle,
                            dimN,
                            (const real **)inout_d,
                            lda,
                            pivot_d,
                            out_d,
                            ldc,
                            info_d,
                            1));

  hl_memcpy(&info_h, info_d, sizeof(int));
  if (info_h != 0) {
    LOG(FATAL) << "Inversion of matrix failed: matrix may be singular.\n";
  }

  hl_free_mem_device(inout_d);
  hl_free_mem_device(pivot_d);
  hl_free_mem_device(out_d);

  CHECK_SYNC("hl_matrix_inverse failed");
}

void hl_matrix_mul(real *A_d,
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
                   int ldc) {
  CHECK_NOTNULL(A_d);
  CHECK_NOTNULL(B_d);
  CHECK_NOTNULL(C_d);

  if (dimN == 1 && dimM != 1 && dimK != 1 && transb == HPPL_OP_N) {
    int m = (transa == HPPL_OP_N) ? dimM : dimK;
    int n = (transa == HPPL_OP_N) ? dimK : dimM;
    hl_matrix_mul_vector(
        A_d, transa, B_d, C_d, m, n, alpha, beta, lda, ldb, ldc);
    return;
  }

  if (dimM == 1 && dimN != 1 && dimK != 1 && transa == HPPL_OP_N) {
    int m = (transb == HPPL_OP_N) ? dimK : dimN;
    int n = (transb == HPPL_OP_N) ? dimN : dimK;
    hl_trans_op_t trans = (transb == HPPL_OP_N) ? HPPL_OP_T : HPPL_OP_N;
    hl_matrix_mul_vector(B_d, trans, A_d, C_d, m, n, alpha, beta, ldb, 1, 1);
    return;
  }

  cublasStatus_t stat;
  if ((HPPL_OP_N == transa) && (HPPL_OP_N == transb)) {
    stat = CUBLAS_GEMM(t_resource.handle,
                       CUBLAS_OP_N,
                       CUBLAS_OP_N,
                       dimN,
                       dimM,
                       dimK,
                       &alpha,
                       B_d,
                       ldb,
                       A_d,
                       lda,
                       &beta,
                       C_d,
                       ldc);
  } else if ((HPPL_OP_T == transa) && (HPPL_OP_N == transb)) {
    stat = CUBLAS_GEMM(t_resource.handle,
                       CUBLAS_OP_N,
                       CUBLAS_OP_T,
                       dimN,
                       dimM,
                       dimK,
                       &alpha,
                       B_d,
                       ldb,
                       A_d,
                       lda,
                       &beta,
                       C_d,
                       ldc);
  } else if ((HPPL_OP_N == transa) && (HPPL_OP_T == transb)) {
    stat = CUBLAS_GEMM(t_resource.handle,
                       CUBLAS_OP_T,
                       CUBLAS_OP_N,
                       dimN,
                       dimM,
                       dimK,
                       &alpha,
                       B_d,
                       ldb,
                       A_d,
                       lda,
                       &beta,
                       C_d,
                       ldc);
  } else {
    LOG(FATAL) << "parameter transa error!";
  }
  CHECK_EQ(stat, CUBLAS_STATUS_SUCCESS) << hl_cublas_get_error_string(stat);
  CHECK_SYNC("hl_matrix_mul failed");
}

void hl_matrix_mul(real *A_d,
                   hl_trans_op_t transa,
                   real *B_d,
                   hl_trans_op_t transb,
                   real *C_d,
                   int dimM,
                   int dimN,
                   int dimK,
                   real alpha,
                   real beta) {
  int lda = (HPPL_OP_N == transa) ? dimK : dimM;
  int ldb = (HPPL_OP_N == transb) ? dimN : dimK;
  int ldc = dimN;

  hl_matrix_mul(A_d,
                transa,
                B_d,
                transb,
                C_d,
                dimM,
                dimN,
                dimK,
                alpha,
                beta,
                lda,
                ldb,
                ldc);
}

void hl_matrix_mul_vector(real *A_d,
                          hl_trans_op_t trans,
                          real *B_d,
                          real *C_d,
                          int dimM,
                          int dimN,
                          real alpha,
                          real beta,
                          int lda,
                          int incb,
                          int incc) {
  CHECK_NOTNULL(A_d);
  CHECK_NOTNULL(B_d);
  CHECK_NOTNULL(C_d);

  cublasStatus_t stat;
  if (HPPL_OP_N == trans) {
    stat = CUBLAS_GEMV(t_resource.handle,
                       CUBLAS_OP_T,
                       dimN,
                       dimM,
                       &alpha,
                       A_d,
                       lda,
                       B_d,
                       incb,
                       &beta,
                       C_d,
                       incc);
  } else if (HPPL_OP_T == trans) {
    stat = CUBLAS_GEMV(t_resource.handle,
                       CUBLAS_OP_N,
                       dimN,
                       dimM,
                       &alpha,
                       A_d,
                       lda,
                       B_d,
                       incb,
                       &beta,
                       C_d,
                       incc);
  } else {
    LOG(FATAL) << "parameter transa error!";
  }

  CHECK_EQ(stat, CUBLAS_STATUS_SUCCESS) << hl_cublas_get_error_string(stat);
  CHECK_SYNC("hl_matrix_mul_vector");
}

void hl_matrix_mul_vector(real *A_d,
                          hl_trans_op_t trans,
                          real *B_d,
                          real *C_d,
                          int dimM,
                          int dimN,
                          real alpha,
                          real beta) {
  hl_matrix_mul_vector(
      A_d, trans, B_d, C_d, dimM, dimN, alpha, beta, dimN, 1, 1);
}
