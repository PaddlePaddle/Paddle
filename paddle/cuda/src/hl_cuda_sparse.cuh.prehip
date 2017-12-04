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


#include "hl_device_functions.cuh"

template <int VALUE_TYPE>
__device__ real findvalue(real* csr_val,
                          int* csr_col,
                          int col_start,
                          int col_end,
                          int index) {
  int start = col_start;
  int end = col_end-1;
  int mid = -1;

  while (start < end) {
    mid = start + ((end - start) / 2);
    if (csr_col[mid] < index)
      start = mid + 1;
    else
      end = mid;
  }

  if ((start < col_end) && (csr_col[start] == index)) {
    real ret = VALUE_TYPE == 0 ? 1.0 : csr_val[start];
    return ret;
  } else {
    return 0.0;
  }
}

#define     CU_CSR2DENSE_THREAD_X   16
#define     CU_CSR2DENSE_THREAD_Y   16
template <int VALUE_TYPE>
__global__ void KeSMatrixCsr2Dense(real * csr_val,
                                   int * csr_row,
                                   int * csr_col,
                                   real * C_d,
                                   const int dimM,
                                   const int dimN) {
  const int row = blockIdx.y*blockDim.y+threadIdx.y;
  const int col = blockIdx.x*blockDim.x+threadIdx.x;

  if (row >= dimM || col >= dimN) {
    return;
  }

  int start = csr_row[row];
  int end = csr_row[row+1];

  real sum = findvalue<VALUE_TYPE>(csr_val, csr_col, start, end, col);
  C_d[row*dimN + col] = sum;
}

template <int VALUE_TYPE>
__global__ void KeSMatrixCsc2Dense(real * csc_val,
                                   int * csc_row,
                                   int * csc_col,
                                   real * C_d,
                                   const int dimM,
                                   const int dimN) {
  const int row = blockIdx.y*blockDim.y+threadIdx.y;
  const int col = blockIdx.x*blockDim.x+threadIdx.x;

  if (row >= dimM || col >= dimN) {
    return;
  }

  int start = csc_col[col];
  int end = csc_col[col+1];

  real sum = findvalue<VALUE_TYPE>(csc_val, csc_row, start, end, row);
  C_d[row*dimN + col] = sum;
}

__device__ __forceinline__
void _calculate_c(real &c, real sum) {
  c = sum;
}
__device__ __forceinline__
void _calculate_c(real &c, real sum, real beta) {
  c = sum + beta * c;
}

#define     CU_CSRMM_N                  4
#define     CU_CSRMM_THREAD_X           32
#define     CU_CSRMM_THREAD_Y           32
#define     CU_CSRMM_BLOCK_N            (32*CU_CSRMM_N)
#define     CU_CSRMM_SHARED_ELEMENT     (2*CU_CSRMM_THREAD_X)
template <int VALUE_TYPE>
__global__ void KeSMatrixCsrMulDense(real *C_d,
                                     real * csr_val,
                                     int * csr_col,
                                     int * csr_row,
                                     real *B_d,
                                     int dimM,
                                     int dimN,
                                     int dimK,
                                     real alpha,
                                     real beta) {
  const int idx = threadIdx.x;
  const int idy = threadIdx.y;
  const int index_m = blockIdx.y*CU_CSRMM_THREAD_Y+threadIdx.y;
  int index_n = blockIdx.x*CU_CSRMM_BLOCK_N+threadIdx.x;

  __shared__ real csr_val_sh[CU_CSRMM_THREAD_Y][CU_CSRMM_SHARED_ELEMENT];
  __shared__ int csr_col_sh[CU_CSRMM_THREAD_Y][CU_CSRMM_SHARED_ELEMENT];

  if (index_m >= dimM) {
    return;
  }

  // possible optimization, cache this in shared memory
  int csr_start = csr_row[index_m];
  int csr_end = csr_row[index_m+1];
  int csr_index =  csr_start + idx;

  int csr_iter = (csr_end-csr_start)/CU_CSRMM_SHARED_ELEMENT;
  int csr_rem = (csr_end-csr_start)%CU_CSRMM_SHARED_ELEMENT;

  int index_k = -1;
  real sum[CU_CSRMM_N] = {0};
  real b_r[CU_CSRMM_N] = {0};

  for (int csr_i = 0; csr_i < csr_iter; csr_i++) {
    #pragma unroll
    for (int i = 0; i < (CU_CSRMM_SHARED_ELEMENT/CU_CSRMM_THREAD_X); i++) {
      if (VALUE_TYPE != 0) {
        csr_val_sh[idy][idx + i*CU_CSRMM_THREAD_X] = csr_val[csr_index];
      }
      csr_col_sh[idy][idx + i*CU_CSRMM_THREAD_X] = csr_col[csr_index];
      csr_index += CU_CSRMM_THREAD_X;
    }

    for (int index = 0; index < CU_CSRMM_SHARED_ELEMENT; index++) {
      index_k = csr_col_sh[idy][index];
      real a_r = VALUE_TYPE == 0 ? 1.0 : csr_val_sh[idy][index];
      int tmp_index = index_n;
      real *B_d_r = B_d + tmp_index;
      #pragma unroll
      for (int n = 0; n < CU_CSRMM_N; n++) {
        if (tmp_index >= dimN) break;
        b_r[n] = B_d_r[index_k*dimN];
        B_d_r += CU_CSRMM_THREAD_X;
        tmp_index += CU_CSRMM_THREAD_X;
      }

      #pragma unroll
      for (int n = 0; n < CU_CSRMM_N; n++) {
        sum[n] = VALUE_TYPE == 0 ? sum[n] + b_r[n] : sum[n] + a_r*b_r[n];
      }
    }
    // __syncthreads();
  }

  if (csr_rem != 0) {
    #pragma unroll
    for (int i = 0; i < (CU_CSRMM_SHARED_ELEMENT/CU_CSRMM_THREAD_X); i++) {
      if (csr_index < csr_end) {
        if (VALUE_TYPE != 0) {
            csr_val_sh[idy][idx + i*CU_CSRMM_THREAD_X] = csr_val[csr_index];
        }
        csr_col_sh[idy][idx + i*CU_CSRMM_THREAD_X] = csr_col[csr_index];
      }
      csr_index += CU_CSRMM_THREAD_X;
    }
    // __syncthreads();

    #pragma unroll
    for (int index = 0; index < csr_rem; index++) {
      index_k = csr_col_sh[idy][index];
      real a_r = VALUE_TYPE == 0 ? 1.0 : csr_val_sh[idy][index];
      int tmp_index = index_n;
      real *B_d_r = B_d + tmp_index;
      #pragma unroll
      for (int n = 0; n < CU_CSRMM_N; n++) {
        if (tmp_index >= dimN) break;
        b_r[n] = B_d_r[index_k*dimN];
        B_d_r += CU_CSRMM_THREAD_X;
        tmp_index += CU_CSRMM_THREAD_X;
      }

      #pragma unroll
      for (int n = 0; n < CU_CSRMM_N; n++) {
        sum[n] = VALUE_TYPE == 0 ? sum[n] + b_r[n] : sum[n] + a_r*b_r[n];
      }
    }
  }

  C_d += __mul24(index_m, dimN);
  if (beta == 0.0) {
    for (int n = 0; n < CU_CSRMM_N; n++) {
      if (index_n < dimN) {
        _calculate_c(C_d[index_n], alpha * sum[n]);
        index_n += CU_CSRMM_THREAD_X;
      }
    }
  } else {
    for (int n = 0; n < CU_CSRMM_N; n++) {
      if (index_n < dimN) {
        _calculate_c(C_d[index_n], alpha * sum[n], beta);
        index_n += CU_CSRMM_THREAD_X;
      }
    }
  }
}

#define CU_CSC_MUL_DENSE_THREAD_N           1
#define CU_CSC_MUL_DENSE_THREAD_X           32
#define CU_CSC_MUL_DENSE_THREAD_Y           4
#define CU_CSC_MUL_DENSE_BLOCK_K            (CU_CSC_MUL_DENSE_THREAD_Y)
#define CU_CSC_MUL_DENSE_BLOCK_N            \
        (CU_CSC_MUL_DENSE_THREAD_N * CU_CSC_MUL_DENSE_THREAD_X)
#define CU_CSC_MUL_DENSE_SHARED_ELEMENT     (CU_CSC_MUL_DENSE_THREAD_X)
template <int VALUE_TYPE>
__global__ void KeSMatrixCscMulDense(real *C_d,
                                     real * csc_val,
                                     int * csc_row,
                                     int * csc_col,
                                     real *B_d,
                                     int dimM,
                                     int dimN,
                                     int dimK,
                                     real alpha,
                                     real beta) {
  const int idx = threadIdx.x;
  const int idy = threadIdx.y;
  const int index_k = blockIdx.y*CU_CSC_MUL_DENSE_BLOCK_K+threadIdx.y;
  const int index_n = blockIdx.x*CU_CSC_MUL_DENSE_BLOCK_N+threadIdx.x;

  if (index_k >= dimK) {
    return;
  }

  __shared__
  real csc_val_sh[CU_CSC_MUL_DENSE_THREAD_Y][CU_CSC_MUL_DENSE_SHARED_ELEMENT];
  __shared__
  int csc_row_sh[CU_CSC_MUL_DENSE_THREAD_Y][CU_CSC_MUL_DENSE_SHARED_ELEMENT];

  // possible optimization, cache this in shared memory
  int csc_start = csc_col[index_k];
  int csc_end = csc_col[index_k+1];
  int csc_index = csc_start + idx;
  int csc_iter = (csc_end-csc_start)/CU_CSC_MUL_DENSE_SHARED_ELEMENT;
  int csc_rem = (csc_end-csc_start)%CU_CSC_MUL_DENSE_SHARED_ELEMENT;
  int index_m = -1;

  real b_r[CU_CSC_MUL_DENSE_THREAD_N] = {0};
  real *B_d_r;
  real *C_d_r;
  int index_n_t;
  B_d += index_n + __mul24(index_k, dimN);
  C_d += index_n;
  for (int csr_i = 0; csr_i < csc_iter; csr_i++) {
    #pragma unroll
    for (int i = 0;
         i < (CU_CSC_MUL_DENSE_SHARED_ELEMENT/CU_CSC_MUL_DENSE_THREAD_X); i++) {
      if (VALUE_TYPE != 0) {
        csc_val_sh[idy][idx + i*CU_CSC_MUL_DENSE_THREAD_X] = csc_val[csc_index];
      }
      csc_row_sh[idy][idx + i*CU_CSC_MUL_DENSE_THREAD_X] = csc_row[csc_index];
      csc_index += CU_CSC_MUL_DENSE_THREAD_X;
    }

    #pragma unroll
    for (int index = 0; index < CU_CSC_MUL_DENSE_SHARED_ELEMENT; index++) {
      index_m = csc_row_sh[idy][index];
      real a_r = VALUE_TYPE == 0 ? 1.0 : csc_val_sh[idy][index];
      B_d_r = B_d;
      C_d_r = C_d + __mul24(index_m, dimN);

      index_n_t = index_n;
      #pragma unroll
      for (int n = 0; n < CU_CSC_MUL_DENSE_THREAD_N; n++) {
        if (index_n_t < dimN) {
          b_r[n] = B_d_r[0];
          B_d_r += CU_CSC_MUL_DENSE_THREAD_X;
          index_n_t += CU_CSC_MUL_DENSE_THREAD_X;
        }
      }

      index_n_t = index_n;
      #pragma unroll
      for (int n = 0; n < CU_CSC_MUL_DENSE_THREAD_N; n++) {
        if (index_n_t < dimN) {
          real tmp;
          tmp = alpha*a_r*b_r[n];
          paddle::paddleAtomicAdd(C_d_r, tmp);
          C_d_r += CU_CSC_MUL_DENSE_THREAD_X;
          index_n_t += CU_CSC_MUL_DENSE_THREAD_X;
        }
      }
    }
    // __syncthreads();
  }

  if (csc_rem != 0) {
    #pragma unroll
    for (int i = 0;
         i < (CU_CSC_MUL_DENSE_SHARED_ELEMENT/CU_CSC_MUL_DENSE_THREAD_X); i++) {
      if (csc_index < csc_end) {
        if (VALUE_TYPE != 0) {
          csc_val_sh[idy][idx + i * CU_CSC_MUL_DENSE_THREAD_X] =
            csc_val[csc_index];
        }
        csc_row_sh[idy][idx + i * CU_CSC_MUL_DENSE_THREAD_X] =
          csc_row[csc_index];
      }
      csc_index += CU_CSC_MUL_DENSE_THREAD_X;
    }
    // __syncthreads();

    #pragma unroll
    for (int index = 0; index < csc_rem; index++) {
      index_m = csc_row_sh[idy][index];
      real a_r = VALUE_TYPE == 0 ? 1.0 : csc_val_sh[idy][index];
      B_d_r = B_d;
      C_d_r = C_d + __mul24(index_m, dimN);

      index_n_t = index_n;
      #pragma unroll
      for (int n = 0; n < CU_CSC_MUL_DENSE_THREAD_N; n++) {
        if (index_n_t < dimN) {
          b_r[n] = B_d_r[0];
          B_d_r += CU_CSC_MUL_DENSE_THREAD_X;
          index_n_t += CU_CSC_MUL_DENSE_THREAD_X;
        }
      }

      index_n_t = index_n;
      #pragma unroll
      for (int n = 0; n < CU_CSC_MUL_DENSE_THREAD_N; n++) {
        if (index_n_t < dimN) {
          real tmp;
          tmp = alpha*a_r*b_r[n];
          paddle::paddleAtomicAdd(C_d_r, tmp);
          C_d_r += CU_CSC_MUL_DENSE_THREAD_X;
          index_n_t += CU_CSC_MUL_DENSE_THREAD_X;
        }
      }
    }
  }
}

/* best perf */
#ifndef PADDLE_TYPE_DOUBLE
#define CU_CSCMM_THREAD_M_BEST          9
#else
#define CU_CSCMM_THREAD_M_BEST          4
#endif
#define CU_CSCMM_THREAD_X_BEST          32
#define CU_CSCMM_THREAD_Y_BEST          32
#define CU_CSCMM_BLOCK_M_BEST  (CU_CSCMM_THREAD_M_BEST * CU_CSCMM_THREAD_X_BEST)
#define CU_CSCMM_BLOCK_N_BEST  (CU_CSCMM_THREAD_Y_BEST)
template <int VALUE_TYPE>
__global__ void KeSMatrixDenseMulCsc(real *C_d,
                                     const real *A_d,
                                     const real *csc_val,
                                     const int *csc_row,
                                     const int *csc_col,
                                     int dimM,
                                     int dimN,
                                     int dimK,
                                     real alpha,
                                     real beta) {
  __shared__ real csc_val_sh[CU_CSCMM_BLOCK_N_BEST][CU_CSCMM_THREAD_X_BEST];
  __shared__ int csc_row_sh[CU_CSCMM_BLOCK_N_BEST][CU_CSCMM_THREAD_X_BEST];
  __shared__ real A_s[CU_CSCMM_BLOCK_M_BEST][CU_CSCMM_THREAD_Y_BEST+1];

  int iter_k = dimK/CU_CSCMM_THREAD_Y_BEST;
  int rem_k = dimK%CU_CSCMM_THREAD_Y_BEST;
  const int idx = threadIdx.x;
  const int idy = threadIdx.y;
  const int index_n = blockIdx.y*CU_CSCMM_BLOCK_N_BEST+threadIdx.y;

  int csc_start;
  int csc_end;
  if (index_n < dimN) {
    csc_start = csc_col[index_n];
    csc_end = csc_col[index_n+1];
  } else {
    csc_start = 0;
    csc_end = 0;
  }
  int csc_index =  csc_start + idx;
  int csc_iter = (csc_end-csc_start)/CU_CSCMM_THREAD_X_BEST;
  int csc_rem = (csc_end-csc_start)%CU_CSCMM_THREAD_X_BEST;
  int index_k = -1;

  if (csc_index < csc_end) {
    if (VALUE_TYPE != 0) {
      csc_val_sh[idy][idx] = csc_val[csc_index];
    }
    csc_row_sh[idy][idx] = csc_row[csc_index];
    csc_index += CU_CSCMM_THREAD_X_BEST;
  }

  const int ibx = blockIdx.x * CU_CSCMM_BLOCK_M_BEST;
  int dim = ibx+idy;
  A_d += idx + __mul24(dim, dimK);
  #pragma unroll
  for (int m = 0; m < CU_CSCMM_THREAD_M_BEST; m++) {
    A_s[idy + m * 32][idx] = 0.0f;
    if (dim + m * 32 < dimM && idx < dimK) {
      A_s[idy + m * 32][idx] = A_d[m * 32 * dimK];
    }
  }
  __syncthreads();

  real b_r;
  real a_r[CU_CSCMM_THREAD_M_BEST] = {0};
  real sum[CU_CSCMM_THREAD_M_BEST] = {0};
  real A_r_s[CU_CSCMM_THREAD_M_BEST] = {0};
  int index = 0;
  int block_end_k = 0;;
  int index_iter_csc = csc_iter;

  for (int i_k = 0; i_k < iter_k; i_k++) {
    A_d += CU_CSCMM_THREAD_Y_BEST;
    block_end_k += CU_CSCMM_THREAD_Y_BEST;
    #pragma unroll
    for (int m = 0; m < CU_CSCMM_THREAD_M_BEST; m++) {
      if (dim + m*32 < dimM && (idx + (i_k+1)*CU_CSCMM_THREAD_Y_BEST < dimK)) {
        A_r_s[m] = A_d[m*32*dimK];
      } else {
        A_r_s[m] = 0.0f;
      }
    }

    if (index_iter_csc > 0) {
      goto WARP_SYNC;
    } else {
      goto WARP_SYNC_2;
    }

    while (index_iter_csc) {
      if (VALUE_TYPE != 0) {
        csc_val_sh[idy][idx] = csc_val[csc_index];
      }
      csc_row_sh[idy][idx] = csc_row[csc_index];
      csc_index += CU_CSCMM_THREAD_X_BEST;
      index = 0;

WARP_SYNC:
      for (; index < CU_CSCMM_THREAD_X_BEST; index++) {
        index_k = csc_row_sh[idy][index];
        if (index_k >= block_end_k) {
          goto BLOCK_SYNC;
        }
        b_r = VALUE_TYPE == 0 ? 1.0 : csc_val_sh[idy][index];
        #pragma unroll
        for (int m = 0; m < CU_CSCMM_THREAD_M_BEST; m++) {
          a_r[m] = A_s[idx+m*32][index_k-i_k*CU_CSCMM_THREAD_Y_BEST];
          sum[m] = VALUE_TYPE == 0 ? sum[m] + a_r[m] : sum[m] + a_r[m]*b_r;
        }
      }
      index_iter_csc--;
    }

    if (csc_rem != 0) {
      if (csc_iter != 0) {
        if (csc_index < csc_end) {
          if (VALUE_TYPE != 0) {
            csc_val_sh[idy][idx] = csc_val[csc_index];
          }
          csc_row_sh[idy][idx] = csc_row[csc_index];
          csc_index += CU_CSCMM_THREAD_X_BEST;
        }
        index = 0;
      }
      __threadfence_block();

WARP_SYNC_2:
      for (; index < csc_rem; index++) {
        index_k = csc_row_sh[idy][index];
        if (index_k >= block_end_k) {
          goto BLOCK_SYNC;
        }
        b_r = VALUE_TYPE == 0 ? 1.0 : csc_val_sh[idy][index];
        #pragma unroll
        for (int m = 0; m < CU_CSCMM_THREAD_M_BEST; m++) {
          a_r[m] = A_s[idx+m*32][index_k-i_k*CU_CSCMM_THREAD_Y_BEST];
          sum[m] = VALUE_TYPE == 0 ? sum[m] + a_r[m] : sum[m] + a_r[m]*b_r;
        }
      }
    }

BLOCK_SYNC:
    __syncthreads();
    #pragma unroll
    for (int m = 0; m < CU_CSCMM_THREAD_M_BEST; m++) {
      A_s[idy+m*32][idx] = A_r_s[m];
    }
    __syncthreads();
  }

  if (rem_k != 0) {
    if (index_iter_csc == 0) {
      goto TEMP_TEST;
    }

    for (; index < CU_CSCMM_THREAD_X_BEST; index++) {
      index_k = csc_row_sh[idy][index];
      if (index_k >= dimK) {
        break;
      }

      b_r = VALUE_TYPE == 0 ? 1.0 : csc_val_sh[idy][index];
      #pragma unroll
      for (int m = 0; m < CU_CSCMM_THREAD_M_BEST; m++) {
        a_r[m] = A_s[idx+m*32][index_k-iter_k*CU_CSCMM_THREAD_Y_BEST];
        sum[m] = VALUE_TYPE == 0 ? sum[m] + a_r[m] : sum[m] + a_r[m]*b_r;
      }
    }

    if (csc_rem != 0) {
      if (csc_index < csc_end) {
        if (VALUE_TYPE != 0) {
          csc_val_sh[idy][idx] = csc_val[csc_index];
        }
        csc_row_sh[idy][idx] = csc_row[csc_index];
        csc_index += CU_CSCMM_THREAD_X_BEST;
      }
      index = 0;

TEMP_TEST:
      for (; index < csc_rem; index++) {
        index_k = csc_row_sh[idy][index];
        if (index_k >= dimK) {
            break;
        }
        b_r = VALUE_TYPE == 0 ? 1.0 : csc_val_sh[idy][index];
        #pragma unroll
        for (int m = 0; m < CU_CSCMM_THREAD_M_BEST; m++) {
          a_r[m] = A_s[idx+m*32][index_k-iter_k*CU_CSCMM_THREAD_Y_BEST];
          sum[m] = VALUE_TYPE == 0 ? sum[m] + a_r[m] : sum[m] + a_r[m]*b_r;
        }
      }
    }
  }

  __syncthreads();
  #pragma unroll
  for (int m = 0; m < CU_CSCMM_THREAD_M_BEST; m++) {
    A_s[idx+m*32][idy] = alpha*sum[m];
  }
  __syncthreads();

  int index_m_c = ibx + idy;
  int index_n_c = blockIdx.y*CU_CSCMM_BLOCK_N_BEST + idx;
  C_d += index_n_c + __mul24(index_m_c, dimN);
  if (beta == 0.0) {
    for (int m = 0; m < CU_CSCMM_THREAD_M_BEST; m++) {
      if (index_m_c < dimM && index_n_c < dimN) {
        _calculate_c(C_d[0], A_s[idy + m * 32][idx]);
      }
      index_m_c += 32;
      C_d += dimN*32;
    }
  } else {
    for (int m = 0; m < CU_CSCMM_THREAD_M_BEST; m++) {
      if (index_m_c < dimM && index_n_c < dimN) {
        _calculate_c(C_d[0], A_s[idy + m * 32][idx], beta);
      }
      index_m_c += 32;
      C_d += dimN*32;
    }
  }
}

#define     CU_DM_CSR_THREAD_X           32
#define     CU_DM_CSR_THREAD_Y           4
#define     CU_DM_CSR_N                  4
#define     CU_DM_CSR_BLOCK_M            (CU_DM_CSR_N*CU_DM_CSR_THREAD_Y)
#define     CU_DM_CSR_BLOCK_K            (CU_DM_CSR_THREAD_X)
#define     CU_DM_CSR_SHARED_ELEMENT     (1*CU_DM_CSR_THREAD_Y)
template <int VALUE_TYPE>
__global__ void KeSMatrixDenseMulCsr(real *C_d,
                                     real *A_d,
                                     real *csr_val,
                                     const int *csr_row,
                                     const int *csr_col,
                                     int dimM,
                                     int dimN,
                                     int dimK,
                                     real alpha,
                                     real beta) {
  const int idx = threadIdx.x;
  const int idy = threadIdx.y;
  int index_k = __mul24(blockIdx.x, CU_DM_CSR_THREAD_X) + threadIdx.x;
  int index_m = __mul24(blockIdx.y, CU_DM_CSR_BLOCK_M) +
    __mul24(threadIdx.y, CU_DM_CSR_N);

  if (index_k >= dimK) {
    return;
  }

  __shared__ real csr_val_sh[CU_DM_CSR_THREAD_X][CU_DM_CSR_SHARED_ELEMENT];
  __shared__ int csr_col_sh[CU_DM_CSR_THREAD_X][CU_DM_CSR_SHARED_ELEMENT];

  // possible optimization, cache this in shared memory
  int csr_start = csr_row[index_k];
  int csr_end = csr_row[index_k+1];
  int csr_index =  csr_start + idy;
  int csr_iter = (csr_end-csr_start)/CU_DM_CSR_SHARED_ELEMENT;
  int csr_rem = (csr_end-csr_start)%CU_DM_CSR_SHARED_ELEMENT;

  real tmp = 0.0;
  int index_n = -1;
  int index_m_t = index_m;
  real a_r[CU_DM_CSR_N] = {0};
  real *A_d_tmp = A_d + __mul24(index_m, dimK) + index_k;
  real *A_d_r = A_d_tmp;

  #pragma unroll
  for (int n=0; n < CU_DM_CSR_N; n++) {
    if ( index_m_t++ < dimM ) {
      a_r[n] = A_d_r[0];
      A_d_r += dimK;
    }
  }

  for (int csr_i = 0; csr_i < csr_iter; csr_i++) {
    #pragma unroll
    for (int i = 0; i < (CU_DM_CSR_SHARED_ELEMENT/CU_DM_CSR_THREAD_Y); i++) {
      if (VALUE_TYPE != 0) {
        csr_val_sh[idx][idy + i*CU_DM_CSR_THREAD_Y] = csr_val
        [csr_index];
      }
      csr_col_sh[idx][idy + i*CU_DM_CSR_THREAD_Y] = csr_col[csr_index];
      csr_index += CU_DM_CSR_THREAD_Y;
    }
    __syncthreads();

    #pragma unroll
    for (int index = 0; index < CU_DM_CSR_SHARED_ELEMENT; index++) {
      index_n = csr_col_sh[idx][index];
      real b_r = VALUE_TYPE == 0 ? 1.0 : csr_val_sh[idx][index];
      real *C_d_r = C_d + __mul24(index_m, dimN) + index_n;

      index_m_t = index_m;
      #pragma unroll
      for (int n=0; n < CU_DM_CSR_N; n++) {
        if (index_m_t++ < dimM) {
          tmp = alpha * b_r * a_r[n];
          paddle::paddleAtomicAdd(C_d_r, tmp);
          C_d_r += dimN;
        }
      }
    }
    __syncthreads();
  }

  if (csr_rem != 0) {
    #pragma unroll
    for (int i = 0; i < (CU_DM_CSR_SHARED_ELEMENT/CU_DM_CSR_THREAD_Y); i++) {
      if (csr_index < csr_end) {
        if (VALUE_TYPE !=0) {
          csr_val_sh[idx][idy + i*CU_DM_CSR_THREAD_Y] = csr_val[csr_index];
        }
        csr_col_sh[idx][idy + i*CU_DM_CSR_THREAD_Y] = csr_col[csr_index];
      }
      csr_index += CU_DM_CSR_THREAD_Y;
    }
    __syncthreads();

    #pragma unroll
    for (int index = 0; index < csr_rem; index++) {
      index_n = csr_col_sh[idx][index];
      real b_r = VALUE_TYPE == 0 ? 1.0 : csr_val_sh[idx][index];
      real *C_d_r = C_d + __mul24(index_m, dimN) + index_n;
      index_m_t = index_m;
      #pragma unroll
      for (int n=0; n < CU_DM_CSR_N; n++) {
        if (index_m_t++ < dimM) {
          tmp = alpha * b_r * a_r[n];
          paddle::paddleAtomicAdd(C_d_r, tmp);
          C_d_r += dimN;
        }
      }
    }
  }
}

#define     CU_CSCMM_DMD2CSC_THREAD_X   128
#define     CU_CSCMM_DMD2CSC_SHARE_X    128
__global__ void KeSMatrixDenseMulDense2CSC(real *csc_val,
                                           const int *csc_row,
                                           const int *csc_col,
                                           real *A_d,
                                           real *B_d,
                                           bool trans_A,
                                           bool trans_B,
                                           int dimM,
                                           int dimN,
                                           int dimK,
                                           real alpha,
                                           real beta) {
  __shared__ real B_s[CU_CSCMM_DMD2CSC_SHARE_X];
  const int idx = threadIdx.x;  // one block compute one column
  const int ibx = blockIdx.x;  // col index
  int csc_start;
  int csc_end;
  if (ibx < dimN) {
    csc_start = csc_col[ibx];
    csc_end = csc_col[ibx + 1];
  } else {
    csc_start = 0;
    csc_end = 0;
  }

  int iter_num = dimK / CU_CSCMM_DMD2CSC_SHARE_X;
  int iter_rem = dimK % CU_CSCMM_DMD2CSC_SHARE_X;
  real * B_tmp = B_d + ibx;  // column index

  for (int j = 0; j < iter_num; j++) {
    int rowStart = (j * CU_CSCMM_DMD2CSC_SHARE_X + idx) * dimN;
    int index = rowStart;
    for (int m = idx;
         m < CU_CSCMM_DMD2CSC_SHARE_X; m += CU_CSCMM_DMD2CSC_THREAD_X) {
     B_s[m] = B_tmp[index];
     index = index + CU_CSCMM_DMD2CSC_THREAD_X * dimN;
    }
    __syncthreads();

    for (int i = csc_col[ibx] + idx;
         i < csc_col[ibx + 1]; i += CU_CSCMM_DMD2CSC_THREAD_X) {
      int row = csc_row[i];  // row Index
      /* compute C[row, ibx] */
      float results = 0;
      if (!trans_A) {
        int index = row * dimK + j * CU_CSCMM_DMD2CSC_SHARE_X;
        for (int k = 0; k < CU_CSCMM_DMD2CSC_SHARE_X; k++) {
          results += A_d[index + k] * B_s[k];
        }
      } else {
        int  index = j * CU_CSCMM_DMD2CSC_SHARE_X;
        for (int k = 0; k < CU_CSCMM_DMD2CSC_SHARE_X; k++) {
          results += A_d[(index + k) * dimM + row] * B_s[k];
        }
      }
      csc_val[i]  += results * alpha;
    }
  }

  if (iter_rem) {
    int rowStart = (iter_num * CU_CSCMM_DMD2CSC_SHARE_X + idx) * dimN;
    int index = rowStart;
    // #pragma unroll
    for (int m = idx; m < iter_rem;  m += CU_CSCMM_DMD2CSC_THREAD_X) {
      B_s[m] = B_tmp[index];
      index = index + CU_CSCMM_DMD2CSC_THREAD_X * dimN;
    }
    __syncthreads();
    for (int i = csc_start + idx;
         i < csc_end; i += CU_CSCMM_DMD2CSC_THREAD_X) {
      int row = csc_row[i];  // row Index
      /* compute C[row, ibx] */
      float results = 0;
      if (!trans_A) {
        int index = row * dimK + iter_num * CU_CSCMM_DMD2CSC_SHARE_X;
        for (int k = 0; k < iter_rem; k++) {
          results += A_d[index + k] * B_s[k];
        }
      } else {
        int  index =  iter_num * CU_CSCMM_DMD2CSC_SHARE_X;
        for (int k = 0; k < iter_rem; k++) {
          results += A_d[(index + k) * dimM + row] * B_s[k];
        }
      }
      csc_val[i] += alpha * results;
    }
  }
}

#define     CU_CSCMM_DMD2CSR_THREAD_X   128
#define     CU_CSCMM_DMD2CSR_SHARE_X    128
__global__ void KeSMatrixDenseMulDense2CSR(real *csr_val,
                                     const int *csr_row,
                                     const int *csr_col,
                                     real *A_d,
                                     real *B_d,
                                     bool  trans_A,
                                     bool  trans_B,
                                     int dimM,
                                     int dimN,
                                     int dimK,
                                     real alpha,
                                     real beta) {
  __shared__ real A_s[CU_CSCMM_DMD2CSR_SHARE_X];
  const int idx = threadIdx.x;  // one block comput one row
  const int ibx = blockIdx.x;  // row index

  int csr_start;
  int csr_end;
  if (ibx < dimM) {
    csr_start = csr_row[ibx];
    csr_end = csr_row[ibx+1];
  } else {
    csr_start = 0;
    csr_end = 0;
  }

  int iter_num = dimK / CU_CSCMM_DMD2CSR_SHARE_X;
  int csr_rem = dimK % CU_CSCMM_DMD2CSR_SHARE_X;
  for (int j = 0; j < iter_num; j++) {
    if (!trans_A) {
      int colStart = j * CU_CSCMM_DMD2CSR_SHARE_X + ibx * dimK;
      int index = colStart + idx;
      #pragma unroll
      for (int m = idx;
           m < CU_CSCMM_DMD2CSR_SHARE_X; m += CU_CSCMM_DMD2CSR_THREAD_X) {
        A_s[m] = A_d[index];
        index = index + CU_CSCMM_DMD2CSR_THREAD_X;
      }
    } else {
      int colStart = (j * CU_CSCMM_DMD2CSR_SHARE_X) * dimM  + ibx;
      int index = colStart + idx * dimM;
      for (int m = idx;
           m < CU_CSCMM_DMD2CSR_SHARE_X; m += CU_CSCMM_DMD2CSR_THREAD_X) {
        A_s[m] = A_d[index];
        index = index + CU_CSCMM_DMD2CSR_THREAD_X * dimM;
      }
    }
    __syncthreads();
    for (int i = csr_start + idx; i < csr_end; i += CU_CSCMM_DMD2CSR_THREAD_X) {
      int col_idx =  csr_col[i];  // col index
      /* comput C[ibx, col_idx] */
      real results = 0;
      int index = (j * CU_CSCMM_DMD2CSR_SHARE_X) * dimN + col_idx;
      for (int k = 0; k < CU_CSCMM_DMD2CSR_SHARE_X; k++) {
        results += A_s[k] * B_d[k * dimN + index];
      }
      csr_val[i] += alpha * results;
    }
  }

  if (csr_rem) {
    if (!trans_A) {
      int colStart = (ibx + 1) * dimK- csr_rem;
      int index = colStart + idx;
      #pragma unroll
      for (int m = idx; m < csr_rem; m += CU_CSCMM_DMD2CSR_THREAD_X) {
        A_s[m] = A_d[index];
        index = index + CU_CSCMM_DMD2CSR_THREAD_X;
      }
     } else {
        int colStart = (iter_num * CU_CSCMM_DMD2CSR_SHARE_X) * dimM  + ibx;
        int index = colStart + idx * dimM;
        for (int m = idx; m < csr_rem;  m += CU_CSCMM_DMD2CSR_THREAD_X) {
          A_s[m] = A_d[index];
          index = index + CU_CSCMM_DMD2CSR_THREAD_X * dimM;
        }
     }
     __syncthreads();
     for (int i = csr_start + idx;
          i < csr_end; i += CU_CSCMM_DMD2CSR_THREAD_X) {
       int col_idx =  csr_col[i];
       float results = 0;
       int  index = (iter_num *CU_CSCMM_DMD2CSR_SHARE_X) * dimN + col_idx;
       for (int k = 0; k < csr_rem; k++) {
         results += A_s[k ] * B_d[k * dimN + index];
       }
       csr_val[i] += alpha * results;
     }
  }
}


/**
 *  @brief  Use to calculate row/col index for CSR/CSC sparse matrix
 *          according to csr_row(csc_col) and
 *          the value position in csr_val/csc_val
 *
 *  @param  indice      csr_row for hl_csr_matrix
 *                      csc_col for hl_csc_matrix
 *  @param  num         length of csr_row/csc_col
 *  @param  index       the value position in csr_val/csc_val
 *                      but need to add 1
 *                      that is, 1,2,3,...,nnz
 *  @note   the following kernels doesn't use findIndex,
 *          but may be used in the future.
 */
__device__ __forceinline__
int findIndex(int* indice, int num, int index) {
  int start = 0;
  int end = num - 1;
  int mid = -1;
  while (start < end) {
    mid = start + ((end - start) / 2);
    if (indice[mid] < index)
      start = mid + 1;
    else
      end = mid;
  }
  return (end - 1);
}


/**
 * @brief sum columns of csr sparse matrix (csr_val), then add to a_val.
 *        This kernel used atomicAdd and adapted to w >> h, w is the
 *        width of csr, and h is the height of csr.
 */
__global__ void KeSMatrixCsrColumnSum(real* a_val, real* csr_val,
                                      int* csr_col, const int dimNNZ) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  for (int idx = gid; idx < dimNNZ; idx += gridDim.x * blockDim.x) {
    int colIdx = csr_col[idx];
    real val = csr_val[idx];
    paddle::paddleAtomicAdd(a_val + colIdx, val);
  }
}

__global__ void KeSMatrixCsrAddBias(real* csr_val, int* csr_col, real* b_d,
                                    real scale, const int nnz) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;  // global index
  for (int idx = gid; idx < nnz; idx += gridDim.x * blockDim.x) {
    int colIdx = csr_col[idx];
    // not coalesced access to b_d
    csr_val[idx] += scale * b_d[colIdx];
  }
}

/**
 * @brief  csr sparse matrix add dense matrix.
 *         This kernel occurs load imbalances
 *         if number of each row is different greatly.
 */
__global__ void KeSMatrixCsrAddDense(real* csr_val, int* csr_row,
                                     int* csr_col, real* b_d, real alpha,
                                     real beta, int dimM, int dimN) {
  int gidx = blockIdx.x * blockDim.x + threadIdx.x;
  int gidy = blockIdx.y;
  if (gidy < dimM) {
    int start = csr_row[gidy];
    int end = csr_row[gidy + 1];
    for (int x = gidx; x < (end - start); x += gridDim.x * blockDim.x) {
      int col = csr_col[start + x];
      real val = csr_val[start + x];
      csr_val[start + x] = beta * val + alpha * b_d[gidy * dimN + col];
    }
  }
}

#define CU_BLOCK_K 16
#define CU_BLOCK_SIZE 128

__global__ void KeSMatrixDenseMulDenseTrans2CSR(
    real* csr_val, const int* csr_row, const int* csr_col, real* A_d,
    real* B_d, bool trans_A, bool trans_B, int dimM, int dimN, int dimK,
    real alpha, real beta) {

  __shared__ real B_s[CU_BLOCK_SIZE][CU_BLOCK_K];
  __shared__ real A_s[CU_BLOCK_K];

  const int idx = threadIdx.x;

  const int gidx_begin = blockIdx.x * CU_BLOCK_SIZE;
  const int gidy = blockIdx.y;
  const int gx_dim = gridDim.x * blockDim.x;

  int start = csr_row[gidy];
  int end = csr_row[gidy + 1];
  int size = end - start;

  int c_iter_num = (size + gx_dim - 1) / gx_dim;
  int iter_num = (dimK + CU_BLOCK_K - 1) / CU_BLOCK_K;
  for (int i = 0; i < c_iter_num; ++i) {
    if ((gidx_begin + i * gx_dim) >= size) {
      return;  // No need to calculate in this block.
    }

    real res = 0.0;
    int c_idx = gidx_begin + i * gx_dim + idx;

    for (int j = 0; j < iter_num; ++j) {
      int col = j * CU_BLOCK_K + idx;
      if (idx < CU_BLOCK_K) {
        A_s[idx] = col < dimK ? A_d[gidy * dimK + col] : 0.0;
      }
      for (int m = 0; m < CU_BLOCK_K; ++m) {
        int row = (idx / CU_BLOCK_K) + m * (CU_BLOCK_SIZE / CU_BLOCK_K);
        col = idx % CU_BLOCK_K;
        int csr_idx = gidx_begin + i * gx_dim + row;
        int ldRow = csr_idx < size ? csr_col[start + csr_idx] : 0;
        int ldCol = j * CU_BLOCK_K + col;
        B_s[row][col] = (csr_idx < size && ldCol < dimK) ?
                        B_d[ldRow * dimK + ldCol] : 0.0;
      }
      __syncthreads();

      for (int k = 0; k < CU_BLOCK_K; k++) {
        res += A_s[k] * B_s[idx][k];
      }
      __syncthreads();
    }

    if (c_idx < size) {
      csr_val[start + c_idx] += alpha * res;
    }
  }
}
