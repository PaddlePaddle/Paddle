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


#ifndef HL_SPARSE_PH_
#define HL_SPARSE_PH_

#include "hl_base.h"

/**
 * @brief   sparse matrix csr format.
 *
 * @param   *csr_val     nonzero values of matrix.
 * @param   *csr_row     row indices.
 * @param   *csr_col     column indices.
 * @param   nnz_s        sizeof of csr_val & csr_col.
 * @param   row_s        sizeof of csr_row.
 * @param   sparsity     sparsity pattern.
 *
 */
typedef struct {
    real                *csr_val;
    int                 *csr_row;
    int                 *csr_col;
    size_t              nnz_s;
    int                 row_s;
    float               sparsity;
}_hl_csr_matrix, *hl_csr_matrix;

/**
 * @brief   sparse matrix csc format.
 *
 * @param   *csc_val      nonzero values of matrix.
 * @param   *csc_row      row indices.
 * @param   *csc_col      column indices.
 * @param   nnz_s         sizeof of csc_val & csc_row.
 * @param   col_s         sizeof of csc_col.
 * @param   sparsity      sparsity pattern.
 *
 */
typedef struct {
    real                *csc_val;
    int                 *csc_row;
    int                 *csc_col;
    size_t              nnz_s;
    int                 col_s;
    float               sparsity;
}_hl_csc_matrix, *hl_csc_matrix;

#define __sparse_get_type_return__(mat, type, field)\
  do {\
    hl_##type##_matrix type##_d = (hl_##type##_matrix)((mat)->matrix);\
    if (type##_d) {\
      return type##_d -> type##_##field;\
    } else {\
      LOG(WARNING) << "parameter " <<  #field << "NULL error!";\
      return NULL;\
    }\
  } while(0)

#define __sparse_get_return__(mat, field)\
  do {\
    if ((mat) == NULL) {\
      LOG(WARNING) << "parameter NULL error!";\
      return NULL;\
    }\
    if ((mat)->format == HL_SPARSE_CSR) {\
      __sparse_get_type_return__(mat, csr, field);\
    } else {\
      __sparse_get_type_return__(mat, csc, field);\
    }\
  } while(0)

#endif  /* HL_SPARSE_PH_ */
