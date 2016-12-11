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

#ifndef HL_AGGREGATE_STUB_H_
#define HL_AGGREGATE_STUB_H_

#include "hl_aggregate.h"

inline void hl_matrix_row_sum(real *A_d, real *C_d, int dimM, int dimN) {}

inline void hl_matrix_row_max(real *A_d, real *C_d, int dimM, int dimN) {}

inline void hl_matrix_row_min(real *A_d, real *C_d, int dimM, int dimN) {}

inline void hl_matrix_column_sum(real *A_d, real *C_d, int dimM, int dimN) {}

inline void hl_matrix_column_max(real *A_d, real *C_d, int dimM, int dimN) {}

inline void hl_matrix_column_min(real *A_d, real *C_d, int dimM, int dimN) {}

inline void hl_vector_sum(real *A_d, real *C_h, int dimM) {}

inline void hl_vector_abs_sum(real *A_d, real *C_h, int dimM) {}

#endif  // HL_AGGREGATE_STUB_H_
