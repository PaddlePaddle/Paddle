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


#include "hl_base.h"
#include "hl_device_functions.cuh"
#include "hl_cuda.h"
#include "paddle/utils/Logging.h"

template<int blockDimX, int blockDimY, int gridDimX, bool AddRow>
__global__ void KeMatrixAddRows(real* output, int ldo,
                                real* table, int ldt,
                                int* ids,
                                int numSamples,
                                int tableSize,
                                int dim) {
  int idx = threadIdx.x;
  int idy = blockIdx.x + threadIdx.y * gridDimX;

  while (idy < numSamples) {
    int tableId = ids[idy];
    if ((0 <= tableId) && (tableId < tableSize)) {
      real *out = output + idy * ldo;
      real *tab = table + tableId * ldt;
      for (int i = idx; i < dim; i += blockDimX) {
        if (AddRow) {
          paddle::paddleAtomicAdd(&tab[i], out[i]);
        } else {
          out[i] += tab[i];
        }
      }
    }
    idy += blockDimY * gridDimX;
  }
}

void hl_matrix_select_rows(real* output, int ldo,
                           real* table, int ldt,
                           int* ids,
                           int numSamples,
                           int tableSize,
                           int dim) {
  CHECK_NOTNULL(output);
  CHECK_NOTNULL(table);
  CHECK_NOTNULL(ids);

  dim3 threads(128, 8);
  dim3 grid(8, 1);
  KeMatrixAddRows<128, 8, 8, 0><<< grid, threads, 0, STREAM_DEFAULT >>>
    (output, ldo, table, ldt, ids, numSamples, tableSize, dim);

  CHECK_SYNC("hl_matrix_select_rows failed");
}

void hl_matrix_add_to_rows(real* table, int ldt,
                           real* input, int ldi,
                           int* ids,
                           int numSamples,
                           int tableSize,
                           int dim) {
  CHECK_NOTNULL(input);
  CHECK_NOTNULL(table);
  CHECK_NOTNULL(ids);

  dim3 threads(128, 8);
  dim3 grid(8, 1);
  KeMatrixAddRows<128, 8, 8, 1><<< grid, threads, 0, STREAM_DEFAULT >>>
    (input, ldi, table, ldt, ids, numSamples, tableSize, dim);

  CHECK_SYNC("hl_matrix_add_to_rows failed");
}

template<class T, int blockDimX, int gridDimX>
__global__ void KeVectorSelect(T* dst, int sized,
                               const T* src, int sizes,
                               const int* ids, int sizei) {
  int idx = threadIdx.x + blockDimX * blockIdx.x;
  while (idx < sizei) {
    int index = ids[idx];
    // check(index < sizes);
    dst[idx] = src[index];
    idx += blockDimX * gridDimX;
  }
}

template <class T>
void hl_vector_select_from(T* dst, int sized,
                           const T* src, int sizes,
                           const int* ids, int sizei) {
  CHECK_NOTNULL(dst);
  CHECK_NOTNULL(src);
  CHECK_NOTNULL(ids);
  CHECK_EQ(sized, sizei);

  dim3 threads(512, 1);
  dim3 grid(8, 1);
  KeVectorSelect<T, 512, 8><<< grid, threads, 0, STREAM_DEFAULT >>>
    (dst, sized, src, sizes, ids, sizei);

  CHECK_SYNC("hl_vector_select_from failed");
}

template
void hl_vector_select_from(real* dst, int sized,
                           const real* src, int sizes,
                           const int* ids, int sizei);
template
void hl_vector_select_from(int* dst, int sized,
                           const int* src, int sizes,
                           const int* ids, int sizei);

