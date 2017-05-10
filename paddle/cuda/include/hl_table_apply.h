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

#ifndef HL_TABLE_APPLY_H_
#define HL_TABLE_APPLY_H_

/**
 * @brief   Get row from table.
 *          output[i] += table[ids[i]]
 *          if ids[i] == -1, it will be ignored
 *
 * @param[out]  output          output matrix.
 * @param[in]   ldo             leading dimension of output.
 * @param[in]   table           table matrix.
 * @param[in]   ldt             leading dimension of table.
 * @param[in]   ids             ids vector.
 * @param[in]   numSamples      height of output.
 * @param[in]   tableSize       height of table.
 * @param[in]   dim             width of table.
 *
 */
extern void hl_matrix_select_rows(real* output,
                                  int ldo,
                                  real* table,
                                  int ldt,
                                  int* ids,
                                  int numSamples,
                                  int tableSize,
                                  int dim);

/**
 * @brief   Add row to table.
 *          table[ids[i]] += output[i]
 *          if ids[i] == -1, it will be ignored
 *
 * @param[out]  table           table matrix.
 * @param[in]   ldt             leading dimension of table.
 * @param[in]   input           input matrix.
 * @param[in]   ldi             leading dimension of input.
 * @param[in]   ids             ids vector.
 * @param[in]   numSamples      height of input.
 * @param[in]   tableSize       height of table.
 * @param[in]   dim             width of table.
 *
 */
extern void hl_matrix_add_to_rows(real* table,
                                  int ldt,
                                  real* input,
                                  int ldi,
                                  int* ids,
                                  int numSamples,
                                  int tableSize,
                                  int dim);

/**
 * @brief   Select element from vector.
 *
 * @param[out]  dst         output vector.
 * @param[in]   sized       size of dst.
 * @param[in]   src         input vector.
 * @param[in]   sizes       size of src.
 * @param[in]   ids         index vector.
 * @param[in]   sizei       size of ids.
 *
 */
template <class T>
extern void hl_vector_select_from(
    T* dst, int sized, const T* src, int sizes, const int* ids, int sizei);

#endif /* HL_TABLE_APPLY_H_ */
