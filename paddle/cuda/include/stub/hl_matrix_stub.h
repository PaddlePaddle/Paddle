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

#ifndef HL_MATRIX_STUB_H_
#define HL_MATRIX_STUB_H_

#include "hl_matrix.h"

inline void hl_matrix_add(real* A_d,
                          real* B_d,
                          real* C_d,
                          int dimM,
                          int dimN,
                          real alpha,
                          real beta) {}

inline void hl_matrix_softmax(real* A_d, real* C_d, int dimM, int dimN) {}

inline void hl_sequence_softmax_forward(real* A_d,
                                        real* C_d,
                                        const int* index,
                                        int numSequence) {}

inline void hl_matrix_softmax_derivative(
    real* grad_d, real* output_d, real* sftmaxSum_d, int dimM, int dimN) {}

inline void hl_matrix_classification_error(real* topVal,
                                           int ldv,
                                           int* topIds,
                                           real* src,
                                           int lds,
                                           int dim,
                                           int topkSize,
                                           int numSamples,
                                           int* label,
                                           real* recResult) {}

inline void hl_matrix_cross_entropy(
    real* A_d, real* C_d, int* label_d, int dimM, int dimN) {}

inline void hl_matrix_cross_entropy_bp(
    real* grad_d, real* output_d, int* label_d, int dimM, int dimN) {}

inline void hl_matrix_multi_binary_cross_entropy(
    real* output, real* entropy, hl_sparse_matrix_s mat, int dimM, int dimN) {}

inline void hl_matrix_multi_binary_cross_entropy_bp(
    real* output, real* grad, hl_sparse_matrix_s mat, int dimM, int dimN) {}

inline void hl_matrix_zero_mem(real* data, int num) {}

inline void hl_param_relu_forward(real* output,
                                  real* input,
                                  real* w,
                                  int width,
                                  int height,
                                  int partial_sum) {}

inline void hl_param_relu_backward_w(real* grad_w,
                                     real* grad_o,
                                     real* input,
                                     int width,
                                     int height,
                                     int partial_sum) {}

inline void hl_param_relu_backward_diff(real* grad_o,
                                        real* input,
                                        real* w,
                                        real* diff,
                                        int width,
                                        int height,
                                        int partial_sum) {}

inline void hl_matrix_add_shared_bias(real* A_d,
                                      real* B_d,
                                      const int channel,
                                      const int dimM,
                                      const int dimN,
                                      real scale) {}

inline void hl_matrix_collect_shared_bias(real* B_d,
                                          real* A_d,
                                          const int channel,
                                          const int dimM,
                                          const int dimN,
                                          real scale) {}

inline void hl_matrix_rotate(
    real* mat, real* matRot, int dimM, int dimN, bool clockWise) {}

#endif  // HL_MATRIX_STUB_H_
