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

#ifndef HL_MATRIX_H_
#define HL_MATRIX_H_

#include "hl_base.h"

/**
 * @brief   Matrix addition: C_d[i] = alpha * A_d[i] + beta * B_d[i].
 *
 * @param[in]   A_d     input matrix (M x N).
 * @param[in]   B_d     input matrix (M x N).
 * @param[out]  C_d     output matrix (M x N).
 * @param[in]   dimM    matrix height.
 * @param[in]   dimN    matrix width.
 * @param[in]   alpha   scalar used for addition.
 * @param[in]   beta    scalar used for addition.
 *
 */
extern void hl_matrix_add(
    real* A_d, real* B_d, real* C_d, int dimM, int dimN, real alpha, real beta);
/**
 * @brief   Matrix Softmax.
 *
 * @param[in]   A_d     input maxtrix (M x N).
 * @param[out]  C_d     output matrix (M x N).
 * @param[in]   dimM    matrix height.
 * @param[in]   dimN    matrix width.
 *
 */
extern void hl_matrix_softmax(real* A_d, real* C_d, int dimM, int dimN);

/**
 * @brief   Matrix softmax derivative.
 *
 * @param[out]  grad_d       intput matrix (M x N).
 * @param[in]   output_d     output matrix (M x N).
 * @param[in]   sftmaxSum_d  softmax sum (M * 1).
 * @param[in]   dimM         matrix height.
 * @param[in]   dimN         matrix width.
 *
 */
extern void hl_matrix_softmax_derivative(
    real* grad_d, real* output_d, real* sftmaxSum_d, int dimM, int dimN);

/**
 * @brief   Sequence softmax.
 *
 * @param[in]   A_d         input vector.
 * @param[out]  C_d         output vector.
 * @param[in]   index       start positions of sequence.
 * @param[in]   numSequence sequence number.
 *
 */
extern void hl_sequence_softmax_forward(real* A_d,
                                        real* C_d,
                                        const int* index,
                                        int numSequence);

/**
 * @brief   Matrix cross entropy.
 *
 * @param[in]   A_d     input matrix (M x N).
 * @param[out]  C_d     output matrix (M X 1).
 * @param[in]   label_d input matrix (M x 1).
 * @param[in]   dimM    matrix height.
 * @param[in]   dimN    matrix width.
 *
 */
extern void hl_matrix_cross_entropy(
    real* A_d, real* C_d, int* label_d, int dimM, int dimN);

/**
 * @brief   Matrix cross entropy back propagation.
 *
 * @param[out]  grad_d      output matrix (M x N).
 * @param[in]   output_d    input matrix (M x N).
 * @param[in]   label_d     input vector (M x 1).
 * @param[in]   dimM        matrix height.
 * @param[in]   dimN        matrix width.
 *
 */
extern void hl_matrix_cross_entropy_bp(
    real* grad_d, real* output_d, int* label_d, int dimM, int dimN);

/**
 * @brief  Matrix multi-binary label cross entropy
 *
 * @param[in]   output    input matrix (M x N).
 * @param[out]  entropy   output matrix (M x 1).
 * @param[in]   mat       input sparse matrix.
 * @param[in]   dimM      matrix height.
 * @param[in]   dimN      matrix width.
 */
extern void hl_matrix_multi_binary_cross_entropy(
    real* output, real* entropy, hl_sparse_matrix_s mat, int dimM, int dimN);

/**
 * @brief  Matrix multi-binary label cross entropy backprop
 *
 * @param[in]   output    input matrix (M x N).
 * @param[out]  grad      output matrix (M x N).
 * @param[in]   mat       input sparse matrix.
 * @param[in]   dimM      matrix height.
 * @param[in]   dimN      matrix width.
 */
extern void hl_matrix_multi_binary_cross_entropy_bp(
    real* output, real* grad, hl_sparse_matrix_s mat, int dimM, int dimN);

/**
 * @brief  Matrix zero memory.
 *
 * @param[in,out]  data   input data.
 * @param[in]      num    length of data.
 *
 */
extern void hl_matrix_zero_mem(real* data, int num);

/**
 * @brief parameter relu forward
 *
 * @param[out] output     output data
 * @param[in]  input      input data
 * @param[in]  w          parameter data
 * @param[in]  width      matrix width
 * @param[in]  height     matrix height
 * @param[in]  partial_sum
 */

extern void hl_param_relu_forward(
    real* output, real* input, real* w, int width, int height, int partial_sum);
/**
 * @brief parameter relu backward w
 *
 * @param[out] grad_w      w grad
 * @param[in]  grad_o      output grad
 * @param[in]  input       input data
 * @param[in]  width       matrix width
 * @param[in]  height      matrix height
 * @param[in]  partial_sum
 */
extern void hl_param_relu_backward_w(real* grad_w,
                                     real* grad_o,
                                     real* input,
                                     int width,
                                     int height,
                                     int partial_sum);
/**
 * @brief parameter relu backward diff
 *
 * @param[in]       grad_o      output grad
 * @param[in]       input       input data
 * @param[in]       w           parameter
 * @param[out]      diff        diff
 * @param[in]       width       matrix width
 * @param[in]       height      matrix height
 * @param[in]       partial_sum
 */
extern void hl_param_relu_backward_diff(real* grad_o,
                                        real* input,
                                        real* w,
                                        real* diff,
                                        int width,
                                        int height,
                                        int partial_sum);

/**
 * @brief   Matrix addition: A_d[i][j] += scale * B_d[j/channel].
 *
 * @param[in]   A_d     input matrix (M x N).
 * @param[in]   B_d     input matrix (1 x channel).
 * @param[in]   channel width of B.
 * @param[in]   dimM    height of A.
 * @param[in]   dimN    width of A.
 * @param[in]   scale   scalar used for addition.
 *
 */
extern void hl_matrix_add_shared_bias(real* A_d,
                                      real* B_d,
                                      const int channel,
                                      const int dimM,
                                      const int dimN,
                                      real scale);

/**
 * @brief   Matrix addition: A_d[i][j] += scale * B_d[j/channel].
 *
 * @param[in]   B_d     input matrix (1 x channel).
 * @param[in]   A_d     input matrix (M x N).
 * @param[in]   channel width of B.
 * @param[in]   dimM    height of A.
 * @param[in]   dimN    width of A.
 * @param[in]   scale   scalar used for addition.
 *
 */
extern void hl_matrix_collect_shared_bias(real* B_d,
                                          real* A_d,
                                          const int channel,
                                          const int dimM,
                                          const int dimN,
                                          real scale);

/**
 * @brief  Matrix rotation in 90 degrees
 *
 * @param[in]   mat       input matrix (M x N).
 * @param[out]  matRot    output matrix (N x M).
 * @param[in]   dimM      input matrix height.
 * @param[in]   dimN      input matrix width.
 * @param[in]   clockWise rotation direction
 */
extern void hl_matrix_rotate(
    real* mat, real* matRot, int dimM, int dimN, bool clockWise);

#endif /* HL_MATRIX_H_ */
