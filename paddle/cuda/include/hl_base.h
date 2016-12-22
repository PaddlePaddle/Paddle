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

#ifndef HL_BASE_H_
#define HL_BASE_H_

#include <cstddef>

#ifdef PADDLE_TYPE_DOUBLE
#define HL_FLOAT_MAX 3.40282347e+38F
#define HL_FLOAT_MIN 1.17549435e-38F
using real = double;
#else
#define HL_FLOAT_MAX 1.7976931348623157e+308
#define HL_FLOAT_MIN 2.2250738585072014e-308
using real = float;
#endif

/**
 * The maximum input value for exp, used to avoid overflow problem.
 * currently only used for tanh function.
 */
#define EXP_MAX_INPUT 40.0

/**
 * @brief DIVUP(x, y) is similar to ceil(x / y).
 * @note  For CUDA, DIVUP will be used to specify
 *        the size of blockDim.
 */
#ifndef DIVUP
#define DIVUP(x, y) (((x) + (y)-1) / (y))
#endif

/**
 * HPPL is an internal high performance parallel computing library
 * for high-level neural network routines, which can support many
 * heterogeneous compute architectures, such as GPU, FPGA, etc.
 */

/**
 * @brief   HPPL CUDA Stream.
 *
 * @note    Each thread can use HPPL_STREAM_* after calling hl_init.
 *          HPPL_STREAM_DEFAULT is HPPL default stream.
 */
typedef enum {
  HPPL_STREAM_DEFAULT = 0, /* Thread Default Stream*/
  HPPL_STREAM_1 = 1,
  HPPL_STREAM_2 = 2,
  HPPL_STREAM_3 = 3,
  HPPL_STREAM_4 = 4,
  HPPL_THREAD_STREAM_1 = 5,
  HPPL_THREAD_STREAM_2 = 6,
  HPPL_THREAD_STREAM_3 = 7,
  HPPL_THREAD_STREAM_4 = 8,
  HPPL_STREAM_END
} hl_stream_t;

/**
 * @brief HPPL activation mode.
 */
typedef enum {
  HL_ACTIVATION_SIGMOID = 0,
  HL_ACTIVATION_RELU = 1,
  HL_ACTIVATION_TANH = 2,
  HL_ACTIVATION_LINEAR = 3,
  HL_ACTIVATION_END
} hl_activation_mode_t;

/**
 * @brief Transpose type.
 */
typedef enum {
  HPPL_OP_N = 0, /* transpose */
  HPPL_OP_T = 1, /* non transpose */
  HPPL_OP_END
} hl_trans_op_t;

/**
 * @brief Lstm value.
 *
 * @param  gateValue         input value.
 * @param  prevStateValue    previous state value.
 * @param  stateValue        state value.
 * @param  stateActiveValue  state active value.
 * @param  outputValue       output value.
 */
typedef struct {
  real *gateValue;
  real *prevStateValue;
  real *stateValue;
  real *stateActiveValue;
  real *outputValue;
  real *checkIg;
  real *checkFg;
  real *checkOg;
} hl_lstm_value;

/**
 * @brief Lstm gradient.
 *
 * @param  gateGrad          input gradient.
 * @param  prevStateGrad     previous state gradient.
 * @param  stateGrad         state gradient.
 * @param  stateActiveGrad   state active gradient.
 * @param  outputGrad        output gradient.
 */
typedef struct {
  real *gateGrad;
  real *prevStateGrad;
  real *stateGrad;
  real *stateActiveGrad;
  real *outputGrad;
  real *checkIgGrad;
  real *checkFgGrad;
  real *checkOgGrad;
} hl_lstm_grad;

/**
 * @brief Gru value.
 *
 * @param  gateWeight           gate weight (updateGate + resetGate).
 * @param  stateWeight          frame state weight.
 * @param  gateValue            gate value results.
 * @param  resetOutputValue     resetOutput value.
 * @param  outputValue          output value.
 * @param  prevOutValue         previous output value.
 *
 */
typedef struct {
  real *gateWeight;
  real *stateWeight;
  real *gateValue;
  real *resetOutputValue;
  real *outputValue;
  real *prevOutValue;
} hl_gru_value;

/**
 * @brief Gru gradient.
 *
 * @param  gateWeightGrad       gate weight gradient.
 * @param  stateWeightGrad      frame state weight gradient.
 * @param  gateGrad             gate gradient results.
 * @param  resetOutputGrad      resetOutput gradient.
 * @param  outputGrad           output gradient.
 * @param  prevOutGrad          previous output gradient.
 */
typedef struct {
  real *gateWeightGrad;
  real *stateWeightGrad;
  real *gateGrad;
  real *resetOutputGrad;
  real *outputGrad;
  real *prevOutGrad;
} hl_gru_grad;

/**
 * @brief  Sparse matrix value type.
 */
typedef enum {
  HL_NO_VALUE = 0, /* matrix values only 0 or 1 */
  HL_FLOAT_VALUE = 1,
  HL_VALUE_END
} hl_matrix_value_t;

/**
 * @brief  HPPL matrix format.
 */
typedef enum {
  HL_SPARSE_CSR = 0,
  HL_SPARSE_CSC = 1,
  HL_SPARSE_END
} hl_matrix_format_t;

typedef struct _hl_matrix_s *hl_matrix_s;

/**
 * @brief   HPPL sparse matrix.
 *
 * @param  matrix     sparse matrix.
 * @param  format     matrix format.
 * @param  type       the type of matrix values.
 * @param  rows       matrix rows.
 * @param  cols       matrix columns.
 * @param  nnz        nonzero values of sparse matrix.
 */
typedef struct {
  hl_matrix_s matrix;
  hl_matrix_format_t format;
  hl_matrix_value_t type;
  int rows;
  int cols;
  size_t nnz;
} _hl_sparse_matrix_s, *hl_sparse_matrix_s;

#ifdef __NVCC__

#include "cuda_runtime.h"
#include "hl_cuda.h"
#include "paddle/utils/Logging.h"

extern __thread bool g_sync_flag;
extern __thread cudaStream_t default_stream;
#define STREAM_DEFAULT default_stream

/**
 * @brief   Check cuda kernel execution.
 * @param   msg   error string
 */
#define CHECK_SYNC(msg)                                               \
  if (true == g_sync_flag) {                                          \
    hl_stream_synchronize(HPPL_STREAM_DEFAULT);                       \
    cudaError_t err = (cudaError_t)hl_get_device_last_error();        \
    CHECK_EQ(cudaSuccess, err)                                        \
        << "[" << msg << "] "                                         \
        << "CUDA error: " << hl_get_device_error_string((size_t)err); \
  }

#endif /* __NVCC__ */

#endif /* HL_BASE_H_ */
