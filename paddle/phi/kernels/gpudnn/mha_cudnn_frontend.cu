// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/phi/kernels/gpudnn/mha_cudnn_frontend.h"

#include <cub/cub.cuh>
#include <map>
#include <unordered_map>
#include <vector>

#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/enforce.h"

#define CUDNN_FRONTEND_UNUSED(X) ((void)X)

#ifdef PADDLE_WITH_CUDNN_FRONTEND

namespace phi {
namespace cudnn_fused_attn {

#define Q_ID 1
#define K_ID 2
#define V_ID 3
#define O_ID 4
#define S_ID 5
#define B_ID 6
#define D_CONST_ID 7
#define S_CONST_ID 8
#define Q_SEQLEN_ID 9
#define K_SEQLEN_ID 10
#define dQ_ID 11
#define dK_ID 12
#define dV_ID 13
#define dO_ID 14
#define MASK_VAL_ID 15
#define dS_ID 16
#define D_SEED_ID 17
#define D_OFFSET_ID 18
#define S_STATS_ID 19
#define S_SUM_ID 20
#define SCALE_PROB 21
#define K_TRANSPOSE_ID 22
#define dQ_ACCUM_ID 23

#define VIRTUAL_ID 30

void generateMatrixStrides(int64_t b,
                           int64_t h,
                           int64_t s_q,
                           int64_t s_kv,
                           int64_t d,
                           int64_t *strideA,
                           MHA_Layout layout,
                           MHA_Matrix matrix) {
  constexpr int batch_dim_idx = 0;
  constexpr int head_dim_idx = 1;
  constexpr int seqlen_dim_idx = 2;
  constexpr int hidden_dim_idx = 3;

  constexpr int seqlen_transpose_dim_idx = 3;
  constexpr int hidden_transpose_dim_idx = 2;

  constexpr int seqlen_q_dim_idx = 2;
  constexpr int seqlen_kv_dim_idx = 3;

  // to be deprecated in the future
  switch (matrix) {
    case MHA_Matrix::Q_Matrix:
      if (layout == MHA_Layout::QKV_INTERLEAVED) {
        strideA[hidden_dim_idx] = 1;
        strideA[seqlen_dim_idx] = 3 * h * d;
        strideA[head_dim_idx] = d;
        strideA[batch_dim_idx] = s_q * 3 * h * d;
      } else if ((layout == MHA_Layout::KV_INTERLEAVED) ||
                 (layout == MHA_Layout::NOT_INTERLEAVED)) {
        strideA[hidden_dim_idx] = 1;
        strideA[seqlen_dim_idx] = h * d;
        strideA[head_dim_idx] = d;
        strideA[batch_dim_idx] = s_q * h * d;
      }
      break;
    case MHA_Matrix::K_Matrix:
      if (layout == MHA_Layout::QKV_INTERLEAVED) {
        strideA[seqlen_dim_idx] = 3 * h * d;
        strideA[hidden_dim_idx] = 1;
        strideA[head_dim_idx] = d;
        strideA[batch_dim_idx] = s_kv * 3 * h * d;
      } else if (layout == MHA_Layout::KV_INTERLEAVED) {
        strideA[seqlen_dim_idx] = 2 * h * d;
        strideA[hidden_dim_idx] = 1;
        strideA[head_dim_idx] = d;
        strideA[batch_dim_idx] = s_kv * 2 * h * d;
      } else if (layout == MHA_Layout::NOT_INTERLEAVED) {
        strideA[seqlen_dim_idx] = h * d;
        strideA[hidden_dim_idx] = 1;
        strideA[head_dim_idx] = d;
        strideA[batch_dim_idx] = s_kv * h * d;
      }
      break;
    case MHA_Matrix::K_Matrix_Transpose:
      if (layout == MHA_Layout::QKV_INTERLEAVED) {
        strideA[seqlen_transpose_dim_idx] = 3 * h * d;
        strideA[hidden_transpose_dim_idx] = 1;
        strideA[head_dim_idx] = d;
        strideA[batch_dim_idx] = s_kv * 3 * h * d;
      } else if (layout == MHA_Layout::KV_INTERLEAVED) {
        strideA[seqlen_transpose_dim_idx] = 2 * h * d;
        strideA[hidden_transpose_dim_idx] = 1;
        strideA[head_dim_idx] = d;
        strideA[batch_dim_idx] = s_kv * 2 * h * d;
      } else if (layout == MHA_Layout::NOT_INTERLEAVED) {
        strideA[seqlen_transpose_dim_idx] = h * d;
        strideA[hidden_transpose_dim_idx] = 1;
        strideA[head_dim_idx] = d;
        strideA[batch_dim_idx] = s_kv * h * d;
      }
      break;
    case MHA_Matrix::V_Matrix:
      if (layout == MHA_Layout::QKV_INTERLEAVED) {
        strideA[hidden_dim_idx] = 1;
        strideA[seqlen_dim_idx] = 3 * h * d;
        strideA[head_dim_idx] = d;
        strideA[batch_dim_idx] = s_kv * 3 * h * d;
      } else if (layout == MHA_Layout::KV_INTERLEAVED) {
        strideA[hidden_dim_idx] = 1;
        strideA[seqlen_dim_idx] = 2 * h * d;
        strideA[head_dim_idx] = d;
        strideA[batch_dim_idx] = s_kv * 2 * h * d;
      } else if (layout == MHA_Layout::NOT_INTERLEAVED) {
        strideA[hidden_dim_idx] = 1;
        strideA[seqlen_dim_idx] = h * d;
        strideA[head_dim_idx] = d;
        strideA[batch_dim_idx] = s_kv * h * d;
      }
      break;
    case MHA_Matrix::V_Matrix_Transpose:
      if (layout == MHA_Layout::QKV_INTERLEAVED) {
        strideA[hidden_transpose_dim_idx] = 1;
        strideA[seqlen_transpose_dim_idx] = 3 * h * d;
        strideA[head_dim_idx] = d;
        strideA[batch_dim_idx] = s_kv * 3 * h * d;
      } else if (layout == MHA_Layout::KV_INTERLEAVED) {
        strideA[hidden_transpose_dim_idx] = 1;
        strideA[seqlen_transpose_dim_idx] = 2 * h * d;
        strideA[head_dim_idx] = d;
        strideA[batch_dim_idx] = s_kv * 2 * h * d;
      } else if (layout == MHA_Layout::NOT_INTERLEAVED) {
        strideA[hidden_transpose_dim_idx] = 1;
        strideA[seqlen_transpose_dim_idx] = h * d;
        strideA[head_dim_idx] = d;
        strideA[batch_dim_idx] = s_kv * h * d;
      }
      break;
    case MHA_Matrix::S_Matrix:
      strideA[seqlen_kv_dim_idx] = 1;
      strideA[seqlen_q_dim_idx] = s_kv;
      strideA[head_dim_idx] = s_q * s_kv;
      strideA[batch_dim_idx] = h * s_q * s_kv;
      break;
    case MHA_Matrix::O_Matrix:
      strideA[seqlen_kv_dim_idx] = 1;
      strideA[seqlen_q_dim_idx] = h * d;
      strideA[head_dim_idx] = d;
      strideA[batch_dim_idx] = s_q * h * d;
      break;
  }
}

static bool allowAllConfig(cudnnBackendDescriptor_t engine_config) {
  (void)engine_config;
  return false;
}

static cudnn_frontend::Tensor tensor_create(cudnnDataType_t type,
                                            int64_t id,
                                            int64_t const *dim,
                                            int64_t const *stride,
                                            bool is_virtual,
                                            bool is_value) {
  int nbDims = 4;
  auto tensor_created =
      cudnn_frontend::TensorBuilder()
          .setDim(nbDims, dim)
          .setStride(nbDims, stride)
          .setId(id)
          .setAlignment(
              16)  // 16B alignment is needed to run a tensor core engine
          .setDataType(type)
          .setVirtual(is_virtual)
          .setByValue(is_value)
          .build();
  VLOG(10) << tensor_created.describe();
  return tensor_created;
}

static cudnn_frontend::PointWiseDesc pw_desc_create(cudnnDataType_t type,
                                                    cudnnPointwiseMode_t mode) {
  auto pw_desc_created = cudnn_frontend::PointWiseDescBuilder()
                             .setMode(mode)
                             .setComputeType(type)
                             .build();

  VLOG(10) << pw_desc_created.describe();
  return pw_desc_created;
}

static cudnn_frontend::Operation unary_pw_op_create(
    cudnn_frontend::Tensor const &xDesc,
    cudnn_frontend::Tensor const &yDesc,
    cudnn_frontend::PointWiseDesc const &pwDesc) {
  auto pw_op_created = cudnn_frontend::OperationBuilder(
                           CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                           .setxDesc(xDesc)
                           .setyDesc(yDesc)
                           .setpwDesc(pwDesc)
                           .build();
  VLOG(10) << pw_op_created.describe();
  return pw_op_created;
}

static cudnn_frontend::Operation binary_pw_op_create(
    cudnn_frontend::Tensor const &xDesc,
    cudnn_frontend::Tensor const &bDesc,
    cudnn_frontend::Tensor const &yDesc,
    cudnn_frontend::PointWiseDesc const &pwDesc) {
  auto pw_op_created = cudnn_frontend::OperationBuilder(
                           CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                           .setxDesc(xDesc)
                           .setbDesc(bDesc)
                           .setyDesc(yDesc)
                           .setpwDesc(pwDesc)
                           .build();
  VLOG(10) << pw_op_created.describe();
  return pw_op_created;
}

static cudnn_frontend::Operation ternary_pw_op_create(
    cudnn_frontend::Tensor const &xDesc,
    cudnn_frontend::Tensor const &bDesc,
    cudnn_frontend::Tensor const &tDesc,
    cudnn_frontend::Tensor const &yDesc,
    cudnn_frontend::PointWiseDesc const &pwDesc) {
  auto pw_op_created = cudnn_frontend::OperationBuilder(
                           CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                           .setxDesc(xDesc)
                           .setbDesc(bDesc)
                           .settDesc(tDesc)
                           .setyDesc(yDesc)
                           .setpwDesc(pwDesc)
                           .build();
  VLOG(10) << pw_op_created.describe();
  return pw_op_created;
}

static cudnn_frontend::Tensor createScale(
    int64_t b,
    int64_t h,
    int64_t s_q,
    int64_t s_kv,
    int64_t d,
    MHA_Layout layout,
    cudnnDataType_t tensorType,
    const cudnn_frontend::Tensor &sTensor,
    std::vector<cudnn_frontend::Operation> *ops) {
  // scale
  int64_t scale_dim[4] = {1, 1, 1, 1};
  int64_t scale_stride[4] = {1, 1, 1, 1};

  int64_t s_dim[4] = {b, h, s_q, s_kv};
  int64_t s_stride[4];
  generateMatrixStrides(
      b, h, s_q, s_kv, d, s_stride, layout, MHA_Matrix::S_Matrix);

  auto scaleTensor = tensor_create(tensorType,
                                   S_CONST_ID,
                                   scale_dim,
                                   scale_stride,
                                   false,
                                   true);  // is by value
  auto sScaleTensor = tensor_create(tensorType,
                                    VIRTUAL_ID + 2000,
                                    s_dim,
                                    s_stride,
                                    true,
                                    false);  // is virtual

  // Define the scale descriptor
  auto scaleDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_MUL);

  // Create a scale node
  auto scale_op =
      binary_pw_op_create(sTensor, scaleTensor, sScaleTensor, scaleDesc);

  ops->push_back(std::move(scale_op));
  return sScaleTensor;
}

static cudnn_frontend::Tensor createQKBMM(
    int64_t b,
    int64_t h,
    int64_t s_q,
    int64_t s_kv,
    int64_t d,
    bool variable_sequence_length,
    MHA_Layout layout,
    cudnnDataType_t tensorType,
    std::vector<cudnn_frontend::Operation> *ops) {
  // Creates the necessary tensor descriptors
  int64_t q_dim[4] = {b, h, s_q, d};
  int64_t q_stride[4];
  generateMatrixStrides(
      b, h, s_q, s_kv, d, q_stride, layout, MHA_Matrix::Q_Matrix);

  int64_t k_dim[4] = {b, h, d, s_kv};
  int64_t k_stride[4];
  generateMatrixStrides(
      b, h, s_q, s_kv, d, k_stride, layout, MHA_Matrix::K_Matrix_Transpose);

  int64_t s_dim[4] = {b, h, s_q, s_kv};
  int64_t s_stride[4];
  generateMatrixStrides(
      b, h, s_q, s_kv, d, s_stride, layout, MHA_Matrix::S_Matrix);

  int64_t seqlen_dim[4] = {b, 1, 1, 1};
  int64_t seqlen_stride[4] = {1, 1, 1, 1};

  auto qTensor = tensor_create(tensorType, Q_ID, q_dim, q_stride, false, false);
  auto kTransposeTensor = tensor_create(
      tensorType, K_ID, k_dim, k_stride, false, false);  // is virtual
  // first GEMM output
  auto sTensor = tensor_create(CUDNN_DATA_FLOAT,
                               VIRTUAL_ID + 1,
                               s_dim,
                               s_stride,
                               true,
                               false);  // is virtual

  // Define the matmul 1 desc
  auto matmul_1_Desc = cudnn_frontend::MatMulDescBuilder()
                           .setComputeType(CUDNN_DATA_FLOAT)
                           .setPaddingValue(0.0f)
                           .build();

  auto seqlenQTensor = tensor_create(
      CUDNN_DATA_INT32, Q_SEQLEN_ID, seqlen_dim, seqlen_stride, false, false);
  auto seqlenKTensor = tensor_create(
      CUDNN_DATA_INT32, K_SEQLEN_ID, seqlen_dim, seqlen_stride, false, false);

  // Create a matmul 1 node
  auto &&matmul_op_builder = cudnn_frontend::OperationBuilder(
      CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR);

  matmul_op_builder.setaMatDesc(qTensor)
      .setbMatDesc(kTransposeTensor)
      .setcMatDesc(sTensor)
      .setmatmulDesc(matmul_1_Desc);

  if (variable_sequence_length) {
    matmul_op_builder.setmOverrideDesc(seqlenQTensor)
        .setnOverrideDesc(seqlenKTensor);
  }

  auto matmul_op1 = matmul_op_builder.build();

  ops->push_back(std::move(matmul_op1));

  return sTensor;
}

static cudnn_frontend::Tensor createPaddingMask(
    int64_t b,
    int64_t h,
    int64_t s_q,
    int64_t s_kv,
    int64_t d,
    MHA_Layout layout,
    cudnnDataType_t tensorType,
    std::vector<cudnn_frontend::Operation> *ops,
    const cudnn_frontend::Tensor &prevBlockOutputTensor) {
  CUDNN_FRONTEND_UNUSED(d);
  CUDNN_FRONTEND_UNUSED(layout);
  CUDNN_FRONTEND_UNUSED(tensorType);

  PADDLE_ENFORCE_EQ(
      (ops->size() != 0),
      true,
      phi::errors::PreconditionNotMet(
          "Padding Mask constructed incorrectly as the first one"));

  // subtraction output
  int64_t afterBMM1_dim[4] = {b, h, s_q, s_kv};
  int64_t afterBMM1_stride[4] = {h * s_q * s_kv, s_q * s_kv, s_kv, 1};

  int64_t maskVal_dim[4] = {1, 1, 1, 1};
  int64_t maskVal_stride[4] = {1, 1, 1, 1};

  int64_t seqlen_dim[4] = {b, 1, 1, 1};
  int64_t seqlen_stride[4] = {1, 1, 1, 1};

  // mask value to put in the masked pixels
  auto maskValTensor = tensor_create(
      CUDNN_DATA_FLOAT, MASK_VAL_ID, maskVal_dim, maskVal_stride, false, true);
  auto seqlenQTensor = tensor_create(
      CUDNN_DATA_INT32, Q_SEQLEN_ID, seqlen_dim, seqlen_stride, false, false);
  auto seqlenKTensor = tensor_create(
      CUDNN_DATA_INT32, K_SEQLEN_ID, seqlen_dim, seqlen_stride, false, false);

  // gen index row output
  auto rowIndexTensor = tensor_create(CUDNN_DATA_FLOAT,
                                      VIRTUAL_ID + 300,
                                      afterBMM1_dim,
                                      afterBMM1_stride,
                                      true,
                                      false);
  // gen index column output
  auto columnIndexTensor = tensor_create(CUDNN_DATA_FLOAT,
                                         VIRTUAL_ID + 301,
                                         afterBMM1_dim,
                                         afterBMM1_stride,
                                         true,
                                         false);
  // less than row output
  auto lessThanRowTensor = tensor_create(CUDNN_DATA_BOOLEAN,
                                         VIRTUAL_ID + 302,
                                         afterBMM1_dim,
                                         afterBMM1_stride,
                                         true,
                                         false);
  // less than column output
  auto lessThanColTensor = tensor_create(CUDNN_DATA_BOOLEAN,
                                         VIRTUAL_ID + 303,
                                         afterBMM1_dim,
                                         afterBMM1_stride,
                                         true,
                                         false);
  // padding mask (lessthanRow && lessthanCol)
  auto paddingMaskTensor = tensor_create(CUDNN_DATA_BOOLEAN,
                                         VIRTUAL_ID + 304,
                                         afterBMM1_dim,
                                         afterBMM1_stride,
                                         true,
                                         false);

  // output after masking
  auto maskOutputTensor = tensor_create(CUDNN_DATA_FLOAT,
                                        VIRTUAL_ID + 305,
                                        afterBMM1_dim,
                                        afterBMM1_stride,
                                        true,
                                        false);

  // Define the gen index for row descriptor
  auto genIndexRowDesc = cudnn_frontend::PointWiseDescBuilder()
                             .setMode(CUDNN_POINTWISE_GEN_INDEX)
                             .setAxis(2)
                             .setComputeType(CUDNN_DATA_FLOAT)
                             .build();

  // Create a gen index Node.
  auto genIndexRow_op = unary_pw_op_create(
      prevBlockOutputTensor, rowIndexTensor, genIndexRowDesc);

  // Define the gen index for row descriptor
  auto genIndexColumnDesc = cudnn_frontend::PointWiseDescBuilder()
                                .setMode(CUDNN_POINTWISE_GEN_INDEX)
                                .setAxis(3)
                                .setComputeType(CUDNN_DATA_FLOAT)
                                .build();

  // Create a gen index Node.
  auto genIndexColumn_op = unary_pw_op_create(
      prevBlockOutputTensor, columnIndexTensor, genIndexColumnDesc);

  // Define the less than comparison for row descriptor
  auto lessThanRowDesc =
      pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_CMP_LT);

  // Create a less than comparison for row Node.
  auto lessThanRow_op = binary_pw_op_create(
      rowIndexTensor, seqlenQTensor, lessThanRowTensor, lessThanRowDesc);

  // Define the less than comparison for column descriptor
  auto lessThanColDesc =
      pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_CMP_LT);

  // Create a less than comparison for col Node.
  auto lessThanCol_op = binary_pw_op_create(
      columnIndexTensor, seqlenKTensor, lessThanColTensor, lessThanColDesc);

  // Define the less than comparison for column descriptor
  auto paddingMaskAndDesc =
      pw_desc_create(CUDNN_DATA_BOOLEAN, CUDNN_POINTWISE_LOGICAL_AND);

  // Create a and node for combining lessThanRow and lessThanCol
  auto paddingMaskAnd_op = binary_pw_op_create(lessThanRowTensor,
                                               lessThanColTensor,
                                               paddingMaskTensor,
                                               paddingMaskAndDesc);

  /////////////////// Apply the mask //////////////////////////

  // Define the binary select to perform masking descriptor
  auto maskDesc =
      pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_BINARY_SELECT);

  // Create a binary select Node.
  auto mask_op = ternary_pw_op_create(prevBlockOutputTensor,
                                      maskValTensor,
                                      paddingMaskTensor,
                                      maskOutputTensor,
                                      maskDesc);

  ops->push_back(std::move(genIndexRow_op));
  ops->push_back(std::move(genIndexColumn_op));
  ops->push_back(std::move(lessThanRow_op));
  ops->push_back(std::move(lessThanCol_op));
  ops->push_back(std::move(paddingMaskAnd_op));
  ops->push_back(std::move(mask_op));

  return maskOutputTensor;
}

static cudnn_frontend::Tensor createCausalMask(
    int64_t b,
    int64_t h,
    int64_t s_q,
    int64_t s_kv,
    int64_t d,
    MHA_Layout layout,
    cudnnDataType_t tensorType,
    std::vector<cudnn_frontend::Operation> *ops,
    const cudnn_frontend::Tensor &prevBlockOutputTensor) {
  CUDNN_FRONTEND_UNUSED(d);
  CUDNN_FRONTEND_UNUSED(layout);
  CUDNN_FRONTEND_UNUSED(tensorType);

  PADDLE_ENFORCE_EQ(
      (ops->size() != 0),
      true,
      phi::errors::PreconditionNotMet(
          "Causal Mask constructed incorrectly as the first one"));

  // subtraction output
  int64_t afterBMM1_dim[4] = {b, h, s_q, s_kv};
  int64_t afterBMM1_stride[4] = {h * s_q * s_kv, s_q * s_kv, s_kv, 1};

  int64_t maskVal_dim[4] = {1, 1, 1, 1};
  int64_t maskVal_stride[4] = {1, 1, 1, 1};

  // mask value to put in the masked pixels
  auto maskValTensor = tensor_create(CUDNN_DATA_FLOAT,
                                     MASK_VAL_ID,
                                     maskVal_dim,
                                     maskVal_stride,
                                     false,
                                     true);  // is by value
  // gen index row output
  auto rowIndexTensor = tensor_create(CUDNN_DATA_FLOAT,
                                      VIRTUAL_ID + 100,
                                      afterBMM1_dim,
                                      afterBMM1_stride,
                                      true,
                                      false);  // is virtual
  // gen index column output
  auto columnIndexTensor = tensor_create(CUDNN_DATA_FLOAT,
                                         VIRTUAL_ID + 101,
                                         afterBMM1_dim,
                                         afterBMM1_stride,
                                         true,
                                         false);  // is virtual
  // create causal mask (row >= col)
  auto causalMaskTensor = tensor_create(CUDNN_DATA_BOOLEAN,
                                        VIRTUAL_ID + 106,
                                        afterBMM1_dim,
                                        afterBMM1_stride,
                                        true,
                                        false);  // is virtual

  // output after masking
  auto maskOutputTensor = tensor_create(CUDNN_DATA_FLOAT,
                                        VIRTUAL_ID + 107,
                                        afterBMM1_dim,
                                        afterBMM1_stride,
                                        true,
                                        false);  // is virtual

  // Define the gen index for row descriptor
  auto genIndexRowDesc = cudnn_frontend::PointWiseDescBuilder()
                             .setMode(CUDNN_POINTWISE_GEN_INDEX)
                             .setAxis(2)
                             .setComputeType(CUDNN_DATA_FLOAT)
                             .build();

  // Create a gen index node
  auto genIndexRow_op = unary_pw_op_create(
      prevBlockOutputTensor, rowIndexTensor, genIndexRowDesc);

  // Define the gen index for row descriptor
  auto genIndexColumnDesc = cudnn_frontend::PointWiseDescBuilder()
                                .setMode(CUDNN_POINTWISE_GEN_INDEX)
                                .setAxis(3)
                                .setComputeType(CUDNN_DATA_FLOAT)
                                .build();

  // Create a gen index node
  auto genIndexColumn_op = unary_pw_op_create(
      prevBlockOutputTensor, columnIndexTensor, genIndexColumnDesc);

  // Define the greater than equal to comparison descriptor
  auto rowGreaterColDesc =
      pw_desc_create(CUDNN_DATA_BOOLEAN, CUDNN_POINTWISE_CMP_GE);

  // Create a greater than equal to node
  auto rowGreaterCol_op = binary_pw_op_create(
      rowIndexTensor, columnIndexTensor, causalMaskTensor, rowGreaterColDesc);

  // Define the binary select to perform masking descriptor
  auto maskDesc =
      pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_BINARY_SELECT);

  // Create a binary select node
  auto mask_op = ternary_pw_op_create(prevBlockOutputTensor,
                                      maskValTensor,
                                      causalMaskTensor,
                                      maskOutputTensor,
                                      maskDesc);

  ops->push_back(std::move(genIndexRow_op));
  ops->push_back(std::move(genIndexColumn_op));
  ops->push_back(std::move(rowGreaterCol_op));
  ops->push_back(std::move(mask_op));

  return maskOutputTensor;
}

static cudnn_frontend::Tensor createSoftmaxForward(
    int64_t b,
    int64_t h,
    int64_t s_q,
    int64_t s_kv,
    bool isTraining,
    std::vector<cudnn_frontend::Operation> *ops,
    const cudnn_frontend::Tensor &sAfterMaskTensor) {
  int64_t afterBMM1_dim[4] = {b, h, s_q, s_kv};
  int64_t afterBMM1_stride[4] = {h * s_q * s_kv, s_q * s_kv, s_kv, 1};

  int64_t afterReduction_dim[4] = {b, h, s_q, 1};
  int64_t afterReduction_stride[4] = {h * s_q, s_q, 1, 1};

  // max (x)
  auto afterMaxReductionTensor = tensor_create(CUDNN_DATA_FLOAT,
                                               VIRTUAL_ID + 150,
                                               afterReduction_dim,
                                               afterReduction_stride,
                                               true,
                                               false);  // is virtual

  // x - max(x)
  auto afterSubtractionTensor = tensor_create(CUDNN_DATA_FLOAT,
                                              VIRTUAL_ID + 151,
                                              afterBMM1_dim,
                                              afterBMM1_stride,
                                              true,
                                              false);  // is virtual

  // e^(x - max(x))
  auto afterExponentTensor = tensor_create(CUDNN_DATA_FLOAT,
                                           VIRTUAL_ID + 152,
                                           afterBMM1_dim,
                                           afterBMM1_stride,
                                           true,
                                           false);  // is virtual;

  // sum (e^(x - max(x)))
  auto afterAddReductionTensor = tensor_create(CUDNN_DATA_FLOAT,
                                               VIRTUAL_ID + 153,
                                               afterReduction_dim,
                                               afterReduction_stride,
                                               true,
                                               false);  // is virtual

  // log (sum (e^(x - max(x))))
  auto afterLogLTensor = tensor_create(CUDNN_DATA_FLOAT,
                                       VIRTUAL_ID + 154,
                                       afterReduction_dim,
                                       afterReduction_stride,
                                       true,
                                       false);

  // M + log (sum (e^(x - max(x))))
  auto softmaxStatsTensor = tensor_create(CUDNN_DATA_FLOAT,
                                          S_STATS_ID,
                                          afterReduction_dim,
                                          afterReduction_stride,
                                          !isTraining,
                                          false);
  // not virtual if training is true, virtual if training is false

  // divide (e/ sum(e))
  auto afterSoftmaxTensor =
      cudnn_frontend::TensorBuilder()
          .setDim(4, afterBMM1_dim)
          .setStride(4, afterBMM1_stride)
          .setId(VIRTUAL_ID + 156)
          .setAlignment(
              16)  // 16B alignment is needed to run a tensor core engine
          .setDataType(CUDNN_DATA_FLOAT)
          .setVirtual(true)
          .setByValue(false)
          .setReorderType(cudnn_frontend::cudnnBackendTensorReordering_t::
                              CUDNN_TENSOR_REORDERING_F16x16)
          .build();

  // Define the reduction descriptor
  auto reductionMaxDesc = cudnn_frontend::ReductionDescBuilder()
                              .setComputeType(CUDNN_DATA_FLOAT)
                              .setReductionOp(CUDNN_REDUCE_TENSOR_MAX)
                              .build();

  // Create a reduction max node
  auto reductionMax_op = cudnn_frontend::OperationBuilder(
                             CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR)
                             .setxDesc(sAfterMaskTensor)
                             .setyDesc(afterMaxReductionTensor)
                             .setreductionDesc(reductionMaxDesc)
                             .build();

  // Define the subtract descriptor
  auto subtractDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_SUB);

  // Create a subtract node
  auto subtract_op = binary_pw_op_create(sAfterMaskTensor,
                                         afterMaxReductionTensor,
                                         afterSubtractionTensor,
                                         subtractDesc);

  // Define the exponent descriptor
  auto exponentDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_EXP);

  // Create a exponent node
  auto exponent_op = unary_pw_op_create(
      afterSubtractionTensor, afterExponentTensor, exponentDesc);

  // Define the reduction descriptor
  auto reductionAddDesc = cudnn_frontend::ReductionDescBuilder()
                              .setComputeType(CUDNN_DATA_FLOAT)
                              .setReductionOp(CUDNN_REDUCE_TENSOR_ADD)
                              .build();

  // Create a reduction add node
  auto reductionAdd_op = cudnn_frontend::OperationBuilder(
                             CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR)
                             .setxDesc(afterExponentTensor)
                             .setyDesc(afterAddReductionTensor)
                             .setreductionDesc(reductionAddDesc)
                             .build();

  // Create log descriptor
  auto logDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_LOG);

  // Create log node
  auto log_op =
      unary_pw_op_create(afterAddReductionTensor, afterLogLTensor, logDesc);

  // Create add descriptor
  auto addDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_ADD);

  // Create add node
  auto add_op = binary_pw_op_create(
      afterMaxReductionTensor, afterLogLTensor, softmaxStatsTensor, addDesc);

  // Define the division descriptor
  auto divisionDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_DIV);

  // Create a subtract node
  auto division_op = binary_pw_op_create(afterExponentTensor,
                                         afterAddReductionTensor,
                                         afterSoftmaxTensor,
                                         divisionDesc);

  ops->push_back(std::move(reductionMax_op));
  ops->push_back(std::move(subtract_op));
  ops->push_back(std::move(exponent_op));
  ops->push_back(std::move(reductionAdd_op));
  ops->push_back(std::move(log_op));
  ops->push_back(std::move(add_op));
  ops->push_back(std::move(division_op));

  return afterSoftmaxTensor;
}

static cudnn_frontend::Tensor createDropoutForward(
    int64_t b,
    int64_t h,
    int64_t s_q,
    int64_t s_kv,
    int64_t d,
    double probability,
    cudnnDataType_t tensorType,
    std::vector<cudnn_frontend::Operation> *ops,
    const cudnn_frontend::Tensor &afterSoftmaxTensor) {
  CUDNN_FRONTEND_UNUSED(d);

  PADDLE_ENFORCE_EQ(
      (ops->size() != 0),
      true,
      phi::errors::PreconditionNotMet(
          "Dropout DAG constructed incorrectly as the first one"));

  int64_t afterBMM1_dim[4] = {b, h, s_q, s_kv};
  int64_t afterBMM1_stride[4] = {h * s_q * s_kv, s_q * s_kv, s_kv, 1};

  int64_t scale_dim[4] = {1, 1, 1, 1};
  int64_t scale_stride[4] = {1, 1, 1, 1};

  auto dropoutSeed = tensor_create(CUDNN_DATA_INT64,
                                   D_SEED_ID,
                                   scale_dim,
                                   scale_stride,
                                   false,
                                   false);  // not virtual
  auto dropoutOffset = tensor_create(CUDNN_DATA_INT64,
                                     D_OFFSET_ID,
                                     scale_dim,
                                     scale_stride,
                                     false,
                                     false);  // not virtual

  // mask for the dropout
  auto dropoutMaskTensor = tensor_create(CUDNN_DATA_FLOAT,
                                         VIRTUAL_ID + 200,
                                         afterBMM1_dim,
                                         afterBMM1_stride,
                                         true,
                                         false);  // is virtual
  // after dropout tensor
  auto afterDropoutTensor =
      cudnn_frontend::TensorBuilder()
          .setDim(4, afterBMM1_dim)
          .setStride(4, afterBMM1_stride)
          .setId(VIRTUAL_ID + 201)
          .setAlignment(
              16)  // 16B alignment is needed to run a tensor core engine
          .setDataType(tensorType)
          .setVirtual(true)
          .setByValue(false)
          .setReorderType(cudnn_frontend::cudnnBackendTensorReordering_t::
                              CUDNN_TENSOR_REORDERING_F16x16)
          .build();
  // scale after dropout
  auto scaleDropoutTensor = tensor_create(CUDNN_DATA_FLOAT,
                                          D_CONST_ID,
                                          scale_dim,
                                          scale_stride,
                                          false,
                                          true);  // is by value
  // after Scale
  auto afterScaleTensor = tensor_create(tensorType,
                                        VIRTUAL_ID + 202,
                                        afterBMM1_dim,
                                        afterBMM1_stride,
                                        true,
                                        false);  // is virtual

  // Define the reduction descriptor
  auto rngDesc = cudnn_frontend::RngDescBuilder()
                     .setRngDistribution(CUDNN_RNG_DISTRIBUTION_BERNOULLI)
                     .setBernoulliDistProbability(1.0 - probability)
                     .build();

  // Create a rng node
  auto rng_op =
      cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_RNG_DESCRIPTOR)
          .setyDesc(dropoutMaskTensor)
          .setSeedDesc(dropoutSeed)
          .setOffsetDesc(dropoutOffset)
          .setRngDesc(rngDesc)
          .build();

  // Define the multiply mask descriptor
  auto maskMulDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_MUL);

  // Create a multiply mask node
  auto maskMul_op = binary_pw_op_create(
      afterSoftmaxTensor, dropoutMaskTensor, afterDropoutTensor, maskMulDesc);

  // Define the multiply scale descriptor
  auto scaleMulDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_MUL);

  // Create a multiply scale node
  auto scaleMul_op = binary_pw_op_create(
      afterDropoutTensor, scaleDropoutTensor, afterScaleTensor, scaleMulDesc);

  ops->push_back(std::move(rng_op));
  ops->push_back(std::move(maskMul_op));
  ops->push_back(std::move(scaleMul_op));

  return afterScaleTensor;
}

static cudnn_frontend::Tensor createDropoutBackward(
    int64_t b,
    int64_t h,
    int64_t s_q,
    int64_t s_kv,
    int64_t d,
    double probability,
    cudnnDataType_t tensorType,
    std::vector<cudnn_frontend::Operation> *ops,
    const cudnn_frontend::Tensor &afterSoftmaxTensor,
    const cudnn_frontend::Tensor &dropoutMaskTensor) {
  CUDNN_FRONTEND_UNUSED(d);

  PADDLE_ENFORCE_EQ(
      (ops->size() != 0),
      true,
      phi::errors::PreconditionNotMet(
          "Dropout DAG constructed incorrectly as the first one"));

  int64_t afterBMM1_dim[4] = {b, h, s_q, s_kv};
  int64_t afterBMM1_stride[4] = {h * s_q * s_kv, s_q * s_kv, s_kv, 1};

  int64_t scale_dim[4] = {1, 1, 1, 1};
  int64_t scale_stride[4] = {1, 1, 1, 1};

  auto dropoutSeed = tensor_create(CUDNN_DATA_INT64,
                                   D_SEED_ID,
                                   scale_dim,
                                   scale_stride,
                                   false,
                                   false);  // not virtual
  auto dropoutOffset = tensor_create(CUDNN_DATA_INT64,
                                     D_OFFSET_ID,
                                     scale_dim,
                                     scale_stride,
                                     false,
                                     false);  // not virtual

  // after dropout tensor
  auto afterDropoutTensor =
      cudnn_frontend::TensorBuilder()
          .setDim(4, afterBMM1_dim)
          .setStride(4, afterBMM1_stride)
          .setId(VIRTUAL_ID + 201)
          .setAlignment(
              16)  // 16B alignment is needed to run a tensor core engine
          .setDataType(tensorType)
          .setVirtual(true)
          .setByValue(false)
          .setReorderType(cudnn_frontend::cudnnBackendTensorReordering_t::
                              CUDNN_TENSOR_REORDERING_F16x16)
          .build();
  // scale after dropout
  auto scaleDropoutTensor = tensor_create(CUDNN_DATA_FLOAT,
                                          D_CONST_ID,
                                          scale_dim,
                                          scale_stride,
                                          false,
                                          true);  // is by value
  // after Scale
  auto afterScaleTensor = tensor_create(tensorType,
                                        VIRTUAL_ID + 202,
                                        afterBMM1_dim,
                                        afterBMM1_stride,
                                        true,
                                        false);  // is virtual

  // Define the reduction descriptor
  auto rngDesc = cudnn_frontend::RngDescBuilder()
                     .setRngDistribution(CUDNN_RNG_DISTRIBUTION_BERNOULLI)
                     .setBernoulliDistProbability(1.0 - probability)
                     .build();

  // Create a rng node
  auto rng_op =
      cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_RNG_DESCRIPTOR)
          .setyDesc(dropoutMaskTensor)
          .setSeedDesc(dropoutSeed)
          .setOffsetDesc(dropoutOffset)
          .setRngDesc(rngDesc)
          .build();

  // Define the multiply mask descriptor
  auto maskMulDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_MUL);

  // Create a multiply mask node
  auto maskMul_op = binary_pw_op_create(
      afterSoftmaxTensor, dropoutMaskTensor, afterDropoutTensor, maskMulDesc);

  // Define the multiply scale descriptor
  auto scaleMulDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_MUL);

  // Create a multiply scale node
  auto scaleMul_op = binary_pw_op_create(
      afterDropoutTensor, scaleDropoutTensor, afterScaleTensor, scaleMulDesc);

  ops->push_back(std::move(rng_op));
  ops->push_back(std::move(maskMul_op));
  ops->push_back(std::move(scaleMul_op));

  return afterScaleTensor;
}

static void createSVBMM(int64_t b,
                        int64_t h,
                        int64_t s_q,
                        int64_t s_kv,
                        int64_t d,
                        bool variable_sequence_length,
                        MHA_Layout layout,
                        cudnnDataType_t tensorType,
                        std::vector<cudnn_frontend::Operation> *ops,
                        cudnn_frontend::Tensor const &afterScaleDropoutTensor) {
  PADDLE_ENFORCE_EQ((ops->size() != 0),
                    true,
                    phi::errors::PreconditionNotMet(
                        "SVBMM op constructed incorrectly as the first one"));

  int64_t v_dim[4] = {b, h, s_kv, d};
  int64_t v_stride[4];
  generateMatrixStrides(
      b, h, s_q, s_kv, d, v_stride, layout, MHA_Matrix::V_Matrix);

  int64_t o_dim[4] = {b, h, s_q, d};
  int64_t o_stride[4];
  generateMatrixStrides(
      b, h, s_q, s_kv, d, o_stride, layout, MHA_Matrix::O_Matrix);

  int64_t seqlen_dim[4] = {b, 1, 1, 1};
  int64_t seqlen_stride[4] = {1, 1, 1, 1};

  auto seqlenQTensor = tensor_create(
      CUDNN_DATA_INT32, Q_SEQLEN_ID, seqlen_dim, seqlen_stride, false, false);
  auto seqlenKTensor = tensor_create(
      CUDNN_DATA_INT32, K_SEQLEN_ID, seqlen_dim, seqlen_stride, false, false);

  auto vTensor = tensor_create(tensorType, V_ID, v_dim, v_stride, false, false);
  // second GEMM output
  auto oTensor = tensor_create(tensorType, O_ID, o_dim, o_stride, false, false);

  // Define the matmul 2 desc
  auto matmul_2_Desc = cudnn_frontend::MatMulDescBuilder()
                           .setComputeType(CUDNN_DATA_FLOAT)
                           .setPaddingValue(0.0f)
                           .build();

  // Create a matmul 2 node
  auto &&matmul_op_builder = cudnn_frontend::OperationBuilder(
      CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR);

  matmul_op_builder.setaMatDesc(afterScaleDropoutTensor)
      .setbMatDesc(vTensor)
      .setcMatDesc(oTensor)
      .setmatmulDesc(matmul_2_Desc);

  if (variable_sequence_length) {
    matmul_op_builder.setmOverrideDesc(seqlenQTensor)
        .setkOverrideDesc(seqlenKTensor);
  }

  auto matmul_op2 = matmul_op_builder.build();

  ops->push_back(std::move(matmul_op2));
}

struct FADescriptor {
  std::int64_t b;
  std::int64_t h;
  std::int64_t s_q;
  std::int64_t s_kv;
  std::int64_t d;
  float attnScale;
  bool isTraining;
  float dropoutProbability;
  MHA_Layout layout;
  MHA_Bias_Type bias_type;
  MHA_Mask_Type mask_type;
  cudnnDataType_t tensor_type;
  bool use_workspace_opt;
  bool variable_sequence_length;

  bool operator<(const FADescriptor &rhs) const {
    return std::tie(b,
                    h,
                    s_q,
                    s_kv,
                    d,
                    attnScale,
                    isTraining,
                    dropoutProbability,
                    layout,
                    mask_type,
                    bias_type,
                    tensor_type,
                    use_workspace_opt,
                    variable_sequence_length) <
           std::tie(rhs.b,
                    rhs.h,
                    rhs.s_q,
                    rhs.s_kv,
                    rhs.d,
                    rhs.attnScale,
                    rhs.isTraining,
                    rhs.dropoutProbability,
                    rhs.layout,
                    rhs.mask_type,
                    rhs.bias_type,
                    rhs.tensor_type,
                    rhs.use_workspace_opt,
                    rhs.variable_sequence_length);
  }
};
}  // namespace cudnn_fused_attn
}  // namespace phi

using namespace phi::cudnn_fused_attn;  // NOLINT

constexpr int BLOCK_SIZE = 512;

__global__ __launch_bounds__(BLOCK_SIZE) void mask_to_actual_seqlens_kernel(
    const int32_t *mask,
    int32_t *q_actual_seqlen,
    int32_t *kv_actual_seqlen,
    int q_seqlen,
    int kv_seqlen,
    bool need_kv) {
  typedef cub::BlockReduce<int, BLOCK_SIZE> BlockReduce;
  __shared__ typename BlockReduce::TempStorage q_smem;
  __shared__ typename BlockReduce::TempStorage kv_smem;
  unsigned int tid = threadIdx.x;
  unsigned int batch_offset = blockIdx.x * q_seqlen * kv_seqlen;

  // load mask, convert to 1/0, do accumulation
  int q = 0, kv = 0;
  for (unsigned int q_idx = tid * kv_seqlen; q_idx < q_seqlen * kv_seqlen;
       q_idx += BLOCK_SIZE * kv_seqlen) {
    q += (mask[q_idx + batch_offset] ? 1 : 0);
  }

  if (need_kv) {
    for (unsigned int kv_idx = tid; kv_idx < kv_seqlen; kv_idx += BLOCK_SIZE) {
      kv += (mask[kv_idx + batch_offset] ? 1 : 0);
    }
  }
  __syncthreads();

  // compute cub::BlockReduce
  int q_sum, kv_sum;
  q_sum = BlockReduce(q_smem).Sum(q);
  if (need_kv) kv_sum = BlockReduce(kv_smem).Sum(kv);

  // write result for this block to global mem
  if (tid == 0) {
    q_actual_seqlen[blockIdx.x] = q_sum;
    if (need_kv) {
      kv_actual_seqlen[blockIdx.x] = kv_sum;
    }
  }
}

void fused_attn_arbitrary_seqlen_fwd(
    int64_t b,
    int64_t h,
    int64_t s_q,
    int64_t s_kv,
    int64_t d,
    bool is_training,
    float scaling_factor,
    float dropout_probability,
    MHA_Layout layout,
    MHA_Mask_Type mask_type,
    void *devPtrQ,
    void *devPtrK,
    void *devPtrV,
    void *devPtrSoftmaxStats,
    void *devPtrO,
    void *devPtrMask,
    // void *devPtrCuSeqlenQ, void *devPtrCuSeqlenKV,
    void *devPtrDropoutSeed,
    void *devPtrDropoutOffset,
    cudnnDataType_t tensorType,
    cudaStream_t stream,
    cudnnHandle_t handle) {
  try {
    CUDNN_CALL(phi::dynload::cudnnSetStream(handle, stream));

    if (!is_training) {
      dropout_probability = 0.0f;
    }

    bool variable_sequence_length =
        CUDNN_VERSION >= 8906 && mask_type == MHA_Mask_Type::PADDING_MASK;

    FADescriptor descriptor{b,
                            h,
                            s_q,
                            s_kv,
                            d,
                            scaling_factor,
                            is_training,
                            dropout_probability,
                            layout,
                            MHA_Bias_Type::NO_BIAS,
                            mask_type,
                            tensorType,
                            false,
                            variable_sequence_length};

    using CacheType = std::map<FADescriptor, cudnn_frontend::ExecutionPlan>;
    static thread_local CacheType fmha_fprop_cache;

    // Get plan from cache if cache is available, otherwise create one
    auto get_plan = [&](CacheType &cache, const FADescriptor &descriptor) {
      // if hit, return
      auto it = cache.find(descriptor);
      if (it != cache.end()) {
        auto plan = it->second;
        return plan;
      }

      // otherwise, build the op_graph and the plan. Then update cache
      std::vector<cudnn_frontend::Operation const *> all_ops;
      std::vector<cudnn_frontend::Operation> ops;

      // Q * K^T
      auto sTensor = createQKBMM(b,
                                 h,
                                 s_q,
                                 s_kv,
                                 d,
                                 variable_sequence_length,
                                 layout,
                                 tensorType,
                                 &ops);

      // Q * K^T * bmmScale
      auto sScaleTensor = createScale(
          b, h, s_q, s_kv, d, layout, CUDNN_DATA_FLOAT, sTensor, &ops);

      auto &sAfterMaskTensor = sScaleTensor;

      if (mask_type == MHA_Mask_Type::CAUSAL_MASK) {
        sAfterMaskTensor = createCausalMask(
            b, h, s_q, s_kv, d, layout, tensorType, &ops, sScaleTensor);
      } else if (variable_sequence_length) {  // padding mask
        sAfterMaskTensor = createPaddingMask(
            b, h, s_q, s_kv, d, layout, tensorType, &ops, sScaleTensor);
      }

      PADDLE_ENFORCE_EQ(
          (dropout_probability >= 0.0f && dropout_probability < 1.0f),
          true,
          phi::errors::PreconditionNotMet(
              "dropout_probability should be in the range [0, 1)"));

      auto softmax_output = createSoftmaxForward(
          b, h, s_q, s_kv, is_training, &ops, sAfterMaskTensor);

      // Dropout(softmax)
      auto dropout_output = createDropoutForward(b,
                                                 h,
                                                 s_q,
                                                 s_kv,
                                                 d,
                                                 dropout_probability,
                                                 tensorType,
                                                 &ops,
                                                 softmax_output);

      createSVBMM(b,
                  h,
                  s_q,
                  s_kv,
                  d,
                  variable_sequence_length,
                  layout,
                  tensorType,
                  &ops,
                  dropout_output);

      for (unsigned int i = 0; i < ops.size(); i++) {
        all_ops.push_back(&ops[i]);
      }

      // Create an Operation Graph
      auto opGraph = cudnn_frontend::OperationGraphBuilder()
                         .setHandle(handle)
                         .setOperationGraph(all_ops.size(), all_ops.data())
                         .build();

      cudnn_frontend::EngineConfigList filtered_configs;
      auto statuses =
          cudnn_frontend::get_heuristics_list<1>({"heuristics_instant"},
                                                 opGraph,
                                                 allowAllConfig,
                                                 filtered_configs,
                                                 true);

      if (filtered_configs.size() == 0) {
        cudnn_frontend::set_error_and_throw_exception(
            nullptr,
            CUDNN_STATUS_NOT_SUPPORTED,
            "run_mha_fprop: No config returned by the heuristics");
      }

      auto plan = cudnn_frontend::ExecutionPlanBuilder()
                      .setHandle(handle)
                      .setEngineConfig(filtered_configs[0], opGraph.getTag())
                      .build();

      cache.insert({descriptor, plan});
      return plan;
    };

    auto plan = get_plan(fmha_fprop_cache, descriptor);
    VLOG(10) << "Plan tag: " << plan.getTag();

    auto plan_workspace_size = plan.getWorkspaceSize();
    VLOG(10) << plan.describe() << " plan requires workspace "
             << plan_workspace_size;

    size_t actual_seqlen_workspace_size = 2 * b * sizeof(int32_t);
    size_t workspace_size = plan_workspace_size + actual_seqlen_workspace_size;

    void *workspace = nullptr;
    if (workspace_size > 0) {
      PADDLE_ENFORCE_GPU_SUCCESS(
          cudaMallocAsync(&workspace, workspace_size, stream));
    }

    // Prepare actual seqlen
    constexpr size_t nthreads_per_block = 512;
    const size_t grid = b;
    void *devActualSeqlenQ =
        static_cast<int8_t *>(workspace) + plan_workspace_size;
    void *devActualSeqlenK =
        static_cast<int8_t *>(devActualSeqlenQ) + b * sizeof(int32_t);

    if (variable_sequence_length) {
      mask_to_actual_seqlens_kernel<<<grid, nthreads_per_block, 0, stream>>>(
          static_cast<const int32_t *>(devPtrMask),
          static_cast<int32_t *>(devActualSeqlenQ),
          static_cast<int32_t *>(devActualSeqlenK),
          s_q,
          s_kv,
          true);
      PADDLE_ENFORCE_GPU_SUCCESS(cudaGetLastError());
    }

    std::set<std::pair<uint64_t, void *>> data_ptrs;
    // Add all the data pointers to be used in the variant pack
    float negInfinity = -1.0E+30f;
    float scale_dropout = 1.0f / (1.0f - dropout_probability);

    data_ptrs.insert(std::pair<uint64_t, void *>(Q_ID, devPtrQ));
    data_ptrs.insert(std::pair<uint64_t, void *>(K_ID, devPtrK));
    data_ptrs.insert(std::pair<uint64_t, void *>(V_ID, devPtrV));
    data_ptrs.insert(std::pair<uint64_t, void *>(MASK_VAL_ID, &negInfinity));
    data_ptrs.insert(std::pair<uint64_t, void *>(S_CONST_ID, &scaling_factor));
    data_ptrs.insert(std::pair<uint64_t, void *>(O_ID, devPtrO));
    data_ptrs.insert(std::pair<uint64_t, void *>(D_SEED_ID, devPtrDropoutSeed));
    data_ptrs.insert(
        std::pair<uint64_t, void *>(D_OFFSET_ID, devPtrDropoutOffset));
    data_ptrs.insert(std::pair<uint64_t, void *>(D_CONST_ID, &scale_dropout));

    if (variable_sequence_length) {
      data_ptrs.insert(
          std::pair<uint64_t, void *>(Q_SEQLEN_ID, devActualSeqlenQ));
      data_ptrs.insert(
          std::pair<uint64_t, void *>(K_SEQLEN_ID, devActualSeqlenK));
    }

    // If training mode, we write out softmax stats
    if (is_training) {
      data_ptrs.insert(
          std::pair<uint64_t, void *>(S_STATS_ID, devPtrSoftmaxStats));
    }

    auto variantPack = cudnn_frontend::VariantPackBuilder()
                           .setWorkspacePointer(workspace)
                           .setDataPointers(data_ptrs)
                           .build();

    CUDNN_CALL(phi::dynload::cudnnBackendExecute(
        handle, plan.get_raw_desc(), variantPack.get_raw_desc()));

    if (workspace_size > 0) {
      PADDLE_ENFORCE_GPU_SUCCESS(cudaFreeAsync(workspace, stream));
    }
  } catch (cudnn_frontend::cudnnException &e) {
    struct cudaDeviceProp prop;
    PADDLE_ENFORCE_GPU_SUCCESS(cudaGetDeviceProperties(&prop, 0));

    // cudnn flash attention is only for GA100 cards and GH100 cards
    if (!((prop.major == 8 && prop.minor == 0) ||
          (prop.major == 9 && prop.minor == 0)) &&
        (e.getCudnnStatus() == CUDNN_STATUS_ARCH_MISMATCH ||
         e.getCudnnStatus() == CUDNN_STATUS_NOT_SUPPORTED)) {
      VLOG(10) << "Only supported for GA100 (cuDNN >= 8900) and "
                  "GH100 (cuDNN >= 8900) GPUs";
    } else {
      VLOG(10) << "[ERROR] Exception " << e.what();
    }
  }
}

void fused_attn_arbitrary_seqlen_bwd(
    int64_t b,
    int64_t h,
    int64_t s_q,
    int64_t s_kv,
    int64_t d,
    float scaling_factor,
    float dropout_probability,
    MHA_Layout layout,
    MHA_Mask_Type mask_type,
    void *devPtrQ,
    void *devPtrKTranspose,
    void *devPtrVTranspose,
    void *devPtrO,
    void *devPtrSoftmaxStats,
    void *devPtrdQ,
    void *devPtrdK,
    void *devPtrdV,
    void *devPtrdO,
    void *devPtrMask,
    // void *devPtrCuSeqlenQ, void *devPtrCuSeqlenKV,
    void *devPtrDropoutSeed,
    void *devPtrDropoutOffset,
    cudnnDataType_t tensorType,
    cudaStream_t stream,
    cudnnHandle_t handle,
    bool use_workspace_opt) {
  try {
    CUDNN_CALL(phi::dynload::cudnnSetStream(handle, stream));

    bool variable_sequence_length =
        CUDNN_VERSION >= 8906 && mask_type == MHA_Mask_Type::PADDING_MASK;

    FADescriptor descriptor{b,
                            h,
                            s_q,
                            s_kv,
                            d,
                            scaling_factor,
                            true,
                            dropout_probability,
                            layout,
                            MHA_Bias_Type::NO_BIAS,
                            mask_type,
                            tensorType,
                            use_workspace_opt,
                            variable_sequence_length};

    using CacheType = std::map<FADescriptor, cudnn_frontend::ExecutionPlan>;
    static thread_local CacheType fmha_bprop_cache;

    auto get_plan = [&](CacheType &cache, const FADescriptor &descriptor) {
      auto it = cache.find(descriptor);
      if (it != cache.end()) {
        return it->second;
      }

      std::vector<cudnn_frontend::Operation const *> all_ops;
      std::vector<cudnn_frontend::Operation> ops;

      // Creates the necessary tensor descriptors
      int64_t q_dim[4] = {b, h, s_q, d};
      int64_t q_stride[4];
      generateMatrixStrides(
          b, h, s_q, s_kv, d, q_stride, layout, MHA_Matrix::Q_Matrix);

      int64_t k_transpose_dim[4] = {b, h, d, s_kv};
      int64_t k_transpose_stride[4];
      generateMatrixStrides(b,
                            h,
                            s_q,
                            s_kv,
                            d,
                            k_transpose_stride,
                            layout,
                            MHA_Matrix::K_Matrix_Transpose);

      int64_t v_transpose_dim[4] = {b, h, d, s_kv};
      int64_t v_transpose_stride[4];
      generateMatrixStrides(b,
                            h,
                            s_q,
                            s_kv,
                            d,
                            v_transpose_stride,
                            layout,
                            MHA_Matrix::V_Matrix_Transpose);

      int64_t p_dim[4] = {b, h, s_q, s_kv};
      int64_t p_stride[4];
      generateMatrixStrides(
          b, h, s_q, s_kv, d, p_stride, layout, MHA_Matrix::S_Matrix);

      int64_t p_transpose_dim[4] = {b, h, s_kv, s_q};
      int64_t p_transpose_stride[4];
      p_transpose_stride[0] = p_stride[0];
      p_transpose_stride[1] = p_stride[1];
      p_transpose_stride[2] = p_stride[3];
      p_transpose_stride[3] = p_stride[2];

      int64_t o_dim[4] = {b, h, s_q, d};
      int64_t o_stride[4];
      generateMatrixStrides(
          b, h, s_q, s_kv, d, o_stride, layout, MHA_Matrix::O_Matrix);

      int64_t dqAccum_dim[4] = {b, h, s_q, d};
      int64_t dqAccum_stride[4];
      generateMatrixStrides(
          b, h, s_q, s_kv, d, dqAccum_stride, layout, MHA_Matrix::O_Matrix);

      int64_t seqlen_dim[4] = {b, 1, 1, 1};
      int64_t seqlen_stride[4] = {1, 1, 1, 1};

      int64_t scale_dim[4] = {1, 1, 1, 1};
      int64_t scale_stride[4] = {1, 1, 1, 1};

      auto seqlenQTensor = tensor_create(CUDNN_DATA_INT32,
                                         Q_SEQLEN_ID,
                                         seqlen_dim,
                                         seqlen_stride,
                                         false,
                                         false);
      auto seqlenKTensor = tensor_create(CUDNN_DATA_INT32,
                                         K_SEQLEN_ID,
                                         seqlen_dim,
                                         seqlen_stride,
                                         false,
                                         false);

      /*******************************************************************************
       *                          Dot product dO * O */

      // output and gradient of the output
      auto oTensor =
          tensor_create(tensorType, O_ID, o_dim, o_stride, false, false);
      auto dOTensor =
          tensor_create(tensorType, dO_ID, o_dim, o_stride, false, false);

      auto dotProductTensor = tensor_create(CUDNN_DATA_FLOAT,
                                            VIRTUAL_ID,
                                            o_dim,
                                            o_stride,
                                            true,
                                            false);  // is virtual

      // Create pointwise mul
      auto multiplyDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_MUL);

      // do * O
      auto dotProductOp = binary_pw_op_create(
          dOTensor, oTensor, dotProductTensor, multiplyDesc);
      ops.push_back(std::move(dotProductOp));

      /*******************************************************************************
       *                         Reduction(dO * O) */

      int64_t reduction_dim[4] = {b, h, s_q, 1};
      int64_t reduction_stride[4] = {h * s_q, s_q, 1, 1};

      // reduction(dO * O)
      auto afterReductionTensor = tensor_create(CUDNN_DATA_FLOAT,
                                                VIRTUAL_ID + 1,
                                                reduction_dim,
                                                reduction_stride,
                                                true,
                                                false);  // is virtual
      auto reductionAddDesc = cudnn_frontend::ReductionDescBuilder()
                                  .setComputeType(CUDNN_DATA_FLOAT)
                                  .setReductionOp(CUDNN_REDUCE_TENSOR_ADD)
                                  .build();

      // Create a reduction add node
      auto reductionAdd_op = cudnn_frontend::OperationBuilder(
                                 CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR)
                                 .setxDesc(dotProductTensor)
                                 .setyDesc(afterReductionTensor)
                                 .setreductionDesc(reductionAddDesc)
                                 .build();
      ops.push_back(std::move(reductionAdd_op));

      /*******************************************************************************
       *                        reduction(dO * O) * scale prob -> softmaxSum */

      auto softmaxSumTensor = tensor_create(CUDNN_DATA_FLOAT,
                                            S_SUM_ID,
                                            reduction_dim,
                                            reduction_stride,
                                            false,
                                            false);  // not virtual
      auto scaleProbTensor = tensor_create(CUDNN_DATA_FLOAT,
                                           SCALE_PROB,
                                           scale_dim,
                                           scale_stride,
                                           false,
                                           true);  // not virtual
      auto softmaxSumOp = binary_pw_op_create(afterReductionTensor,
                                              scaleProbTensor,
                                              softmaxSumTensor,
                                              multiplyDesc);
      ops.push_back(std::move(softmaxSumOp));

      /*******************************************************************************
       *                        Q @ K.T -> P */

      // Inputs from fprop
      auto qTensor =
          tensor_create(tensorType, Q_ID, q_dim, q_stride, false, false);
      auto kTransposeTensor = tensor_create(
          tensorType, K_ID, k_transpose_dim, k_transpose_stride, false, false);
      auto pTensor = tensor_create(CUDNN_DATA_FLOAT,
                                   VIRTUAL_ID + 2,
                                   p_dim,
                                   p_stride,
                                   true,
                                   false);  // is virtual

      // matmul to calculate dvTensor
      auto matmul_0_Desc = cudnn_frontend::MatMulDescBuilder()
                               .setComputeType(CUDNN_DATA_FLOAT)
                               .setPaddingValue(0.0f)
                               .build();

      auto &&matmul_op_builder = cudnn_frontend::OperationBuilder(
          CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR);

      matmul_op_builder.setaMatDesc(qTensor)
          .setbMatDesc(kTransposeTensor)
          .setcMatDesc(pTensor)
          .setmatmulDesc(matmul_0_Desc);

      if (variable_sequence_length) {
        matmul_op_builder.setmOverrideDesc(seqlenQTensor)
            .setnOverrideDesc(seqlenKTensor);
      }

      auto matmul_op0 = matmul_op_builder.build();

      ops.push_back(std::move(matmul_op0));

      /*******************************************************************************
       *                        P * bmmScale -> pAfterScale */

      auto bmmScaleTensor = tensor_create(CUDNN_DATA_FLOAT,
                                          S_CONST_ID,
                                          scale_dim,
                                          scale_stride,
                                          false,
                                          true);  // not virtual and by value
      auto pAfterScaleTensor = tensor_create(CUDNN_DATA_FLOAT,
                                             VIRTUAL_ID + 2000,
                                             p_dim,
                                             p_stride,
                                             true,
                                             false);  // virtual
      auto scaleOp = binary_pw_op_create(
          pTensor, bmmScaleTensor, pAfterScaleTensor, multiplyDesc);
      ops.push_back(std::move(scaleOp));

      /*******************************************************************************
       *                          Causal masking -> pAfterMaskTensor */

      auto &pAfterMaskTensor = pAfterScaleTensor;
      if (mask_type == MHA_Mask_Type::CAUSAL_MASK) {  // causal mask
        pAfterMaskTensor = createCausalMask(
            b, h, s_q, s_kv, d, layout, tensorType, &ops, pAfterScaleTensor);
      } else if (variable_sequence_length) {  // padding mask
        pAfterMaskTensor = createPaddingMask(
            b, h, s_q, s_kv, d, layout, tensorType, &ops, pAfterScaleTensor);
      }

      /*******************************************************************************
       *                          pAfterMaskTensor - softmaxStats ->
       * pAfterSubtract */

      auto pAfterSubtractTensor = tensor_create(CUDNN_DATA_FLOAT,
                                                VIRTUAL_ID + 3,
                                                p_dim,
                                                p_stride,
                                                true,
                                                false);  // is virtual
      auto softmaxStatsTensor = tensor_create(CUDNN_DATA_FLOAT,
                                              S_STATS_ID,
                                              reduction_dim,
                                              reduction_stride,
                                              false,
                                              false);  // not virtual
      auto subtractDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_SUB);
      auto subtract_op = binary_pw_op_create(pAfterMaskTensor,
                                             softmaxStatsTensor,
                                             pAfterSubtractTensor,
                                             subtractDesc);
      ops.push_back(std::move(subtract_op));

      /*******************************************************************************
       *                          e^(pAfterSubtract) -> pAfterSoftmax */

      auto pAfterSoftmaxTensor = tensor_create(CUDNN_DATA_FLOAT,
                                               VIRTUAL_ID + 4,
                                               p_dim,
                                               p_stride,
                                               true,
                                               false);  // is virtual
      auto expDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_EXP);
      auto exp_op = unary_pw_op_create(
          pAfterSubtractTensor, pAfterSoftmaxTensor, expDesc);
      ops.push_back(std::move(exp_op));

      /*******************************************************************************
       *                          Dropout -> afterScaleDropout */

      auto dropoutMaskTensor = tensor_create(CUDNN_DATA_FLOAT,
                                             VIRTUAL_ID + 5,
                                             p_dim,
                                             p_stride,
                                             true,
                                             false);  // is virtual
      auto afterScaleDropoutTensor = createDropoutBackward(b,
                                                           h,
                                                           s_q,
                                                           s_kv,
                                                           d,
                                                           dropout_probability,
                                                           tensorType,
                                                           &ops,
                                                           pAfterSoftmaxTensor,
                                                           dropoutMaskTensor);

      /*******************************************************************************
       *                          afterScaleDropout -> sTransposeTensor */

      auto sTransposeTensor = tensor_create(tensorType,
                                            VIRTUAL_ID + 6,
                                            p_transpose_dim,
                                            p_transpose_stride,
                                            true,
                                            false);  // is virtual
      auto reshape_op = cudnn_frontend::OperationBuilder(
                            CUDNN_BACKEND_OPERATION_RESHAPE_DESCRIPTOR)
                            .setxDesc(afterScaleDropoutTensor)
                            .setyDesc(sTransposeTensor)
                            .build();
      ops.push_back(std::move(reshape_op));

      // Outputs of bprop
      int64_t dq_dim[4] = {b, h, s_q, d};
      int64_t dq_stride[4];
      generateMatrixStrides(
          b, h, s_q, s_kv, d, dq_stride, layout, MHA_Matrix::Q_Matrix);

      int64_t dk_dim[4] = {b, h, s_kv, d};
      int64_t dk_stride[4];
      generateMatrixStrides(
          b, h, s_q, s_kv, d, dk_stride, layout, MHA_Matrix::K_Matrix);

      int64_t dv_dim[4] = {b, h, s_kv, d};
      int64_t dv_stride[4];
      generateMatrixStrides(
          b, h, s_q, s_kv, d, dv_stride, layout, MHA_Matrix::V_Matrix);

      // Outputs of backprop
      auto dQTensor =
          tensor_create(tensorType, dQ_ID, dq_dim, dq_stride, false, false);
      auto dKTensor =
          tensor_create(tensorType, dK_ID, dk_dim, dk_stride, false, false);
      auto dVTensor =
          tensor_create(tensorType, dV_ID, dv_dim, dv_stride, false, false);
      // not virtual

      /*******************************************************************************
       *                          sTransposeTensor @ dO -> dV */

      auto matmul_1_Desc = cudnn_frontend::MatMulDescBuilder()
                               .setComputeType(CUDNN_DATA_FLOAT)
                               .setPaddingValue(0.0f)
                               .build();

      auto &&matmul_op1_builder = cudnn_frontend::OperationBuilder(
          CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR);

      matmul_op1_builder.setaMatDesc(sTransposeTensor)
          .setbMatDesc(dOTensor)
          .setcMatDesc(dVTensor)
          .setmatmulDesc(matmul_1_Desc);

      if (variable_sequence_length) {
        matmul_op1_builder.setmOverrideDesc(seqlenKTensor)
            .setkOverrideDesc(seqlenQTensor);
      }

      auto matmul_op1 = matmul_op1_builder.build();

      ops.push_back(std::move(matmul_op1));

      /*******************************************************************************
       *                          dO @ V.T -> dS */

      auto vTransposeTensor = tensor_create(
          tensorType, V_ID, v_transpose_dim, v_transpose_stride, false, false);
      auto dSTensor = tensor_create(CUDNN_DATA_FLOAT,
                                    VIRTUAL_ID + 7,
                                    p_dim,
                                    p_stride,
                                    true,
                                    false);  // is virtual

      auto matmul_2_Desc = cudnn_frontend::MatMulDescBuilder()
                               .setComputeType(CUDNN_DATA_FLOAT)
                               .setPaddingValue(0.0f)
                               .build();

      auto &&matmul_op2_builder = cudnn_frontend::OperationBuilder(
          CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR);

      matmul_op2_builder.setaMatDesc(dOTensor)
          .setbMatDesc(vTransposeTensor)
          .setcMatDesc(dSTensor)
          .setmatmulDesc(matmul_2_Desc);

      if (variable_sequence_length) {
        matmul_op2_builder.setmOverrideDesc(seqlenQTensor)
            .setnOverrideDesc(seqlenKTensor);
      }

      auto matmul_op2 = matmul_op2_builder.build();

      ops.push_back(std::move(matmul_op2));

      /*******************************************************************************
       *                          dS * dropoutMask -> dSAfterDropout */

      auto dSAfterDropoutTensor = tensor_create(CUDNN_DATA_FLOAT,
                                                VIRTUAL_ID + 8,
                                                p_dim,
                                                p_stride,
                                                true,
                                                false);  // is virtual
      auto multiply_op = binary_pw_op_create(
          dSTensor, dropoutMaskTensor, dSAfterDropoutTensor, multiplyDesc);
      ops.push_back(std::move(multiply_op));

      /*******************************************************************************
       *                          dSAfterDropout - softmaxSum -> dsAfterSubtract
       */

      auto dsAfterSubtractTensor = tensor_create(CUDNN_DATA_FLOAT,
                                                 VIRTUAL_ID + 9,
                                                 p_dim,
                                                 p_stride,
                                                 true,
                                                 false);  // is virtual
      auto subtract_op2 = binary_pw_op_create(dSAfterDropoutTensor,
                                              softmaxSumTensor,
                                              dsAfterSubtractTensor,
                                              subtractDesc);
      ops.push_back(std::move(subtract_op2));

      /*******************************************************************************
       *                          dsAfterSubtract * afterSoftmax -> dP */

      auto dPTensor = tensor_create(CUDNN_DATA_FLOAT,
                                    VIRTUAL_ID + 10,
                                    p_dim,
                                    p_stride,
                                    true,
                                    false);  // is virtual
      auto multiply_op2 = binary_pw_op_create(
          dsAfterSubtractTensor, pAfterSoftmaxTensor, dPTensor, multiplyDesc);
      ops.push_back(std::move(multiply_op2));

      /*******************************************************************************
       *                          dP * scaleDropout -> dPAfterDropoutScale */
      auto dPAfterDropoutScaleTensor = tensor_create(CUDNN_DATA_FLOAT,
                                                     VIRTUAL_ID + 11,
                                                     p_dim,
                                                     p_stride,
                                                     true,
                                                     false);  // is virtual
      auto scaleDropoutTensor = tensor_create(CUDNN_DATA_FLOAT,
                                              D_CONST_ID,
                                              scale_dim,
                                              scale_stride,
                                              false,
                                              true);  // is by value
      auto multiply_op3 = binary_pw_op_create(dPTensor,
                                              scaleDropoutTensor,
                                              dPAfterDropoutScaleTensor,
                                              multiplyDesc);
      ops.push_back(std::move(multiply_op3));

      /*******************************************************************************
       *                          dPAfterDropoutScale * bmmScale ->
       * dPScaledTensor  */

      auto dPScaledTensor = tensor_create(CUDNN_DATA_FLOAT,
                                          VIRTUAL_ID + 12,
                                          p_dim,
                                          p_stride,
                                          true,
                                          false);  // is virtual
      auto multiply_op4 = binary_pw_op_create(dPAfterDropoutScaleTensor,
                                              bmmScaleTensor,
                                              dPScaledTensor,
                                              multiplyDesc);
      ops.push_back(std::move(multiply_op4));

      /*******************************************************************************
       *                          K.T -> K */
      int64_t kDim[4] = {b, h, s_kv, d};
      int64_t kStride[4];
      generateMatrixStrides(
          b, h, s_q, s_kv, d, kStride, layout, MHA_Matrix::K_Matrix);
      auto kTensor = tensor_create(tensorType,
                                   VIRTUAL_ID + 13,
                                   kDim,
                                   kStride,
                                   true,
                                   false);  // is virtual
      auto reshape_op2 = cudnn_frontend::OperationBuilder(
                             CUDNN_BACKEND_OPERATION_RESHAPE_DESCRIPTOR)
                             .setxDesc(kTransposeTensor)
                             .setyDesc(kTensor)
                             .build();
      ops.push_back(std::move(reshape_op2));

      /*******************************************************************************
       *                          dP @ K -> dqAccumTensor / dqTensor */
      auto dqAccumTensor =
          cudnn_frontend::TensorBuilder()
              .setDim(4, dqAccum_dim)
              .setStride(4, dqAccum_stride)
              .setId(dQ_ACCUM_ID)
              .setAlignment(
                  16)  // 16B alignment is needed to run a tensor core engine
              .setDataType(CUDNN_DATA_FLOAT)
              .setVirtual(false)
              .setByValue(false)
              .setReorderType(cudnn_frontend::cudnnBackendTensorReordering_t::
                                  CUDNN_TENSOR_REORDERING_F16x16)
              .build();

      auto matmul_3_Desc = cudnn_frontend::MatMulDescBuilder()
                               .setComputeType(CUDNN_DATA_FLOAT)
                               .setPaddingValue(0.0f)
                               .build();

      auto &&matmul_op3_builder = cudnn_frontend::OperationBuilder(
          CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR);

      matmul_op3_builder.setaMatDesc(dPScaledTensor)
          .setbMatDesc(kTensor)
          .setmatmulDesc(matmul_3_Desc);

      if (use_workspace_opt) {
        matmul_op3_builder.setcMatDesc(dQTensor);
      } else {
        matmul_op3_builder.setcMatDesc(dqAccumTensor);
      }

      if (variable_sequence_length) {
        matmul_op3_builder.setmOverrideDesc(seqlenQTensor)
            .setkOverrideDesc(seqlenKTensor);
      }

      auto matmul_op3 = matmul_op3_builder.build();

      ops.push_back(std::move(matmul_op3));

      /*******************************************************************************
       *                          dP.T @ Q -> dK */
      auto dPTransposeTensor = tensor_create(CUDNN_DATA_FLOAT,
                                             VIRTUAL_ID + 14,
                                             p_transpose_dim,
                                             p_transpose_stride,
                                             true,
                                             false);  // is virtual
      auto reshape_op3 = cudnn_frontend::OperationBuilder(
                             CUDNN_BACKEND_OPERATION_RESHAPE_DESCRIPTOR)
                             .setxDesc(dPScaledTensor)
                             .setyDesc(dPTransposeTensor)
                             .build();
      ops.push_back(std::move(reshape_op3));

      auto matmul_4_Desc = cudnn_frontend::MatMulDescBuilder()
                               .setComputeType(CUDNN_DATA_FLOAT)
                               .setPaddingValue(0.0f)
                               .build();

      auto &&matmul_op4_builder = cudnn_frontend::OperationBuilder(
          CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR);

      matmul_op4_builder.setaMatDesc(dPTransposeTensor)
          .setbMatDesc(qTensor)
          .setcMatDesc(dKTensor)
          .setmatmulDesc(matmul_4_Desc);

      if (variable_sequence_length) {
        matmul_op4_builder.setmOverrideDesc(seqlenKTensor)
            .setkOverrideDesc(seqlenQTensor);
      }

      auto matmul_op4 = matmul_op4_builder.build();

      ops.push_back(std::move(matmul_op4));

      /*******************************************************************************
       *                          dqAccumTensor @ identity -> dqTensor */
      if (!use_workspace_opt) {
        auto identityDesc =
            pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_IDENTITY);
        auto identity_op =
            unary_pw_op_create(dqAccumTensor, dQTensor, identityDesc);
        ops.push_back(std::move(identity_op));
      }

      for (unsigned int i = 0; i < ops.size(); i++) {
        all_ops.push_back(&ops[i]);
      }

      // Create an Operation Graph
      auto opGraph = cudnn_frontend::OperationGraphBuilder()
                         .setHandle(handle)
                         .setOperationGraph(all_ops.size(), all_ops.data())
                         .build();

      cudnn_frontend::EngineConfigList filtered_configs;
      auto statuses =
          cudnn_frontend::get_heuristics_list<1>({"heuristics_instant"},
                                                 opGraph,
                                                 allowAllConfig,
                                                 filtered_configs,
                                                 true);

      if (filtered_configs.size() == 0) {
        cudnn_frontend::set_error_and_throw_exception(
            nullptr,
            CUDNN_STATUS_NOT_SUPPORTED,
            "run_mha_bprop: No config returned by the heuristics");
      }

      auto plan = cudnn_frontend::ExecutionPlanBuilder()
                      .setHandle(handle)
                      .setEngineConfig(filtered_configs[0], opGraph.getTag())
                      .build();

      cache.insert({descriptor, plan});
      return plan;
    };

    auto plan = get_plan(fmha_bprop_cache, descriptor);
    VLOG(10) << "Plan tag: " << plan.getTag();

    auto plan_workspace_size = plan.getWorkspaceSize();
    size_t softmaxSum_workspace_size = b * h * s_q * sizeof(float);
    size_t dqAccum_workspace_size =
        use_workspace_opt ? 0 : b * s_q * h * d * sizeof(float);
    size_t actual_seqlen_workspace_size = 2 * b * sizeof(int32_t);
    size_t workspace_size = plan_workspace_size + softmaxSum_workspace_size +
                            dqAccum_workspace_size +
                            actual_seqlen_workspace_size;
    void *workspace = nullptr;
    VLOG(10) << "Malloc workspace size: " << workspace_size << " bytes";
    if (workspace_size > 0) {
      PADDLE_ENFORCE_GPU_SUCCESS(
          cudaMallocAsync(&workspace, workspace_size, stream));
    }

    void *devPtrSoftmaxSum =
        static_cast<int8_t *>(workspace) + plan_workspace_size;
    void *devPtrdQAccumulator =
        static_cast<int8_t *>(devPtrSoftmaxSum) + softmaxSum_workspace_size;
    if (!use_workspace_opt) {
      PADDLE_ENFORCE_GPU_SUCCESS(cudaMemsetAsync(
          devPtrdQAccumulator, 0, dqAccum_workspace_size, stream));
    }

    constexpr size_t nthreads_per_block = 512;
    const size_t grid = b;
    void *devActualSeqlenQ =
        static_cast<int8_t *>(devPtrdQAccumulator) + dqAccum_workspace_size;
    void *devActualSeqlenK =
        static_cast<int8_t *>(devActualSeqlenQ) + b * sizeof(int32_t);
    mask_to_actual_seqlens_kernel<<<grid, nthreads_per_block, 0, stream>>>(
        static_cast<const int32_t *>(devPtrMask),
        static_cast<int32_t *>(devActualSeqlenQ),
        static_cast<int32_t *>(devActualSeqlenK),
        s_q,
        s_kv,
        true);
    PADDLE_ENFORCE_GPU_SUCCESS(cudaGetLastError());

    std::set<std::pair<uint64_t, void *>> data_ptrs;
    // add all the data pointers to be used in the variant pack
    float negInfinity = -1.0E+31f;
    float scale_dropout = 1.0f / (1.0f - dropout_probability);
    data_ptrs.insert(std::pair<uint64_t, void *>(dQ_ID, devPtrdQ));
    if (!use_workspace_opt) {
      data_ptrs.insert(
          std::pair<uint64_t, void *>(dQ_ACCUM_ID, devPtrdQAccumulator));
    }
    data_ptrs.insert(std::pair<uint64_t, void *>(dK_ID, devPtrdK));
    data_ptrs.insert(std::pair<uint64_t, void *>(dV_ID, devPtrdV));

    data_ptrs.insert(std::pair<uint64_t, void *>(Q_ID, devPtrQ));
    data_ptrs.insert(std::pair<uint64_t, void *>(K_ID, devPtrKTranspose));
    data_ptrs.insert(std::pair<uint64_t, void *>(V_ID, devPtrVTranspose));
    data_ptrs.insert(std::pair<uint64_t, void *>(O_ID, devPtrO));
    data_ptrs.insert(std::pair<uint64_t, void *>(dO_ID, devPtrdO));
    data_ptrs.insert(
        std::pair<uint64_t, void *>(S_STATS_ID, devPtrSoftmaxStats));
    data_ptrs.insert(std::pair<uint64_t, void *>(S_SUM_ID, devPtrSoftmaxSum));
    data_ptrs.insert(std::pair<uint64_t, void *>(D_SEED_ID, devPtrDropoutSeed));
    data_ptrs.insert(
        std::pair<uint64_t, void *>(D_OFFSET_ID, devPtrDropoutOffset));
    data_ptrs.insert(std::pair<uint64_t, void *>(MASK_VAL_ID, &negInfinity));
    if (variable_sequence_length) {
      data_ptrs.insert(
          std::pair<uint64_t, void *>(Q_SEQLEN_ID, devActualSeqlenQ));
      data_ptrs.insert(
          std::pair<uint64_t, void *>(K_SEQLEN_ID, devActualSeqlenK));
    }

    float scaleProb = 1.0f - dropout_probability;
    data_ptrs.insert(std::pair<uint64_t, void *>(D_CONST_ID, &scale_dropout));
    data_ptrs.insert(std::pair<uint64_t, void *>(S_CONST_ID, &scaling_factor));
    data_ptrs.insert(std::pair<uint64_t, void *>(SCALE_PROB, &scaleProb));

    auto variantPack = cudnn_frontend::VariantPackBuilder()
                           .setWorkspacePointer(workspace)
                           .setDataPointers(data_ptrs)
                           .build();

    CUDNN_CALL(phi::dynload::cudnnBackendExecute(
        handle, plan.get_raw_desc(), variantPack.get_raw_desc()));
    if (workspace_size > 0) {
      PADDLE_ENFORCE_GPU_SUCCESS(cudaFreeAsync(workspace, stream));
    }
  } catch (cudnn_frontend::cudnnException &e) {
    struct cudaDeviceProp prop;
    PADDLE_ENFORCE_GPU_SUCCESS(cudaGetDeviceProperties(&prop, 0));

    // cudnn flash attention is only for GA100 cards and GH100 cards
    if (!((prop.major == 8 && prop.minor == 0) ||
          (prop.major == 9 && prop.minor == 0)) &&
        (e.getCudnnStatus() == CUDNN_STATUS_ARCH_MISMATCH ||
         e.getCudnnStatus() == CUDNN_STATUS_NOT_SUPPORTED)) {
      VLOG(10) << "Only supported for GA100 (cuDNN >= 8900) and "
                  "GH100 (cuDNN >= 8900) GPUs";
    } else {
      VLOG(10) << "[ERROR] Exception " << e.what();
    }
  }
}

#endif
