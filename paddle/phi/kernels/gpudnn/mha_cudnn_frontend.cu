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

#include <map>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/platform/bfloat16.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/float16.h"
#include "paddle/phi/kernels/gpudnn/mha_cudnn_frontend.h"

#define CUDNN_FRONTEND_UNUSED(X) ((void)X)

#ifdef PADDLE_WITH_CUDNN_FRONTEND

namespace phi {
namespace cudnn_fmha {

// Used for MHA
void generateMHAStrides(int64_t b,
                        int64_t h,
                        int64_t s_q,
                        int64_t s_kv,
                        int64_t d,
                        int64_t *strideA,
                        MHA_Layout layout,
                        MHA_Matrix matrix) {
  CUDNN_FRONTEND_UNUSED(b);
  constexpr int batch_dim_idx = 0;
  constexpr int head_dim_idx = 1;
  constexpr int seqlen_dim_idx = 2;
  constexpr int hidden_dim_idx = 3;

  constexpr int seqlen_transpose_dim_idx = 3;
  constexpr int hidden_transpose_dim_idx = 2;

  constexpr int seqlen_q_dim_idx = 2;
  constexpr int seqlen_kv_dim_idx = 3;

  switch (matrix) {
    case MHA_Matrix::Q_Matrix:
      if (layout == MHA_Layout::QKV_INTERLEAVED) {
        strideA[hidden_dim_idx] = 1;
        strideA[seqlen_dim_idx] = 3 * h * d;
        strideA[head_dim_idx] = d;
        strideA[batch_dim_idx] = s_q * 3 * h * d;
      } else {
        strideA[hidden_dim_idx] = 1;
        strideA[seqlen_dim_idx] = h * d;
        strideA[head_dim_idx] = d;
        strideA[batch_dim_idx] = s_q * h * d;
      }
      break;
    case MHA_Matrix::K_Matrix:
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
      } else {
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
      } else {
        strideA[hidden_dim_idx] = 1;
        strideA[seqlen_dim_idx] = h * d;
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

#define VIRTUAL_ID 20

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

static void createScale(int64_t b,
                        int64_t h,
                        int64_t s_q,
                        int64_t s_kv,
                        int64_t d,
                        MHA_Layout layout,
                        cudnnDataType_t tensorType,
                        std::vector<cudnn_frontend::Operation> &ops  // NOLINT
) {
  // scale
  int64_t scale_dim[4] = {1, 1, 1, 1};
  int64_t scale_stride[4] = {1, 1, 1, 1};

  int64_t k_dim[4] = {b, h, d, s_kv};
  int64_t k_stride[4];
  generateMHAStrides(
      b, h, s_q, s_kv, d, k_stride, layout, MHA_Matrix::K_Matrix);

  auto scaleTensor = tensor_create(tensorType,
                                   S_CONST_ID,
                                   scale_dim,
                                   scale_stride,
                                   false,
                                   true);  // is by value
  auto kTensor = tensor_create(tensorType, K_ID, k_dim, k_stride, false, false);
  auto afterScaleKTensor = tensor_create(
      tensorType, VIRTUAL_ID, k_dim, k_stride, true, false);  // is virtual

  // Define the scale descriptor
  auto scaleDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_MUL);

  // Create a Scale Node.
  auto scale_op =
      binary_pw_op_create(kTensor, scaleTensor, afterScaleKTensor, scaleDesc);

  ops.push_back(std::move(scale_op));
}

static cudnn_frontend::Tensor createBMM1(
    int64_t b,
    int64_t h,
    int64_t s_q,
    int64_t s_kv,
    int64_t d,
    MHA_Layout layout,
    cudnnDataType_t tensorType,
    bool zero_s,
    std::vector<cudnn_frontend::Operation> &ops  // NOLINT
) {
  // Creates the necessary tensor descriptors
  int64_t q_dim[4] = {b, h, s_q, d};
  int64_t q_stride[4];
  generateMHAStrides(
      b, h, s_q, s_kv, d, q_stride, layout, MHA_Matrix::Q_Matrix);

  int64_t k_dim[4] = {b, h, d, s_kv};
  int64_t k_stride[4];
  generateMHAStrides(
      b, h, s_q, s_kv, d, k_stride, layout, MHA_Matrix::K_Matrix);

  int64_t p_dim[4] = {b, h, s_q, s_kv};
  int64_t p_stride[4];
  generateMHAStrides(
      b, h, s_q, s_kv, d, p_stride, layout, MHA_Matrix::S_Matrix);

  int64_t seqlen_dim[4] = {b, 1, 1, 1};
  int64_t seqlen_stride[4] = {1, 1, 1, 1};

  auto qTensor = tensor_create(tensorType, Q_ID, q_dim, q_stride, false, false);
  auto afterScaleKTensor = tensor_create(
      tensorType, VIRTUAL_ID, k_dim, k_stride, true, false);  // is virtual
  // first GEMM output
  auto pTensor = tensor_create(CUDNN_DATA_FLOAT,
                               VIRTUAL_ID + 1,
                               p_dim,
                               p_stride,
                               true,
                               false);  // is virtual

  auto seqlenQTensor = tensor_create(
      CUDNN_DATA_INT32, Q_SEQLEN_ID, seqlen_dim, seqlen_stride, false, false);
  auto seqlenKTensor = tensor_create(
      CUDNN_DATA_INT32, K_SEQLEN_ID, seqlen_dim, seqlen_stride, false, false);

  // Define the matmul 1 desc
  auto matmul_1_Desc = cudnn_frontend::MatMulDescBuilder()
                           .setComputeType(CUDNN_DATA_FLOAT)
                           .build();
  if (zero_s) {
    auto matmul_1_Desc = cudnn_frontend::MatMulDescBuilder()
                             .setComputeType(CUDNN_DATA_FLOAT)
                             .setPaddingValue(0.0f)
                             .build();
  }

  VLOG(10) << matmul_1_Desc.describe();

  // Create a matmul 1 Node
  auto matmul_op1 = cudnn_frontend::OperationBuilder(
                        CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR)
                        .setaMatDesc(qTensor)
                        .setbMatDesc(afterScaleKTensor)
                        .setcMatDesc(pTensor)
                        .setmOverrideDesc(seqlenQTensor)
                        .setnOverrideDesc(seqlenKTensor)
                        .setmatmulDesc(matmul_1_Desc)
                        .build();

  VLOG(10) << matmul_op1.describe();

  ops.push_back(std::move(matmul_op1));

  return pTensor;
}

static cudnn_frontend::Tensor createBias(
    int64_t b,
    int64_t h,
    int64_t s_q,
    int64_t s_kv,
    int64_t d,
    MHA_Layout layout,
    cudnnDataType_t tensorType,
    std::vector<cudnn_frontend::Operation> &ops,  // NOLINT
    cudnn_frontend::Tensor const &prevBlockOutputTensor) {
  cudnn_frontend::throw_if(ops.size() == 0,
                           "Bias op constructed incorrectly as the first one",
                           CUDNN_STATUS_BAD_PARAM);

  int64_t b_dim[4] = {1, h, s_q, s_kv};
  int64_t b_stride[4] = {h * s_q * s_kv, s_q * s_kv, s_kv, 1};

  int64_t afterBias_dim[4] = {b, h, s_q, s_kv};
  int64_t afterBias_stride[4];
  generateMHAStrides(
      b, h, s_q, s_kv, d, afterBias_stride, layout, MHA_Matrix::S_Matrix);

  // bias
  auto bTensor = tensor_create(tensorType, B_ID, b_dim, b_stride, false, false);
  // output
  auto afterBiasTensor = tensor_create(CUDNN_DATA_FLOAT,
                                       VIRTUAL_ID + 50,
                                       afterBias_dim,
                                       afterBias_stride,
                                       true,
                                       false);  // is virtual

  // Define the bias descriptor
  auto biasDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_ADD);

  // Create a Bias Node.
  auto bias_op = binary_pw_op_create(
      prevBlockOutputTensor, bTensor, afterBiasTensor, biasDesc);

  ops.push_back(std::move(bias_op));

  return afterBiasTensor;
}

static cudnn_frontend::Tensor createMask(
    int64_t b,
    int64_t h,
    int64_t s_q,
    int64_t s_kv,
    int64_t d,
    MHA_Layout layout,
    bool is_causal_masking,
    cudnnDataType_t tensorType,
    std::vector<cudnn_frontend::Operation> &ops,  // NOLINT
    cudnn_frontend::Tensor const &prevBlockOutputTensor,
    bool is_bprop) {
  CUDNN_FRONTEND_UNUSED(d);
  CUDNN_FRONTEND_UNUSED(layout);
  CUDNN_FRONTEND_UNUSED(tensorType);
  CUDNN_FRONTEND_UNUSED(is_bprop);

  cudnn_frontend::throw_if(
      ops.size() == 0,
      "Padding Mask constructed incorrectly as the first one",
      CUDNN_STATUS_BAD_PARAM);

  // subtraction output
  int64_t afterBMM1_dim[4] = {b, h, s_q, s_kv};
  int64_t afterBMM1_stride[4] = {h * s_q * s_kv, s_q * s_kv, s_kv, 1};

  int64_t seqlen_dim[4] = {b, 1, 1, 1};
  int64_t seqlen_stride[4] = {1, 1, 1, 1};

  int64_t maskVal_dim[4] = {1, 1, 1, 1};
  int64_t maskVal_stride[4] = {1, 1, 1, 1};

  // mask value to put in the masked pixels
  auto maskValTensor = tensor_create(CUDNN_DATA_FLOAT,
                                     MASK_VAL_ID,
                                     maskVal_dim,
                                     maskVal_stride,
                                     false,
                                     true);  // is by value

  auto seqlenQTensor = tensor_create(
      CUDNN_DATA_INT32, Q_SEQLEN_ID, seqlen_dim, seqlen_stride, false, false);
  auto seqlenKTensor = tensor_create(
      CUDNN_DATA_INT32, K_SEQLEN_ID, seqlen_dim, seqlen_stride, false, false);
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
  // less than row output
  auto lessThanRowTensor = tensor_create(CUDNN_DATA_BOOLEAN,
                                         VIRTUAL_ID + 102,
                                         afterBMM1_dim,
                                         afterBMM1_stride,
                                         true,
                                         false);  // is virtual
                                                  // less than column output
  auto lessThanColTensor = tensor_create(CUDNN_DATA_BOOLEAN,
                                         VIRTUAL_ID + 103,
                                         afterBMM1_dim,
                                         afterBMM1_stride,
                                         true,
                                         false);  // is virtual
  // padding mask (lessthanRow && lessthanCol)
  auto paddingMaskTensor = tensor_create(CUDNN_DATA_BOOLEAN,
                                         VIRTUAL_ID + 104,
                                         afterBMM1_dim,
                                         afterBMM1_stride,
                                         true,
                                         false);  // is virtual
  // row >= col check for causal mask
  auto rowGreaterColTensor = tensor_create(CUDNN_DATA_BOOLEAN,
                                           VIRTUAL_ID + 105,
                                           afterBMM1_dim,
                                           afterBMM1_stride,
                                           true,
                                           false);  // is virtual
  // create causal mask (padding && row >= col)
  auto causalMaskTensor = tensor_create(CUDNN_DATA_BOOLEAN,
                                        VIRTUAL_ID + 106,
                                        afterBMM1_dim,
                                        afterBMM1_stride,
                                        true,
                                        false);  // is virtual

  // output after masking
  int64_t maskOutputTensor_id = VIRTUAL_ID + 107;
  int64_t maskOutputTensor_virtual = true;
  cudnnDataType_t maskOutputTensor_dataType = CUDNN_DATA_FLOAT;
  auto maskOutputTensor_reorderType = cudnn_frontend::
      cudnnBackendTensorReordering_t::CUDNN_TENSOR_REORDERING_NONE;

  if (is_bprop) {
    maskOutputTensor_id = dS_ID;
    maskOutputTensor_virtual = false;
    maskOutputTensor_dataType = tensorType;
    maskOutputTensor_reorderType = cudnn_frontend::
        cudnnBackendTensorReordering_t::CUDNN_TENSOR_REORDERING_F16x16;
  }

  auto maskOutputTensor =
      cudnn_frontend::TensorBuilder()
          .setDim(4, afterBMM1_dim)
          .setStride(4, afterBMM1_stride)
          .setAlignment(
              16)  // 16B alignment is needed to run a tensor core engine
          .setByValue(false)
          .setDataType(maskOutputTensor_dataType)
          .setVirtual(maskOutputTensor_virtual)
          .setId(maskOutputTensor_id)
          .setReorderType(maskOutputTensor_reorderType)
          .build();

  // Define the gen index for row descriptor
  auto genIndexRowDesc = cudnn_frontend::PointWiseDescBuilder()
                             .setMode(CUDNN_POINTWISE_GEN_INDEX)
                             .setAxis(2)
                             .setComputeType(CUDNN_DATA_FLOAT)
                             .build();
  VLOG(10) << genIndexRowDesc.describe();

  // Create a gen index Node.
  auto genIndexRow_op = unary_pw_op_create(
      prevBlockOutputTensor, rowIndexTensor, genIndexRowDesc);
  VLOG(10) << genIndexRow_op.describe();

  // Define the gen index for row descriptor
  auto genIndexColumnDesc = cudnn_frontend::PointWiseDescBuilder()
                                .setMode(CUDNN_POINTWISE_GEN_INDEX)
                                .setAxis(3)
                                .setComputeType(CUDNN_DATA_FLOAT)
                                .build();
  VLOG(10) << genIndexColumnDesc.describe();

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

  // Define the greater than equal to comparison descriptor
  auto rowGreaterColDesc =
      pw_desc_create(CUDNN_DATA_BOOLEAN, CUDNN_POINTWISE_CMP_GE);

  // Create a greater than equal to Node.
  auto rowGreaterCol_op = binary_pw_op_create(rowIndexTensor,
                                              columnIndexTensor,
                                              rowGreaterColTensor,
                                              rowGreaterColDesc);

  // Define the and to create causal mask descriptor
  auto causalMaskAndDesc =
      pw_desc_create(CUDNN_DATA_BOOLEAN, CUDNN_POINTWISE_LOGICAL_AND);

  // Create a causal Mask Node.
  auto causalMaskAnd_op = binary_pw_op_create(paddingMaskTensor,
                                              rowGreaterColTensor,
                                              causalMaskTensor,
                                              causalMaskAndDesc);

  /////////////////// Apply the mask //////////////////////////

  auto maskTensor = (is_causal_masking) ? std::move(causalMaskTensor)
                                        : std::move(paddingMaskTensor);

  // Define the binary select to perform masking descriptor
  auto maskDesc =
      pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_BINARY_SELECT);

  // Create a binary select Node.
  auto mask_op = ternary_pw_op_create(prevBlockOutputTensor,
                                      maskValTensor,
                                      maskTensor,
                                      maskOutputTensor,
                                      maskDesc);

  ops.push_back(std::move(genIndexRow_op));
  ops.push_back(std::move(genIndexColumn_op));
  ops.push_back(std::move(lessThanRow_op));
  ops.push_back(std::move(lessThanCol_op));
  ops.push_back(std::move(paddingMaskAnd_op));
  if (is_causal_masking) ops.push_back(std::move(rowGreaterCol_op));
  if (is_causal_masking) ops.push_back(std::move(causalMaskAnd_op));
  ops.push_back(std::move(mask_op));

  return maskOutputTensor;
}

static cudnn_frontend::Tensor createSoftmaxForward(
    int64_t b,
    int64_t h,
    int64_t s_q,
    int64_t s_kv,
    int64_t d,
    MHA_Layout layout,
    bool enable_dropout,
    bool softmax_output_virtual,
    cudnnDataType_t tensorType,
    std::vector<cudnn_frontend::Operation> &ops,  // NOLINT
    cudnn_frontend::Tensor const &prevBlockOutputTensor) {
  CUDNN_FRONTEND_UNUSED(d);
  CUDNN_FRONTEND_UNUSED(layout);

  int64_t afterBMM1_dim[4] = {b, h, s_q, s_kv};
  int64_t afterBMM1_stride[4] = {h * s_q * s_kv, s_q * s_kv, s_kv, 1};

  int64_t afterReduction_dim[4] = {b, h, s_q, 1};
  int64_t afterReduction_stride[4] = {h * s_q, s_q, 1, 1};

  cudnnDataType_t softmaxOutputType = (enable_dropout || softmax_output_virtual)
                                          ? CUDNN_DATA_FLOAT
                                          : tensorType;
  uint64_t softmaxOutputName = softmax_output_virtual ? VIRTUAL_ID + 154 : S_ID;

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
  // divide (e/ sum(e))

  auto reorder_type = cudnn_frontend::cudnnBackendTensorReordering_t::
      CUDNN_TENSOR_REORDERING_F16x16;

  auto afterDivisionTensor =
      cudnn_frontend::TensorBuilder()
          .setDim(4, afterBMM1_dim)
          .setStride(4, afterBMM1_stride)
          .setId(softmaxOutputName)
          .setAlignment(
              16)  // 16B alignment is needed to run a tensor core engine
          .setDataType(softmaxOutputType)
          .setVirtual(softmax_output_virtual)
          .setByValue(false)
          .setReorderType(reorder_type)
          .build();

  // Define the reduction descriptor
  auto reductionMaxDesc = cudnn_frontend::ReductionDescBuilder()
                              .setComputeType(CUDNN_DATA_FLOAT)
                              .setReductionOp(CUDNN_REDUCE_TENSOR_MAX)
                              .build();
  VLOG(10) << reductionMaxDesc.describe();

  // Create a reduction max Node.
  auto reductionMax_op = cudnn_frontend::OperationBuilder(
                             CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR)
                             .setxDesc(prevBlockOutputTensor)
                             .setyDesc(afterMaxReductionTensor)
                             .setreductionDesc(reductionMaxDesc)
                             .build();
  VLOG(10) << reductionMax_op.describe();

  // Define the subtract descriptor
  auto subtractDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_SUB);

  // Create a subtract Node.
  auto subtract_op = binary_pw_op_create(prevBlockOutputTensor,
                                         afterMaxReductionTensor,
                                         afterSubtractionTensor,
                                         subtractDesc);

  // Define the exponent descriptor
  auto exponentDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_EXP);

  // Create a exponent Node.
  auto exponent_op = unary_pw_op_create(
      afterSubtractionTensor, afterExponentTensor, exponentDesc);

  // Define the reduction descriptor
  auto reductionAddDesc = cudnn_frontend::ReductionDescBuilder()
                              .setComputeType(CUDNN_DATA_FLOAT)
                              .setReductionOp(CUDNN_REDUCE_TENSOR_ADD)
                              .build();
  VLOG(10) << reductionAddDesc.describe();

  // Create a reduction add Node.
  auto reductionAdd_op = cudnn_frontend::OperationBuilder(
                             CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR)
                             .setxDesc(afterExponentTensor)
                             .setyDesc(afterAddReductionTensor)
                             .setreductionDesc(reductionAddDesc)
                             .build();

  VLOG(10) << reductionAdd_op.describe();

  // Define the division descriptor
  auto divisionDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_DIV);

  // Create a subtract Node.
  auto division_op = binary_pw_op_create(afterExponentTensor,
                                         afterAddReductionTensor,
                                         afterDivisionTensor,
                                         divisionDesc);

  ops.push_back(std::move(reductionMax_op));
  ops.push_back(std::move(subtract_op));
  ops.push_back(std::move(exponent_op));
  ops.push_back(std::move(reductionAdd_op));
  ops.push_back(std::move(division_op));

  return afterDivisionTensor;
}

static cudnn_frontend::Tensor createDropout(
    int64_t b,
    int64_t h,
    int64_t s_q,
    int64_t s_kv,
    int64_t d,
    int64_t seed,
    double probability,
    cudnnDataType_t tensorType,
    // NOLINTNEXTLINE(runtime/references)
    std::vector<cudnn_frontend::Operation> &ops,
    cudnn_frontend::Tensor const &prevBlockOutputTensor) {
  CUDNN_FRONTEND_UNUSED(d);

  cudnn_frontend::throw_if(
      ops.size() == 0,
      "Dropout DAG constructed incorrectly as the first one",
      CUDNN_STATUS_BAD_PARAM);

  int64_t afterBMM1_dim[4] = {b, h, s_q, s_kv};
  int64_t afterBMM1_stride[4] = {h * s_q * s_kv, s_q * s_kv, s_kv, 1};

  int64_t scale_dim[4] = {1, 1, 1, 1};
  int64_t scale_stride[4] = {1, 1, 1, 1};

  // mask for the dropout
  auto dropoutMaskTensor = tensor_create(CUDNN_DATA_FLOAT,
                                         VIRTUAL_ID + 200,
                                         afterBMM1_dim,
                                         afterBMM1_stride,
                                         true,
                                         false);  // is virtual

  auto reorder_type = cudnn_frontend::cudnnBackendTensorReordering_t::
      CUDNN_TENSOR_REORDERING_F16x16;

  // after dropout tensor
  auto afterDropoutTensor =
      cudnn_frontend::TensorBuilder()
          .setDim(4, afterBMM1_dim)
          .setStride(4, afterBMM1_stride)
          .setId(S_ID)
          .setAlignment(
              16)  // 16B alignment is needed to run a tensor core engine
          .setDataType(tensorType)
          .setVirtual(false)
          .setByValue(false)
          .setReorderType(reorder_type)
          .build();
  // scale after dropout
  auto scaleDropoutTensor = tensor_create(tensorType,
                                          D_CONST_ID,
                                          scale_dim,
                                          scale_stride,
                                          false,
                                          true);  // is by value
  // after Scale
  auto afterScaleTensor = tensor_create(tensorType,
                                        VIRTUAL_ID + 201,
                                        afterBMM1_dim,
                                        afterBMM1_stride,
                                        true,
                                        false);  // is virtual

  // Define the reduction descriptor
  auto rngDesc = cudnn_frontend::RngDescBuilder()
                     .setRngDistribution(CUDNN_RNG_DISTRIBUTION_BERNOULLI)
                     .setBernoulliDistProbability(1.0 - probability)
                     .build();
  VLOG(10) << rngDesc.describe();

  // Create a rng Node.
  auto rng_op =
      cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_RNG_DESCRIPTOR)
          .setyDesc(dropoutMaskTensor)
          .setSeed(seed)
          .setRngDesc(rngDesc)
          .build();

  VLOG(10) << rng_op.describe();

  // Define the multiply mask descriptor
  auto maskMulDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_MUL);

  // Create a multiply mask Node.
  auto maskMul_op = binary_pw_op_create(prevBlockOutputTensor,
                                        dropoutMaskTensor,
                                        afterDropoutTensor,
                                        maskMulDesc);

  // Define the multiply scale descriptor
  auto scaleMulDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_MUL);

  // Create a multiply mask Node.
  auto scaleMul_op = binary_pw_op_create(
      afterDropoutTensor, scaleDropoutTensor, afterScaleTensor, scaleMulDesc);

  ops.push_back(std::move(rng_op));
  ops.push_back(std::move(maskMul_op));
  ops.push_back(std::move(scaleMul_op));

  return afterScaleTensor;
}

static void createBMM2(int64_t b,
                       int64_t h,
                       int64_t s_q,
                       int64_t s_kv,
                       int64_t d,
                       MHA_Layout layout,
                       cudnnDataType_t tensorType,
                       std::vector<cudnn_frontend::Operation> &ops,  // NOLINT
                       cudnn_frontend::Tensor const &prevBlockOutputTensor) {
  cudnn_frontend::throw_if(ops.size() == 0,
                           "BMM2 op constructed incorrectly as the first one",
                           CUDNN_STATUS_BAD_PARAM);

  int64_t seqlen_dim[4] = {b, 1, 1, 1};
  int64_t seqlen_stride[4] = {1, 1, 1, 1};

  int64_t v_dim[4] = {b, h, s_kv, d};
  int64_t v_stride[4];
  generateMHAStrides(
      b, h, s_q, s_kv, d, v_stride, layout, MHA_Matrix::V_Matrix);

  int64_t o_dim[4] = {b, h, s_q, d};
  int64_t o_stride[4];
  generateMHAStrides(
      b, h, s_q, s_kv, d, o_stride, layout, MHA_Matrix::O_Matrix);

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
  VLOG(10) << matmul_2_Desc.describe();

  // Create a matmul 2 Node
  auto matmul_op2 = cudnn_frontend::OperationBuilder(
                        CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR)
                        .setaMatDesc(prevBlockOutputTensor)
                        .setbMatDesc(vTensor)
                        .setcMatDesc(oTensor)
                        .setmOverrideDesc(seqlenQTensor)
                        .setkOverrideDesc(seqlenKTensor)
                        .setmatmulDesc(matmul_2_Desc)
                        .build();

  VLOG(10) << matmul_op2.describe();

  ops.push_back(std::move(matmul_op2));
}

static cudnn_frontend::Tensor createSoftmaxBackward(
    int64_t b,
    int64_t h,
    int64_t s_q,
    int64_t s_kv,
    int64_t d,
    MHA_Layout layout,
    cudnnDataType_t tensorType,
    // NOLINTNEXTLINE(runtime/references)
    std::vector<cudnn_frontend::Operation> &ops,
    cudnn_frontend::Tensor const &yTensor,
    cudnn_frontend::Tensor const &dyTensor) {
  CUDNN_FRONTEND_UNUSED(tensorType);

  cudnn_frontend::throw_if(
      ops.size() == 0,
      "Softmax backward constructed incorrectly as the first one",
      CUDNN_STATUS_BAD_PARAM);

  int64_t p_dim[4] = {b, h, s_q, s_kv};
  int64_t p_stride[4];
  generateMHAStrides(
      b, h, s_q, s_kv, d, p_stride, layout, MHA_Matrix::S_Matrix);

  int64_t p_reduction_dim[4] = {b, h, s_q, 1};
  int64_t p_reduction_stride[4];

  p_reduction_stride[3] = 1;
  p_reduction_stride[2] = 1;
  p_reduction_stride[1] = s_q;
  p_reduction_stride[0] = s_q * h;

  int64_t const_dim[4] = {1, 1, 1, 1};
  int64_t const_stride[4] = {1, 1, 1, 1};

  // creating all tensors
  auto softmaxScaleTensor = tensor_create(
      CUDNN_DATA_FLOAT, S_CONST_ID, const_dim, const_stride, false, true);
  auto dyMulYTensor = tensor_create(
      CUDNN_DATA_FLOAT, VIRTUAL_ID + 250, p_dim, p_stride, true, false);
  auto dxAfterReductionTensor = tensor_create(CUDNN_DATA_FLOAT,
                                              VIRTUAL_ID + 251,
                                              p_reduction_dim,
                                              p_reduction_stride,
                                              true,
                                              false);
  auto dxAfterSubtractionTensor = tensor_create(
      CUDNN_DATA_FLOAT, VIRTUAL_ID + 252, p_dim, p_stride, true, false);
  auto dxUnscaleTensor = tensor_create(
      CUDNN_DATA_FLOAT, VIRTUAL_ID + 253, p_dim, p_stride, true, false);
  auto dxTensor = tensor_create(
      CUDNN_DATA_FLOAT, VIRTUAL_ID + 254, p_dim, p_stride, true, false);

  // creating all ops
  // mul (y * dy)
  auto mul_1_desc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_MUL);
  auto mul_1_op =
      binary_pw_op_create(yTensor, dyTensor, dyMulYTensor, mul_1_desc);

  // reduction add sum (y * dy)
  auto reductionAddDesc = cudnn_frontend::ReductionDescBuilder()
                              .setComputeType(CUDNN_DATA_FLOAT)
                              .setReductionOp(CUDNN_REDUCE_TENSOR_ADD)
                              .build();
  VLOG(10) << reductionAddDesc.describe();

  auto reductionAdd_op = cudnn_frontend::OperationBuilder(
                             CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR)
                             .setxDesc(dyMulYTensor)
                             .setyDesc(dxAfterReductionTensor)
                             .setreductionDesc(reductionAddDesc)
                             .build();

  VLOG(10) << reductionAdd_op.describe();

  // subtraction (dy - sum(y * dy))
  auto sub_0_desc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_SUB);
  auto sub_0_op = binary_pw_op_create(
      dyTensor, dxAfterReductionTensor, dxAfterSubtractionTensor, sub_0_desc);

  // mul (y * (dy - sum(y * dy)))
  auto mul_2_desc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_MUL);
  auto mul_2_op = binary_pw_op_create(
      yTensor, dxAfterSubtractionTensor, dxUnscaleTensor, mul_2_desc);

  // mul (scale * dx)
  auto mul_3_desc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_MUL);
  auto mul_3_op = binary_pw_op_create(
      dxUnscaleTensor, softmaxScaleTensor, dxTensor, mul_3_desc);

  ops.push_back(std::move(mul_1_op));
  ops.push_back(std::move(reductionAdd_op));
  ops.push_back(std::move(sub_0_op));
  ops.push_back(std::move(mul_2_op));
  ops.push_back(std::move(mul_3_op));

  return dxTensor;
}

struct FMHADescriptor {
  std::int64_t b;
  std::int64_t h;
  std::int64_t s_q;
  std::int64_t s_kv;
  std::int64_t d;
  std::int64_t seed;
  float scaling_factor;
  float dropout_probability;
  bool is_causal_masking;
  MHA_Layout layout;
  MHA_Bias_Type bias_type;
  cudnnDataType_t tensor_type;

  bool operator<(const FMHADescriptor &rhs) const {
    return std::tie(b,
                    h,
                    s_q,
                    s_kv,
                    d,
                    seed,
                    scaling_factor,
                    dropout_probability,
                    is_causal_masking,
                    layout,
                    bias_type,
                    tensor_type) < std::tie(rhs.b,
                                            rhs.h,
                                            rhs.s_q,
                                            rhs.s_kv,
                                            rhs.d,
                                            rhs.seed,
                                            rhs.scaling_factor,
                                            rhs.dropout_probability,
                                            rhs.is_causal_masking,
                                            rhs.layout,
                                            rhs.bias_type,
                                            rhs.tensor_type);
  }
};
}  // namespace cudnn_fmha
}  // namespace phi

using namespace phi::cudnn_fmha;  // NOLINT

void run_cudnn_fmha_fwd(int64_t b,
                        int64_t h,
                        int64_t s_q,
                        int64_t s_kv,
                        int64_t d,
                        int64_t seed,
                        MHA_Layout layout,
                        float scaling_factor,
                        double dropout_probability,
                        MHA_Bias_Type bias_type,
                        bool is_causal_masking,
                        void *devPtrQ,
                        void *devPtrK,
                        void *devPtrV,
                        void *devPtrS,
                        void *devPtrO,
                        void *devPtrBias,
                        void *devActualSeqlenQ,
                        void *devActualSeqlenKV,
                        cudnnDataType_t tensorType,
                        cudnnHandle_t handle_) {
  // cudnnHandle_t handle_;
  try {
    // Create cudnn handle
    // checkCudnnErr(cudnnCreate(&handle_));

    FMHADescriptor descriptor{b,
                              h,
                              s_q,
                              s_kv,
                              d,
                              seed,
                              scaling_factor,
                              static_cast<float>(dropout_probability),
                              is_causal_masking,
                              layout,
                              bias_type,
                              tensorType};

    using CacheType = std::map<FMHADescriptor, cudnn_frontend::ExecutionPlan>;
    static CacheType fmha_fprop_cache;

    bool enable_dropout = (dropout_probability != 0.0f);

    // Get plan from cache if cache is available, otherwise create one
    auto get_plan = [&](CacheType &cache, const FMHADescriptor &descriptor) {
      // if hit, return
      auto it = cache.find(descriptor);
      if (it != cache.end()) {
        auto plan = it->second;
        return plan;
      }

      // otherwise, build the op_graph and the plan. Then update cache
      std::vector<cudnn_frontend::Operation const *> all_ops;
      std::vector<cudnn_frontend::Operation> ops;

      createScale(b, h, s_q, s_kv, d, layout, tensorType, ops);

      auto zero_s = (bias_type != MHA_Bias_Type::NO_BIAS);
      auto bmm1_output =
          createBMM1(b, h, s_q, s_kv, d, layout, tensorType, zero_s, ops);

      if (bias_type != MHA_Bias_Type::NO_BIAS) {
        createBias(b, h, s_q, s_kv, d, layout, tensorType, ops, bmm1_output);
      }

      auto mask_output = createMask(b,
                                    h,
                                    s_q,
                                    s_kv,
                                    d,
                                    layout,
                                    is_causal_masking,
                                    tensorType,
                                    ops,
                                    bmm1_output,
                                    false);

      cudnn_frontend::throw_if(dropout_probability == 1.0f,
                               "Dropout probability cannot be 1.0",
                               CUDNN_STATUS_BAD_PARAM);

      bool softmax_output_virtual = enable_dropout || devPtrS == nullptr;
      auto softmax_output = createSoftmaxForward(b,
                                                 h,
                                                 s_q,
                                                 s_kv,
                                                 d,
                                                 layout,
                                                 enable_dropout,
                                                 softmax_output_virtual,
                                                 tensorType,
                                                 ops,
                                                 mask_output);

      if (dropout_probability != 0.0f) {
        auto dropout_output = createDropout(b,
                                            h,
                                            s_q,
                                            s_kv,
                                            d,
                                            seed,
                                            dropout_probability,
                                            tensorType,
                                            ops,
                                            softmax_output);
        createBMM2(b, h, s_q, s_kv, d, layout, tensorType, ops, dropout_output);
      } else {
        createBMM2(b, h, s_q, s_kv, d, layout, tensorType, ops, softmax_output);
      }

      for (unsigned int i = 0; i < ops.size(); i++) {
        all_ops.push_back(&ops[i]);
      }
      VLOG(10) << "Total ops created: " << ops.size();

      // Create an Operation Graph
      auto opGraph = cudnn_frontend::OperationGraphBuilder()
                         .setHandle(handle_)
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
                      .setHandle(handle_)
                      .setEngineConfig(filtered_configs[0], opGraph.getTag())
                      .build();
      cache.insert({descriptor, plan});
      return plan;
    };

    auto plan = get_plan(fmha_fprop_cache, descriptor);
    VLOG(10) << "Plan tag: " << plan.getTag();

    auto workspace_size = plan.getWorkspaceSize();
    VLOG(10) << plan.describe() << " requires workspace " << workspace_size;

    void *workspace_ptr = nullptr;
    if (workspace_size > 0) {
      PADDLE_ENFORCE_GPU_SUCCESS(cudaMalloc(&workspace_ptr, workspace_size));
    }

    std::set<std::pair<uint64_t, void *>> data_ptrs;
    // change this if you have access to float_min
    float negInfinity = -1.0E+20;
    auto scale_dropout = paddle::platform::float16(
                             static_cast<float>(1 / (1 - dropout_probability)))
                             .x;

    // add all the data pointers to be used in the variant pack
    data_ptrs.insert(std::pair<uint64_t, void *>(Q_ID, devPtrQ));
    data_ptrs.insert(std::pair<uint64_t, void *>(K_ID, devPtrK));
    data_ptrs.insert(std::pair<uint64_t, void *>(V_ID, devPtrV));
    data_ptrs.insert(
        std::pair<uint64_t, void *>(Q_SEQLEN_ID, devActualSeqlenQ));
    data_ptrs.insert(
        std::pair<uint64_t, void *>(K_SEQLEN_ID, devActualSeqlenKV));
    data_ptrs.insert(std::pair<uint64_t, void *>(MASK_VAL_ID, &negInfinity));

    if (tensorType == CUDNN_DATA_FLOAT) {
      data_ptrs.insert(
          std::pair<uint64_t, void *>(S_CONST_ID, &scaling_factor));
    } else if (tensorType == CUDNN_DATA_HALF) {
      auto scaling_factor_fp16 = paddle::platform::float16(scaling_factor).x;
      data_ptrs.insert(
          std::pair<uint64_t, void *>(S_CONST_ID, &scaling_factor_fp16));
    } else if (tensorType == CUDNN_DATA_BFLOAT16) {
      auto scaling_factor_bf16 = paddle::platform::bfloat16(scaling_factor).x;
      data_ptrs.insert(
          std::pair<uint64_t, void *>(S_CONST_ID, &scaling_factor_bf16));
    } else {
      std::cerr << "Not supported tensorType." << std::endl;
    }

    data_ptrs.insert(std::pair<uint64_t, void *>(O_ID, devPtrO));

    if (bias_type != MHA_Bias_Type::NO_BIAS) {
      data_ptrs.insert(std::pair<uint64_t, void *>(B_ID, devPtrBias));
    }

    if (devPtrS != nullptr) {
      data_ptrs.insert(std::pair<uint64_t, void *>(S_ID, devPtrS));
    }

    if (enable_dropout) {
      data_ptrs.insert(std::pair<uint64_t, void *>(D_CONST_ID, &scale_dropout));
    }

    auto variantPack = cudnn_frontend::VariantPackBuilder()
                           .setWorkspacePointer(workspace_ptr)
                           .setDataPointers(data_ptrs)
                           .build();
    VLOG(10) << "variantPack " << variantPack.describe();
    cudnnStatus_t status = phi::dynload::cudnnBackendExecute(
        handle_, plan.get_raw_desc(), variantPack.get_raw_desc());
    if (workspace_size > 0) {
      PADDLE_ENFORCE_GPU_SUCCESS(cudaFree(workspace_ptr));
    }

    // checkCudnnErr(cudnnDestroy(handle_));

    cudnn_frontend::throw_if(
        [status]() { return (status != CUDNN_STATUS_SUCCESS); },
        "Plan execute error",
        status);
  } catch (cudnn_frontend::cudnnException &e) {
    struct cudaDeviceProp prop;
    PADDLE_ENFORCE_GPU_SUCCESS(cudaGetDeviceProperties(&prop, 0));

    // this example is only for GA100 cards (cudnn Version >= 8700) and GH100
    // cards (cudnn Version >= 8800)
    if (!((prop.major == 8 && prop.minor == 0) ||
          (prop.major == 9 && prop.minor == 0 && CUDNN_VERSION >= 8800)) &&
        (e.getCudnnStatus() == CUDNN_STATUS_ARCH_MISMATCH ||
         e.getCudnnStatus() == CUDNN_STATUS_NOT_SUPPORTED)) {
      std::cout << "Only supported for GA100 (cuDNN >= 8700) and "
                   "GH100 (cuDNN >= 8800) GPUs"
                << std::endl;
    } else {
      std::cout << "[ERROR] Exception " << e.what() << std::endl;
      // CHECK(false);
    }
  }
}

void run_cudnn_fmha_bwd(int64_t b,
                        int64_t h,
                        int64_t s_q,
                        int64_t s_kv,
                        int64_t d,
                        MHA_Layout layout,
                        float scaling_factor,
                        float dropout_probability,
                        bool is_causal_masking,
                        void *devPtrQ,
                        void *devPtrK,
                        void *devPtrV,
                        void *devPtrS,
                        void *devPtrdQ,
                        void *devPtrdK,
                        void *devPtrdV,
                        void *devPtrdO,
                        void *devPtrdS,
                        void *devActualSeqlenQ,
                        void *devActualSeqlenKV,
                        cudnnDataType_t tensorType,
                        cudnnHandle_t handle_) {
  // cudnnHandle_t handle_;
  try {
    // Create cudnn handle
    // checkCudnnErr(cudnnCreate(&handle_));

    FMHADescriptor descriptor{b,
                              h,
                              s_q,
                              s_kv,
                              d,
                              0,
                              scaling_factor,
                              static_cast<float>(dropout_probability),
                              is_causal_masking,
                              layout,
                              MHA_Bias_Type::NO_BIAS,
                              tensorType};

    using CacheType = std::map<FMHADescriptor, cudnn_frontend::ExecutionPlan>;
    static CacheType fmha_bprop_cache;

    auto get_plan = [&](CacheType &cache, const FMHADescriptor &descriptor) {
      auto it = cache.find(descriptor);
      if (it != cache.end()) {
        return it->second;
      }

      std::vector<cudnn_frontend::Operation const *> all_ops;
      std::vector<cudnn_frontend::Operation> ops;

      // Creates the necessary tensor descriptors
      int64_t q_dim[4] = {b, h, s_q, d};
      int64_t q_stride[4];
      generateMHAStrides(
          b, h, s_q, s_kv, d, q_stride, layout, MHA_Matrix::Q_Matrix);

      int64_t k_dim[4] = {b, h, s_kv, d};
      int64_t k_stride[4];
      generateMHAStrides(
          b,
          h,
          s_q,
          s_kv,
          d,
          k_stride,
          layout,
          MHA_Matrix::V_Matrix);  // type is correct as K is not transposed

      int64_t v_dim[4] = {b, h, d, s_kv};
      int64_t v_stride[4];
      generateMHAStrides(
          b,
          h,
          s_q,
          s_kv,
          d,
          v_stride,
          layout,
          MHA_Matrix::K_Matrix);  // type is correct as V is transposed

      int64_t p_dim[4] = {b, h, s_q, s_kv};
      int64_t p_stride[4];
      generateMHAStrides(
          b, h, s_q, s_kv, d, p_stride, layout, MHA_Matrix::S_Matrix);

      int64_t p_transpose_dim[4] = {b, h, s_kv, s_q};
      int64_t p_transpose_stride[4];
      p_transpose_stride[0] = p_stride[0];
      p_transpose_stride[1] = p_stride[1];
      p_transpose_stride[2] = p_stride[3];
      p_transpose_stride[3] = p_stride[2];

      int64_t o_dim[4] = {b, h, s_q, d};
      int64_t o_stride[4];
      generateMHAStrides(
          b, h, s_q, s_kv, d, o_stride, layout, MHA_Matrix::O_Matrix);

      int64_t seqlen_dim[4] = {b, 1, 1, 1};
      int64_t seqlen_stride[4] = {1, 1, 1, 1};

      int64_t scale_dim[4] = {1, 1, 1, 1};
      int64_t scale_stride[4] = {1, 1, 1, 1};

      // inputs to fprop
      auto qTensor =
          tensor_create(tensorType, Q_ID, q_dim, q_stride, false, false);
      auto kTensor =
          tensor_create(tensorType, K_ID, k_dim, k_stride, false, false);
      auto vTensor =
          tensor_create(tensorType, V_ID, v_dim, v_stride, false, false);
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

      // gradient of the output
      auto doTensor =
          tensor_create(tensorType, dO_ID, o_dim, o_stride, false, false);

      auto reorder_type = cudnn_frontend::cudnnBackendTensorReordering_t::
          CUDNN_TENSOR_REORDERING_F16x16;

      // activation from fprop
      auto pTensor =
          cudnn_frontend::TensorBuilder()
              .setDim(4, p_dim)
              .setStride(4, p_stride)
              .setId(S_ID)
              .setAlignment(
                  16)  // 16B alignment is needed to run a tensor core engine
              .setDataType(tensorType)
              .setVirtual(false)
              .setByValue(false)
              .setReorderType(reorder_type)
              .build();

      // outputs from bprop
      auto dqTensor =
          tensor_create(tensorType, dQ_ID, q_dim, q_stride, false, false);
      auto dkTensor =
          tensor_create(tensorType, dK_ID, k_dim, k_stride, false, false);
      auto dvTensor =
          tensor_create(tensorType,
                        dV_ID,
                        k_dim,
                        k_stride,
                        false,
                        false);  // not transposed therefore k_dim and k_stride

      ////////////////////////////////////////////////////////
      // start creating the ops and the intermediate tensors
      auto pReshapeTensor = tensor_create(tensorType,
                                          VIRTUAL_ID + 300,
                                          p_transpose_dim,
                                          p_transpose_stride,
                                          true,
                                          false);

      // reshape to perform transpose and make pReshape
      auto reshape_op = cudnn_frontend::OperationBuilder(
                            CUDNN_BACKEND_OPERATION_RESHAPE_DESCRIPTOR)
                            .setxDesc(pTensor)
                            .setyDesc(pReshapeTensor)
                            .build();

      VLOG(10) << reshape_op.describe();
      ops.push_back(std::move(reshape_op));

      // scale dropout
      auto dropoutScaleTensor = tensor_create(CUDNN_DATA_FLOAT,
                                              D_CONST_ID,
                                              scale_dim,
                                              scale_stride,
                                              false,
                                              true);  // is by value
      auto pAfterScaleTensor = tensor_create(tensorType,
                                             VIRTUAL_ID + 301,
                                             p_transpose_dim,
                                             p_transpose_stride,
                                             true,
                                             false);

      auto scaleMulDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_MUL);
      auto scaleMul_op = binary_pw_op_create(
          pReshapeTensor, dropoutScaleTensor, pAfterScaleTensor, scaleMulDesc);
      ops.push_back(std::move(scaleMul_op));

      // perform absolute operation to remove the mask bit
      auto pTransposeAfterAbsTensor = tensor_create(tensorType,
                                                    VIRTUAL_ID + 302,
                                                    p_transpose_dim,
                                                    p_transpose_stride,
                                                    true,
                                                    false);

      auto absDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_ABS);
      auto abs_op = unary_pw_op_create(
          pAfterScaleTensor, pTransposeAfterAbsTensor, absDesc);
      ops.push_back(std::move(abs_op));

      // matmul to calculate dvTensor
      auto matmul_0_Desc = cudnn_frontend::MatMulDescBuilder()
                               .setComputeType(CUDNN_DATA_FLOAT)
                               .setPaddingValue(0.0f)
                               .build();
      VLOG(10) << matmul_0_Desc.describe();

      auto matmul_op0 = cudnn_frontend::OperationBuilder(
                            CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR)
                            .setaMatDesc(pTransposeAfterAbsTensor)
                            .setbMatDesc(doTensor)
                            .setcMatDesc(dvTensor)
                            .setmOverrideDesc(seqlenKTensor)
                            .setkOverrideDesc(seqlenQTensor)
                            .setmatmulDesc(matmul_0_Desc)
                            .build();

      VLOG(10) << matmul_op0.describe();

      ops.push_back(std::move(matmul_op0));

      // matmul to calculate dpTensor
      auto dpTensor = tensor_create(
          CUDNN_DATA_FLOAT, VIRTUAL_ID + 303, p_dim, p_stride, true, false);

      auto matmul_1_Desc = cudnn_frontend::MatMulDescBuilder()
                               .setComputeType(CUDNN_DATA_FLOAT)
                               .build();
      VLOG(10) << matmul_1_Desc.describe();

      auto matmul_op1 = cudnn_frontend::OperationBuilder(
                            CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR)
                            .setaMatDesc(doTensor)
                            .setbMatDesc(vTensor)
                            .setcMatDesc(dpTensor)
                            .setmOverrideDesc(seqlenQTensor)
                            .setnOverrideDesc(seqlenKTensor)
                            .setmatmulDesc(matmul_1_Desc)
                            .build();

      VLOG(10) << matmul_op1.describe();

      ops.push_back(std::move(matmul_op1));

      // mask the values which were dropped in dropout
      auto pAbsTensor = tensor_create(
          tensorType, VIRTUAL_ID + 304, p_dim, p_stride, true, false);

      auto p_absDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_ABS);
      auto p_abs_op = unary_pw_op_create(pTensor, pAbsTensor, p_absDesc);
      ops.push_back(std::move(p_abs_op));

      // create the dropout mask
      auto zeroTensor = tensor_create(CUDNN_DATA_FLOAT,
                                      MASK_VAL_ID,
                                      scale_dim,
                                      scale_stride,
                                      false,
                                      true);  // is by value
      auto dropoutMaskTensor = tensor_create(
          CUDNN_DATA_BOOLEAN, VIRTUAL_ID + 305, p_dim, p_stride, true, false);

      auto greater_than_0_desc =
          pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_CMP_GT);
      auto greater_than_0_op = binary_pw_op_create(
          pTensor, zeroTensor, dropoutMaskTensor, greater_than_0_desc);
      ops.push_back(std::move(greater_than_0_op));

      // scale for the dropout
      auto dpAfterScaleTensor = tensor_create(
          CUDNN_DATA_FLOAT, VIRTUAL_ID + 306, p_dim, p_stride, true, false);

      auto mul_0_desc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_MUL);
      auto mul_0_op = binary_pw_op_create(
          dpTensor, dropoutScaleTensor, dpAfterScaleTensor, mul_0_desc);
      ops.push_back(std::move(mul_0_op));

      // drop the values based on the dropout mask
      auto dpAfterDropoutTensor = tensor_create(
          CUDNN_DATA_FLOAT, VIRTUAL_ID + 307, p_dim, p_stride, true, false);

      auto selection_0_desc =
          pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_BINARY_SELECT);
      auto selection_0_op = ternary_pw_op_create(dpAfterScaleTensor,
                                                 zeroTensor,
                                                 dropoutMaskTensor,
                                                 dpAfterDropoutTensor,
                                                 selection_0_desc);
      ops.push_back(std::move(selection_0_op));

      // softmax backward
      auto dsTensor = createSoftmaxBackward(b,
                                            h,
                                            s_q,
                                            s_kv,
                                            d,
                                            layout,
                                            tensorType,
                                            ops,
                                            pAbsTensor,
                                            dpAfterDropoutTensor);

      // mask
      auto dsAfterMaskTensor = createMask(b,
                                          h,
                                          s_q,
                                          s_kv,
                                          d,
                                          layout,
                                          is_causal_masking,
                                          tensorType,
                                          ops,
                                          dsTensor,
                                          true);

      // matmul to calculate dqTensor
      auto matmul_2_Desc = cudnn_frontend::MatMulDescBuilder()
                               .setComputeType(CUDNN_DATA_FLOAT)
                               .setPaddingValue(0.0f)
                               .build();
      VLOG(10) << matmul_2_Desc.describe();

      auto matmul_op2 = cudnn_frontend::OperationBuilder(
                            CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR)
                            .setaMatDesc(dsAfterMaskTensor)
                            .setbMatDesc(kTensor)
                            .setcMatDesc(dqTensor)
                            .setmOverrideDesc(seqlenQTensor)
                            .setkOverrideDesc(seqlenKTensor)
                            .setmatmulDesc(matmul_2_Desc)
                            .build();

      VLOG(10) << matmul_op2.describe();

      ops.push_back(std::move(matmul_op2));

      // reshape for transpose of ds
      auto dsAfterMaskReshapeTensor = tensor_create(tensorType,
                                                    VIRTUAL_ID + 308,
                                                    p_transpose_dim,
                                                    p_transpose_stride,
                                                    true,
                                                    false);

      auto reshape_2_op = cudnn_frontend::OperationBuilder(
                              CUDNN_BACKEND_OPERATION_RESHAPE_DESCRIPTOR)
                              .setxDesc(dsAfterMaskTensor)
                              .setyDesc(dsAfterMaskReshapeTensor)
                              .build();

      VLOG(10) << reshape_2_op.describe();
      ops.push_back(std::move(reshape_2_op));

      // matmul to calculate dkTensor
      auto matmul_3_Desc = cudnn_frontend::MatMulDescBuilder()
                               .setComputeType(CUDNN_DATA_FLOAT)
                               .setPaddingValue(0.0f)
                               .build();
      VLOG(10) << matmul_3_Desc.describe();

      auto matmul_op3 = cudnn_frontend::OperationBuilder(
                            CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR)
                            .setaMatDesc(dsAfterMaskReshapeTensor)
                            .setbMatDesc(qTensor)
                            .setcMatDesc(dkTensor)
                            .setmOverrideDesc(seqlenKTensor)
                            .setkOverrideDesc(seqlenQTensor)
                            .setmatmulDesc(matmul_3_Desc)
                            .build();

      VLOG(10) << matmul_op3.describe();

      ops.push_back(std::move(matmul_op3));

      /////////////////////////////////////////////////////////////////

      VLOG(10) << "Total ops created: " << ops.size();

      for (unsigned int i = 0; i < ops.size(); i++) {
        all_ops.push_back(&ops[i]);
      }

      // Create an Operation Graph
      auto opGraph = cudnn_frontend::OperationGraphBuilder()
                         .setHandle(handle_)
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
                      .setHandle(handle_)
                      .setEngineConfig(filtered_configs[0], opGraph.getTag())
                      .build();
      cache.insert({descriptor, plan});
      return plan;
    };

    auto plan = get_plan(fmha_bprop_cache, descriptor);
    VLOG(10) << "Plan tag: " << plan.getTag();

    auto workspace_size = plan.getWorkspaceSize();
    VLOG(10) << plan.describe() << " requires workspace " << workspace_size;

    void *workspace_ptr = nullptr;
    if (workspace_size > 0) {
      PADDLE_ENFORCE_GPU_SUCCESS(cudaMalloc(&workspace_ptr, workspace_size));
    }

    std::set<std::pair<uint64_t, void *>> data_ptrs;
    // add all the data pointers to be used in the variant pack
    data_ptrs.insert(std::pair<uint64_t, void *>(dQ_ID, devPtrdQ));
    data_ptrs.insert(std::pair<uint64_t, void *>(dK_ID, devPtrdK));
    data_ptrs.insert(std::pair<uint64_t, void *>(dV_ID, devPtrdV));

    data_ptrs.insert(std::pair<uint64_t, void *>(Q_ID, devPtrQ));
    data_ptrs.insert(std::pair<uint64_t, void *>(K_ID, devPtrK));
    data_ptrs.insert(std::pair<uint64_t, void *>(V_ID, devPtrV));
    data_ptrs.insert(std::pair<uint64_t, void *>(S_ID, devPtrS));
    data_ptrs.insert(std::pair<uint64_t, void *>(dO_ID, devPtrdO));
    data_ptrs.insert(std::pair<uint64_t, void *>(dS_ID, devPtrdS));
    data_ptrs.insert(
        std::pair<uint64_t, void *>(Q_SEQLEN_ID, devActualSeqlenQ));
    data_ptrs.insert(
        std::pair<uint64_t, void *>(K_SEQLEN_ID, devActualSeqlenKV));

    float zeroVal = 0.0f;
    float dropoutScale = 1.0f / (1.0f - dropout_probability);

    data_ptrs.insert(std::pair<uint64_t, void *>(D_CONST_ID, &dropoutScale));
    data_ptrs.insert(std::pair<uint64_t, void *>(S_CONST_ID, &scaling_factor));
    data_ptrs.insert(std::pair<uint64_t, void *>(MASK_VAL_ID, &zeroVal));

    auto variantPack = cudnn_frontend::VariantPackBuilder()
                           .setWorkspacePointer(workspace_ptr)
                           .setDataPointers(data_ptrs)
                           .build();
    VLOG(10) << "variantPack " << variantPack.describe();
    cudnnStatus_t status = phi::dynload::cudnnBackendExecute(
        handle_, plan.get_raw_desc(), variantPack.get_raw_desc());
    if (workspace_size > 0) {
      PADDLE_ENFORCE_GPU_SUCCESS(cudaFree(workspace_ptr));
    }

    // checkCudnnErr(cudnnDestroy(handle_));

    cudnn_frontend::throw_if(
        [status]() { return (status != CUDNN_STATUS_SUCCESS); },
        "Plan execute error",
        status);
  } catch (cudnn_frontend::cudnnException &e) {
    struct cudaDeviceProp prop;
    PADDLE_ENFORCE_GPU_SUCCESS(cudaGetDeviceProperties(&prop, 0));

    // this example is only for GA100 cards and GH100 cards
    if (!((prop.major == 8 && prop.minor == 0) ||
          (prop.major == 9 && prop.minor == 0 && CUDNN_VERSION >= 8800)) &&
        (e.getCudnnStatus() == CUDNN_STATUS_ARCH_MISMATCH ||
         e.getCudnnStatus() == CUDNN_STATUS_NOT_SUPPORTED)) {
      std::cout << "Only supported for GA100 (cuDNN >= 8700) and "
                   "GH100 (cuDNN >= 8800) GPUs"
                << std::endl;
    } else {
      std::cout << "[ERROR] Exception " << e.what() << std::endl;
      // CHECK(false);
    }
  }
}

#endif
