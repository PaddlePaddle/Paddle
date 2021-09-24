/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <glog/logging.h>
#include <cassert>
#include "paddle/fluid/inference/tensorrt/plugin/matmul_op_int8_plugin.h"

#include <cublasLt.h>
#include <cuda_runtime.h>

namespace plf = paddle::platform;

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

int round(int in, int n) { return (in + n - 1) / n * n; }
void Ltgemm_int8_linear(cublasLtHandle_t ltHandle, int m, int n, int k,
                        const int8_t* A, bool transA, const int8_t* B,
                        bool transB, int8_t* C, int batch, float inscale_0,
                        float inscale_1, float outscale, float alpha_op,
                        void* workspace, cudaStream_t stream) {
  cublasOperation_t AopTranspose, BopTranspose;
  if (transA) {
    AopTranspose = CUBLAS_OP_T;
  } else {
    AopTranspose = CUBLAS_OP_N;
  }
  if (transB) {
    BopTranspose = CUBLAS_OP_T;
  } else {
    BopTranspose = CUBLAS_OP_N;
  }

  int8_t *Atransform = nullptr, *Btransform = nullptr, *Ctransform = nullptr;

  cublasLtMatrixLayout_t AtransformDesc = nullptr, BtransformDesc = nullptr,
                         CtransformDesc = nullptr;
  cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr;

  float alpha = alpha_op * inscale_0 * inscale_1 / outscale;
  float beta = 0;
  float transformAlpha = 1, transformBeta = 0;

  int64_t stridea = k * n;
  int64_t strideb = k * m;
  int64_t stridec = m * n;

  cudaDataType_t cudadataTypeIO = CUDA_R_8I;
  cudaDataType_t cudaDataTypeS = CUDA_R_32F;
  cublasComputeType_t cudaComputeType = CUBLAS_COMPUTE_32I;
  cublasLtOrder_t COL32 = CUBLASLT_ORDER_COL32;
  cublasLtOrder_t COL4_4R2_8C = CUBLASLT_ORDER_COL4_4R2_8C;

  int const ldatransform = 32 * n;
  int const ldbtransform = 32 * round(m, 8);
  int const ldctransform = 32 * n;

  CUDA_RT_CALL(cudaMalloc((void**)&Atransform,
                          sizeof(int8_t) * round(k, 32) / 32 * ldatransform));
  CUDA_RT_CALL(cudaMalloc((void**)&Btransform,
                          sizeof(int8_t) * round(k, 32) / 32 * ldbtransform));
  CUDA_RT_CALL(cudaMalloc((void**)&Ctransform,
                          sizeof(int8_t) * round(m, 32) / 32 * ldctransform));

  CUDA_RT_CALL(cublasLtMatrixLayoutCreate(&Adesc, cudadataTypeIO,
                                          AopTranspose == CUBLAS_OP_N ? n : k,
                                          AopTranspose == CUBLAS_OP_N ? k : n,
                                          AopTranspose == CUBLAS_OP_N ? n : k));
  CUDA_RT_CALL(cublasLtMatrixLayoutSetAttribute(
      Adesc, CUBLASLT_MATRIX_LAYOUT_TYPE, &cudadataTypeIO,
      sizeof(cudadataTypeIO)));
  CUDA_RT_CALL(cublasLtMatrixLayoutSetAttribute(
      Adesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &(batch), sizeof(batch)));
  CUDA_RT_CALL(cublasLtMatrixLayoutSetAttribute(
      Adesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &(stridea),
      sizeof(stridea)));

  CUDA_RT_CALL(cublasLtMatrixLayoutCreate(&Bdesc, cudadataTypeIO,
                                          BopTranspose == CUBLAS_OP_N ? k : m,
                                          BopTranspose == CUBLAS_OP_N ? m : k,
                                          BopTranspose == CUBLAS_OP_N ? k : m));
  CUDA_RT_CALL(cublasLtMatrixLayoutSetAttribute(
      Bdesc, CUBLASLT_MATRIX_LAYOUT_TYPE, &cudadataTypeIO,
      sizeof(cudadataTypeIO)));
  CUDA_RT_CALL(cublasLtMatrixLayoutSetAttribute(
      Bdesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &(batch), sizeof(batch)));
  CUDA_RT_CALL(cublasLtMatrixLayoutSetAttribute(
      Bdesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &(strideb),
      sizeof(strideb)));

  CUDA_RT_CALL(cublasLtMatrixLayoutCreate(&Cdesc, cudadataTypeIO, n, m, n));
  CUDA_RT_CALL(cublasLtMatrixLayoutSetAttribute(
      Cdesc, CUBLASLT_MATRIX_LAYOUT_TYPE, &cudadataTypeIO,
      sizeof(cudadataTypeIO)));
  CUDA_RT_CALL(cublasLtMatrixLayoutSetAttribute(
      Cdesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &(batch), sizeof(batch)));
  CUDA_RT_CALL(cublasLtMatrixLayoutSetAttribute(
      Cdesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &(stridec),
      sizeof(stridec)));

  CUDA_RT_CALL(cublasLtMatrixLayoutCreate(&AtransformDesc, cudadataTypeIO, n, k,
                                          ldatransform));
  CUDA_RT_CALL(cublasLtMatrixLayoutSetAttribute(
      AtransformDesc, CUBLASLT_MATRIX_LAYOUT_TYPE, &cudadataTypeIO,
      sizeof(cudadataTypeIO)));
  CUDA_RT_CALL(cublasLtMatrixLayoutSetAttribute(
      AtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &COL32, sizeof(COL32)));

  CUDA_RT_CALL(cublasLtMatrixLayoutCreate(&BtransformDesc, cudadataTypeIO, m, k,
                                          ldbtransform));
  CUDA_RT_CALL(cublasLtMatrixLayoutSetAttribute(
      BtransformDesc, CUBLASLT_MATRIX_LAYOUT_TYPE, &cudadataTypeIO,
      sizeof(cudadataTypeIO)));
  CUDA_RT_CALL(cublasLtMatrixLayoutSetAttribute(
      BtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &COL4_4R2_8C,
      sizeof(COL4_4R2_8C)));

  CUDA_RT_CALL(cublasLtMatrixLayoutCreate(&CtransformDesc, cudadataTypeIO, n, m,
                                          ldctransform));
  CUDA_RT_CALL(cublasLtMatrixLayoutSetAttribute(
      CtransformDesc, CUBLASLT_MATRIX_LAYOUT_TYPE, &cudadataTypeIO,
      sizeof(cudadataTypeIO)));
  CUDA_RT_CALL(cublasLtMatrixLayoutSetAttribute(
      CtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &COL32, sizeof(COL32)));

  cublasLtMatrixTransformDesc_t transformDescT = nullptr;
  cublasOperation_t Transpose = CUBLAS_OP_T;
  CUDA_RT_CALL(
      cublasLtMatrixTransformDescCreate(&transformDescT, cudaDataTypeS));
  CUDA_RT_CALL(cublasLtMatrixTransformDescSetAttribute(
      transformDescT, CUBLASLT_MATRIX_TRANSFORM_DESC_SCALE_TYPE, &cudaDataTypeS,
      sizeof(cudaDataTypeS)));
  CUDA_RT_CALL(cublasLtMatrixTransformDescSetAttribute(
      transformDescT, CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &Transpose,
      sizeof(Transpose)));

  cublasLtMatrixTransformDesc_t transformDescN = nullptr;
  CUDA_RT_CALL(
      cublasLtMatrixTransformDescCreate(&transformDescN, cudaDataTypeS));
  CUDA_RT_CALL(cublasLtMatrixTransformDescSetAttribute(
      transformDescN, CUBLASLT_MATRIX_TRANSFORM_DESC_SCALE_TYPE, &cudaDataTypeS,
      sizeof(cudaDataTypeS)));

  if (AopTranspose == CUBLAS_OP_N) {
    CUDA_RT_CALL(cublasLtMatrixTransform(
        ltHandle, transformDescN, &transformAlpha, A, Adesc, &transformBeta,
        nullptr, nullptr, Atransform, AtransformDesc, stream));
  } else {
    CUDA_RT_CALL(cublasLtMatrixTransform(
        ltHandle, transformDescT, &transformAlpha, A, Adesc, &transformBeta,
        nullptr, nullptr, Atransform, AtransformDesc, stream));
  }

  if (BopTranspose == CUBLAS_OP_T) {
    CUDA_RT_CALL(cublasLtMatrixTransform(
        ltHandle, transformDescN, &transformAlpha, B, Bdesc, &transformBeta,
        nullptr, nullptr, Btransform, BtransformDesc, stream));
  } else {
    CUDA_RT_CALL(cublasLtMatrixTransform(
        ltHandle, transformDescT, &transformAlpha, B, Bdesc, &transformBeta,
        nullptr, nullptr, Btransform, BtransformDesc, stream));
  }

  cublasLtMatmulDesc_t matmulDesc = nullptr;
  cublasOperation_t ATranspose = CUBLAS_OP_N, BTranspose = CUBLAS_OP_T;
  CUDA_RT_CALL(
      cublasLtMatmulDescCreate(&matmulDesc, cudaComputeType, cudaDataTypeS));
  CUDA_RT_CALL(cublasLtMatmulDescSetAttribute(matmulDesc,
                                              CUBLASLT_MATMUL_DESC_TRANSA,
                                              &ATranspose, sizeof(ATranspose)));
  CUDA_RT_CALL(cublasLtMatmulDescSetAttribute(matmulDesc,
                                              CUBLASLT_MATMUL_DESC_TRANSB,
                                              &BTranspose, sizeof(BTranspose)));

  CUDA_RT_CALL(cublasLtMatmul(ltHandle, matmulDesc, &alpha, Atransform,
                              AtransformDesc, Btransform, BtransformDesc, &beta,
                              Ctransform, CtransformDesc, Ctransform,
                              CtransformDesc, nullptr, workspace, 0, stream));

  CUDA_RT_CALL(cublasLtMatrixTransform(
      ltHandle, transformDescN, &transformAlpha, Ctransform, CtransformDesc,
      &transformBeta, nullptr, nullptr, C, Cdesc, stream));

  CUDA_RT_CALL(cublasLtMatrixLayoutDestroy(AtransformDesc));
  CUDA_RT_CALL(cublasLtMatrixLayoutDestroy(BtransformDesc));
  CUDA_RT_CALL(cublasLtMatrixLayoutDestroy(CtransformDesc));
  CUDA_RT_CALL(cublasLtMatrixLayoutDestroy(Adesc));
  CUDA_RT_CALL(cublasLtMatrixLayoutDestroy(Bdesc));
  CUDA_RT_CALL(cublasLtMatrixLayoutDestroy(Cdesc));
  CUDA_RT_CALL(cublasLtMatmulDescDestroy(matmulDesc));
  CUDA_RT_CALL(cublasLtMatrixTransformDescDestroy(transformDescT));
  CUDA_RT_CALL(cublasLtMatrixTransformDescDestroy(transformDescN));
  cudaDeviceSynchronize();
}

void Ltgemm_fp32_linear(cublasLtHandle_t ltHandle, int m, int n, int k,
                        const float* A, bool transA, const float* B,
                        bool transB, float* C, int batch, float alpha_op,
                        void* workspace, cudaStream_t stream) {
  cublasOperation_t AopTranspose, BopTranspose;
  if (transA) {
    AopTranspose = CUBLAS_OP_T;
  } else {
    AopTranspose = CUBLAS_OP_N;
  }
  if (transB) {
    BopTranspose = CUBLAS_OP_T;
  } else {
    BopTranspose = CUBLAS_OP_N;
  }

  cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr;

  float alpha = alpha_op, beta = 0;
  int64_t stridea = k * n;
  int64_t strideb = k * m;
  int64_t stridec = m * n;

  cudaDataType_t cudadataTypeIO = CUDA_R_32F;
  cudaDataType_t cudaDataTypeS = CUDA_R_32F;
  cublasComputeType_t cudaComputeType = CUBLAS_COMPUTE_32F_FAST_16F;

  CUDA_RT_CALL(cublasLtMatrixLayoutCreate(&Adesc, cudadataTypeIO,
                                          AopTranspose == CUBLAS_OP_N ? n : k,
                                          AopTranspose == CUBLAS_OP_N ? k : n,
                                          AopTranspose == CUBLAS_OP_N ? n : k));
  CUDA_RT_CALL(cublasLtMatrixLayoutSetAttribute(
      Adesc, CUBLASLT_MATRIX_LAYOUT_TYPE, &cudadataTypeIO,
      sizeof(cudadataTypeIO)));
  CUDA_RT_CALL(cublasLtMatrixLayoutSetAttribute(
      Adesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &(batch), sizeof(batch)));
  CUDA_RT_CALL(cublasLtMatrixLayoutSetAttribute(
      Adesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &(stridea),
      sizeof(stridea)));

  CUDA_RT_CALL(cublasLtMatrixLayoutCreate(&Bdesc, cudadataTypeIO,
                                          BopTranspose == CUBLAS_OP_N ? k : m,
                                          BopTranspose == CUBLAS_OP_N ? m : k,
                                          BopTranspose == CUBLAS_OP_N ? k : m));
  CUDA_RT_CALL(cublasLtMatrixLayoutSetAttribute(
      Bdesc, CUBLASLT_MATRIX_LAYOUT_TYPE, &cudadataTypeIO,
      sizeof(cudadataTypeIO)));
  CUDA_RT_CALL(cublasLtMatrixLayoutSetAttribute(
      Bdesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &(batch), sizeof(batch)));
  CUDA_RT_CALL(cublasLtMatrixLayoutSetAttribute(
      Bdesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &(strideb),
      sizeof(strideb)));

  CUDA_RT_CALL(cublasLtMatrixLayoutCreate(&Cdesc, cudadataTypeIO, n, m, n));
  CUDA_RT_CALL(cublasLtMatrixLayoutSetAttribute(
      Cdesc, CUBLASLT_MATRIX_LAYOUT_TYPE, &cudadataTypeIO,
      sizeof(cudadataTypeIO)));
  CUDA_RT_CALL(cublasLtMatrixLayoutSetAttribute(
      Cdesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &(batch), sizeof(batch)));
  CUDA_RT_CALL(cublasLtMatrixLayoutSetAttribute(
      Cdesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &(stridec),
      sizeof(stridec)));

  cublasLtMatmulDesc_t matmulDesc = nullptr;
  CUDA_RT_CALL(
      cublasLtMatmulDescCreate(&matmulDesc, cudaComputeType, cudaDataTypeS));
  CUDA_RT_CALL(
      cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA,
                                     &AopTranspose, sizeof(AopTranspose)));
  CUDA_RT_CALL(
      cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB,
                                     &BopTranspose, sizeof(BopTranspose)));

  CUDA_RT_CALL(cublasLtMatmul(ltHandle, matmulDesc, &alpha, A, Adesc, B, Bdesc,
                              &beta, C, Cdesc, C, Cdesc, nullptr, workspace, 0,
                              stream));

  CUDA_RT_CALL(cublasLtMatrixLayoutDestroy(Adesc));
  CUDA_RT_CALL(cublasLtMatrixLayoutDestroy(Bdesc));
  CUDA_RT_CALL(cublasLtMatrixLayoutDestroy(Cdesc));
  CUDA_RT_CALL(cublasLtMatmulDescDestroy(matmulDesc));
  cudaDeviceSynchronize();
}

void Ltgemm_fp16_linear(cublasLtHandle_t ltHandle, int m, int n, int k,
                        const half* A, bool transA, const half* B, bool transB,
                        half* C, int batch, float alpha_op, void* workspace,
                        cudaStream_t stream) {
  cublasOperation_t AopTranspose, BopTranspose;
  if (transA) {
    AopTranspose = CUBLAS_OP_T;
  } else {
    AopTranspose = CUBLAS_OP_N;
  }
  if (transB) {
    BopTranspose = CUBLAS_OP_T;
  } else {
    BopTranspose = CUBLAS_OP_N;
  }

  cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr;

  half alpha = alpha_op, beta = 0;
  int64_t stridea = k * n;
  int64_t strideb = k * m;
  int64_t stridec = m * n;

  cudaDataType_t cudadataTypeIO = CUDA_R_16F;
  cudaDataType_t cudaDataTypeS = CUDA_R_16F;
  cublasComputeType_t cudaComputeType = CUBLAS_COMPUTE_16F;

  CUDA_RT_CALL(cublasLtMatrixLayoutCreate(&Adesc, cudadataTypeIO,
                                          AopTranspose == CUBLAS_OP_N ? n : k,
                                          AopTranspose == CUBLAS_OP_N ? k : n,
                                          AopTranspose == CUBLAS_OP_N ? n : k));
  CUDA_RT_CALL(cublasLtMatrixLayoutSetAttribute(
      Adesc, CUBLASLT_MATRIX_LAYOUT_TYPE, &cudadataTypeIO,
      sizeof(cudadataTypeIO)));
  CUDA_RT_CALL(cublasLtMatrixLayoutSetAttribute(
      Adesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &(batch), sizeof(batch)));
  CUDA_RT_CALL(cublasLtMatrixLayoutSetAttribute(
      Adesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &(stridea),
      sizeof(stridea)));

  CUDA_RT_CALL(cublasLtMatrixLayoutCreate(&Bdesc, cudadataTypeIO,
                                          BopTranspose == CUBLAS_OP_N ? k : m,
                                          BopTranspose == CUBLAS_OP_N ? m : k,
                                          BopTranspose == CUBLAS_OP_N ? k : m));
  CUDA_RT_CALL(cublasLtMatrixLayoutSetAttribute(
      Bdesc, CUBLASLT_MATRIX_LAYOUT_TYPE, &cudadataTypeIO,
      sizeof(cudadataTypeIO)));
  CUDA_RT_CALL(cublasLtMatrixLayoutSetAttribute(
      Bdesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &(batch), sizeof(batch)));
  CUDA_RT_CALL(cublasLtMatrixLayoutSetAttribute(
      Bdesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &(strideb),
      sizeof(strideb)));

  CUDA_RT_CALL(cublasLtMatrixLayoutCreate(&Cdesc, cudadataTypeIO, n, m, n));
  CUDA_RT_CALL(cublasLtMatrixLayoutSetAttribute(
      Cdesc, CUBLASLT_MATRIX_LAYOUT_TYPE, &cudadataTypeIO,
      sizeof(cudadataTypeIO)));
  CUDA_RT_CALL(cublasLtMatrixLayoutSetAttribute(
      Cdesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &(batch), sizeof(batch)));
  CUDA_RT_CALL(cublasLtMatrixLayoutSetAttribute(
      Cdesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &(stridec),
      sizeof(stridec)));

  cublasLtMatmulDesc_t matmulDesc = nullptr;
  CUDA_RT_CALL(
      cublasLtMatmulDescCreate(&matmulDesc, cudaComputeType, cudaDataTypeS));
  CUDA_RT_CALL(
      cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA,
                                     &AopTranspose, sizeof(AopTranspose)));
  CUDA_RT_CALL(
      cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB,
                                     &BopTranspose, sizeof(BopTranspose)));

  CUDA_RT_CALL(cublasLtMatmul(ltHandle, matmulDesc, &alpha, A, Adesc, B, Bdesc,
                              &beta, C, Cdesc, C, Cdesc, nullptr, workspace, 0,
                              stream));

  CUDA_RT_CALL(cublasLtMatrixLayoutDestroy(Adesc));
  CUDA_RT_CALL(cublasLtMatrixLayoutDestroy(Bdesc));
  CUDA_RT_CALL(cublasLtMatrixLayoutDestroy(Cdesc));
  CUDA_RT_CALL(cublasLtMatmulDescDestroy(matmulDesc));
  cudaDeviceSynchronize();
}

nvinfer1::DataType MatmulPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* input_types,
    int nb_inputs) const TRT_NOEXCEPT {
  return input_types[0];
}

nvinfer1::Dims MatmulPlugin::getOutputDimensions(
    int index, const nvinfer1::Dims* input_dims, int num_inputs) TRT_NOEXCEPT {
  if (transB_) {
    m_ = dims_x_.d[dims_x_.nbDims - 1];
    k_ = dims_x_.d[dims_x_.nbDims - 2];
  } else {
    m_ = dims_x_.d[dims_x_.nbDims - 2];
    k_ = dims_x_.d[dims_x_.nbDims - 1];
  }
  if (transA_) {
    n_ = dims_y_.d[dims_y_.nbDims - 2];
  } else {
    n_ = dims_y_.d[dims_y_.nbDims - 1];
  }

  batch_ = 1;
  for (int i = 0; i < dims_x_.nbDims - 2; i++) {
    batch_ *= dims_x_.d[i];
  }
  nvinfer1::Dims output_dims;
  output_dims.nbDims = dims_x_.nbDims;
  for (int i = 0; i < output_dims.nbDims - 2; i++) {
    output_dims.d[i] = dims_x_.d[i];
  }
  output_dims.d[output_dims.nbDims - 2] = m_;
  output_dims.d[output_dims.nbDims - 1] = n_;

  return output_dims;
}

bool MatmulPlugin::supportsFormatCombination(
    int32_t pos, nvinfer1::PluginTensorDesc const* inOut, int32_t nbInputs,
    int32_t nbOutputs) const TRT_NOEXCEPT {
  PADDLE_ENFORCE_EQ(nbInputs, 2, platform::errors::InvalidArgument("......."
                                                                   ",,,,,,,"));
  PADDLE_ENFORCE_EQ(nbOutputs, getNbOutputs(),
                    platform::errors::InvalidArgument("......."
                                                      ",,,,,,,"));
  if (pos == 0) {
    return (inOut[pos].type == nvinfer1::DataType::kINT8 ||
            inOut[pos].type == nvinfer1::DataType::kHALF ||
            inOut[pos].type == nvinfer1::DataType::kFLOAT) &&
           inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
  } else {
    return inOut[pos].type == inOut[0].type &&
           inOut[pos].format == inOut[0].format;
  }
}

void MatmulPlugin::configurePlugin(const nvinfer1::PluginTensorDesc* inputs,
                                   int32_t nbInputs,
                                   const nvinfer1::PluginTensorDesc* out,
                                   int32_t nbOutputs) TRT_NOEXCEPT {
  inscale_0_ = inputs[0].scale;
  inscale_1_ = inputs[1].scale;
  outscale_ = out[0].scale;
  type_ = inputs[0].type;
  if (inputs[0].type == nvinfer1::DataType::kINT8) {
    std::cout << "configurePlugin: kINT8" << std::endl;
  } else if (inputs[0].type == nvinfer1::DataType::kFLOAT) {
    std::cout << "configurePlugin: kFLOAT" << std::endl;
  } else if (inputs[0].type == nvinfer1::DataType::kHALF) {
    std::cout << "configurePlugin: kHALF" << std::endl;
  }
  // assert(inputs[0].type == inputs[1].type == out[0].type);
}

int MatmulPlugin::enqueue(int batchSize, const void* const* inputs,
#if IS_TRT_VERSION_LT(8000)
                          void** outputs, void* workspace,
                          cudaStream_t stream) {
#else
                          void* const* outputs, void* workspace,
                          cudaStream_t stream) TRT_NOEXCEPT {
#endif
  cublasLtHandle_t handle;
  CUDA_RT_CALL(cublasLtCreate(&handle));
  if (type_ == nvinfer1::DataType::kINT8) {
    std::cout << "type_: kINT8" << std::endl;
  } else if (type_ == nvinfer1::DataType::kFLOAT) {
    std::cout << "type_: kFLOAT" << std::endl;
  } else if (type_ == nvinfer1::DataType::kHALF) {
    std::cout << "type_: kHALF" << std::endl;
  }

  if (type_ == nvinfer1::DataType::kINT8) {
    std::cout << "int_8_m: " << m_ << "   n: " << n_ << "   k: " << k_
              << "  inscale_0_: " << inscale_0_
              << "  inscale_1_: " << inscale_1_ << "  outscale_: " << outscale_
              << std::endl;
    const int8_t* B = static_cast<const int8_t*>(inputs[0]);
    const int8_t* A = static_cast<const int8_t*>(inputs[1]);
    int8_t* C = static_cast<int8_t*>(outputs[0]);
    Ltgemm_int8_linear(handle, m_, n_, k_, A, transA_, B, transB_, C, batch_,
                       inscale_0_, inscale_1_, outscale_, alpha_, workspace,
                       stream);
  } else if (type_ == nvinfer1::DataType::kFLOAT) {
    std::cout << "float_m: " << m_ << "   n: " << n_ << "   k: " << k_
              << "  inscale_0_: " << inscale_0_
              << "  inscale_1_: " << inscale_1_ << "  outscale_: " << outscale_
              << std::endl;
    const float* B = static_cast<const float*>(inputs[0]);
    const float* A = static_cast<const float*>(inputs[1]);
    float* C = static_cast<float*>(outputs[0]);
    Ltgemm_fp32_linear(handle, m_, n_, k_, A, transA_, B, transB_, C, batch_,
                       alpha_, workspace, stream);
  } else if (type_ == nvinfer1::DataType::kHALF) {
    std::cout << "half_m: " << m_ << "   n: " << n_ << "   k: " << k_
              << "  inscale_0_: " << inscale_0_
              << "  inscale_1_: " << inscale_1_ << "  outscale_: " << outscale_
              << std::endl;
    const half* B = static_cast<const half*>(inputs[0]);
    const half* A = static_cast<const half*>(inputs[1]);
    half* C = static_cast<half*>(outputs[0]);
    Ltgemm_fp16_linear(handle, m_, n_, k_, A, transA_, B, transB_, C, batch_,
                       alpha_, workspace, stream);
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "VarMessageToVarType:Unsupported type"));
  }
  return cudaGetLastError() != cudaSuccess;
}

nvinfer1::DataType MatmulPluginDynamic::getOutputDataType(
    int index, const nvinfer1::DataType* input_types,
    int nb_inputs) const TRT_NOEXCEPT {
  return input_types[0];
}

nvinfer1::DimsExprs MatmulPluginDynamic::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs,
    nvinfer1::IExprBuilder& exprBuilder) TRT_NOEXCEPT {
  nvinfer1::DimsExprs output_dims(inputs[0]);
  if (transB_) {
    output_dims.d[output_dims.nbDims - 2] = inputs[0].d[inputs[0].nbDims - 1];
  } else {
    output_dims.d[output_dims.nbDims - 2] = inputs[0].d[inputs[0].nbDims - 2];
  }
  if (transA_) {
    output_dims.d[output_dims.nbDims - 1] = inputs[1].d[inputs[1].nbDims - 2];
  } else {
    output_dims.d[output_dims.nbDims - 1] = inputs[1].d[inputs[1].nbDims - 1];
  }
  return output_dims;
}

bool MatmulPluginDynamic::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs,
    int nbOutputs) TRT_NOEXCEPT {
  PADDLE_ENFORCE_EQ(nbInputs, 2, platform::errors::InvalidArgument("......."
                                                                   ",,,,,,,"));
  PADDLE_ENFORCE_EQ(nbOutputs, getNbOutputs(),
                    platform::errors::InvalidArgument("......."
                                                      ",,,,,,,"));
  if (pos == 0) {
    return (inOut[pos].type == nvinfer1::DataType::kINT8 ||
            inOut[pos].type == nvinfer1::DataType::kHALF ||
            inOut[pos].type == nvinfer1::DataType::kFLOAT) &&
           inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
  } else {
    return inOut[pos].type == inOut[0].type &&
           inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
  }
}

int MatmulPluginDynamic::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
                                 const nvinfer1::PluginTensorDesc* outputDesc,
                                 const void* const* inputs,
                                 void* const* outputs, void* workspace,
                                 cudaStream_t stream) TRT_NOEXCEPT {
  const auto Input0Desc = inputDesc[0];
  const auto Input1Desc = inputDesc[1];
  const auto OutDesc = outputDesc[0];

  inscale_0_ = Input0Desc.scale;
  inscale_1_ = Input1Desc.scale;
  outscale_ = OutDesc.scale;
  type_ = Input0Desc.type;

  if (transB_) {
    m_ = Input0Desc.dims.d[Input0Desc.dims.nbDims - 1];
    k_ = Input0Desc.dims.d[Input0Desc.dims.nbDims - 2];
  } else {
    m_ = Input0Desc.dims.d[Input0Desc.dims.nbDims - 2];
    k_ = Input0Desc.dims.d[Input0Desc.dims.nbDims - 1];
  }
  if (transA_) {
    n_ = Input1Desc.dims.d[Input1Desc.dims.nbDims - 2];
  } else {
    n_ = Input1Desc.dims.d[Input1Desc.dims.nbDims - 1];
  }

  batch_ = 1;
  for (int i = 0; i < Input0Desc.dims.nbDims - 2; i++) {
    batch_ *= Input0Desc.dims.d[i];
  }

  cublasLtHandle_t handle;
  CUDA_RT_CALL(cublasLtCreate(&handle));
  if (type_ == nvinfer1::DataType::kINT8) {
    std::cout << "int_8_m: " << m_ << "   n: " << n_ << "   k: " << k_
              << "  inscale_0_: " << inscale_0_
              << "  inscale_1_: " << inscale_1_ << "  outscale_: " << outscale_
              << std::endl;
    const int8_t* B = static_cast<const int8_t*>(inputs[0]);
    const int8_t* A = static_cast<const int8_t*>(inputs[1]);
    int8_t* C = static_cast<int8_t*>(outputs[0]);
    Ltgemm_int8_linear(handle, m_, n_, k_, A, transA_, B, transB_, C, batch_,
                       inscale_0_, inscale_1_, outscale_, alpha_, workspace,
                       stream);
  } else if (type_ == nvinfer1::DataType::kFLOAT) {
    std::cout << "float_m: " << m_ << "   n: " << n_ << "   k: " << k_
              << "  inscale_0_: " << inscale_0_
              << "  inscale_1_: " << inscale_1_ << "  outscale_: " << outscale_
              << std::endl;
    const float* B = static_cast<const float*>(inputs[0]);
    const float* A = static_cast<const float*>(inputs[1]);
    float* C = static_cast<float*>(outputs[0]);
    Ltgemm_fp32_linear(handle, m_, n_, k_, A, transA_, B, transB_, C, batch_,
                       alpha_, workspace, stream);
  } else if (type_ == nvinfer1::DataType::kHALF) {
    std::cout << "half_m: " << m_ << "   n: " << n_ << "   k: " << k_
              << "  inscale_0_: " << inscale_0_
              << "  inscale_1_: " << inscale_1_ << "  outscale_: " << outscale_
              << std::endl;
    const half* B = static_cast<const half*>(inputs[0]);
    const half* A = static_cast<const half*>(inputs[1]);
    half* C = static_cast<half*>(outputs[0]);
    Ltgemm_fp16_linear(handle, m_, n_, k_, A, transA_, B, transB_, C, batch_,
                       alpha_, workspace, stream);
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "VarMessageToVarType:Unsupported type"));
  }
  return cudaGetLastError() != cudaSuccess;
}

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
