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
void Ltgemm_int8_linear(
    cublasLtHandle_t ltHandle, const int8_t* A, cublasLtMatrixLayout_t Adesc,
    int8_t* Atransform, cublasLtMatrixLayout_t AtransformDesc, bool transA_,
    const int8_t* B, cublasLtMatrixLayout_t Bdesc, int8_t* Btransform,
    cublasLtMatrixLayout_t BtransformDesc, bool transB_, int8_t* C,
    cublasLtMatrixLayout_t Cdesc, int8_t* Ctransform,
    cublasLtMatrixLayout_t CtransformDesc,
    cublasLtMatrixTransformDesc_t transformDescT,
    cublasLtMatrixTransformDesc_t transformDescN,
    cublasLtMatmulDesc_t matmulDesc, void* alpha_scale, void* alpha_zero,
    void* alpha_one, void* workspace, cudaStream_t stream) {
  if (transA_) {
    CUDA_RT_CALL(cublasLtMatrixTransform(ltHandle, transformDescT, alpha_one, A,
                                         Adesc, alpha_zero, nullptr, nullptr,
                                         Atransform, AtransformDesc, stream));
  } else {
    CUDA_RT_CALL(cublasLtMatrixTransform(ltHandle, transformDescN, alpha_one, A,
                                         Adesc, alpha_zero, nullptr, nullptr,
                                         Atransform, AtransformDesc, stream));
  }

  if (transB_) {
    CUDA_RT_CALL(cublasLtMatrixTransform(ltHandle, transformDescN, alpha_one, B,
                                         Bdesc, alpha_zero, nullptr, nullptr,
                                         Btransform, BtransformDesc, stream));
  } else {
    CUDA_RT_CALL(cublasLtMatrixTransform(ltHandle, transformDescT, alpha_one, B,
                                         Bdesc, alpha_zero, nullptr, nullptr,
                                         Btransform, BtransformDesc, stream));
  }

  CUDA_RT_CALL(cublasLtMatmul(ltHandle, matmulDesc, alpha_scale, Atransform,
                              AtransformDesc, Btransform, BtransformDesc,
                              nullptr, Ctransform, CtransformDesc, Ctransform,
                              CtransformDesc, nullptr, workspace, 0, stream));

  CUDA_RT_CALL(cublasLtMatrixTransform(ltHandle, transformDescN, alpha_one,
                                       Ctransform, CtransformDesc, alpha_zero,
                                       nullptr, nullptr, C, Cdesc, stream));
}

void Ltgemm_fp32_linear(cublasLtHandle_t ltHandle, const float* A,
                        cublasLtMatrixLayout_t Adesc, const float* B,
                        cublasLtMatrixLayout_t Bdesc, float* C,
                        cublasLtMatrixLayout_t Cdesc,
                        cublasLtMatmulDesc_t matmulDesc, void* alpha_scale,
                        void* alpha_zero, void* workspace,
                        cudaStream_t stream) {
  CUDA_RT_CALL(cublasLtMatmul(ltHandle, matmulDesc, alpha_scale, A, Adesc, B,
                              Bdesc, alpha_zero, C, Cdesc, C, Cdesc, nullptr,
                              workspace, 0, stream));
}

void Ltgemm_fp16_linear(cublasLtHandle_t ltHandle, const half* A,
                        cublasLtMatrixLayout_t Adesc, const half* B,
                        cublasLtMatrixLayout_t Bdesc, half* C,
                        cublasLtMatrixLayout_t Cdesc,
                        cublasLtMatmulDesc_t matmulDesc, void* alpha_scale,
                        void* alpha_zero, void* workspace,
                        cudaStream_t stream) {
  CUDA_RT_CALL(cublasLtMatmul(ltHandle, matmulDesc, alpha_scale, A, Adesc, B,
                              Bdesc, alpha_zero, C, Cdesc, C, Cdesc, nullptr,
                              workspace, 0, stream));
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
    return (inOut[pos].type == nvinfer1::DataType::kHALF ||
            inOut[pos].type == nvinfer1::DataType::kFLOAT ||
            inOut[pos].type == nvinfer1::DataType::kINT8) &&
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
  std::cout << "configurePlugin(): this: " << this << std::endl;
  float inscale_0 = inputs[0].scale;
  float inscale_1 = inputs[1].scale;
  float outscale = out[0].scale;
  type_ = inputs[0].type;
  int64_t stridea = k_ * n_;
  int64_t strideb = k_ * m_;
  int64_t stridec = m_ * n_;

  cublasOperation_t AopTranspose, BopTranspose;
  if (transA_) {
    AopTranspose = CUBLAS_OP_T;
  } else {
    AopTranspose = CUBLAS_OP_N;
  }
  if (transB_) {
    BopTranspose = CUBLAS_OP_T;
  } else {
    BopTranspose = CUBLAS_OP_N;
  }

  if (type_ == nvinfer1::DataType::kINT8) {
    cudaDataType_t cudadataTypeIO = CUDA_R_8I;
    cudaDataType_t cudaDataTypeS = CUDA_R_32F;
    cublasComputeType_t cudaComputeType = CUBLAS_COMPUTE_32I;
    cublasLtOrder_t COL32 = CUBLASLT_ORDER_COL32;
    cublasLtOrder_t COL4_4R2_8C = CUBLASLT_ORDER_COL4_4R2_8C;

    int const ldatransform = 32 * n_;
    int const ldbtransform = 32 * ((m_ + 8 - 1) / 8 * 8);
    int const ldctransform = 32 * n_;
    CUDA_RT_CALL(cudaMalloc(
        (void**)&Atransform_,
        sizeof(int8_t) * ((k_ + 32 - 1) / 32 * 32) / 32 * ldatransform));
    CUDA_RT_CALL(cudaMalloc(
        (void**)&Btransform_,
        sizeof(int8_t) * ((k_ + 32 - 1) / 32 * 32) / 32 * ldbtransform));
    CUDA_RT_CALL(cudaMalloc(
        (void**)&Ctransform_,
        sizeof(int8_t) * ((m_ + 32 - 1) / 32 * 32) / 32 * ldctransform));

    CUDA_RT_CALL(cublasLtMatrixLayoutCreate(
        &Adesc_, cudadataTypeIO, AopTranspose == CUBLAS_OP_N ? n_ : k_,
        AopTranspose == CUBLAS_OP_N ? k_ : n_,
        AopTranspose == CUBLAS_OP_N ? n_ : k_));
    CUDA_RT_CALL(cublasLtMatrixLayoutSetAttribute(
        Adesc_, CUBLASLT_MATRIX_LAYOUT_TYPE, &cudadataTypeIO,
        sizeof(cudadataTypeIO)));
    CUDA_RT_CALL(cublasLtMatrixLayoutSetAttribute(
        Adesc_, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &(batch_), sizeof(batch_)));
    CUDA_RT_CALL(cublasLtMatrixLayoutSetAttribute(
        Adesc_, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &(stridea),
        sizeof(stridea)));

    CUDA_RT_CALL(cublasLtMatrixLayoutCreate(
        &Bdesc_, cudadataTypeIO, BopTranspose == CUBLAS_OP_N ? k_ : m_,
        BopTranspose == CUBLAS_OP_N ? m_ : k_,
        BopTranspose == CUBLAS_OP_N ? k_ : m_));
    CUDA_RT_CALL(cublasLtMatrixLayoutSetAttribute(
        Bdesc_, CUBLASLT_MATRIX_LAYOUT_TYPE, &cudadataTypeIO,
        sizeof(cudadataTypeIO)));
    CUDA_RT_CALL(cublasLtMatrixLayoutSetAttribute(
        Bdesc_, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &(batch_), sizeof(batch_)));
    CUDA_RT_CALL(cublasLtMatrixLayoutSetAttribute(
        Bdesc_, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &(strideb),
        sizeof(strideb)));

    CUDA_RT_CALL(
        cublasLtMatrixLayoutCreate(&Cdesc_, cudadataTypeIO, n_, m_, n_));
    CUDA_RT_CALL(cublasLtMatrixLayoutSetAttribute(
        Cdesc_, CUBLASLT_MATRIX_LAYOUT_TYPE, &cudadataTypeIO,
        sizeof(cudadataTypeIO)));
    CUDA_RT_CALL(cublasLtMatrixLayoutSetAttribute(
        Cdesc_, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &(batch_), sizeof(batch_)));
    CUDA_RT_CALL(cublasLtMatrixLayoutSetAttribute(
        Cdesc_, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &(stridec),
        sizeof(stridec)));

    CUDA_RT_CALL(cublasLtMatrixLayoutCreate(&AtransformDesc_, cudadataTypeIO,
                                            n_, k_, ldatransform));
    CUDA_RT_CALL(cublasLtMatrixLayoutSetAttribute(
        AtransformDesc_, CUBLASLT_MATRIX_LAYOUT_TYPE, &cudadataTypeIO,
        sizeof(cudadataTypeIO)));
    CUDA_RT_CALL(cublasLtMatrixLayoutSetAttribute(
        AtransformDesc_, CUBLASLT_MATRIX_LAYOUT_ORDER, &COL32, sizeof(COL32)));

    CUDA_RT_CALL(cublasLtMatrixLayoutCreate(&BtransformDesc_, cudadataTypeIO,
                                            m_, k_, ldbtransform));
    CUDA_RT_CALL(cublasLtMatrixLayoutSetAttribute(
        BtransformDesc_, CUBLASLT_MATRIX_LAYOUT_TYPE, &cudadataTypeIO,
        sizeof(cudadataTypeIO)));
    CUDA_RT_CALL(cublasLtMatrixLayoutSetAttribute(
        BtransformDesc_, CUBLASLT_MATRIX_LAYOUT_ORDER, &COL4_4R2_8C,
        sizeof(COL4_4R2_8C)));

    CUDA_RT_CALL(cublasLtMatrixLayoutCreate(&CtransformDesc_, cudadataTypeIO,
                                            n_, m_, ldctransform));
    CUDA_RT_CALL(cublasLtMatrixLayoutSetAttribute(
        CtransformDesc_, CUBLASLT_MATRIX_LAYOUT_TYPE, &cudadataTypeIO,
        sizeof(cudadataTypeIO)));
    CUDA_RT_CALL(cublasLtMatrixLayoutSetAttribute(
        CtransformDesc_, CUBLASLT_MATRIX_LAYOUT_ORDER, &COL32, sizeof(COL32)));

    cublasOperation_t Transpose = CUBLAS_OP_T;
    cublasLtPointerMode_t transform_model = CUBLASLT_POINTER_MODE_DEVICE;
    CUDA_RT_CALL(
        cublasLtMatrixTransformDescCreate(&transformDescT_, cudaDataTypeS));
    CUDA_RT_CALL(cublasLtMatrixTransformDescSetAttribute(
        transformDescT_, CUBLASLT_MATRIX_TRANSFORM_DESC_SCALE_TYPE,
        &cudaDataTypeS, sizeof(cudaDataTypeS)));
    CUDA_RT_CALL(cublasLtMatrixTransformDescSetAttribute(
        transformDescT_, CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &Transpose,
        sizeof(Transpose)));
    CUDA_RT_CALL(cublasLtMatrixTransformDescSetAttribute(
        transformDescT_, CUBLASLT_MATRIX_TRANSFORM_DESC_POINTER_MODE,
        &transform_model, sizeof(transform_model)));

    CUDA_RT_CALL(
        cublasLtMatrixTransformDescCreate(&transformDescN_, cudaDataTypeS));
    CUDA_RT_CALL(cublasLtMatrixTransformDescSetAttribute(
        transformDescN_, CUBLASLT_MATRIX_TRANSFORM_DESC_SCALE_TYPE,
        &cudaDataTypeS, sizeof(cudaDataTypeS)));
    CUDA_RT_CALL(cublasLtMatrixTransformDescSetAttribute(
        transformDescN_, CUBLASLT_MATRIX_TRANSFORM_DESC_POINTER_MODE,
        &transform_model, sizeof(transform_model)));

    cublasOperation_t ATranspose = CUBLAS_OP_N, BTranspose = CUBLAS_OP_T;
    cublasLtPointerMode_t matmul_model =
        CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_ZERO;
    CUDA_RT_CALL(
        cublasLtMatmulDescCreate(&matmulDesc_, cudaComputeType, cudaDataTypeS));
    CUDA_RT_CALL(
        cublasLtMatmulDescSetAttribute(matmulDesc_, CUBLASLT_MATMUL_DESC_TRANSA,
                                       &ATranspose, sizeof(ATranspose)));
    CUDA_RT_CALL(
        cublasLtMatmulDescSetAttribute(matmulDesc_, CUBLASLT_MATMUL_DESC_TRANSB,
                                       &BTranspose, sizeof(BTranspose)));
    CUDA_RT_CALL(cublasLtMatmulDescSetAttribute(
        matmulDesc_, CUBLASLT_MATMUL_DESC_POINTER_MODE, &matmul_model,
        sizeof(matmul_model)));

    float alpha_tem[n_];
    for (int i = 0; i < n_; i++) {
      alpha_tem[i] = alpha_ * inscale_0 * inscale_1 / outscale;
    }
    CUDA_RT_CALL(cudaMalloc((void**)&alpha_scale_, n_ * sizeof(float)));
    cudaMemcpyAsync(alpha_scale_, alpha_tem, n_ * sizeof(float),
                    cudaMemcpyHostToDevice);
    float zero_tem = 0;
    CUDA_RT_CALL(cudaMalloc((void**)&alpha_zero_, sizeof(float)));
    cudaMemcpyAsync(alpha_zero_, &zero_tem, sizeof(float),
                    cudaMemcpyHostToDevice);
    float one_tem = 1;
    CUDA_RT_CALL(cudaMalloc((void**)&alpha_one_, sizeof(float)));
    cudaMemcpyAsync(alpha_one_, &one_tem, sizeof(float),
                    cudaMemcpyHostToDevice);
  } else if (type_ == nvinfer1::DataType::kHALF) {
    cudaDataType_t cudadataTypeIO = CUDA_R_16F;
    cudaDataType_t cudaDataTypeS = CUDA_R_16F;
    cublasComputeType_t cudaComputeType = CUBLAS_COMPUTE_16F;

    CUDA_RT_CALL(cublasLtMatrixLayoutCreate(
        &Adesc_, cudadataTypeIO, AopTranspose == CUBLAS_OP_N ? n_ : k_,
        AopTranspose == CUBLAS_OP_N ? k_ : n_,
        AopTranspose == CUBLAS_OP_N ? n_ : k_));
    CUDA_RT_CALL(cublasLtMatrixLayoutSetAttribute(
        Adesc_, CUBLASLT_MATRIX_LAYOUT_TYPE, &cudadataTypeIO,
        sizeof(cudadataTypeIO)));
    CUDA_RT_CALL(cublasLtMatrixLayoutSetAttribute(
        Adesc_, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &(batch_), sizeof(batch_)));
    CUDA_RT_CALL(cublasLtMatrixLayoutSetAttribute(
        Adesc_, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &(stridea),
        sizeof(stridea)));

    CUDA_RT_CALL(cublasLtMatrixLayoutCreate(
        &Bdesc_, cudadataTypeIO, BopTranspose == CUBLAS_OP_N ? k_ : m_,
        BopTranspose == CUBLAS_OP_N ? m_ : k_,
        BopTranspose == CUBLAS_OP_N ? k_ : m_));
    CUDA_RT_CALL(cublasLtMatrixLayoutSetAttribute(
        Bdesc_, CUBLASLT_MATRIX_LAYOUT_TYPE, &cudadataTypeIO,
        sizeof(cudadataTypeIO)));
    CUDA_RT_CALL(cublasLtMatrixLayoutSetAttribute(
        Bdesc_, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &(batch_), sizeof(batch_)));
    CUDA_RT_CALL(cublasLtMatrixLayoutSetAttribute(
        Bdesc_, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &(strideb),
        sizeof(strideb)));

    CUDA_RT_CALL(
        cublasLtMatrixLayoutCreate(&Cdesc_, cudadataTypeIO, n_, m_, n_));
    CUDA_RT_CALL(cublasLtMatrixLayoutSetAttribute(
        Cdesc_, CUBLASLT_MATRIX_LAYOUT_TYPE, &cudadataTypeIO,
        sizeof(cudadataTypeIO)));
    CUDA_RT_CALL(cublasLtMatrixLayoutSetAttribute(
        Cdesc_, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &(batch_), sizeof(batch_)));
    CUDA_RT_CALL(cublasLtMatrixLayoutSetAttribute(
        Cdesc_, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &(stridec),
        sizeof(stridec)));

    cublasLtPointerMode_t matmul_model = CUBLASLT_POINTER_MODE_DEVICE;
    CUDA_RT_CALL(
        cublasLtMatmulDescCreate(&matmulDesc_, cudaComputeType, cudaDataTypeS));
    CUDA_RT_CALL(
        cublasLtMatmulDescSetAttribute(matmulDesc_, CUBLASLT_MATMUL_DESC_TRANSA,
                                       &AopTranspose, sizeof(AopTranspose)));
    CUDA_RT_CALL(
        cublasLtMatmulDescSetAttribute(matmulDesc_, CUBLASLT_MATMUL_DESC_TRANSB,
                                       &BopTranspose, sizeof(BopTranspose)));
    CUDA_RT_CALL(cublasLtMatmulDescSetAttribute(
        matmulDesc_, CUBLASLT_MATMUL_DESC_POINTER_MODE, &matmul_model,
        sizeof(matmul_model)));

    half alpha_tem = static_cast<half>(alpha_);
    CUDA_RT_CALL(cudaMalloc((void**)&alpha_scale_, sizeof(half)));
    cudaMemcpyAsync(alpha_scale_, &alpha_tem, sizeof(half),
                    cudaMemcpyHostToDevice);
    half zero_tem = 0;
    CUDA_RT_CALL(cudaMalloc((void**)&alpha_zero_, sizeof(half)));
    cudaMemcpyAsync(alpha_zero_, &zero_tem, sizeof(half),
                    cudaMemcpyHostToDevice);
    float one_tem = 1;
    CUDA_RT_CALL(cudaMalloc((void**)&alpha_one_, sizeof(float)));
    cudaMemcpyAsync(alpha_one_, &one_tem, sizeof(float),
                    cudaMemcpyHostToDevice);
  } else {
    cudaDataType_t cudadataTypeIO = CUDA_R_32F;
    cudaDataType_t cudaDataTypeS = CUDA_R_32F;
    cublasComputeType_t cudaComputeType = CUBLAS_COMPUTE_32F_FAST_16F;

    CUDA_RT_CALL(cublasLtMatrixLayoutCreate(
        &Adesc_, cudadataTypeIO, AopTranspose == CUBLAS_OP_N ? n_ : k_,
        AopTranspose == CUBLAS_OP_N ? k_ : n_,
        AopTranspose == CUBLAS_OP_N ? n_ : k_));
    CUDA_RT_CALL(cublasLtMatrixLayoutSetAttribute(
        Adesc_, CUBLASLT_MATRIX_LAYOUT_TYPE, &cudadataTypeIO,
        sizeof(cudadataTypeIO)));
    CUDA_RT_CALL(cublasLtMatrixLayoutSetAttribute(
        Adesc_, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &(batch_), sizeof(batch_)));
    CUDA_RT_CALL(cublasLtMatrixLayoutSetAttribute(
        Adesc_, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &(stridea),
        sizeof(stridea)));

    CUDA_RT_CALL(cublasLtMatrixLayoutCreate(
        &Bdesc_, cudadataTypeIO, BopTranspose == CUBLAS_OP_N ? k_ : m_,
        BopTranspose == CUBLAS_OP_N ? m_ : k_,
        BopTranspose == CUBLAS_OP_N ? k_ : m_));
    CUDA_RT_CALL(cublasLtMatrixLayoutSetAttribute(
        Bdesc_, CUBLASLT_MATRIX_LAYOUT_TYPE, &cudadataTypeIO,
        sizeof(cudadataTypeIO)));
    CUDA_RT_CALL(cublasLtMatrixLayoutSetAttribute(
        Bdesc_, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &(batch_), sizeof(batch_)));
    CUDA_RT_CALL(cublasLtMatrixLayoutSetAttribute(
        Bdesc_, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &(strideb),
        sizeof(strideb)));

    CUDA_RT_CALL(
        cublasLtMatrixLayoutCreate(&Cdesc_, cudadataTypeIO, n_, m_, n_));
    CUDA_RT_CALL(cublasLtMatrixLayoutSetAttribute(
        Cdesc_, CUBLASLT_MATRIX_LAYOUT_TYPE, &cudadataTypeIO,
        sizeof(cudadataTypeIO)));
    CUDA_RT_CALL(cublasLtMatrixLayoutSetAttribute(
        Cdesc_, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &(batch_), sizeof(batch_)));
    CUDA_RT_CALL(cublasLtMatrixLayoutSetAttribute(
        Cdesc_, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &(stridec),
        sizeof(stridec)));

    cublasLtPointerMode_t matmul_model = CUBLASLT_POINTER_MODE_DEVICE;
    CUDA_RT_CALL(
        cublasLtMatmulDescCreate(&matmulDesc_, cudaComputeType, cudaDataTypeS));
    CUDA_RT_CALL(
        cublasLtMatmulDescSetAttribute(matmulDesc_, CUBLASLT_MATMUL_DESC_TRANSA,
                                       &AopTranspose, sizeof(AopTranspose)));
    CUDA_RT_CALL(
        cublasLtMatmulDescSetAttribute(matmulDesc_, CUBLASLT_MATMUL_DESC_TRANSB,
                                       &BopTranspose, sizeof(BopTranspose)));
    CUDA_RT_CALL(cublasLtMatmulDescSetAttribute(
        matmulDesc_, CUBLASLT_MATMUL_DESC_POINTER_MODE, &matmul_model,
        sizeof(matmul_model)));

    float alpha_tem = alpha_;
    CUDA_RT_CALL(cudaMalloc((void**)&alpha_scale_, sizeof(float)));
    cudaMemcpyAsync(alpha_scale_, &alpha_tem, sizeof(float),
                    cudaMemcpyHostToDevice);
    float zero_tem = 0;
    CUDA_RT_CALL(cudaMalloc((void**)&alpha_zero_, sizeof(float)));
    cudaMemcpyAsync(alpha_zero_, &zero_tem, sizeof(float),
                    cudaMemcpyHostToDevice);
    float one_tem = 1;
    CUDA_RT_CALL(cudaMalloc((void**)&alpha_one_, sizeof(float)));
    cudaMemcpyAsync(alpha_one_, &one_tem, sizeof(float),
                    cudaMemcpyHostToDevice);
  }
}

void MatmulPlugin::attachToContext(
    cudnnContext* cudnnContext, cublasContext* cublasContext,
    nvinfer1::IGpuAllocator* gpuAllocator) TRT_NOEXCEPT {
  cublasLtCreate(&cublas_);
}

void MatmulPlugin::detachFromContext() TRT_NOEXCEPT {
  cublasLtDestroy(cublas_);
}

void MatmulPlugin::terminate() TRT_NOEXCEPT {
  std::cout << "terminate(): this: " << this << std::endl;
  /*
   if(alpha_scale_){
     cudaFree((void *)alpha_scale_);
     alpha_scale_ = nullptr;
   }
   if(alpha_zero_){
     cudaFree((void *)alpha_zero_);
     alpha_zero_ = nullptr;
   }
   if(alpha_one_){
     cudaFree((void *)alpha_one_);
     alpha_one_ = nullptr;
   }
   if(Atransform_){
     cudaFree((void *)Atransform_);
     Atransform_ = nullptr;
   }
   if(Btransform_){
     cudaFree((void *)Btransform_);
     Btransform_ = nullptr;
   }
   if(Ctransform_){
     cudaFree((void *)Ctransform_);
     Ctransform_ = nullptr;
   }   */
}

int MatmulPlugin::enqueue(int batchSize, const void* const* inputs,
#if IS_TRT_VERSION_LT(8000)
                          void** outputs, void* workspace,
                          cudaStream_t stream) {
#else
                          void* const* outputs, void* workspace,
                          cudaStream_t stream) TRT_NOEXCEPT {
#endif
  std::cout << "enqueue(): " << this << std::endl;
  if (type_ == nvinfer1::DataType::kINT8) {
    std::cout << "type_: kINT8" << std::endl;
  } else if (type_ == nvinfer1::DataType::kFLOAT) {
    std::cout << "type_: kFLOAT" << std::endl;
  } else if (type_ == nvinfer1::DataType::kHALF) {
    std::cout << "type_: kHALF" << std::endl;
  }

  if (type_ == nvinfer1::DataType::kINT8) {
    const int8_t* B = static_cast<const int8_t*>(inputs[0]);
    const int8_t* A = static_cast<const int8_t*>(inputs[1]);
    int8_t* C = static_cast<int8_t*>(outputs[0]);
    Ltgemm_int8_linear(
        cublas_, A, Adesc_, Atransform_, AtransformDesc_, transA_, B, Bdesc_,
        Btransform_, BtransformDesc_, transB_, C, Cdesc_, Ctransform_,
        CtransformDesc_, transformDescT_, transformDescN_, matmulDesc_,
        alpha_scale_, alpha_zero_, alpha_one_, workspace, stream);
  } else if (type_ == nvinfer1::DataType::kFLOAT) {
    const float* B = static_cast<const float*>(inputs[0]);
    const float* A = static_cast<const float*>(inputs[1]);
    float* C = static_cast<float*>(outputs[0]);
    Ltgemm_fp32_linear(cublas_, A, Adesc_, B, Bdesc_, C, Cdesc_, matmulDesc_,
                       alpha_scale_, alpha_zero_, workspace, stream);
  } else if (type_ == nvinfer1::DataType::kHALF) {
    const half* B = static_cast<const half*>(inputs[0]);
    const half* A = static_cast<const half*>(inputs[1]);
    half* C = static_cast<half*>(outputs[0]);
    Ltgemm_fp16_linear(cublas_, A, Adesc_, B, Bdesc_, C, Cdesc_, matmulDesc_,
                       alpha_scale_, alpha_zero_, workspace, stream);
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
    return (inOut[pos].type == nvinfer1::DataType::kHALF ||
            inOut[pos].type == nvinfer1::DataType::kFLOAT ||
            inOut[pos].type == nvinfer1::DataType::kINT8) &&
           inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
  } else {
    return inOut[pos].type == inOut[0].type &&
           inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
  }
}

void MatmulPluginDynamic::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* outputs,
    int nbOutputs) TRT_NOEXCEPT {
  float inscale_0 = inputs[0].desc.scale;
  float inscale_1 = inputs[1].desc.scale;
  float outscale = outputs[0].desc.scale;
  type_ = inputs[0].desc.type;
  uint64_t m_max, n_max, k_max;
  if (transB_) {
    m_max = inputs[0].max.d[inputs[0].max.nbDims - 1];
    k_max = inputs[0].max.d[inputs[0].max.nbDims - 2];
  } else {
    m_max = inputs[0].max.d[inputs[0].max.nbDims - 2];
    k_max = inputs[0].max.d[inputs[0].max.nbDims - 1];
  }
  if (transA_) {
    n_max = inputs[1].max.d[inputs[1].max.nbDims - 2];
  } else {
    n_max = inputs[1].max.d[inputs[1].max.nbDims - 1];
  }

  int const ldatransform = 32 * n_max;
  int const ldbtransform = 32 * ((m_max + 8 - 1) / 8 * 8);
  int const ldctransform = 32 * n_max;
  CUDA_RT_CALL(cudaMalloc(
      (void**)&Atransform_,
      sizeof(int8_t) * ((k_max + 32 - 1) / 32 * 32) / 32 * ldatransform));
  CUDA_RT_CALL(cudaMalloc(
      (void**)&Btransform_,
      sizeof(int8_t) * ((k_max + 32 - 1) / 32 * 32) / 32 * ldbtransform));
  CUDA_RT_CALL(cudaMalloc(
      (void**)&Ctransform_,
      sizeof(int8_t) * ((m_max + 32 - 1) / 32 * 32) / 32 * ldctransform));

  if (type_ == nvinfer1::DataType::kINT8) {
    float alpha_tem[n_max];
    for (int i = 0; i < n_max; i++) {
      alpha_tem[i] = alpha_ * inscale_0 * inscale_1 / outscale;
    }
    CUDA_RT_CALL(cudaMalloc((void**)&alpha_scale_, n_max * sizeof(float)));
    cudaMemcpyAsync(alpha_scale_, alpha_tem, n_max * sizeof(float),
                    cudaMemcpyHostToDevice);
    float zero_tem = 0;
    CUDA_RT_CALL(cudaMalloc((void**)&alpha_zero_, sizeof(float)));
    cudaMemcpyAsync(alpha_zero_, &zero_tem, sizeof(float),
                    cudaMemcpyHostToDevice);
    float one_tem = 1;
    CUDA_RT_CALL(cudaMalloc((void**)&alpha_one_, sizeof(float)));
    cudaMemcpyAsync(alpha_one_, &one_tem, sizeof(float),
                    cudaMemcpyHostToDevice);
  } else if (type_ == nvinfer1::DataType::kHALF) {
    half alpha_tem = static_cast<half>(alpha_);
    CUDA_RT_CALL(cudaMalloc((void**)&alpha_scale_, sizeof(half)));
    cudaMemcpyAsync(alpha_scale_, &alpha_tem, sizeof(half),
                    cudaMemcpyHostToDevice);
    half zero_tem = 0;
    CUDA_RT_CALL(cudaMalloc((void**)&alpha_zero_, sizeof(half)));
    cudaMemcpyAsync(alpha_zero_, &zero_tem, sizeof(half),
                    cudaMemcpyHostToDevice);
    float one_tem = 1;
    CUDA_RT_CALL(cudaMalloc((void**)&alpha_one_, sizeof(float)));
    cudaMemcpyAsync(alpha_one_, &one_tem, sizeof(float),
                    cudaMemcpyHostToDevice);
  } else {
    float alpha_tem = alpha_;
    CUDA_RT_CALL(cudaMalloc((void**)&alpha_scale_, sizeof(float)));
    cudaMemcpyAsync(alpha_scale_, &alpha_tem, sizeof(float),
                    cudaMemcpyHostToDevice);
    float zero_tem = 0;
    CUDA_RT_CALL(cudaMalloc((void**)&alpha_zero_, sizeof(float)));
    cudaMemcpyAsync(alpha_zero_, &zero_tem, sizeof(float),
                    cudaMemcpyHostToDevice);
    float one_tem = 1;
    CUDA_RT_CALL(cudaMalloc((void**)&alpha_one_, sizeof(float)));
    cudaMemcpyAsync(alpha_one_, &one_tem, sizeof(float),
                    cudaMemcpyHostToDevice);
  }
}

void MatmulPluginDynamic::attachToContext(
    cudnnContext* cudnnContext, cublasContext* cublasContext,
    nvinfer1::IGpuAllocator* gpuAllocator) TRT_NOEXCEPT {
  cublasLtCreate(&cublas_);
}

void MatmulPluginDynamic::detachFromContext() TRT_NOEXCEPT {
  cublasLtDestroy(cublas_);
}

void MatmulPluginDynamic::terminate() TRT_NOEXCEPT {
  std::cout << "terminate: " << this << std::endl;
  /*if(alpha_scale_){
    cudaFree((void *)alpha_scale_);
    alpha_scale_ = nullptr;
  }
  if(alpha_zero_){
    cudaFree((void *)alpha_zero_);
    alpha_zero_ = nullptr;
  }
  if(alpha_one_){
    cudaFree((void *)alpha_one_);
    alpha_one_ = nullptr;
  }
  if(Atransform_){
    cudaFree((void *)Atransform_);
    Atransform_ = nullptr;
  }
  if(Btransform_){
    cudaFree((void *)Btransform_);
    Btransform_ = nullptr;
  }
  if(Ctransform_){
    cudaFree((void *)Ctransform_);
    Ctransform_ = nullptr;
  } */
}

int MatmulPluginDynamic::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
                                 const nvinfer1::PluginTensorDesc* outputDesc,
                                 const void* const* inputs,
                                 void* const* outputs, void* workspace,
                                 cudaStream_t stream) TRT_NOEXCEPT {
  std::cout << "enqueue(): " << this << std::endl;
  if (type_ == nvinfer1::DataType::kINT8) {
    std::cout << "type_: kINT8" << std::endl;
  } else if (type_ == nvinfer1::DataType::kFLOAT) {
    std::cout << "type_: kFLOAT" << std::endl;
  } else if (type_ == nvinfer1::DataType::kHALF) {
    std::cout << "type_: kHALF" << std::endl;
  }
  const auto Input0Desc = inputDesc[0];
  const auto Input1Desc = inputDesc[1];
  uint64_t m, n, k;
  if (transB_) {
    m = Input0Desc.dims.d[Input0Desc.dims.nbDims - 1];
    k = Input0Desc.dims.d[Input0Desc.dims.nbDims - 2];
  } else {
    m = Input0Desc.dims.d[Input0Desc.dims.nbDims - 2];
    k = Input0Desc.dims.d[Input0Desc.dims.nbDims - 1];
  }
  if (transA_) {
    n = Input1Desc.dims.d[Input1Desc.dims.nbDims - 2];
  } else {
    n = Input1Desc.dims.d[Input1Desc.dims.nbDims - 1];
  }

  int batch = 1;
  for (int i = 0; i < Input0Desc.dims.nbDims - 2; i++) {
    batch *= Input0Desc.dims.d[i];
  }
  int const ldatransform = 32 * n;
  int const ldbtransform = 32 * ((m + 8 - 1) / 8 * 8);
  int const ldctransform = 32 * n;

  int64_t stridea = k * n;
  int64_t strideb = k * m;
  int64_t stridec = m * n;

  cublasOperation_t AopTranspose, BopTranspose;
  if (transA_) {
    AopTranspose = CUBLAS_OP_T;
  } else {
    AopTranspose = CUBLAS_OP_N;
  }
  if (transB_) {
    BopTranspose = CUBLAS_OP_T;
  } else {
    BopTranspose = CUBLAS_OP_N;
  }

  cublasLtMatrixLayout_t Adesc{nullptr}, Bdesc{nullptr}, Cdesc{nullptr};
  cublasLtMatmulDesc_t matmulDesc{nullptr};
  cublasLtMatrixLayout_t AtransformDesc{nullptr}, BtransformDesc{nullptr},
      CtransformDesc{nullptr};
  int8_t *Atransform{nullptr}, *Btransform{nullptr}, *Ctransform{nullptr};
  cublasLtMatrixTransformDesc_t transformDescT{nullptr},
      transformDescN{nullptr};
  if (type_ == nvinfer1::DataType::kINT8) {
    cudaDataType_t cudadataTypeIO = CUDA_R_8I;
    cudaDataType_t cudaDataTypeS = CUDA_R_32F;
    cublasComputeType_t cudaComputeType = CUBLAS_COMPUTE_32I;
    cublasLtOrder_t COL32 = CUBLASLT_ORDER_COL32;
    cublasLtOrder_t COL4_4R2_8C = CUBLASLT_ORDER_COL4_4R2_8C;

    CUDA_RT_CALL(cublasLtMatrixLayoutCreate(
        &Adesc, cudadataTypeIO, AopTranspose == CUBLAS_OP_N ? n : k,
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

    CUDA_RT_CALL(cublasLtMatrixLayoutCreate(
        &Bdesc, cudadataTypeIO, BopTranspose == CUBLAS_OP_N ? k : m,
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

    CUDA_RT_CALL(cublasLtMatrixLayoutCreate(&AtransformDesc, cudadataTypeIO, n,
                                            k, ldatransform));
    CUDA_RT_CALL(cublasLtMatrixLayoutSetAttribute(
        AtransformDesc, CUBLASLT_MATRIX_LAYOUT_TYPE, &cudadataTypeIO,
        sizeof(cudadataTypeIO)));
    CUDA_RT_CALL(cublasLtMatrixLayoutSetAttribute(
        AtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &COL32, sizeof(COL32)));

    CUDA_RT_CALL(cublasLtMatrixLayoutCreate(&BtransformDesc, cudadataTypeIO, m,
                                            k, ldbtransform));
    CUDA_RT_CALL(cublasLtMatrixLayoutSetAttribute(
        BtransformDesc, CUBLASLT_MATRIX_LAYOUT_TYPE, &cudadataTypeIO,
        sizeof(cudadataTypeIO)));
    CUDA_RT_CALL(cublasLtMatrixLayoutSetAttribute(
        BtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &COL4_4R2_8C,
        sizeof(COL4_4R2_8C)));

    CUDA_RT_CALL(cublasLtMatrixLayoutCreate(&CtransformDesc, cudadataTypeIO, n,
                                            m, ldctransform));
    CUDA_RT_CALL(cublasLtMatrixLayoutSetAttribute(
        CtransformDesc, CUBLASLT_MATRIX_LAYOUT_TYPE, &cudadataTypeIO,
        sizeof(cudadataTypeIO)));
    CUDA_RT_CALL(cublasLtMatrixLayoutSetAttribute(
        CtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &COL32, sizeof(COL32)));

    cublasOperation_t Transpose = CUBLAS_OP_T;
    cublasLtPointerMode_t transform_model = CUBLASLT_POINTER_MODE_DEVICE;
    CUDA_RT_CALL(
        cublasLtMatrixTransformDescCreate(&transformDescT, cudaDataTypeS));
    CUDA_RT_CALL(cublasLtMatrixTransformDescSetAttribute(
        transformDescT, CUBLASLT_MATRIX_TRANSFORM_DESC_SCALE_TYPE,
        &cudaDataTypeS, sizeof(cudaDataTypeS)));
    CUDA_RT_CALL(cublasLtMatrixTransformDescSetAttribute(
        transformDescT, CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &Transpose,
        sizeof(Transpose)));
    CUDA_RT_CALL(cublasLtMatrixTransformDescSetAttribute(
        transformDescT, CUBLASLT_MATRIX_TRANSFORM_DESC_POINTER_MODE,
        &transform_model, sizeof(transform_model)));

    CUDA_RT_CALL(
        cublasLtMatrixTransformDescCreate(&transformDescN, cudaDataTypeS));
    CUDA_RT_CALL(cublasLtMatrixTransformDescSetAttribute(
        transformDescN, CUBLASLT_MATRIX_TRANSFORM_DESC_SCALE_TYPE,
        &cudaDataTypeS, sizeof(cudaDataTypeS)));
    CUDA_RT_CALL(cublasLtMatrixTransformDescSetAttribute(
        transformDescN, CUBLASLT_MATRIX_TRANSFORM_DESC_POINTER_MODE,
        &transform_model, sizeof(transform_model)));

    cublasOperation_t ATranspose = CUBLAS_OP_N, BTranspose = CUBLAS_OP_T;
    cublasLtPointerMode_t matmul_model =
        CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_ZERO;
    CUDA_RT_CALL(
        cublasLtMatmulDescCreate(&matmulDesc, cudaComputeType, cudaDataTypeS));
    CUDA_RT_CALL(
        cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA,
                                       &ATranspose, sizeof(ATranspose)));
    CUDA_RT_CALL(
        cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB,
                                       &BTranspose, sizeof(BTranspose)));
    CUDA_RT_CALL(cublasLtMatmulDescSetAttribute(
        matmulDesc, CUBLASLT_MATMUL_DESC_POINTER_MODE, &matmul_model,
        sizeof(matmul_model)));

    const int8_t* B = static_cast<const int8_t*>(inputs[0]);
    const int8_t* A = static_cast<const int8_t*>(inputs[1]);
    int8_t* C = static_cast<int8_t*>(outputs[0]);
    Ltgemm_int8_linear(cublas_, A, Adesc, Atransform_, AtransformDesc, transA_,
                       B, Bdesc, Btransform_, BtransformDesc, transB_, C, Cdesc,
                       Ctransform_, CtransformDesc, transformDescT,
                       transformDescN, matmulDesc, alpha_scale_, alpha_zero_,
                       alpha_one_, workspace, stream);
  } else if (type_ == nvinfer1::DataType::kHALF) {
    cudaDataType_t cudadataTypeIO = CUDA_R_16F;
    cudaDataType_t cudaDataTypeS = CUDA_R_16F;
    cublasComputeType_t cudaComputeType = CUBLAS_COMPUTE_16F;

    CUDA_RT_CALL(cublasLtMatrixLayoutCreate(
        &Adesc, cudadataTypeIO, AopTranspose == CUBLAS_OP_N ? n : k,
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

    CUDA_RT_CALL(cublasLtMatrixLayoutCreate(
        &Bdesc, cudadataTypeIO, BopTranspose == CUBLAS_OP_N ? k : m,
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

    cublasLtPointerMode_t matmul_model = CUBLASLT_POINTER_MODE_DEVICE;
    CUDA_RT_CALL(
        cublasLtMatmulDescCreate(&matmulDesc, cudaComputeType, cudaDataTypeS));
    CUDA_RT_CALL(
        cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA,
                                       &AopTranspose, sizeof(AopTranspose)));
    CUDA_RT_CALL(
        cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB,
                                       &BopTranspose, sizeof(BopTranspose)));
    CUDA_RT_CALL(cublasLtMatmulDescSetAttribute(
        matmulDesc, CUBLASLT_MATMUL_DESC_POINTER_MODE, &matmul_model,
        sizeof(matmul_model)));

    const half* B = static_cast<const half*>(inputs[0]);
    const half* A = static_cast<const half*>(inputs[1]);
    half* C = static_cast<half*>(outputs[0]);
    Ltgemm_fp16_linear(cublas_, A, Adesc, B, Bdesc, C, Cdesc, matmulDesc,
                       alpha_scale_, alpha_zero_, workspace, stream);
  } else {
    cudaDataType_t cudadataTypeIO = CUDA_R_32F;
    cudaDataType_t cudaDataTypeS = CUDA_R_32F;
    cublasComputeType_t cudaComputeType = CUBLAS_COMPUTE_32F_FAST_16F;

    CUDA_RT_CALL(cublasLtMatrixLayoutCreate(
        &Adesc, cudadataTypeIO, AopTranspose == CUBLAS_OP_N ? n : k,
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

    CUDA_RT_CALL(cublasLtMatrixLayoutCreate(
        &Bdesc, cudadataTypeIO, BopTranspose == CUBLAS_OP_N ? k : m,
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

    cublasLtPointerMode_t matmul_model = CUBLASLT_POINTER_MODE_DEVICE;
    CUDA_RT_CALL(
        cublasLtMatmulDescCreate(&matmulDesc, cudaComputeType, cudaDataTypeS));
    CUDA_RT_CALL(
        cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA,
                                       &AopTranspose, sizeof(AopTranspose)));
    CUDA_RT_CALL(
        cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB,
                                       &BopTranspose, sizeof(BopTranspose)));
    CUDA_RT_CALL(cublasLtMatmulDescSetAttribute(
        matmulDesc, CUBLASLT_MATMUL_DESC_POINTER_MODE, &matmul_model,
        sizeof(matmul_model)));

    const float* B = static_cast<const float*>(inputs[0]);
    const float* A = static_cast<const float*>(inputs[1]);
    float* C = static_cast<float*>(outputs[0]);
    Ltgemm_fp32_linear(cublas_, A, Adesc, B, Bdesc, C, Cdesc, matmulDesc,
                       alpha_scale_, alpha_zero_, workspace, stream);
  }
  return cudaGetLastError() != cudaSuccess;
}

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
