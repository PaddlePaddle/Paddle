// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
// SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION &
// AFFILIATES. All rights reserved.
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

#ifndef PADDLE_FLUID_INFERENCE_TENSORRT_PLUGIN_COMMON_BERTCOMMON_H_
#define PADDLE_FLUID_INFERENCE_TENSORRT_PLUGIN_COMMON_BERTCOMMON_H_

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cassert>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>
#include "NvInfer.h"
#include "NvInferRuntimeCommon.h"
#include "paddle/fluid/inference/tensorrt/plugin/common/plugin.h"

#define TRT_UNUSED (void)
#define BERT_PRINT_DEBUG_MSG 0
#if BERT_PRINT_DEBUG_MSG
#define TRANSFORMER_DEBUG_MSG(msg) (gLogVerbose << (msg) << std::endl)
#define BERT_DEBUG_VALUE(key, value) (gLogVerbose << key << value << std::endl)
#else
#define TRANSFORMER_DEBUG_MSG(msg) TRT_UNUSED(msg)
#define BERT_DEBUG_VALUE(key, value) \
  TRT_UNUSED(key);                   \
  TRT_UNUSED(value)
#endif
using half = __half;
constexpr uint32_t BDIM = 1;  // batch dimension
constexpr uint32_t SDIM = 0;  // seq len dimension
constexpr uint32_t HDIM = 2;  // hidden dimension
constexpr int32_t kSM_53 = 53;
constexpr int32_t kSM_70 = 70;
constexpr int32_t kSM_72 = 72;
constexpr int32_t kSM_75 = 75;
constexpr int32_t kSM_80 = 80;
constexpr int32_t kSM_86 = 86;
constexpr int32_t kSM_87 = 87;
constexpr size_t threadsPerCta128 = 2 * 2 * 32;
constexpr size_t threadsPerCta384 = 1 * 8 * 32;

// The number of xmmas in the M dimension. We use one uint32_t per XMMA in the M
// dimension: (s + 16*warps_m - 1) / (16*warps_m);
constexpr size_t xmmasM128 = 4;
constexpr size_t xmmasM384 = 24;

// Packed mask size per batch. Layout is XMMAS_M * THREADS_PER_CTA.
constexpr size_t unfusedMaskSize = 1;
constexpr size_t packedMaskSize64 = xmmasM128 * threadsPerCta128;
constexpr size_t packedMaskSize96 = xmmasM128 * threadsPerCta128;
constexpr size_t packedMaskSize128 = xmmasM128 * threadsPerCta128;
constexpr size_t packedMaskSize384 = xmmasM384 * threadsPerCta384;

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

inline uint32_t getElementSize(nvinfer1::DataType t) noexcept {
  switch (t) {
    case nvinfer1::DataType::kINT32:
      return 4;
    case nvinfer1::DataType::kFLOAT:
      return 4;
    case nvinfer1::DataType::kHALF:
      return 2;
    case nvinfer1::DataType::kBOOL:
    case nvinfer1::DataType::kINT8:
      return 1;
  }
  return 0;
}

inline int64_t getWeightsSize(const nvinfer1::Weights& w,
                              nvinfer1::DataType type) {
  return w.count * getElementSize(type);
}

template <typename T>
inline void serFromDev(char** buffer, const T* data, size_t nbElem) {
  const size_t len = sizeof(T) * nbElem;
  cudaMemcpy(
      *buffer, static_cast<const void*>(data), len, cudaMemcpyDeviceToHost);
  *buffer += len;
}

template <typename T>
struct CudaDeleter {
  void operator()(T* buf) { cudaFree(buf); }
};

template <typename T>
using cuda_unique_ptr = std::unique_ptr<T, CudaDeleter<T>>;

template <typename T>
using cuda_shared_ptr = std::shared_ptr<T>;

template <typename T>
void make_cuda_shared(cuda_shared_ptr<T>* ptr, void* cudaMem) {
  ptr->reset(static_cast<T*>(cudaMem), CudaDeleter<T>());
}

struct WeightsWithOwnership : public nvinfer1::Weights {
  WeightsWithOwnership() {
    values = nullptr;
    count = 0;
  }
  ~WeightsWithOwnership() { operator delete[](const_cast<void*>(values)); }

  WeightsWithOwnership(const WeightsWithOwnership&) = delete;
  WeightsWithOwnership operator=(const WeightsWithOwnership&) = delete;
  WeightsWithOwnership(const WeightsWithOwnership&&) = delete;
  WeightsWithOwnership operator=(const WeightsWithOwnership&&) = delete;

  void convertAndCopy(const nvinfer1::Weights& src, nvinfer1::DataType type) {
    this->type = type;
    this->count = src.count;
    if (type == nvinfer1::DataType::kFLOAT) {
      auto destBuf = new float[src.count];
      this->values = destBuf;
      if (src.type == nvinfer1::DataType::kFLOAT) {
        TRANSFORMER_DEBUG_MSG("Float Weights(Host) => Float Array(Host)");
        std::copy_n(static_cast<const float*>(src.values), src.count, destBuf);
      } else {
        assert(src.type == nvinfer1::DataType::kHALF);
        TRANSFORMER_DEBUG_MSG("Half Weights(Host) => Float Array(Host)");
        const auto s = static_cast<const half*>(src.values);
        auto d = static_cast<float*>(const_cast<void*>(this->values));
        for (auto it = 0; it < src.count; it++) {
          d[it] = __half2float(s[it]);
        }
      }
    } else if (type == nvinfer1::DataType::kHALF) {
      auto destBuf = new half[src.count];
      this->values = destBuf;
      if (src.type == nvinfer1::DataType::kHALF) {
        TRANSFORMER_DEBUG_MSG("Half Weights(Host) => Half Array(Host)");
        std::copy_n(static_cast<const half*>(src.values), src.count, destBuf);
      } else {
        assert(src.type == nvinfer1::DataType::kFLOAT);
        TRANSFORMER_DEBUG_MSG("Float Weights(Host) => Half Array(Host)");
        const auto s = static_cast<const float*>(src.values);
        auto d = static_cast<half*>(const_cast<void*>(this->values));
        for (auto it = 0; it < src.count; it++) {
          d[it] = __float2half(s[it]);
        }
      }
    } else {
      throw std::runtime_error("Unsupported DataType specified for plugin.");
    }
  }

  void convertAndCopy(const char** srcBuf,
                      size_t count,
                      nvinfer1::DataType type) noexcept {
    this->type = type;
    this->count = count;
    const auto nbBytes = getWeightsSize(*this, type);
    auto destBuf = new char[nbBytes];
    this->values = destBuf;
    std::copy_n(*srcBuf, nbBytes, destBuf);
    *srcBuf += nbBytes;
  }
};

template <typename T>
inline void copyToDevice(WeightsWithOwnership* hostWeights,
                         size_t nbBytes,
                         cuda_unique_ptr<T>* cudaWeights) {
  if (hostWeights->values) {
    void* cudaMem{nullptr};
    cudaMalloc(&cudaMem, nbBytes);
    cudaMemcpy(cudaMem, hostWeights->values, nbBytes, cudaMemcpyHostToDevice);
    cudaWeights->reset(static_cast<T*>(cudaMem));
  }
}

inline nvinfer1::DataType fieldTypeToDataType(
    const nvinfer1::PluginFieldType ftype) {
  switch (ftype) {
    case nvinfer1::PluginFieldType::kFLOAT32: {
      TRANSFORMER_DEBUG_MSG("PluginFieldType is Float32");
      return nvinfer1::DataType::kFLOAT;
    }
    case nvinfer1::PluginFieldType::kFLOAT16: {
      TRANSFORMER_DEBUG_MSG("PluginFieldType is Float16");
      return nvinfer1::DataType::kHALF;
    }
    case nvinfer1::PluginFieldType::kINT32: {
      TRANSFORMER_DEBUG_MSG("PluginFieldType is Int32");
      return nvinfer1::DataType::kINT32;
    }
    case nvinfer1::PluginFieldType::kINT8: {
      TRANSFORMER_DEBUG_MSG("PluginFieldType is Int8");
      return nvinfer1::DataType::kINT8;
    }
    default:
      TRANSFORMER_DEBUG_MSG("PluginFieldType is Float32");
      return nvinfer1::DataType::kFLOAT;
  }
}

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

#endif  // PADDLE_FLUID_INFERENCE_TENSORRT_PLUGIN_COMMON_BERTCOMMON_H_
