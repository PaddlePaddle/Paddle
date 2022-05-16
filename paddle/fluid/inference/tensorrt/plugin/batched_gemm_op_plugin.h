/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//#include <NvInfer.h>
//#include <cassert>
//#include <cublas_v2.h>
//#include <iostream>
//#include <string>
//#include <vector>
#pragma once
#include <algorithm>
#include <string>
#include <vector>
#include "paddle/fluid/inference/tensorrt/engine.h"
#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin.h"
//#include "paddle/fluid/platform/dynload/cublasLt.h"
#include "paddle/fluid/platform/dynload/cublas.h"
#include "paddle/fluid/platform/enforce.h"

#define DEBUG
#ifdef DEBUG
    #define WHERE_AM_I()                               \
        do                                             \
        {                                              \
            printf("[%s]: this=%p\n", __func__, this); \
        } while (0);
#else
    #define WHERE_AM_I()
#endif // ifdef DEBUG

/*
inline void check(cublasStatus_t ret, int line)
{
    if (ret != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "cuBLAS Error: " << ret << ", line: " << line << std::endl;
    }
}

inline void check(cudaError_t ret, int line)
{
    if (ret != cudaSuccess)
    {
        std::cerr << "CUDA Error: " << cudaGetErrorString(ret) << ", line: " << line << std::endl;
    }
}

#define CHECK(_x) check((_x), __LINE__)
*/

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

class CuBLASBatchedGemmPlugin : public DynamicPluginTensorRT
{
private:
    const std::string name_;
    std::string       namespace_;

public:
    bool           bOwnWeight_ {false};
    bool           has_relu_ {false};
    int            nK_ {0}; // shape of the weight, B_{nK,nN}
    int            nN_ {0};
    int            batchcount_ {0};
    float *        pGPUWeight_ {nullptr};
    nvinfer1::Weights        weight_ {nvinfer1::DataType::kFLOAT, nullptr, 0};
    float *        pGPUBias_ {nullptr};
    nvinfer1::Weights        bias_ {nvinfer1::DataType::kFLOAT, nullptr, 0};
    cublasHandle_t handle_ {nullptr};

public:
    CuBLASBatchedGemmPlugin() = delete;
    CuBLASBatchedGemmPlugin(const std::string &name, nvinfer1::Weights weight, nvinfer1::Weights bias, int k, int n, int batchcount, bool has_relu, bool needDeepCopy);
    CuBLASBatchedGemmPlugin(const std::string &name, const void *buffer, size_t length);
    ~CuBLASBatchedGemmPlugin();

    // Method inherited from IPluginV2
    const char *getPluginType() const noexcept override;
    const char *getPluginVersion() const noexcept override;
    int32_t     getNbOutputs() const noexcept override;
    int32_t     initialize() noexcept override;
    void        terminate() noexcept override;
    size_t      getSerializationSize() const noexcept override;
    void        serialize(void *buffer) const noexcept override;
    void        destroy() noexcept override;
    void        setPluginNamespace(const char *pluginNamespace) noexcept override;
    const char *getPluginNamespace() const noexcept override;

    // Method inherited from IPluginV2Ext
    nvinfer1::DataType getOutputDataType(int32_t index, nvinfer1::DataType const *inputTypes, int32_t nbInputs) const noexcept override;
    //void     attachToContext(cudnnContext *contextCudnn, cublasContext *contextCublas, nvinfer1::IGpuAllocator *gpuAllocator) noexcept override;
    //void     detachFromContext() noexcept override;

    //Method inherited from IPluginV2DynamicExt
    nvinfer1::IPluginV2DynamicExt *clone() const noexcept override;
    nvinfer1::DimsExprs            getOutputDimensions(int32_t outputIndex, const nvinfer1::DimsExprs *inputs, int32_t nbInputs, nvinfer1::IExprBuilder &exprBuilder) noexcept override;
    bool                 supportsFormatCombination(int32_t pos, const nvinfer1::PluginTensorDesc *inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override;
    void                 configurePlugin(const nvinfer1::DynamicPluginTensorDesc *in, int32_t nbInputs, const nvinfer1::DynamicPluginTensorDesc *out, int32_t nbOutputs) noexcept override;
    size_t               getWorkspaceSize(const nvinfer1::PluginTensorDesc *inputs, int32_t nbInputs, const nvinfer1::PluginTensorDesc *outputs, int32_t nbOutputs) const noexcept override;
    int32_t              enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc, const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept override;
};

class CuBLASBatchedGemmPluginCreator : public nvinfer1::IPluginCreator
{
private:
    nvinfer1::PluginFieldCollection    fc_ {0, nullptr};
    std::vector<nvinfer1::PluginField> attr_;
    std::string                     namespace_;
    std::string plugin_name_;

public:
    CuBLASBatchedGemmPluginCreator() {}
    //~CuBLASBatchedGemmPluginCreator() {}
    const char *                 getPluginName() const noexcept override
    {
        return "batchedgemmplugin";
    }
    const char *                 getPluginVersion() const noexcept override
    {
        return "1";
    }
    const nvinfer1::PluginFieldCollection *getFieldNames() noexcept override
    {
        return &fc_;
    }
    nvinfer1::IPluginV2 *                  createPlugin(const char *name, const nvinfer1::PluginFieldCollection *fc) noexcept override
    {
        return nullptr;
    }
    nvinfer1::IPluginV2 *                  deserializePlugin(const char *name, const void *serialData, size_t serialLength) noexcept override
    {
        WHERE_AM_I()
        return new CuBLASBatchedGemmPlugin(name, serialData, serialLength);
    }
    void                         setPluginNamespace(const char *pluginNamespace) noexcept override
    {
        WHERE_AM_I()
        namespace_ = pluginNamespace;
    }
    const char *                 getPluginNamespace() const noexcept override
    {
        return namespace_.c_str();
    }
};

REGISTER_TRT_PLUGIN_V2(CuBLASBatchedGemmPluginCreator);
}//plugin
}//tensorrt
}//inference
}//paddle
