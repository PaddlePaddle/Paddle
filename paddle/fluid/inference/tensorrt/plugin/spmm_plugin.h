/* Copyright (c) 2022, PaddlePaddle Authors, NVIDIA CORPORATION. All rights
reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#pragma once

#include <algorithm>
#include <cassert>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "paddle/fluid/inference/tensorrt/engine.h"
#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin.h"
#include "paddle/phi/backends/dynload/cusparseLt.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

class SpmmPluginDynamic : public nvinfer1::IPluginV2DynamicExt {
 public:
  enum class Activation { kNone, kRelu, kGelu };
  SpmmPluginDynamic(const std::string& name,
                    const nvinfer1::DataType precision,
                    const int out_dim,
                    const nvinfer1::Weights& weight,
                    const nvinfer1::Weights& bias,
                    Activation activation);
  // The second constructor is for clone member function
  SpmmPluginDynamic(const std::string& name,
                    const nvinfer1::DataType precision,
                    const int out_dim,
                    const int k,
                    const void* weight,
                    size_t compressed_size,
                    const void* bias,
                    bool is_configured,
                    const int m_max,
                    const int optim_alg,
                    Activation activation);
  SpmmPluginDynamic(const std::string name, const void* data, size_t length);
  SpmmPluginDynamic() = delete;
  nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
  nvinfer1::DimsExprs getOutputDimensions(
      int outputIndex,
      const nvinfer1::DimsExprs* inputs,
      int nbInputs,
      nvinfer1::IExprBuilder& exprBuilder) noexcept override;
  bool supportsFormatCombination(int pos,
                                 const nvinfer1::PluginTensorDesc* inOut,
                                 int nbInputs,
                                 int nbOutputs) noexcept override;
  void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in,
                       int nbInputs,
                       const nvinfer1::DynamicPluginTensorDesc* out,
                       int nbOutputs) noexcept override;
  size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs,
                          int nbInputs,
                          const nvinfer1::PluginTensorDesc* outputs,
                          int nbOutputs) const noexcept override;
  int enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
              const nvinfer1::PluginTensorDesc* outputDesc,
              const void* const* inputs,
              void* const* outputs,
              void* workspace,
              cudaStream_t stream) noexcept override;

  nvinfer1::DataType getOutputDataType(int index,
                                       const nvinfer1::DataType* inputTypes,
                                       int nbInputs) const noexcept override;
  const char* getPluginType() const noexcept override;
  const char* getPluginVersion() const noexcept override;
  int getNbOutputs() const noexcept override;
  int initialize() noexcept override;
  void terminate() noexcept override;
  size_t getSerializationSize() const noexcept override;
  void serialize(void* buffer) const noexcept override;
  void destroy() noexcept override;
  void setPluginNamespace(const char* pluginNamespace) noexcept override;
  const char* getPluginNamespace() const noexcept override;

 private:
  struct cusparseLtContext {
    cusparseLtHandle_t handle;
    cusparseLtMatDescriptor_t matA;
    cusparseLtMatDescriptor_t matB;
    cusparseLtMatDescriptor_t matC;
    cusparseLtMatmulDescriptor_t matmul;
    cusparseLtMatmulAlgSelection_t alg_sel;
    cusparseLtMatmulPlan_t plan;
    cusparseLtContext();
    ~cusparseLtContext();
    size_t workspace_size{0};
    bool is_initialized{false};
    int activation{0};
    float relu_upper_bound{0};
    float relu_threshold{0};
    void init(int m,
              int n,
              int k,
              cudaDataType_t type,
              void* bias_ptr,
              SpmmPluginDynamic::Activation activation);
    void setAlgo(int id);
    void destroy();
    void compressMatB(int n,
                      int k,
                      cudaDataType_t type,
                      void* src,
                      void** dest,
                      size_t* compressed_size);
  };  // struct SpmmPluginDynamic::cusparseLtContext
  const std::string layer_name_;
  std::string namespace_;
  nvinfer1::DataType precision_;
  size_t precision_size_;
  size_t
      element_size_;  // size of weight (float if INT8 or FLOAT; half if HALF)
  int out_dim_;
  int k_;
  int m_max_;
  bool is_configured_;  // already get m, scale bias, and search the optim alg
                        // or not
  int optim_alg_;       // the index of optimal algorithm
  float weight_scale_;  // record the weight scale from constructor
  void* weight_compressed_;      // host compressed weight
  void* weight_compressed_dev_;  //  device compressed weight
  std::shared_ptr<void>
      weight_compressed_dev_global_;  // shared pointer to the
                                      // device compressed weight
  size_t compressed_size_;            // size of compressed weight
  bool has_bias_;                     // there is bias or not
  void* bias_;                        // host bias
  void* bias_dev_;                    // device bias
  Activation activation_;             // record the activation type
  cusparseLtContext spmm_context_;
};  // class SpmmPluginDynamic

class SpmmPluginDynamicCreator : public nvinfer1::IPluginCreator {
 public:
  SpmmPluginDynamicCreator();
  const char* getPluginName() const noexcept override;
  const char* getPluginVersion() const noexcept override;
  const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;
  nvinfer1::IPluginV2* createPlugin(
      const char* name,
      const nvinfer1::PluginFieldCollection* fc) noexcept override;
  nvinfer1::IPluginV2* deserializePlugin(const char* name,
                                         const void* serialData,
                                         size_t serialLength) noexcept override;
  void setPluginNamespace(const char* pluginNamespace) noexcept override;
  const char* getPluginNamespace() const noexcept override;

 private:
  static nvinfer1::PluginFieldCollection field_collection_;
  static std::vector<nvinfer1::PluginField> plugin_attr_;
  std::string namespace_;
};  // class SpmmPluginDynamicCreator

REGISTER_TRT_PLUGIN_V2(SpmmPluginDynamicCreator);
}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
