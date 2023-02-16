/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include "paddle/fluid/framework/phi_utils.h"
#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin.h"
#include "paddle/fluid/platform/enforce.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/cuda_stream.h"
#include "paddle/phi/kernels/fusion/cutlass/int4_gemm/int4_gemm_decl.h"
#include "paddle/phi/kernels/fusion/cutlass/int4_gemm/int4_gemm_util.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {
class MatmulInt4Plugin : public nvinfer1::IPluginV2IOExt {
 public:
  MatmulInt4Plugin(nvinfer1::Dims const& dims_x, nvinfer1::Dims const& dims_y);
  MatmulInt4Plugin(void const* data, size_t length);

  // IPluginV2IOExt Methods
  void configurePlugin(nvinfer1::PluginTensorDesc const* in,
                       int32_t nb_inputs,
                       nvinfer1::PluginTensorDesc const* out,
                       int32_t nb_outputs) noexcept override;
  bool supportsFormatCombination(int32_t pos,
                                 nvinfer1::PluginTensorDesc const* in_out,
                                 int32_t nb_inputs,
                                 int32_t nb_outputs) const noexcept override;

  // IPluginV2Ext Methods
  nvinfer1::DataType getOutputDataType(
      int32_t index,
      nvinfer1::DataType const* input_types,
      int32_t nb_inputs) const noexcept override;
  bool isOutputBroadcastAcrossBatch(int32_t output_index,
                                    const bool* input_is_broadcasted,
                                    int32_t nb_inputs) const noexcept override;
  bool canBroadcastInputAcrossBatch(
      int32_t input_index) const noexcept override;
  void attachToContext(cudnnContext* cudnnContext,
                       cublasContext* cublasContext,
                       nvinfer1::IGpuAllocator* gpuAllocator) noexcept override;
  void detachFromContext() noexcept override;
  nvinfer1::IPluginV2Ext* clone() const noexcept override;

  // IPluginV2 Methods
  const char* getPluginType() const noexcept override;
  const char* getPluginVersion() const noexcept override;
  int32_t getNbOutputs() const noexcept override;
  nvinfer1::Dims getOutputDimensions(int32_t index,
                                     nvinfer1::Dims const* inputs,
                                     int32_t nb_input_dims) noexcept override;
  int32_t initialize() noexcept override;
  void terminate() noexcept override;
  size_t getWorkspaceSize(int32_t max_batch_size) const noexcept override;
  size_t getSerializationSize() const noexcept override;
  void serialize(void* buffer) const noexcept override;
  void destroy() noexcept override;
  char const* getPluginNamespace() const noexcept override;
  void setPluginNamespace(char const* plugin_name_space) noexcept override;
  int32_t enqueue(int32_t batch_size,
                  void const* const* inputs,
                  void* const* outputs,
                  void* workspace,
                  cudaStream_t stream) noexcept override;

 protected:
  nvinfer1::Dims dims_x_;
  nvinfer1::Dims dims_y_;
  int batch_;
  uint64_t m_;
  uint64_t n_;
  uint64_t k_;
  int32_t *Atmp_, *Btmp_, *Cres_;
  cutlass::int4b_t *Aconvert_, *Bconvert_;
  nvinfer1::DataType type_;
  std::string namespace_;
};

class MatmulInt4PluginCreator : public nvinfer1::IPluginCreator {
 public:
  MatmulInt4PluginCreator();
  char const* getPluginName() const noexcept override;
  char const* getPluginVersion() const noexcept override;
  const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;
  void setPluginNamespace(char const* plugin_namespace) noexcept override;
  char const* getPluginNamespace() const noexcept override;
  nvinfer1::IPluginV2* createPlugin(
      char const* name,
      const nvinfer1::PluginFieldCollection* fc) noexcept override;
  nvinfer1::IPluginV2* deserializePlugin(
      char const* name,
      void const* serial_data,
      size_t serial_length) noexcept override;

 protected:
  std::string plugin_namespace_;
};

REGISTER_TRT_PLUGIN_V2(MatmulInt4PluginCreator);

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
