// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once
#include <stdio.h>

#include <cassert>
#include <string>
#include <vector>

#include <NvInfer.h>

#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/type_defs.h"
#include "paddle/fluid/inference/tensorrt/helper.h"
#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin_utils.h"
#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/phi/core/kernel_context.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/memory/allocation/cuda_allocator.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/core/compat/arg_map_context.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/inference/tensorrt/plugin/generic_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

class GenericDynamicPlugin : public DynamicPluginTensorRT {
 public:
  
  GenericDynamicPlugin() { }
  
  GenericDynamicPlugin(const paddle::framework::proto::OpDesc &proto_op_desc) {
    proto_op_desc_ = proto_op_desc;
    op_desc_ = std::move(framework::OpDesc(proto_op_desc_, nullptr));
    proto_op_desc_.SerializeToString(&op_meta_data_);  
  }
  
  // It was used for tensorrt deserialization.
  // It should not be called by users.
  GenericDynamicPlugin(void const* serialData, size_t serialLength) {
    std::string op_meta_data((char*)serialData, serialLength);
    op_meta_data_ = std::move(op_meta_data);
  }

  // IPluginV2 method
  const char* getPluginType() const TRT_NOEXCEPT override {
    return "generic_dynamic_plugin";
  }
  
  int getNbOutputs() const TRT_NOEXCEPT override { 
    // framework::OpDesc op_desc(proto_op_desc_, nullptr);
    return op_desc_.OutputNames().size();
  }
  
  // Initialize the layer for execution.
  int initialize() TRT_NOEXCEPT override;

  // Shutdown the layer. This is called when the engine is destroyed
  void terminate() TRT_NOEXCEPT override { free(dev_ctx_); }

  void destroy() TRT_NOEXCEPT { };

  size_t getSerializationSize() const TRT_NOEXCEPT {
    size_t res = 0;
    res += op_meta_data_.size();
    return res;
  }

  void serialize(void* buffer) const TRT_NOEXCEPT {
    //serialize op_meta_data_
    std::memcpy(buffer, op_meta_data_.c_str(), op_meta_data_.size());
    reinterpret_cast<char*&>(buffer) += op_meta_data_.size();
  }

  // The Func in IPluginV2
  IPluginV2DynamicExt* clone() const TRT_NOEXCEPT {
    IPluginV2DynamicExt *plugin = new GenericDynamicPlugin(proto_op_desc_);
    plugin->initialize();
    return plugin;
  }

  nvinfer1::DimsExprs getOutputDimensions(
      int output_index,
      const nvinfer1::DimsExprs* inputs,
      int nb_inputs,
      nvinfer1::IExprBuilder& expr_builder) TRT_NOEXCEPT; 

  bool supportsFormatCombination(
      int pos,
      const nvinfer1::PluginTensorDesc* in_out,
      int nb_inputs,
      int nb_outputs) TRT_NOEXCEPT  { return true; }

  void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in,
                               int nb_inputs,
                               const nvinfer1::DynamicPluginTensorDesc* out,
                               int nb_outputs) TRT_NOEXCEPT;

  int enqueue(const nvinfer1::PluginTensorDesc* input_desc,
                      const nvinfer1::PluginTensorDesc* output_desc,
                      const void* const* inputs,
                      void* const* outputs,
                      void* workspace,
                      cudaStream_t stream) TRT_NOEXCEPT;
  
  nvinfer1::DataType getOutputDataType(
      int index,
      const nvinfer1::DataType* input_types,
      int nb_inputs) const TRT_NOEXCEPT {
        return input_types[0];
  }
  
 private:
  std::string op_meta_data_;
  framework::proto::OpDesc proto_op_desc_;
  framework::OpDesc op_desc_;
 private:
  std::shared_ptr<phi::KernelContext> phi_kernel_context_;
  paddle::platform::CUDADeviceContext *dev_ctx_;
 private:
  const phi::Kernel *phi_kernel_;
  static platform::CUDAPlace place_;
  static paddle::memory::allocation::CUDAAllocator allocator_;
};

class GenericDynamicPluginCreator : public TensorRTPluginCreator {
 public:
  const char* getPluginName() const TRT_NOEXCEPT override {
    return "generic_dynamic_plugin";
  }

  const char* getPluginVersion() const TRT_NOEXCEPT override { return "1"; }
  
  nvinfer1::IPluginV2* createPlugin(
      const char* name,
      const nvinfer1::PluginFieldCollection* fc) noexcept override {
    CHECK(false) << "not implement";
  }

  nvinfer1::IPluginV2DynamicExt* deserializePlugin(const char* name,
                                         const void* serial_data,
                                         size_t serial_length)
      TRT_NOEXCEPT override {
    auto* plugin = new GenericDynamicPlugin(serial_data, serial_length);
    return plugin;
  }
};
REGISTER_TRT_PLUGIN_V2(GenericDynamicPluginCreator);

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
