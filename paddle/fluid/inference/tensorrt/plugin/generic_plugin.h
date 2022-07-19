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

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

phi::Attribute Convert2PhiAttribute(framework::OpDesc& op_desc, std::string attr_name);

class PluginArgumentMappingContext : public ::phi::ArgumentMappingContext {
 public:
// only support op in pd dialect
  explicit PluginArgumentMappingContext(framework::OpDesc *op_desc_ptr) : op_desc_ptr_(op_desc_ptr) { }

  bool HasInput(const std::string& name) const override {
    auto inputs = op_desc_ptr_->InputNames();
    for(auto &i : inputs) {
      if(i == name)
        return true;
    }
    return false;
  }
  bool HasOutput(const std::string& name) const override {
    return op_desc_ptr_->HasOutput(name);
  }
  bool HasAttr(const std::string& name) const override {
    return op_desc_ptr_->HasAttr(name);
  }

  // now we can't use Attribute here, it will cause phi relay on
  // paddle::variant and BlockDesc
  paddle::any Attr(const std::string& attr_name) const override { 
    auto attr_type = op_desc_ptr_->GetAttrType(attr_name);
    switch (attr_type) {
      case framework::proto::AttrType::INT: {
        return PADDLE_GET_CONST(int, op_desc_ptr_->GetAttr(attr_name));
        break;
      };
      case framework::proto::AttrType::FLOAT: {
        return PADDLE_GET_CONST(float, op_desc_ptr_->GetAttr(attr_name));
        break;
      };
      case framework::proto::AttrType::STRING: {
        return PADDLE_GET_CONST(std::string, op_desc_ptr_->GetAttr(attr_name));
        break;
      };
      case framework::proto::AttrType::INTS: {
        return PADDLE_GET_CONST(std::vector<int>, op_desc_ptr_->GetAttr(attr_name));
        break;
      };
      case framework::proto::AttrType::FLOATS: {
        return PADDLE_GET_CONST(std::vector<float>, op_desc_ptr_->GetAttr(attr_name));
        break;
      };
      case framework::proto::AttrType::STRINGS: {
        return PADDLE_GET_CONST(std::vector<std::string>, op_desc_ptr_->GetAttr(attr_name));
        break;
      };
      case framework::proto::AttrType::BOOLEAN: {
        return PADDLE_GET_CONST(bool, op_desc_ptr_->GetAttr(attr_name));
        break;
      };
      case framework::proto::AttrType::BOOLEANS: {
        return PADDLE_GET_CONST(std::vector<bool>, op_desc_ptr_->GetAttr(attr_name));
        break;
      };
      default:{
        LOG(ERROR) << "Can't conver op's attribute ["<< attr_name << "] to paddle any.";
      }
    }
    return paddle::any();
  }

  size_t InputSize(const std::string& name) const override {
    return op_desc_ptr_->InputNames().size();
  }
  size_t OutputSize(const std::string& name) const override {
    return op_desc_ptr_->OutputNames().size();
  }

  bool IsDenseTensorInput(const std::string& name) const override {
    return true;
  }
  bool IsDenseTensorInputs(const std::string& name) const override {
    return true;
  }
  bool IsSelectedRowsInput(const std::string& name) const override {
    return true;
  }
  bool IsDenseTensorVectorInput(const std::string& name) const override {
    return true;
  }

  bool IsDenseTensorOutput(const std::string& name) const override {
    return true;
  }
  bool IsSelectedRowsOutput(const std::string& name) const override { 
    return false; 
  }

  bool IsForInferShape() const override { return false; }

 private:
  framework::OpDesc *op_desc_ptr_;
};

class GenericPlugin : public PluginTensorRT {
 public:
  GenericPlugin() { }
  
  GenericPlugin(const paddle::framework::proto::OpDesc &proto_op_desc) {
    proto_op_desc_ = proto_op_desc;
    op_desc_ = std::move(framework::OpDesc(proto_op_desc_, nullptr));
    proto_op_desc_.SerializeToString(&op_meta_data_);  
  }

  void InferShape(const paddle::framework::proto::OpDesc &proto_op_desc, const nvinfer1::Dims* inputDims, int nbInputDims);

  // It was used for tensorrt deserialization.
  // It should not be called by users.
  GenericPlugin(void const* serialData, size_t serialLength) {
    deserializeBase(serialData, serialLength);
    std::string op_meta_data((char*)serialData, serialLength);
    op_meta_data_ = std::move(op_meta_data);
  }
  
  // IPluginV2 method
  const char* getPluginType() const TRT_NOEXCEPT override {
    return "generic_plugin";
  }
  
  int getNbOutputs() const TRT_NOEXCEPT override { 
    // framework::OpDesc op_desc(proto_op_desc_, nullptr);
    return op_desc_.OutputNames().size();
  }
  
  nvinfer1::Dims getOutputDimensions(int index,
                                     const nvinfer1::Dims* inputDims,
                                     int nbInputDims) TRT_NOEXCEPT override {
    InferShape(proto_op_desc_, inputDims, nbInputDims);
    return output_dims_.at(index);
  }
  
  // Initialize the layer for execution.
  int initialize() TRT_NOEXCEPT override;

  // Shutdown the layer. This is called when the engine is destroyed
  void terminate() TRT_NOEXCEPT override { free(dev_ctx_); }

  // Execute the layer
  #if IS_TRT_VERSION_LT(8000)
    int enqueue(int batch_size,
                const void* const* inputs,
                void** outputs,
  #else
    int enqueue(int batch_size,
                const void* const* inputs,
                void* const* outputs,
  #endif
                void* workspace,
                cudaStream_t stream) TRT_NOEXCEPT override;
  
  // Find the size of the serialization buffer required
  size_t getSerializationSize() const TRT_NOEXCEPT override {
    size_t res = 0;
    res += getBaseSerializationSize();
    res += op_meta_data_.size();
    return res;
  }

  // Serialize the layer config to buffer.
  // TensorRT will call this func to serialize the configuration of TensorRT
  // engine. It should not be called by users.
  void serialize(void* buffer) const TRT_NOEXCEPT override {
    serializeBase(buffer);
    //serialize op_meta_data_
    std::memcpy(buffer, op_meta_data_.c_str(), op_meta_data_.size());
    reinterpret_cast<char*&>(buffer) += op_meta_data_.size();
  }

  nvinfer1::IPluginV2* clone() const TRT_NOEXCEPT override {
    return new GenericPlugin(proto_op_desc_);
  }

 private:
  std::string op_meta_data_;
  framework::proto::OpDesc proto_op_desc_;
  framework::OpDesc op_desc_;
 private:
  std::shared_ptr<phi::KernelContext>phi_kernel_context_;
  paddle::platform::CUDADeviceContext *dev_ctx_;
 private:
  const phi::Kernel *phi_kernel_;
  static platform::CUDAPlace place_;
  static paddle::memory::allocation::CUDAAllocator allocator_;
};

class GenericPluginCreator : public TensorRTPluginCreator {
 public:
  const char* getPluginName() const TRT_NOEXCEPT override {
    return "generic_plugin";
  }

  const char* getPluginVersion() const TRT_NOEXCEPT override { return "1"; }
  
  nvinfer1::IPluginV2* createPlugin(
      const char* name,
      const nvinfer1::PluginFieldCollection* fc) noexcept override {
    CHECK(false) << "not implement";
  }

  nvinfer1::IPluginV2* deserializePlugin(const char* name,
                                         const void* serial_data,
                                         size_t serial_length)
      TRT_NOEXCEPT override {
    auto* plugin = new GenericPlugin(serial_data, serial_length);
    return plugin;
  }
};
REGISTER_TRT_PLUGIN_V2(GenericPluginCreator);

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
