// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include <NvInfer.h>
#include <stdio.h>
#include <cassert>
#include <string>
#include <vector>

#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/type_defs.h"
#include "paddle/fluid/inference/tensorrt/engine.h"
#include "paddle/fluid/inference/tensorrt/helper.h"
#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin.h"
#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin_utils.h"
#include "paddle/fluid/inference/tensorrt/plugin_arg_mapping_context.h"
#include "paddle/fluid/memory/allocation/cuda_allocator.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_context.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

void BuildPhiKernelContextAttr(const framework::OpDesc& op_desc,
                               phi::KernelContext* kernel_context,
                               const phi::KernelSignature& signature,
                               const phi::Kernel& phi_kernel);

class GenericPlugin : public DynamicPluginTensorRT {
 public:
  struct InputOutPutVarInfo {
    std::vector<int> inputs_data_type;
    std::vector<int> outputs_data_type;
  };

 public:
  GenericPlugin() {}

  GenericPlugin(const paddle::framework::proto::OpDesc& proto_op_desc,
                const InputOutPutVarInfo& in_out_info);

  GenericPlugin(const paddle::framework::proto::OpDesc& proto_op_desc,
                const std::vector<int>& inputs_data_type,
                const std::vector<int>& outputs_data_type);

  // It was used for tensorrt deserialization.
  // It should not be called by users.
  GenericPlugin(void const* serialData, size_t serialLength);

  // IPluginV2 method
  const char* getPluginType() const TRT_NOEXCEPT override {
    return "generic_plugin";
  }

  int getNbOutputs() const TRT_NOEXCEPT override;

  int getNbInputs() const TRT_NOEXCEPT;

  // Initialize the layer for execution.
  int initialize() TRT_NOEXCEPT override;

  // Shutdown the layer. This is called when the engine is destroyed
  void terminate() TRT_NOEXCEPT override;

  void destroy() TRT_NOEXCEPT{};

  size_t getSerializationSize() const TRT_NOEXCEPT {
    return op_meta_data_.size() + SerializedSize(inputs_data_type_) +
           SerializedSize(outputs_data_type_);
  }

  void serialize(void* buffer) const TRT_NOEXCEPT;

  // The Func in IPluginV2
  nvinfer1::IPluginV2DynamicExt* clone() const TRT_NOEXCEPT;

  nvinfer1::DimsExprs getOutputDimensions(
      int output_index,
      const nvinfer1::DimsExprs* inputs,
      int nb_inputs,
      nvinfer1::IExprBuilder& expr_builder)  // NOLINT
      TRT_NOEXCEPT;

  bool supportsFormatCombination(int pos,
                                 const nvinfer1::PluginTensorDesc* in_out,
                                 int nb_inputs,
                                 int nb_outputs) TRT_NOEXCEPT;

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

  nvinfer1::DataType getOutputDataType(int index,
                                       const nvinfer1::DataType* input_types,
                                       int nb_inputs) const TRT_NOEXCEPT;

 private:
  std::string op_meta_data_;
  framework::proto::OpDesc proto_op_desc_;
  framework::OpDesc op_desc_;

 private:
  const phi::Kernel* phi_kernel_{nullptr};

  phi::KernelContext* phi_kernel_context_{nullptr};
  std::vector<phi::DenseTensor>* dense_tensor_inputs_{nullptr};
  std::vector<phi::DenseTensor>* dense_tensor_outputs_{nullptr};

 private:
  InputOutPutVarInfo in_out_info_;
  std::vector<int> inputs_data_type_;
  std::vector<int> outputs_data_type_;
};

class GenericPluginCreator : public TensorRTPluginCreator {
 public:
  const char* getPluginName() const TRT_NOEXCEPT override {
    return "generic_plugin";
  }

  const char* getPluginVersion() const TRT_NOEXCEPT override { return "1"; }

  nvinfer1::IPluginV2DynamicExt* deserializePlugin(const char* name,
                                                   const void* serial_data,
                                                   size_t serial_length)
      TRT_NOEXCEPT override {
    return new GenericPlugin(serial_data, serial_length);
  }
};
REGISTER_TRT_PLUGIN_V2(GenericPluginCreator);

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
