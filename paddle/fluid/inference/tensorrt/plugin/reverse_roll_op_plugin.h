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

#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/inference/tensorrt/engine.h"
#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

class ReverseRollPluginDynamic : public DynamicPluginTensorRT{
 public:
  ReverseRollPluginDynamic(int window_num,
                           int window_len,
                           int window_size,
                           int input_resolution,
                           int shift_size,
                           bool with_fp16) 
    : window_num_(window_num),
      window_len_(window_len),
      window_size_(window_size),
      input_resolution_(input_resolution),
      shift_size_(shift_size),
      with_fp16_(with_fp16) {};
  ReverseRollPluginDynamic(void const* serialData,
                           size_t serialLength){
    DeserializeValue(&serialData, &serialLength, &window_num_);
    DeserializeValue(&serialData, &serialLength, &window_len_);
    DeserializeValue(&serialData, &serialLength, &window_size_);
    DeserializeValue(&serialData, &serialLength, &input_resolution_);
    DeserializeValue(&serialData, &serialLength, &shift_size_);
    DeserializeValue(&serialData, &serialLength, &with_fp16_);
};

nvinfer1::IPluginV2DynamicExt* clone() const TRT_NOEXCEPT override {
    return new ReverseRollPluginDynamic(window_num_,
                                        window_len_,
                                        window_size_,
                                        input_resolution_,
                                        shift_size_,
                                        with_fp16_);
  }
  const char* getPluginType() const TRT_NOEXCEPT override {
    return "reverse_roll_dynamic";
  }
  int getNbOutputs() const TRT_NOEXCEPT override { return 1; }
  int initialize() TRT_NOEXCEPT override { return 0; }
  size_t getSerializationSize() const TRT_NOEXCEPT override {
      return SerializedSize(window_num_)+SerializedSize(window_len_)+
             SerializedSize(window_size_)+SerializedSize(input_resolution_)+
             SerializedSize(shift_size_)+SerializedSize(with_fp16_);
  }
  void serialize(void* buffer) const TRT_NOEXCEPT override {
    SerializeValue(&buffer, window_num_);
    SerializeValue(&buffer, window_len_);
    SerializeValue(&buffer, window_size_);
    SerializeValue(&buffer, input_resolution_);
    SerializeValue(&buffer, shift_size_);
    SerializeValue(&buffer, with_fp16_);
  }


  nvinfer1::DimsExprs getOutputDimensions(int output_index,
                                          const nvinfer1::DimsExprs* inputs,
                                          int nb_inputs,
                                          nvinfer1::IExprBuilder& expr_builder)
        TRT_NOEXCEPT override;
  bool supportsFormatCombination(int pos,
                                 const nvinfer1::PluginTensorDesc* inOut,
                                 int nbInputs,
                                 int nbOutputs) TRT_NOEXCEPT override;
  void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in,
                       int nbInputs,
                       const nvinfer1::DynamicPluginTensorDesc* out,
                       int nbOutputs) TRT_NOEXCEPT override;
  size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs,
                          int nbInputs,
                          const nvinfer1::PluginTensorDesc* outputs,
                          int nbOutputs) const TRT_NOEXCEPT override {
    return 0;
  }

  int enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
              const nvinfer1::PluginTensorDesc* outputDesc,
              const void* const* inputs,
              void* const* outputs,
              void* workspace,
              cudaStream_t stream) TRT_NOEXCEPT override;
  nvinfer1::DataType getOutputDataType(int index,
                                       const nvinfer1::DataType* inputTypes,
                                       int nbInputs) const
      TRT_NOEXCEPT override;

  void destroy() TRT_NOEXCEPT override { delete this; }
private:
  int window_num_;
  int window_len_;
  int window_size_;
  int input_resolution_;
  int shift_size_;
  bool with_fp16_;
};

class ReverseRollPluginDynamicCreater
    : public TensorRTPluginCreator {
    public:
      const char* getPluginName() const TRT_NOEXCEPT override {
        return "reverse_roll_dynamic";
      }
    const char* getPluginVersion() const TRT_NOEXCEPT override { return "1"; }
    nvinfer1::IPluginV2* deserializePlugin(const char* name,
                                           const void* serial_data,
                                           size_t serial_length)
        TRT_NOEXCEPT override {
        return new ReverseRollPluginDynamic(serial_data, serial_length);
    }
};
REGISTER_TRT_PLUGIN_V2(ReverseRollPluginDynamicCreater);

} // plugin
} // tensorrt
} // inference
} // paddle