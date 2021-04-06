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

#include <string>
#include <vector>

#include "paddle/fluid/inference/tensorrt/engine.h"
#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

class YoloBoxPlugin : public nvinfer1::IPluginV2Ext {
 public:
  explicit YoloBoxPlugin(const nvinfer1::DataType data_type,
                         const std::vector<int>& anchors, const int class_num,
                         const float conf_thresh, const int downsample_ratio,
                         const bool clip_bbox, const float scale_x_y,
                         const int input_h, const int input_w);
  YoloBoxPlugin(const void* data, size_t length);
  ~YoloBoxPlugin() override;

  const char* getPluginType() const override;
  const char* getPluginVersion() const override;
  int getNbOutputs() const override;
  nvinfer1::Dims getOutputDimensions(int index, const nvinfer1::Dims* inputs,
                                     int nb_input_dims) override;
  bool supportsFormat(nvinfer1::DataType type,
                      nvinfer1::TensorFormat format) const override;
  size_t getWorkspaceSize(int max_batch_size) const override;
  int enqueue(int batch_size, const void* const* inputs, void** outputs,
              void* workspace, cudaStream_t stream) override;
  template <typename T>
  int enqueue_impl(int batch_size, const void* const* inputs, void** outputs,
                   void* workspace, cudaStream_t stream);
  int initialize() override;
  void terminate() override;
  size_t getSerializationSize() const override;
  void serialize(void* buffer) const override;
  void destroy() override;
  void setPluginNamespace(const char* lib_namespace) override;
  const char* getPluginNamespace() const override;

  nvinfer1::DataType getOutputDataType(int index,
                                       const nvinfer1::DataType* input_type,
                                       int nb_inputs) const override;
  bool isOutputBroadcastAcrossBatch(int output_index,
                                    const bool* input_is_broadcast,
                                    int nb_inputs) const override;
  bool canBroadcastInputAcrossBatch(int input_index) const override;
  void configurePlugin(const nvinfer1::Dims* input_dims, int nb_inputs,
                       const nvinfer1::Dims* output_dims, int nb_outputs,
                       const nvinfer1::DataType* input_types,
                       const nvinfer1::DataType* output_types,
                       const bool* input_is_broadcast,
                       const bool* output_is_broadcast,
                       nvinfer1::PluginFormat float_format,
                       int max_batct_size) override;
  nvinfer1::IPluginV2Ext* clone() const override;

 private:
  nvinfer1::DataType data_type_;
  std::vector<int> anchors_;
  int* anchors_device_;
  int class_num_;
  float conf_thresh_;
  int downsample_ratio_;
  bool clip_bbox_;
  float scale_x_y_;
  int input_h_;
  int input_w_;
  std::string namespace_;
};

class YoloBoxPluginCreator : public nvinfer1::IPluginCreator {
 public:
  YoloBoxPluginCreator();
  ~YoloBoxPluginCreator() override = default;

  void setPluginNamespace(const char* lib_namespace) override;
  const char* getPluginNamespace() const override;
  const char* getPluginName() const override;
  const char* getPluginVersion() const override;
  const nvinfer1::PluginFieldCollection* getFieldNames() override;

  nvinfer1::IPluginV2Ext* createPlugin(
      const char* name, const nvinfer1::PluginFieldCollection* fc) override;
  nvinfer1::IPluginV2Ext* deserializePlugin(const char* name,
                                            const void* serial_data,
                                            size_t serial_length) override;

 private:
  std::string namespace_;
  nvinfer1::PluginFieldCollection field_collection_;
};

REGISTER_TRT_PLUGIN_V2(YoloBoxPluginCreator);

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
