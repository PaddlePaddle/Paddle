// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

/*
 * Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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

#pragma once
#include <stdio.h>
#include <cassert>
#include <string>
#include <vector>
#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

// One of the preferred ways of making TensorRT to be able to see
// our custom layer requires extending IPluginV2 and IPluginCreator classes.
// For requirements for overriden functions, check TensorRT API docs.

class GeluPlugin : public IPluginV2 {
 public:
  explicit GeluPlugin(const std::string name);

  GeluPlugin(const std::string name, const void* data, size_t length);

  // It doesn't make sense to make GeluPlugin without arguments, so we delete
  // default constructor.
  GeluPlugin() = delete;

  int getNbOutputs() const override;

  Dims getOutputDimensions(int index, const Dims* inputs,
                           int nbInputDims) override;

  int initialize() override;

  void terminate() override;

  size_t getWorkspaceSize(int) const override { return 0; };

  int enqueue(int batchSize, const void* const* inputs, void** outputs,
              void* workspace, cudaStream_t stream) override;

  size_t getSerializationSize() const override;

  void serialize(void* buffer) const override;

  void configureWithFormat(const Dims* inputDims, int nbInputs,
                           const Dims* outputDims, int nbOutputs, DataType type,
                           PluginFormat format, int maxBatchSize) override;

  bool supportsFormat(DataType type, PluginFormat format) const override;

  const char* getPluginType() const override;

  const char* getPluginVersion() const override;

  void destroy() override;

  nvinfer1::IPluginV2* clone() const override;

  void setPluginNamespace(const char* pluginNamespace) override;

  const char* getPluginNamespace() const override;

 private:
  const std::string mLayerName;
  size_t mInputVolume;
  std::string mNamespace;

  DataType mType;
};

class GeluPluginCreator : public IPluginCreator {
 public:
  GeluPluginCreator();

  const char* getPluginName() const override;

  const char* getPluginVersion() const override;

  const PluginFieldCollection* getFieldNames() override;

  IPluginV2* createPlugin(const char* name,
                          const PluginFieldCollection* fc) override;

  IPluginV2* deserializePlugin(const char* name, const void* serialData,
                               size_t serialLength) override;

  void setPluginNamespace(const char* pluginNamespace) override;

  const char* getPluginNamespace() const override;

 private:
  static PluginFieldCollection mFC;
  static std::vector<PluginField> mPluginAttributes;
  std::string mNamespace;
};

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
