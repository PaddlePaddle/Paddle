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

#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {

void PluginTensorRT::serializeBase(void*& buffer) {
  serialize_value(&buffer, input_dims_);
  serialize_value(&buffer, max_batch_size_);
  serialize_value(&buffer, data_type_);
  serialize_value(&buffer, data_format_);
}

void PluginTensorRT::deserializeBase(void const*& serialData,
                                     size_t& serialLength) {
  deserialize_value(&serialData, &serialLength, &input_dims_);
  deserialize_value(&serialData, &serialLength, &max_batch_size_);
  deserialize_value(&serialData, &serialLength, &data_type_);
  deserialize_value(&serialData, &serialLength, &data_format_);
}

size_t PluginTensorRT::getBaseSerializationSize() {
  return (serialized_size(input_dims_) + serialized_size(max_batch_size_) +
          serialized_size(data_type_) + serialized_size(data_format_));
}

bool PluginTensorRT::supportsFormat(nvinfer1::DataType type,
                                    nvinfer1::PluginFormat format) const {
  return ((type == nvinfer1::DataType::kFLOAT) &&
          (format == nvinfer1::PluginFormat::kNCHW));
}

void PluginTensorRT::configureWithFormat(const nvinfer1::Dims* inputDims,
                                         int nbInputs,
                                         const nvinfer1::Dims* outputDims,
                                         int nbOutputs, nvinfer1::DataType type,
                                         nvinfer1::PluginFormat format,
                                         int maxBatchSize) {
  data_type_ = type;
  data_format_ = format;
  input_dims_.assign(inputDims, inputDims + nbInputs);
  max_batch_size_ = maxBatchSize;
}

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
