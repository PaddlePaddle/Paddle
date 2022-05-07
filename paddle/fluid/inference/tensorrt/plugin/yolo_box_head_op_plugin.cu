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

#include <algorithm>
#include <cassert>
#include "paddle/fluid/inference/tensorrt/plugin/yolo_box_head_op_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

YoloBoxHeadPlugin::YoloBoxHeadPlugin(
    const nvinfer1::DataType data_type, const std::vector<int>& anchors,
    const int class_num, const float conf_thresh, const int downsample_ratio,
    const bool clip_bbox, const float scale_x_y)
    : data_type_(data_type),
      class_num_(class_num),
      conf_thresh_(conf_thresh),
      downsample_ratio_(downsample_ratio),
      clip_bbox_(clip_bbox),
      scale_x_y_(scale_x_y) {
  assert(data_type_ == nvinfer1::DataType::kFLOAT);
  assert(class_num_ > 0);
}

YoloBoxHeadPlugin::YoloBoxHeadPlugin(const void* data, size_t length) {
  DeserializeValue(&data, &length, &data_type_);
  DeserializeValue(&data, &length, &anchors_);
  DeserializeValue(&data, &length, &class_num_);
  DeserializeValue(&data, &length, &conf_thresh_);
  DeserializeValue(&data, &length, &downsample_ratio_);
  DeserializeValue(&data, &length, &clip_bbox_);
  DeserializeValue(&data, &length, &scale_x_y_);
}

YoloBoxHeadPlugin::~YoloBoxHeadPlugin() {}

const char* YoloBoxHeadPlugin::getPluginType() const TRT_NOEXCEPT {
  return "yolo_box_head_plugin";
}

const char* YoloBoxHeadPlugin::getPluginVersion() const TRT_NOEXCEPT {
  return "1";
}

int YoloBoxHeadPlugin::getNbOutputs() const TRT_NOEXCEPT { return 1; }

nvinfer1::Dims YoloBoxHeadPlugin::getOutputDimensions(
    int index, const nvinfer1::Dims* inputs, int nb_input_dims) TRT_NOEXCEPT {
  assert(index == 0);
  assert(nb_input_dims == 1);
  return inputs[0];
}

bool YoloBoxHeadPlugin::supportsFormat(
    nvinfer1::DataType type, nvinfer1::TensorFormat format) const TRT_NOEXCEPT {
  return type == data_type_ && format == nvinfer1::TensorFormat::kLINEAR;
}

size_t YoloBoxHeadPlugin::getWorkspaceSize(int max_batch_size) const
    TRT_NOEXCEPT {
  return 0;
}

__global__ void YoloBoxHeadV3Kernel(const float* input, float* output,
                                    const uint grid_size_x,
                                    const uint grid_size_y,
                                    const uint class_num,
                                    const uint anchors_num) {
  uint x_id = blockIdx.x * blockDim.x + threadIdx.x;
  uint y_id = blockIdx.y * blockDim.y + threadIdx.y;
  uint z_id = blockIdx.z * blockDim.z + threadIdx.z;
  if ((x_id >= grid_size_x) || (y_id >= grid_size_y) || (z_id >= anchors_num)) {
    return;
  }
  const int grids_num = grid_size_x * grid_size_y;
  const int bbindex = y_id * grid_size_x + x_id;

  // objectness
  output[bbindex + grids_num * (z_id * (5 + class_num) + 4)] =
      SigmoidGPU(input[bbindex + grids_num * (z_id * (5 + class_num) + 4)]);
  // x
  output[bbindex + grids_num * (z_id * (5 + class_num) + 0)] =
      SigmoidGPU(input[bbindex + grids_num * (z_id * (5 + class_num) + 0)]);
  // y
  output[bbindex + grids_num * (z_id * (5 + class_num) + 1)] =
      SigmoidGPU(input[bbindex + grids_num * (z_id * (5 + class_num) + 1)]);
  // w
  output[bbindex + grids_num * (z_id * (5 + class_num) + 2)] =
      __expf(input[bbindex + grids_num * (z_id * (5 + class_num) + 2)]);
  // h
  output[bbindex + grids_num * (z_id * (5 + class_num) + 3)] =
      __expf(input[bbindex + grids_num * (z_id * (5 + class_num) + 3)]);
  // Probabilities of classes
  for (uint i = 0; i < class_num; ++i) {
    output[bbindex + grids_num * (z_id * (5 + class_num) + (5 + i))] =
        SigmoidGPU(
            input[bbindex + grids_num * (z_id * (5 + class_num) + (5 + i))]);
  }
}

int YoloBoxHeadPlugin::enqueue(int batch_size, const void* const* inputs,
#if IS_TRT_VERSION_LT(8000)
                               void** outputs, void* workspace,
#else
                               void* const* outputs, void* workspace,
#endif
                               cudaStream_t stream) TRT_NOEXCEPT {
  const int h = input_dims_[0].d[1];
  const int w = input_dims_[0].d[2];
  const int grid_size_x = w;
  const int grid_size_y = h;
  const int anchors_num = anchors_.size() / 2;
  const float* input_data = static_cast<const float*>(inputs[0]);
  float* output_data = static_cast<float*>(outputs[0]);
  const int volume = input_dims_[0].d[0] * h * w;
  dim3 block(16, 16, 4);
  dim3 grid((grid_size_x / block.x) + 1, (grid_size_y / block.y) + 1,
            (anchors_num / block.z) + 1);
  for (int n = 0; n < batch_size; n++) {
    YoloBoxHeadV3Kernel<<<grid, block, 0, stream>>>(
        input_data + batch * volume, output_data + batch * volume, grid_size_x,
        grid_size_y, class_num_, anchors_num);
  }
  return 0;
}

int YoloBoxHeadPlugin::initialize() TRT_NOEXCEPT { return 0; }

void YoloBoxHeadPlugin::terminate() TRT_NOEXCEPT {}

size_t YoloBoxHeadPlugin::getSerializationSize() const TRT_NOEXCEPT {
  size_t serialize_size = 0;
  serialize_size += SerializedSize(data_type_);
  serialize_size += SerializedSize(anchors_);
  serialize_size += SerializedSize(class_num_);
  serialize_size += SerializedSize(conf_thresh_);
  serialize_size += SerializedSize(downsample_ratio_);
  serialize_size += SerializedSize(clip_bbox_);
  serialize_size += SerializedSize(scale_x_y_);
  return serialize_size;
}

void YoloBoxHeadPlugin::serialize(void* buffer) const TRT_NOEXCEPT {
  SerializeValue(&buffer, data_type_);
  SerializeValue(&buffer, anchors_);
  SerializeValue(&buffer, class_num_);
  SerializeValue(&buffer, conf_thresh_);
  SerializeValue(&buffer, downsample_ratio_);
  SerializeValue(&buffer, clip_bbox_);
  SerializeValue(&buffer, scale_x_y_);
}

void YoloBoxHeadPlugin::destroy() TRT_NOEXCEPT {}

void YoloBoxHeadPlugin::setPluginNamespace(const char* lib_namespace)
    TRT_NOEXCEPT {
  namespace_ = std::string(lib_namespace);
}

const char* YoloBoxHeadPlugin::getPluginNamespace() const TRT_NOEXCEPT {
  return namespace_.c_str();
}

nvinfer1::DataType YoloBoxHeadPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* input_type,
    int nb_inputs) const TRT_NOEXCEPT {
  return input_type[0];
}

bool YoloBoxHeadPlugin::isOutputBroadcastAcrossBatch(
    int output_index, const bool* input_is_broadcast,
    int nb_inputs) const TRT_NOEXCEPT {
  return false;
}

bool YoloBoxHeadPlugin::canBroadcastInputAcrossBatch(int input_index) const
    TRT_NOEXCEPT {
  return false;
}

void YoloBoxHeadPlugin::configurePlugin(
    const nvinfer1::Dims* input_dims, int nb_inputs,
    const nvinfer1::Dims* output_dims, int nb_outputs,
    const nvinfer1::DataType* input_types,
    const nvinfer1::DataType* output_types, const bool* input_is_broadcast,
    const bool* output_is_broadcast, nvinfer1::PluginFormat float_format,
    int max_batct_size) TRT_NOEXCEPT {}

nvinfer1::IPluginV2Ext* YoloBoxHeadPlugin::clone() const TRT_NOEXCEPT {
  return new YoloBoxHeadPlugin(data_type_, anchors_, class_num_, conf_thresh_,
                               downsample_ratio_, clip_bbox_, scale_x_y_);
}

YoloBoxHeadPluginCreator::YoloBoxHeadPluginCreator() {}

void YoloBoxHeadPluginCreator::setPluginNamespace(const char* lib_namespace)
    TRT_NOEXCEPT {
  namespace_ = std::string(lib_namespace);
}

const char* YoloBoxHeadPluginCreator::getPluginNamespace() const TRT_NOEXCEPT {
  return namespace_.c_str();
}

const char* YoloBoxHeadPluginCreator::getPluginName() const TRT_NOEXCEPT {
  return "yolo_box_head_plugin";
}

const char* YoloBoxHeadPluginCreator::getPluginVersion() const TRT_NOEXCEPT {
  return "1";
}

const nvinfer1::PluginFieldCollection* YoloBoxHeadPluginCreator::getFieldNames()
    TRT_NOEXCEPT {
  return &field_collection_;
}

nvinfer1::IPluginV2Ext* YoloBoxHeadPluginCreator::createPlugin(
    const char* name, const nvinfer1::PluginFieldCollection* fc) TRT_NOEXCEPT {
  const nvinfer1::PluginField* fields = fc->fields;

  int type_id = -1;
  std::vector<int> anchors;
  int class_num = -1;
  float conf_thresh = 0.01;
  int downsample_ratio = 32;
  bool clip_bbox = true;
  float scale_x_y = 1.;

  for (int i = 0; i < fc->nbFields; ++i) {
    const std::string field_name(fc->fields[i].name);
    if (field_name.compare("type_id") == 0) {
      type_id = *static_cast<const int*>(fc->fields[i].data);
    } else if (field_name.compare("anchors")) {
      const int length = fc->fields[i].length;
      const int* data = static_cast<const int*>(fc->fields[i].data);
      anchors.insert(anchors.end(), data, data + length);
    } else if (field_name.compare("class_num")) {
      class_num = *static_cast<const int*>(fc->fields[i].data);
    } else if (field_name.compare("conf_thresh")) {
      conf_thresh = *static_cast<const float*>(fc->fields[i].data);
    } else if (field_name.compare("downsample_ratio")) {
      downsample_ratio = *static_cast<const int*>(fc->fields[i].data);
    } else if (field_name.compare("clip_bbox")) {
      clip_bbox = *static_cast<const bool*>(fc->fields[i].data);
    } else if (field_name.compare("scale_x_y")) {
      scale_x_y = *static_cast<const float*>(fc->fields[i].data);
    } else {
      assert(false && "unknown plugin field name.");
    }
  }

  return new YoloBoxHeadPlugin(
      type_id ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT, anchors,
      class_num, conf_thresh, downsample_ratio, clip_bbox, scale_x_y);
}

nvinfer1::IPluginV2Ext* YoloBoxHeadPluginCreator::deserializePlugin(
    const char* name, const void* serial_data,
    size_t serial_length) TRT_NOEXCEPT {
  auto plugin = new YoloBoxHeadPlugin(serial_data, serial_length);
  plugin->setPluginNamespace(namespace_.c_str());
  return plugin;
}

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
