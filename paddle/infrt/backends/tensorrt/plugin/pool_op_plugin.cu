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

#include "glog/logging.h"
#include "paddle/infrt/backends/tensorrt/plugin/plugin_utils.h"
#include "paddle/infrt/backends/tensorrt/plugin/pool_op_plugin.h"
#include "paddle/phi/kernels/funcs/pooling.h"

namespace infrt {
namespace backends {
namespace tensorrt {
namespace plugin {

PoolPlugin::PoolPlugin(bool ceil_mode,
                       PoolType pool_type,
                       bool adaptive,
                       bool exclusive,
                       std::vector<int> ksize,
                       std::vector<int> strides,
                       std::vector<int> paddings,
                       std::vector<int> input_shape,
                       std::vector<int> real_paddings)
    : ceil_mode_(ceil_mode),
      pool_type_(pool_type),
      adaptive_(adaptive),
      exclusive_(exclusive),
      ksize_(ksize),
      strides_(strides),
      paddings_(paddings),
      real_paddings_(real_paddings),
      input_shape_(input_shape) {
  output_shape_ = input_shape_;
  std::vector<int> output_shape =
      CalcOutputSize({input_shape_[1], input_shape_[2]},
                     ceil_mode_,
                     adaptive_,
                     ksize_,
                     strides_,
                     real_paddings_);
  output_shape_[1] = output_shape[0];
  output_shape_[2] = output_shape[1];
}

PoolPlugin::PoolPlugin(void const* serialData, size_t serialLength) {
  // deserializeBase(serialData, serialLength);
  DeserializeValue(&serialData, &serialLength, &ceil_mode_);
  DeserializeValue(&serialData, &serialLength, &pool_type_);
  DeserializeValue(&serialData, &serialLength, &adaptive_);
  DeserializeValue(&serialData, &serialLength, &exclusive_);
  DeserializeValue(&serialData, &serialLength, &ksize_);
  DeserializeValue(&serialData, &serialLength, &strides_);
  DeserializeValue(&serialData, &serialLength, &paddings_);
  DeserializeValue(&serialData, &serialLength, &real_paddings_);
  DeserializeValue(&serialData, &serialLength, &input_shape_);
  DeserializeValue(&serialData, &serialLength, &output_shape_);
}

const char* PoolPlugin::getPluginType() const noexcept { return "pool_plugin"; }

const char* PoolPlugin::getPluginVersion() const noexcept { return "1"; }

int PoolPlugin::getNbOutputs() const noexcept { return 1; }

nvinfer1::Dims PoolPlugin::getOutputDimensions(int outputIndex,
                                               const nvinfer1::Dims* inputs,
                                               int nbInputs) noexcept {
  assert(nbInputs == 1);
  assert(index == 0);
  assert(inputs[0].nbDims == 3);
  nvinfer1::Dims const& input_dims = inputs[0];

  nvinfer1::Dims output_dims = input_dims;

  output_dims.d[1] = output_shape_[1];
  output_dims.d[2] = output_shape_[2];
  return output_dims;
}

int32_t PoolPlugin::initialize() noexcept { return 0; }

void PoolPlugin::terminate() noexcept {}

size_t PoolPlugin::getWorkspaceSize(int32_t maxBatchSize) const noexcept {
  return 0;
}

#if IS_TRT_VERSION_LT(8000)
int PoolPlugin::enqueue(int batch_size,
                        const void* const* inputs,
                        void** outputs,
#else
int PoolPlugin::enqueue(int batch_size,
                        const void* const* inputs,
                        void* const* outputs,
#endif
                        void* workspace,
                        cudaStream_t stream) noexcept {
  // TODO(wilber)
  int input_size = 0;
  float const* idata = reinterpret_cast<float const*>(inputs[0]);
  float* const* odatas = reinterpret_cast<float* const*>(outputs);

  std::vector<int> input_shape = input_shape_;
  std::vector<int> output_shape = output_shape_;
  input_shape.insert(input_shape.begin(), batch_size);
  output_shape.insert(output_shape.begin(), batch_size);

  if (pool_type_ == PoolType::max) {
    ::phi::funcs::MaxPool<float> pool_process;
    ::phi::funcs::Pool2dDirectCUDAFunctor<phi::funcs::MaxPool<float>, float>
        pool2d_forward;
    pool2d_forward(idata,
                   input_shape,
                   output_shape,
                   ksize_,
                   strides_,
                   paddings_,
                   true,
                   false,
                   odatas[0],
                   stream,
                   pool_process);
  } else if (pool_type_ == PoolType::avg) {
    ::phi::funcs::AvgPool<float> pool_process;
    ::phi::funcs::Pool2dDirectCUDAFunctor<phi::funcs::AvgPool<float>, float>
        pool2d_forward;
    pool2d_forward(idata,
                   input_shape,
                   output_shape,
                   ksize_,
                   strides_,
                   paddings_,
                   exclusive_,
                   adaptive_,
                   odatas[0],
                   stream,
                   pool_process);
  }

  return cudaGetLastError() != cudaSuccess;
}

// TODO(wilber): serialize base info?
size_t PoolPlugin::getSerializationSize() const noexcept {
  return SerializedSize(ceil_mode_) + SerializedSize(pool_type_) +
         SerializedSize(adaptive_) + SerializedSize(exclusive_) +
         SerializedSize(ksize_) + SerializedSize(strides_) +
         SerializedSize(paddings_) + SerializedSize(real_paddings_) +
         SerializedSize(input_shape_) + SerializedSize(output_shape_);
}
// TODO(wilber): serialize base info?
void PoolPlugin::serialize(void* buffer) const noexcept {
  // serializeBase(buffer);
  SerializeValue(&buffer, ceil_mode_);
  SerializeValue(&buffer, pool_type_);
  SerializeValue(&buffer, adaptive_);
  SerializeValue(&buffer, exclusive_);
  SerializeValue(&buffer, ksize_);
  SerializeValue(&buffer, strides_);
  SerializeValue(&buffer, paddings_);
  SerializeValue(&buffer, real_paddings_);
  SerializeValue(&buffer, input_shape_);
  SerializeValue(&buffer, output_shape_);
}

void PoolPlugin::destroy() noexcept { delete this; }

void PoolPlugin::setPluginNamespace(char const* plugin_namespace) noexcept {
  namespace_ = plugin_namespace;
}

char const* PoolPlugin::getPluginNamespace() const noexcept {
  return namespace_.c_str();
}

nvinfer1::DataType PoolPlugin::getOutputDataType(
    int32_t index,
    nvinfer1::DataType const* input_types,
    int32_t nbInputs) const noexcept {
  CHECK_EQ(index, 0);
  CHECK_EQ((input_types[0] == nvinfer1::DataType::kFLOAT), true);
  return input_types[0];
}

bool PoolPlugin::isOutputBroadcastAcrossBatch(int32_t outputIndex,
                                              bool const* inputIsBroadcasted,
                                              int32_t nbInputs) const noexcept {
  return false;
}

bool PoolPlugin::canBroadcastInputAcrossBatch(int32_t inputIndex) const
    noexcept {
  return false;
}

nvinfer1::IPluginV2Ext* PoolPlugin::clone() const noexcept {
  auto* plugin = new PoolPlugin(ceil_mode_,
                                pool_type_,
                                adaptive_,
                                exclusive_,
                                ksize_,
                                strides_,
                                paddings_,
                                input_shape_,
                                real_paddings_);
  plugin->setPluginNamespace(namespace_.c_str());
  return plugin;
}

void PoolPlugin::configurePlugin(nvinfer1::PluginTensorDesc const* in,
                                 int32_t nb_input,
                                 nvinfer1::PluginTensorDesc const* out,
                                 int32_t nb_output) noexcept {
  CHECK_EQ(nb_input, 1);
  CHECK_EQ(nb_output, 1);

  input_dims_ = in[0].dims;
  data_format_ = in[0].format;
  data_type_ = in[0].type;
}

bool PoolPlugin::supportsFormatCombination(
    int32_t pos,
    nvinfer1::PluginTensorDesc const* in_out,
    int32_t nb_inputs,
    int32_t nb_outputs) const noexcept {
  CHECK_LT(pos, nb_inputs + nb_outputs);
  CHECK_NOTNULL(in_out);

  return ((in_out[pos].type == nvinfer1::DataType::kFLOAT) &&
          in_out[pos].format == nvinfer1::PluginFormat::kLINEAR);
}

nvinfer1::IPluginV2* PoolPluginCreator::createPlugin(
    const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept {
  // auto* plugin = new UffPoolPluginV2(*fc);
  field_collection_ = *fc;
  plugin_name_ = name;
  const nvinfer1::PluginField* fields = fc->fields;

  bool ceil_mode;
  PoolPlugin::PoolType pool_type;
  bool adaptive;
  bool exclusive;
  std::vector<int> ksize;
  std::vector<int> strides;
  std::vector<int> paddings;
  std::vector<int> real_paddings;
  std::vector<int> input_shape;
  std::vector<int> output_shape;

  // TODO(wilber): add implement.
  CHECK(false) << "not implement";
  // for (int i = 0; i < fc->nbFields; ++i) {
  //   const char* attr_name = fields[i].name;
  //   if (!strcmp(attr_name, "ceil_mode")) {
  //     CHECK_EQ(fields[i].type == nvinfer1::PluginFieldType::kINT8, true);
  //     ceil_mode = *static_cast<const bool*>(fields[i].data);
  //     // mParam.numOutputBoxesPerClass =
  //     //     *(static_cast<const int*>(fields[i].data));
  //   }
  // }

  return nullptr;
}

nvinfer1::IPluginV2* PoolPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept {
  auto* plugin = new PoolPlugin(serialData, serialLength);
  plugin_name_ = name;
  return plugin;
}

}  // namespace plugin
}  // namespace tensorrt
}  // namespace backends
}  // namespace infrt
