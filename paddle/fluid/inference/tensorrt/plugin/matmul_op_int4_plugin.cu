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

#include "paddle/fluid/inference/tensorrt/plugin/matmul_op_int4_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

char const* PLUGINVERSION{"1"};
char const* MATMULINT4PLUGINNAME{"MatmulInt4Plugin"};

MatmulInt4Plugin::MatmulInt4Plugin(nvinfer1::Dims const& dims_x,
                                   nvinfer1::Dims const& dims_y)
    : dims_x_(dims_x),
      dims_y_(dims_y),
      Atmp_(nullptr),
      Btmp_(nullptr),
      Cres_(nullptr),
      Aconvert_(nullptr),
      Bconvert_(nullptr) {}

MatmulInt4Plugin::MatmulInt4Plugin(void const* data, size_t length) {
  DeserializeValue(&data, &length, &dims_x_);
  DeserializeValue(&data, &length, &dims_y_);
  DeserializeValue(&data, &length, &Atmp_);
  DeserializeValue(&data, &length, &Btmp_);
  DeserializeValue(&data, &length, &Cres_);
  DeserializeValue(&data, &length, &Aconvert_);
  DeserializeValue(&data, &length, &Bconvert_);
  DeserializeValue(&data, &length, &type_);
}

void MatmulInt4Plugin::configurePlugin(nvinfer1::PluginTensorDesc const* in,
                                       int32_t nb_inputs,
                                       nvinfer1::PluginTensorDesc const* out,
                                       int32_t nb_outputs) noexcept {
  m_ = dims_x_.d[dims_x_.nbDims - 2];
  k_ = dims_x_.d[dims_x_.nbDims - 1];
  n_ = dims_y_.d[dims_y_.nbDims - 1];
  type_ = in[0].type;
  uint64_t mk = m_ * k_;
  uint64_t kn = k_ * n_;
  uint64_t mn = m_ * n_;
  PADDLE_ENFORCE_GPU_SUCCESS(
      cudaMalloc(reinterpret_cast<void**>(&Aconvert_), mk / 2));
  PADDLE_ENFORCE_GPU_SUCCESS(
      cudaMalloc(reinterpret_cast<void**>(&Bconvert_), kn / 2));
  PADDLE_ENFORCE_GPU_SUCCESS(
      cudaMalloc(reinterpret_cast<void**>(&Cres_), mn * 4));
}

bool MatmulInt4Plugin::supportsFormatCombination(
    int32_t pos,
    nvinfer1::PluginTensorDesc const* in_out,
    int32_t nb_inputs,
    int32_t nb_outputs) const noexcept {
  PADDLE_ENFORCE_EQ(nb_inputs,
                    2,
                    platform::errors::InvalidArgument("Must have 2 inputs, "
                                                      "but got %d input(s). ",
                                                      nb_inputs));
  PADDLE_ENFORCE_EQ(nb_outputs,
                    getNbOutputs(),
                    platform::errors::InvalidArgument("Must have 1 output, "
                                                      "but got %d output(s). ",
                                                      nb_outputs));
  if (pos == 0) {
    return (in_out[pos].type == nvinfer1::DataType::kHALF ||
            in_out[pos].type == nvinfer1::DataType::kINT8 ||
            in_out[pos].type == nvinfer1::DataType::kFLOAT) &&
           in_out[pos].format == nvinfer1::TensorFormat::kLINEAR;
  } else {
    return in_out[pos].type == in_out[0].type &&
           in_out[pos].format == in_out[pos].format;
  }
}

nvinfer1::DataType MatmulInt4Plugin::getOutputDataType(
    int32_t index,
    nvinfer1::DataType const* input_types,
    int32_t nb_inputs) const noexcept {
  return input_types[0];
}

bool MatmulInt4Plugin::isOutputBroadcastAcrossBatch(
    int32_t output_index,
    const bool* input_is_broadcasted,
    int32_t nb_inputs) const noexcept {
  return false;
}

bool MatmulInt4Plugin::canBroadcastInputAcrossBatch(
    int32_t input_index) const noexcept {
  return false;
}

void MatmulInt4Plugin::attachToContext(
    cudnnContext* cudnnContext,
    cublasContext* cublasContext,
    nvinfer1::IGpuAllocator* gpuAllocator) noexcept {}

void MatmulInt4Plugin::detachFromContext() noexcept {}

nvinfer1::IPluginV2Ext* MatmulInt4Plugin::clone() const noexcept {
  auto p = new MatmulInt4Plugin(dims_x_, dims_y_);
  p->setPluginNamespace(namespace_.c_str());
  p->Atmp_ = Atmp_;
  p->Btmp_ = Btmp_;
  p->Cres_ = Cres_;
  p->Aconvert_ = Aconvert_;
  p->Bconvert_ = Bconvert_;
  return p;
}

const char* MatmulInt4Plugin::getPluginType() const noexcept {
  return MATMULINT4PLUGINNAME;
}

const char* MatmulInt4Plugin::getPluginVersion() const noexcept {
  return PLUGINVERSION;
}

int32_t MatmulInt4Plugin::getNbOutputs() const noexcept { return 1; }

nvinfer1::Dims MatmulInt4Plugin::getOutputDimensions(
    int32_t index,
    nvinfer1::Dims const* inputs,
    int32_t nb_input_dims) noexcept {
  batch_ = 1;
  for (int i = 0; i < inputs[0].nbDims - 2; ++i) {
    batch_ *= inputs[0].d[i];
  }
  m_ = inputs[0].d[inputs[0].nbDims - 2];
  k_ = inputs[0].d[inputs[0].nbDims - 1];
  n_ = inputs[1].d[inputs[1].nbDims - 1];
  nvinfer1::Dims output_dims;
  output_dims.nbDims = inputs[0].nbDims;
  for (int i = 0; i < inputs[0].nbDims - 2; ++i) {
    output_dims.d[i] = inputs[0].d[i];
  }
  output_dims.d[output_dims.nbDims - 2] = m_;
  output_dims.d[output_dims.nbDims - 1] = n_;
  return output_dims;
}

int32_t MatmulInt4Plugin::initialize() noexcept { return 0; }

void MatmulInt4Plugin::terminate() noexcept {
  cudaFree(reinterpret_cast<void*>(Aconvert_));
  cudaFree(reinterpret_cast<void*>(Bconvert_));
  cudaFree(reinterpret_cast<void*>(Cres_));
}

size_t MatmulInt4Plugin::getWorkspaceSize(
    int32_t max_batch_size) const noexcept {
  return 0;
}

size_t MatmulInt4Plugin::getSerializationSize() const noexcept {
  return SerializedSize(dims_x_) + SerializedSize(dims_y_) +
         SerializedSize(batch_) + SerializedSize(m_) + SerializedSize(n_) +
         SerializedSize(k_) + SerializedSize(Atmp_) + SerializedSize(Btmp_) +
         SerializedSize(Cres_) + SerializedSize(Aconvert_) +
         SerializedSize(Bconvert_) + SerializedSize(type_);
}

void MatmulInt4Plugin::serialize(void* buffer) const noexcept {
  SerializeValue(&buffer, dims_x_);
  SerializeValue(&buffer, dims_y_);
  SerializeValue(&buffer, batch_);
  SerializeValue(&buffer, m_);
  SerializeValue(&buffer, n_);
  SerializeValue(&buffer, k_);
  SerializeValue(&buffer, Atmp_);
  SerializeValue(&buffer, Btmp_);
  SerializeValue(&buffer, Cres_);
  SerializeValue(&buffer, Aconvert_);
  SerializeValue(&buffer, Bconvert_);
  SerializeValue(&buffer, type_);
}

void MatmulInt4Plugin::destroy() noexcept { delete this; }

char const* MatmulInt4Plugin::getPluginNamespace() const noexcept {
  return namespace_.c_str();
}

void MatmulInt4Plugin::setPluginNamespace(
    char const* plugin_name_space) noexcept {
  namespace_ = plugin_name_space;
}

int32_t MatmulInt4Plugin::enqueue(int32_t batch_size,
                                  void const* const* inputs,
                                  void* const* outputs,
                                  void* workspace,
                                  cudaStream_t stream) noexcept {
  platform::CUDAPlace place(platform::GetCurrentDeviceId());

  phi::GPUContext ctx(place, false);
  phi::CUDAStream stream_(place,
                          phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
  ctx.SetCUDAStream(&stream_, false);

  if (type_ == nvinfer1::DataType::kFLOAT) {
    const float* A = static_cast<const float*>(inputs[0]);
    const float* B = static_cast<const float*>(inputs[1]);
    phi::fusion::cutlass_gemm_internal::ConvertDataToInt4<float>(
        A, Aconvert_, m_ * k_);
    phi::fusion::cutlass_gemm_internal::ConvertDataToInt4<float>(
        B, Bconvert_, k_ * n_);
  } else if (type_ == nvinfer1::DataType::kHALF) {
    const cutlass::half_t* A = static_cast<const cutlass::half_t*>(inputs[0]);
    const cutlass::half_t* B = static_cast<const cutlass::half_t*>(inputs[1]);
    phi::fusion::cutlass_gemm_internal::ConvertDataToInt4<cutlass::half_t>(
        A, Aconvert_, m_ * k_);
    phi::fusion::cutlass_gemm_internal::ConvertDataToInt4<cutlass::half_t>(
        B, Bconvert_, k_ * n_);
  } else if (type_ == nvinfer1::DataType::kINT8) {
    const int8_t* A = static_cast<const int8_t*>(inputs[0]);
    const int8_t* B = static_cast<const int8_t*>(inputs[1]);
    phi::fusion::cutlass_gemm_internal::ConvertDataToInt4<int8_t>(
        A, Aconvert_, m_ * k_);
    phi::fusion::cutlass_gemm_internal::ConvertDataToInt4<int8_t>(
        B, Bconvert_, k_ * n_);
  }
  phi::fusion::cutlass_gemm_internal::GemmAllParams params{
      Aconvert_, Bconvert_, nullptr, Cres_, batch_, m_, n_, k_, &ctx};
  int sm = phi::fusion::cutlass_gemm_internal::getSMVersion();
  phi::fusion::cutlass_gemm_internal::Int4Gemm(params, sm);
  if (type_ == nvinfer1::DataType::kFLOAT) {
    float* C = static_cast<float*>(outputs[0]);
    phi::fusion::cutlass_gemm_internal::ConvertData<int32_t, float>(
        Cres_, C, m_ * n_);
  } else if (type_ == nvinfer1::DataType::kHALF) {
    auto C = static_cast<cutlass::half_t*>(outputs[0]);
    phi::fusion::cutlass_gemm_internal::ConvertData<int32_t, cutlass::half_t>(
        Cres_, C, m_ * n_);
  } else if (type_ == nvinfer1::DataType::kINT8) {
    auto C = static_cast<int8_t*>(outputs[0]);
    phi::fusion::cutlass_gemm_internal::ConvertData<int32_t, int8_t>(
        Cres_, C, m_ * n_);
  }
  return cudaGetLastError() != cudaSuccess;
}

static nvinfer1::PluginFieldCollection field_collection_{0, nullptr};

MatmulInt4PluginCreator::MatmulInt4PluginCreator() {}

char const* MatmulInt4PluginCreator::getPluginName() const noexcept {
  return MATMULINT4PLUGINNAME;
}

char const* MatmulInt4PluginCreator::getPluginVersion() const noexcept {
  return PLUGINVERSION;
}

const nvinfer1::PluginFieldCollection*
MatmulInt4PluginCreator::getFieldNames() noexcept {
  return &field_collection_;
}

void MatmulInt4PluginCreator::setPluginNamespace(
    char const* plugin_namespace) noexcept {
  plugin_namespace_ = plugin_namespace;
}

char const* MatmulInt4PluginCreator::getPluginNamespace() const noexcept {
  return plugin_namespace_.c_str();
}

nvinfer1::IPluginV2* MatmulInt4PluginCreator::createPlugin(
    char const* name, const nvinfer1::PluginFieldCollection* fc) noexcept {
  return nullptr;
}

nvinfer1::IPluginV2* MatmulInt4PluginCreator::deserializePlugin(
    char const* name, void const* serial_data, size_t serial_length) noexcept {
  return new MatmulInt4Plugin(serial_data, serial_length);
}
}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
