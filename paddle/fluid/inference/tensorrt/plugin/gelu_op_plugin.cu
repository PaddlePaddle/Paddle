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

#include <cassert>
#include <cstring>
#include <vector>
#include "paddle/fluid/inference/tensorrt/plugin/gelu_op_plugin.h"
#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin_factory.h"
#include "pluginKernels.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

// constants for approximating the normal cdf
constexpr float A = 0.5;

constexpr float B = 0.7978845608028654;  // sqrt(2.0/M_PI)

constexpr float C = 0.035677408136300125;  // 0.044715 * sqrt(2.0/M_PI)

template <typename T, unsigned TPB>
__global__ void geluKernel(const T a, const T b, const T c, int n,
                           const T* input, T* output) {
  const int idx = blockIdx.x * TPB + threadIdx.x;

  if (idx < n) {
    const T in = input[idx];
    const T cdf = a + a * tanh(in * (c * in * in + b));
    output[idx] = in * cdf;
  }
}

int computeGelu(cudaStream_t stream, int n, const float* input, float* output) {
  constexpr int blockSize = 256;
  const int gridSize = (n + blockSize - 1) / blockSize;
  geluKernel<float, blockSize><<<gridSize, blockSize, 0, stream>>>(
      A, B, C, n, input, output);

  CHECK(cudaPeekAtLastError());
  return 0;
}

int computeGelu(cudaStream_t stream, int n, const half* input, half* output) {
  const int blockSize = 256;

  if (0 == (n & 1)) {
    const int n2 = n / 2;

    const int gridSize = (n2 + blockSize - 1) / blockSize;
    const half2 A2 = __floats2half2_rn(A, A);
    const half2 B2 = __floats2half2_rn(B, B);
    const half2 C2 = __floats2half2_rn(C, C);
    const half2* input2 = reinterpret_cast<const half2*>(input);
    half2* output2 = reinterpret_cast<half2*>(output);
    geluKernel<half2, blockSize><<<gridSize, blockSize, 0, stream>>>(
        A2, B2, C2, n2, input2, output2);
  } else {
    const int gridSize = (n + blockSize - 1) / blockSize;
    geluKernel<half, blockSize><<<gridSize, blockSize, 0, stream>>>(
        A, B, C, n, input, output);
  }

  CHECK(cudaPeekAtLastError());
  return 0;
}

namespace {
static const char* GELU_PLUGIN_VERSION{"1"};
static const char* GELU_PLUGIN_NAME{"CustomGeluPlugin"};
}  // namespace

// Static class fields initialization
PluginFieldCollection GeluPluginCreator::mFC{};
std::vector<PluginField> GeluPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(GeluPluginCreator);

GeluPlugin::GeluPlugin(const std::string name) : mLayerName(name) {}

GeluPlugin::GeluPlugin(const std::string name, const void* data, size_t length)
    : mLayerName(name) {
  gLogVerbose << "Gelu Deser start" << std::endl;
  const char* d = static_cast<const char*>(data);
  const char* a = d;
  mInputVolume = readFromBuffer<decltype(mInputVolume)>(d);
  mType = readFromBuffer<DataType>(d);
  assert(d == a + length);
  gLogVerbose << "Gelu Deser done" << std::endl;
}

const char* GeluPlugin::getPluginType() const { return GELU_PLUGIN_NAME; }

const char* GeluPlugin::getPluginVersion() const { return GELU_PLUGIN_VERSION; }

int GeluPlugin::getNbOutputs() const { return 1; }

Dims GeluPlugin::getOutputDimensions(int index, const Dims* inputs,
                                     int nbInputDims) {
  // Validate input arguments
  assert(nbInputDims == 1);
  assert(index == 0);

  // doesn't change input dimension, so output Dims will be the same as
  // input Dims
  return *inputs;
}

int GeluPlugin::initialize() { return 0; }

int GeluPlugin::enqueue(int batchSize, const void* const* inputs,
                        void** outputs, void*, cudaStream_t stream) {
  int status = -1;

  // Our plugin outputs only one tensor
  // Launch CUDA kernel wrapper and save its return value
  if (mType == DataType::kFLOAT) {
    const float* input = static_cast<const float*>(inputs[0]);
    float* output = static_cast<float*>(outputs[0]);
    status = computeGelu(stream, mInputVolume * batchSize, input, output);
  } else if (mType == DataType::kHALF) {
    const half* input = static_cast<const half*>(inputs[0]);
    half* output = static_cast<half*>(outputs[0]);
    status = computeGelu(stream, mInputVolume * batchSize, input, output);
  } else {
    assert(false);
  }

  return status;
}

size_t GeluPlugin::getSerializationSize() const {
  return sizeof(mInputVolume) + sizeof(DataType);
}

void GeluPlugin::serialize(void* buffer) const {
  char *d = static_cast<char *>(buffer), *a = d;
  writeToBuffer(d, mInputVolume);
  writeToBuffer(d, mType);
  assert(d == a + getSerializationSize());
}

void GeluPlugin::configureWithFormat(const Dims* inputs, int nbInputs,
                                     const Dims* outputs, int nbOutputs,
                                     DataType type, PluginFormat format, int) {
  // Validate input arguments
  assert(nbOutputs == 1);
  assert(format == PluginFormat::kNCHW);

  // Fetch volume for future enqueue() operations
  size_t volume = 1;
  for (int i = 0; i < inputs->nbDims; i++) {
    volume *= inputs->d[i];
  }
  mInputVolume = volume;
  mType = type;
}

bool GeluPlugin::supportsFormat(DataType type, PluginFormat format) const {
  if (type == DataType::kFLOAT || type == DataType::kHALF)
    return format == PluginFormat::kNCHW;
  else
    return false;
}

void GeluPlugin::terminate() {}

void GeluPlugin::destroy() {
  // This gets called when the network containing plugin is destroyed
  delete this;
}

IPluginV2* GeluPlugin::clone() const { return new GeluPlugin(mLayerName); }

void GeluPlugin::setPluginNamespace(const char* libNamespace) {
  mNamespace = libNamespace;
}

const char* GeluPlugin::getPluginNamespace() const {
  return mNamespace.c_str();
}

GeluPluginCreator::GeluPluginCreator() {
  // Fill PluginFieldCollection with PluginField arguments metadata
  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char* GeluPluginCreator::getPluginName() const {
  return GELU_PLUGIN_NAME;
}

const char* GeluPluginCreator::getPluginVersion() const {
  return GELU_PLUGIN_VERSION;
}

const PluginFieldCollection* GeluPluginCreator::getFieldNames() { return &mFC; }

IPluginV2* GeluPluginCreator::createPlugin(const char* name,
                                           const PluginFieldCollection* fc) {
  gLogVerbose << "Creating GeluPlugin...\n";
  GeluPlugin* p = new GeluPlugin(name);
  return p;
}

IPluginV2* GeluPluginCreator::deserializePlugin(const char* name,
                                                const void* serialData,
                                                size_t serialLength) {
  // This object will be deleted when the network is destroyed, which will
  // call GeluPlugin::destroy()
  return new GeluPlugin(name, serialData, serialLength);
}

void GeluPluginCreator::setPluginNamespace(const char* libNamespace) {
  mNamespace = libNamespace;
}

const char* GeluPluginCreator::getPluginNamespace() const {
  return mNamespace.c_str();
}

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
