/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/inference/tensorrt/engine.h"

#include <NvInfer.h>
#include <cuda.h>
#include <glog/logging.h>
#include <string>
#include "paddle/fluid/inference/analysis/helper.h"
#include "paddle/fluid/inference/tensorrt/helper.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace inference {
namespace tensorrt {

int TensorRTEngine::runtime_batch_ = 1;

void TensorRTEngine::Build(const DescType &paddle_model) {
  PADDLE_ENFORCE(false, "not implemented");
}

void TensorRTEngine::Execute(int batch_size, std::vector<void *> *buffers,
                             cudaStream_t stream) {
  freshDeviceId();
  batch_size_ = batch_size;
  infer_context_->enqueue(batch_size, buffers->data(), stream, nullptr);
  cudaStreamSynchronize(stream);
  SetRuntimeBatch(batch_size);
}

void TensorRTEngine::FreezeNetwork() {
  freshDeviceId();
  VLOG(3) << "TRT to freeze network";
  PADDLE_ENFORCE(infer_builder_ != nullptr,
                 "Call InitNetwork first to initialize network.");
  PADDLE_ENFORCE(infer_network_ != nullptr,
                 "Call InitNetwork first to initialize network.");
  // build engine.
  infer_builder_->setMaxBatchSize(max_batch_);
  infer_builder_->setMaxWorkspaceSize(max_workspace_);
  if (enable_int8_) {
    infer_builder_->setInt8Mode(true);
    PADDLE_ENFORCE(
        calibrator_ != nullptr,
        "The precision mode is 'INT8', the calibrator should not be nullptr");
    infer_builder_->setInt8Calibrator(calibrator_);
  }

  infer_engine_.reset(infer_builder_->buildCudaEngine(*infer_network_));
  PADDLE_ENFORCE(infer_engine_ != nullptr, "build cuda engine failed!");

  infer_context_.reset(infer_engine_->createExecutionContext());
}

nvinfer1::ITensor *TensorRTEngine::DeclareInput(const std::string &name,
                                                nvinfer1::DataType dtype,
                                                const nvinfer1::Dims &dims) {
  PADDLE_ENFORCE_EQ(0, buffer_sizes_.count(name), "duplicate input name %s",
                    name);

  PADDLE_ENFORCE(infer_network_ != nullptr, "should initnetwork first");
  auto *input = infer_network_->addInput(name.c_str(), dtype, dims);
  PADDLE_ENFORCE(input, "infer network add input %s failed", name);
  buffer_sizes_[name] = kDataTypeSize[static_cast<int>(dtype)] *
                        analysis::AccuDims(dims.d, dims.nbDims) * max_batch_;
  PADDLE_ENFORCE(input->isNetworkInput());
  TensorRTEngine::SetITensor(name, input);
  return input;
}

void TensorRTEngine::DeclareOutput(const nvinfer1::ILayer *layer, int offset,
                                   const std::string &name) {
  PADDLE_ENFORCE_EQ(0, buffer_sizes_.count(name), "duplicate output name %s",
                    name);

  auto *output = layer->getOutput(offset);
  SetITensor(name, output);
  PADDLE_ENFORCE(output != nullptr);
  output->setName(name.c_str());
  PADDLE_ENFORCE(!output->isNetworkInput());
  infer_network_->markOutput(*output);
  PADDLE_ENFORCE(output->isNetworkOutput());
  // output buffers' size can only be decided latter, set zero here to mark this
  // and will reset latter.
  buffer_sizes_[name] = 0;
}

bool TensorRTEngine::HasDeclared(const std::string &name) {
  return buffer_sizes_.count(name) > 0;
}

void TensorRTEngine::DeclareOutput(const std::string &name) {
  PADDLE_ENFORCE_EQ(0, buffer_sizes_.count(name), "duplicate output name %s",
                    name);

  auto *output = TensorRTEngine::GetITensor(name);
  PADDLE_ENFORCE(output != nullptr);
  output->setName(name.c_str());
  PADDLE_ENFORCE(!output->isNetworkInput());
  infer_network_->markOutput(*output);
  // output buffers' size can only be decided latter, set zero here to mark this
  // and will reset latter.
  buffer_sizes_[name] = 0;
}

void TensorRTEngine::SetITensor(const std::string &name,
                                nvinfer1::ITensor *tensor) {
  PADDLE_ENFORCE(tensor != nullptr);
  PADDLE_ENFORCE_EQ(0, itensor_map_.count(name), "duplicate ITensor name %s",
                    name);
  itensor_map_[name] = tensor;
}

nvinfer1::ITensor *TensorRTEngine::GetITensor(const std::string &name) {
  PADDLE_ENFORCE(itensor_map_.count(name), "no ITensor %s", name);
  return itensor_map_[name];
}

void TensorRTEngine::SetRuntimeBatch(size_t batch_size) {
  runtime_batch_ = batch_size;
}

int TensorRTEngine::GetRuntimeBatch() { return runtime_batch_; }

nvinfer1::IPluginLayer *TensorRTEngine::AddPlugin(
    nvinfer1::ITensor *const *inputs, int num_inputs,
    plugin::PluginTensorRT *plugin) {
  owned_plugin_.emplace_back(plugin);
  return infer_network_.get()->addPluginExt(inputs, num_inputs, *plugin);
}

void TensorRTEngine::freshDeviceId() {
  int count;
  cudaGetDeviceCount(&count);
  PADDLE_ENFORCE_LT(device_id_, count);
  cudaSetDevice(device_id_);
}

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
