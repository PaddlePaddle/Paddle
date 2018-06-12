/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
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

void TensorRTEngine::Build(const DescType& paddle_model) {
  PADDLE_ENFORCE(false, "not implemented");
}

void TensorRTEngine::Execute(int batch_size) {
  std::vector<void*> buffers;
  for (auto& buf : buffers_) {
    PADDLE_ENFORCE_NOT_NULL(buf.buffer, "buffer should be allocated");
    PADDLE_ENFORCE_GT(buf.max_size, 0);
    PADDLE_ENFORCE(buf.device == DeviceType::GPU);
    buffers.push_back(buf.buffer);
  }
  infer_context_->enqueue(batch_size, buffers.data(), *stream_, nullptr);
  cudaStreamSynchronize(*stream_);
}

TensorRTEngine::~TensorRTEngine() {
  cudaStreamSynchronize(*stream_);
  // clean buffer
  for (auto& buf : buffers_) {
    if (buf.device == DeviceType::GPU && buf.buffer != nullptr) {
      PADDLE_ENFORCE_EQ(0, cudaFree(buf.buffer));
      buf.buffer = nullptr;
      buf.max_size = 0;
    }
  }
}

void TensorRTEngine::FreezeNetwork() {
  PADDLE_ENFORCE(infer_builder_ != nullptr,
                 "Call InitNetwork first to initialize network.");
  PADDLE_ENFORCE(infer_network_ != nullptr,
                 "Call InitNetwork first to initialize network.");
  // build engine.
  infer_builder_->setMaxBatchSize(max_batch_);
  infer_builder_->setMaxWorkspaceSize(max_workspace_);

  infer_engine_.reset(infer_builder_->buildCudaEngine(*infer_network_));
  PADDLE_ENFORCE(infer_engine_ != nullptr, "build cuda engine failed!");

  infer_context_.reset(infer_engine_->createExecutionContext());

  // allocate GPU buffers.
  buffers_.resize(buffer_sizes_.size());
  for (auto& item : buffer_sizes_) {
    if (item.second == 0) {
      auto slot_offset = infer_engine_->getBindingIndex(item.first.c_str());
      auto dims = infer_engine_->getBindingDimensions(slot_offset);
      item.second = kDataTypeSize[static_cast<int>(
                        infer_engine_->getBindingDataType(slot_offset))] *
                    analysis::AccuDims(dims.d, dims.nbDims);
    }
    auto& buf = buffer(item.first);
    CHECK(buf.buffer == nullptr);  // buffer should be allocated only once.
    PADDLE_ENFORCE_EQ(0, cudaMalloc(&buf.buffer, item.second));
    VLOG(4) << "buffer malloc " << item.first << " " << item.second << " "
            << buf.buffer;
    buf.size = buf.max_size = item.second;
    buf.device = DeviceType::GPU;
  }
}

nvinfer1::ITensor* TensorRTEngine::DeclareInput(const std::string& name,
                                                nvinfer1::DataType dtype,
                                                const nvinfer1::Dims& dims) {
  PADDLE_ENFORCE_EQ(0, buffer_sizes_.count(name), "duplicate input name %s",
                    name);

  PADDLE_ENFORCE(infer_network_ != nullptr, "should initnetwork first");
  auto* input = infer_network_->addInput(name.c_str(), dtype, dims);
  PADDLE_ENFORCE(input, "infer network add input %s failed", name);
  buffer_sizes_[name] = kDataTypeSize[static_cast<int>(dtype)] *
                        analysis::AccuDims(dims.d, dims.nbDims);
  PADDLE_ENFORCE(input->isNetworkInput());
  TensorRTEngine::SetITensor(name, input);
  return input;
}

void TensorRTEngine::DeclareOutput(const nvinfer1::ILayer* layer, int offset,
                                   const std::string& name) {
  PADDLE_ENFORCE_EQ(0, buffer_sizes_.count(name), "duplicate output name %s",
                    name);

  auto* output = layer->getOutput(offset);
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

void TensorRTEngine::DeclareOutput(const std::string& name) {
  PADDLE_ENFORCE_EQ(0, buffer_sizes_.count(name), "duplicate output name %s",
                    name);

  auto* output = TensorRTEngine::GetITensor(name);
  PADDLE_ENFORCE(output != nullptr);
  output->setName(name.c_str());
  PADDLE_ENFORCE(!output->isNetworkInput());
  infer_network_->markOutput(*output);
  // output buffers' size can only be decided latter, set zero here to mark this
  // and will reset latter.
  buffer_sizes_[name] = 0;
}

void* TensorRTEngine::GetOutputInGPU(const std::string& name) {
  return buffer(name).buffer;
}

void TensorRTEngine::GetOutputInGPU(const std::string& name, void* dst,
                                    size_t max_size) {
  // determine data size
  auto it = buffer_sizes_.find(name);
  PADDLE_ENFORCE(it != buffer_sizes_.end());
  PADDLE_ENFORCE_GT(it->second, 0);
  PADDLE_ENFORCE_GE(max_size, it->second);
  auto& buf = buffer(name);
  PADDLE_ENFORCE_NOT_NULL(buf.buffer, "buffer should be allocated before");
  PADDLE_ENFORCE_EQ(cudaMemcpyAsync(dst, buf.buffer, it->second,
                                    cudaMemcpyDeviceToDevice, *stream_),
                    0);
}

void TensorRTEngine::GetOutputInCPU(const std::string& name, void* dst,
                                    size_t max_size) {
  // determine data size
  auto it = buffer_sizes_.find(name);
  PADDLE_ENFORCE(it != buffer_sizes_.end());
  PADDLE_ENFORCE_GT(it->second, 0);
  PADDLE_ENFORCE_GE(max_size, it->second);
  auto& buf = buffer(name);
  PADDLE_ENFORCE_NOT_NULL(buf.buffer, "buffer should be allocated before");
  PADDLE_ENFORCE_EQ(0, cudaMemcpyAsync(dst, buf.buffer, it->second,
                                       cudaMemcpyDeviceToHost, *stream_));
}

Buffer& TensorRTEngine::buffer(const std::string& name) {
  PADDLE_ENFORCE(infer_engine_ != nullptr, "call FreezeNetwork first.");
  auto it = buffer_sizes_.find(name);
  PADDLE_ENFORCE(it != buffer_sizes_.end());
  auto slot_offset = infer_engine_->getBindingIndex(name.c_str());
  return buffers_[slot_offset];
}

void TensorRTEngine::SetInputFromCPU(const std::string& name, const void* data,
                                     size_t size) {
  auto& buf = buffer(name);
  PADDLE_ENFORCE_NOT_NULL(buf.buffer);
  PADDLE_ENFORCE_LE(size, buf.max_size, "buffer is too small");
  PADDLE_ENFORCE(buf.device == DeviceType::GPU);
  PADDLE_ENFORCE_EQ(0, cudaMemcpyAsync(buf.buffer, data, size,
                                       cudaMemcpyHostToDevice, *stream_));
}

void TensorRTEngine::SetInputFromGPU(const std::string& name, const void* data,
                                     size_t size) {
  auto& buf = buffer(name);
  PADDLE_ENFORCE_NOT_NULL(buf.buffer);
  PADDLE_ENFORCE_LE(size, buf.max_size, "buffer is too small");
  PADDLE_ENFORCE(buf.device == DeviceType::GPU);
  PADDLE_ENFORCE_EQ(0, cudaMemcpyAsync(buf.buffer, data, size,
                                       cudaMemcpyDeviceToDevice, *stream_));
}

void TensorRTEngine::SetITensor(const std::string& name,
                                nvinfer1::ITensor* tensor) {
  PADDLE_ENFORCE(tensor != nullptr);
  PADDLE_ENFORCE_EQ(0, itensor_map_.count(name), "duplicate ITensor name %s",
                    name);
  itensor_map_[name] = tensor;
}

nvinfer1::ITensor* TensorRTEngine::GetITensor(const std::string& name) {
  PADDLE_ENFORCE(itensor_map_.count(name), "no ITensor %s", name);
  return itensor_map_[name];
}

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
