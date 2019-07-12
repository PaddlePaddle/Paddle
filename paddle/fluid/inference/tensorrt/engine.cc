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
    if (calibrator_) {
      infer_builder_->setInt8Calibrator(calibrator_);
    } else {
      infer_builder_->setInt8Calibrator(nullptr);

#if IS_TRT_VERSION_GE(5000)
      infer_builder_->setStrictTypeConstraints(true);
      for (auto &quant_range : quant_dynamic_range_) {
        auto tensor = quant_range.first;
        float range = quant_range.second;
        tensor->setDynamicRange(-range, range);
      }

      std::unordered_set<nvinfer1::ITensor *> all_t;
      for (int i = 0; i < infer_network_->getNbLayers(); i++) {
        auto layer = infer_network_->getLayer(i);
        for (int j = 0; j < layer->getNbOutputs(); j++) {
          all_t.insert(layer->getOutput(j));
        }
      }
      for (int i = 0; i < infer_network_->getNbInputs(); i++) {
        all_t.insert(infer_network_->getInput(i));
      }

      for (auto &t : all_t) {
        if (!quant_dynamic_range_.count(t)) {
          LOG(WARNING)
              << "We are in trt int8 mode(not calibration), scale not setted"
              << " for tensor " << t->getName()
              << ", this might be ok when trt does not need this range";
        }
      }
#endif
    }
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

float *TensorRTEngine::GetWeightCPUData(const std::string &name,
                                        framework::Tensor *weight_tensor,
                                        bool enable_int8,
                                        const std::vector<float> &scale) {
  auto w_dims = weight_tensor->dims();
  platform::CPUPlace cpu_place;
  PADDLE_ENFORCE(!weight_map.count(name),
                 "During TRT Op converter: We set weight %s with the same name "
                 "twice into the weight_map",
                 name);
  weight_map[name].reset(new framework::Tensor());
  weight_map[name]->Resize(weight_tensor->dims());
  TensorCopySync(*weight_tensor, cpu_place, weight_map[name].get());
  float *weight_data = weight_map[name]->mutable_data<float>(cpu_place);

  if (enable_int8) {
    // when the op is fc, scale's size should be 1
    // when the op is conv, the scale's size should be w_dims[0]
    bool valid_scale_size =
        (scale.size() == 1 || scale.size() == static_cast<size_t>(w_dims[0]));
    PADDLE_ENFORCE(valid_scale_size, "TRT int8 quant: invalid scale size");
    for (int i = 0; i < weight_tensor->numel(); i++) {
      bool is_valid_int8 =
          ((weight_data[i] >= -128) && (weight_data[i] <= 127));
      PADDLE_ENFORCE(is_valid_int8,
                     "We are in anakin subgraph int8 mode, the weight of conv "
                     "should be in range [-128, 127]");
      if (scale.size() == 1) {
        weight_data[i] *= (scale[0] / 127);
      } else {
        PADDLE_ENFORCE(w_dims.size() == 4,
                       "TRT int8 quant : We only use the channel quant for "
                       "conv op, so the weight dims should be 4.");
        int inner_size = w_dims[1] * w_dims[2] * w_dims[3];
        weight_data[i] *= (scale[i / inner_size] / 127);
      }
    }
  }
  return weight_data;
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
