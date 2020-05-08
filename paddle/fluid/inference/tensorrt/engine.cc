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

void TensorRTEngine::InitNetwork() {
  freshDeviceId();
  infer_builder_.reset(createInferBuilder(&logger_));

  if (with_dynamic_shape_) {
#if IS_TRT_VERSION_GE(6000)
    infer_networkv2_.reset(infer_builder_->createNetworkV2(
        1U << static_cast<int>(
            nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)));
    infer_builder_config_.reset(infer_builder_->createBuilderConfig());
    infer_ptr<nvinfer1::IBuilderConfig> infer_builder_config_;
    optim_profile_.reset(infer_builder_->createOptimizationProfile());
#endif
  } else {
    infer_network_.reset(infer_builder_->createNetwork());
  }
}

void TensorRTEngine::Execute(int batch_size, std::vector<void *> *buffers,
                             cudaStream_t stream) {
  freshDeviceId();
  auto infer_context = context();
  if (!with_dynamic_shape()) {
    infer_context->enqueue(batch_size, buffers->data(), stream, nullptr);
  } else {
#if IS_TRT_VERSION_GE(6000)
    infer_context->enqueueV2(buffers->data(), stream, nullptr);
#endif
  }
  SetRuntimeBatch(batch_size);
}

void TensorRTEngine::FreezeNetwork() {
  freshDeviceId();
  VLOG(3) << "TRT to freeze network";
  PADDLE_ENFORCE(infer_builder_ != nullptr,
                 "Call InitNetwork first to initialize network.");
  PADDLE_ENFORCE_EQ(network() != nullptr, true,
                    platform::errors::InvalidArgument(
                        "Call InitNetwork first to initialize network."));
  // build engine.
  infer_builder_->setMaxBatchSize(max_batch_);
  infer_builder_->setMaxWorkspaceSize(max_workspace_);
  bool enable_fp16 = (precision_ == AnalysisConfig::Precision::kHalf);
#if IS_TRT_VERSION_GE(5000)
  if (enable_fp16) {
    bool support_fp16 = infer_builder_->platformHasFastFp16();
    infer_builder_->setFp16Mode(support_fp16);
    if (!support_fp16) {
      LOG(INFO) << "You specify FP16 mode, but the hardware do not support "
                   "FP16 speed up, use FP32 instead.";
    } else {
      LOG(INFO) << "Run Paddle-TRT FP16 mode";
    }
  }
#else
  if (enable_fp16)
    LOG(INFO) << "Using FP16 in Paddle-TRT must ensure that the version of TRT "
                 "is at least 5."
                 "So, use FP32 to run.";
#endif
  bool enable_int8 = (precision_ == AnalysisConfig::Precision::kInt8);

  if (enable_int8) {
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
      for (int i = 0; i < network()->getNbLayers(); i++) {
        auto layer = network()->getLayer(i);
        for (int j = 0; j < layer->getNbOutputs(); j++) {
          all_t.insert(layer->getOutput(j));
        }
      }
      for (int i = 0; i < network()->getNbInputs(); i++) {
        all_t.insert(network()->getInput(i));
      }

      for (auto &t : all_t) {
        if (!quant_dynamic_range_.count(t)) {
          VLOG(3) << "We are in trt int8 mode(not calibration), scale not set"
                  << " for tensor " << t->getName()
                  << ", this might be ok when trt does not need this range";
        }
      }
      std::unordered_set<std::string> all_out_t_name;
      for (int i = 0; i < network()->getNbOutputs(); i++) {
        auto *temp = network()->getOutput(i);
        temp->setDynamicRange(-1, 1);
        all_out_t_name.insert(temp->getName());
      }

      for (int i = 0; i < network()->getNbLayers(); i++) {
        auto layer = network()->getLayer(i);
        for (int j = 0; j < layer->getNbOutputs(); j++) {
          auto *temp_out = layer->getOutput(j);
          if (std::find(all_out_t_name.begin(), all_out_t_name.end(),
                        temp_out->getName()) != all_out_t_name.end()) {
            layer->setPrecision(nvinfer1::DataType::kFLOAT);
            layer->setOutputType(j, nvinfer1::DataType::kFLOAT);
          }
        }
      }
#endif
    }
  }

  if (with_dynamic_shape_) {
#if IS_TRT_VERSION_GE(6000)
    LOG(INFO) << "Run Paddle-TRT Dynamic Shape mode.";
    for (auto &input : min_input_shape_) {
      optim_profile_->setDimensions(
          input.first.c_str(), nvinfer1::OptProfileSelector::kMIN,
          Vec2TRT_Dims(input.second, input.first, true));
      optim_profile_->setDimensions(
          input.first.c_str(), nvinfer1::OptProfileSelector::kMAX,
          Vec2TRT_Dims(max_input_shape_[input.first], input.first, true));
      optim_profile_->setDimensions(
          input.first.c_str(), nvinfer1::OptProfileSelector::kOPT,
          Vec2TRT_Dims(optim_input_shape_[input.first], input.first, true));
    }
    infer_builder_config_->addOptimizationProfile(optim_profile_.get());
    if (WithFp16()) {
      infer_builder_config_->setFlag(nvinfer1::BuilderFlag::kFP16);
      if (disable_trt_plugin_fp16()) {
        LOG(INFO) << "NOTE: In order to achieve higher accuracy, you have "
                     "disabled the fp16 mode of TRT Plugin,\n"
                  << "you can reopen it with "
                     "'config.SetDynamicShapeInfo(min_shape, max_shape, "
                     "opt_shape, false /*disable_trt_plugin_fp16*/)'";
      }
    }
    infer_engine_.reset(infer_builder_->buildEngineWithConfig(
        *network(), *infer_builder_config_));
#endif
  } else {
    infer_engine_.reset(infer_builder_->buildCudaEngine(*network()));
  }
  PADDLE_ENFORCE(infer_engine_ != nullptr, "build cuda engine failed!");
}

nvinfer1::ITensor *TensorRTEngine::DeclareInput(const std::string &name,
                                                nvinfer1::DataType dtype,
                                                const nvinfer1::Dims &dims) {
  PADDLE_ENFORCE_EQ(network() != nullptr, true,
                    platform::errors::InvalidArgument(
                        "The TRT network should be initialized first."));
  auto *input = network()->addInput(name.c_str(), dtype, dims);
  PADDLE_ENFORCE(input, "infer network add input %s failed", name);
  PADDLE_ENFORCE(input->isNetworkInput());
  TensorRTEngine::SetITensor(name, input);
  return input;
}

void TensorRTEngine::DeclareOutput(const nvinfer1::ILayer *layer, int offset,
                                   const std::string &name) {
  auto *output = layer->getOutput(offset);
  SetITensor(name, output);
  PADDLE_ENFORCE(output != nullptr);
  output->setName(name.c_str());
  PADDLE_ENFORCE(!output->isNetworkInput());
  network()->markOutput(*output);
  PADDLE_ENFORCE(output->isNetworkOutput());
}

void TensorRTEngine::DeclareOutput(const std::string &name) {
  auto *output = TensorRTEngine::GetITensor(name);
  PADDLE_ENFORCE(output != nullptr);
  output->setName(name.c_str());
  PADDLE_ENFORCE(!output->isNetworkInput());
  network()->markOutput(*output);
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
  static int name_suffix_counter = 0;
  std::string name_suffix = std::to_string(name_suffix_counter);
  std::string splitter = "__";
  std::string name_with_suffix = name + splitter + name_suffix;
  auto w_dims = weight_tensor->dims();
  platform::CPUPlace cpu_place;
  PADDLE_ENFORCE_EQ(
      weight_map.count(name_with_suffix), 0,
      "During TRT Op converter: We set weight %s with the same name "
      "twice into the weight_map",
      name_with_suffix);
  weight_map[name_with_suffix].reset(new framework::Tensor());
  weight_map[name_with_suffix]->Resize(weight_tensor->dims());
  TensorCopySync(*weight_tensor, cpu_place, weight_map[name_with_suffix].get());
  float *weight_data =
      weight_map[name_with_suffix]->mutable_data<float>(cpu_place);
  name_suffix_counter += 1;

  if (enable_int8) {
    // when the op is fc, scale's size should be 1
    // when the op is conv, scale's size should be w_dims[0]
    bool valid_scale_size =
        (scale.size() == 1 || scale.size() == static_cast<size_t>(w_dims[0]));
    PADDLE_ENFORCE(valid_scale_size, "TRT int8 quant: invalid scale size");
    for (int i = 0; i < weight_tensor->numel(); i++) {
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
  return network()->addPluginExt(inputs, num_inputs, *plugin);
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
