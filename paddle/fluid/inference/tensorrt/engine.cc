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
#include <glog/logging.h>

#include <string>

#include "NvInferRuntimeCommon.h"
#include "cuda_runtime_api.h"  // NOLINT
#include "paddle/fluid/inference/tensorrt/helper.h"
#include "paddle/fluid/platform/device/gpu/gpu_info.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/phi/common/data_type.h"

namespace paddle {
namespace inference {
namespace tensorrt {

int TensorRTEngine::runtime_batch_ = 1;
thread_local int TensorRTEngine::predictor_id_per_thread = -1;

void TensorRTEngine::Weight::SetDataType(phi::DataType type) {
  nvinfer1::DataType nv_type = nvinfer1::DataType::kFLOAT;
  switch (type) {
    case phi::DataType::FLOAT32:
      nv_type = nvinfer1::DataType::kFLOAT;
      break;
    case phi::DataType::FLOAT16:
      nv_type = nvinfer1::DataType::kHALF;
      break;
    case phi::DataType::INT32:
      nv_type = nvinfer1::DataType::kINT32;
      break;
    case phi::DataType::INT8:
      nv_type = nvinfer1::DataType::kINT8;
      break;
#if IS_TRT_VERSION_GE(7000)
    case phi::DataType::BOOL:
      nv_type = nvinfer1::DataType::kBOOL;
      break;
#endif
    default:
      paddle::platform::errors::InvalidArgument(
          "Paddle-TRT loads weighths failed, found not supported data type %s.",
          type);
      break;
  }
  w_.type = nv_type;
}

void TensorRTEngine::InitNetwork() {
  freshDeviceId();
  infer_builder_.reset(createInferBuilder(&logger_));

  if (with_dynamic_shape_) {
    infer_network_.reset(infer_builder_->createNetworkV2(
        1U << static_cast<int>(
            nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)));
  } else {
    infer_network_.reset(infer_builder_->createNetworkV2(0U));
  }

  infer_builder_config_.reset(infer_builder_->createBuilderConfig());
  // optim_profile_ = infer_builder_->createOptimizationProfile();
  optim_profiles_.resize(max_profile_num_);
  for (int i = 0; i < max_profile_num_; i++)
    optim_profiles_[i] = infer_builder_->createOptimizationProfile();
}

nvinfer1::IExecutionContext *TensorRTEngine::context() {
  std::unique_lock<std::mutex> lock(mutex_);
  if (infer_context_.find(predictor_id_per_thread) == infer_context_.end()) {
    PADDLE_ENFORCE_NOT_NULL(
        infer_engine_,
        platform::errors::InvalidArgument(
            "You should build engine first and then set the context."));
    // We may see trt warning: Profile 0 has been chosen by another
    // IExecutionContext...
    // It's ok. We will set it later.
    nvinfer1::IExecutionContext *infer_context{nullptr};
    if (context_memory_sharing_) {
      infer_context =
          infer_engine_->createExecutionContextWithoutDeviceMemory();
    } else {
      infer_context = infer_engine_->createExecutionContext();
    }
    PADDLE_ENFORCE_NOT_NULL(
        infer_context,
        platform::errors::InvalidArgument(
            "TensorRT engine can not build execution context."));
    if (with_dynamic_shape_) {
      // need new profile if it's not the first
      if (cur_profile_num_ > 0) {
        infer_context->setOptimizationProfile(cur_profile_num_);
      }
      profile_index_[predictor_id_per_thread] = cur_profile_num_;
      ++cur_profile_num_;
    }
    infer_context_[predictor_id_per_thread].reset(infer_context);
  }
  return infer_context_[predictor_id_per_thread].get();
}

void TensorRTEngine::Execute(int batch_size,
                             std::vector<void *> *buffers,
                             cudaStream_t stream) {
  freshDeviceId();
  auto infer_context = context();
  if (context_memory_sharing_) {
    void *context_memory{nullptr};
    context_memory =
        inference::Singleton<inference::tensorrt::TRTEngineManager>::Global()
            .getContextMemory(
                predictor_id_per_thread,
                phi::GPUPlace(device_id_),
                phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
    infer_context->setDeviceMemory(context_memory);
  }
  if (!with_dynamic_shape()) {
    infer_context->enqueue(batch_size, buffers->data(), stream, nullptr);
  } else {
    infer_context->enqueueV2(buffers->data(), stream, nullptr);
  }
  SetRuntimeBatch(batch_size);
}

void TensorRTEngine::FreezeNetwork() {
  freshDeviceId();
  VLOG(3) << "TRT to freeze network";
  PADDLE_ENFORCE_NOT_NULL(infer_builder_,
                          platform::errors::InvalidArgument(
                              "Inference builder of TRT is null. Please make "
                              "sure you call InitNetwork first."));
  PADDLE_ENFORCE_NOT_NULL(network(),
                          platform::errors::InvalidArgument(
                              "Call InitNetwork first to initialize network."));
  // build engine.
  infer_builder_->setMaxBatchSize(max_batch_);
  infer_builder_config_->setMaxWorkspaceSize(max_workspace_);

  bool enable_fp16 = (precision_ == AnalysisConfig::Precision::kHalf);
  if (enable_fp16) {
    bool support_fp16 = infer_builder_->platformHasFastFp16();
    infer_builder_config_->setFlag(nvinfer1::BuilderFlag::kFP16);
    if (!support_fp16) {
      LOG(INFO) << "You specify FP16 mode, but the hardware do not support "
                   "FP16 speed up, use FP32 instead.";
    } else {
      LOG(INFO) << "Run Paddle-TRT FP16 mode";
    }
  }

  bool enable_int8 = (precision_ == AnalysisConfig::Precision::kInt8);
  if (enable_int8) {
    if (!use_dla_) {
      infer_builder_config_->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
    infer_builder_config_->setFlag(nvinfer1::BuilderFlag::kINT8);

    if (calibrator_) {
      infer_builder_config_->setInt8Calibrator(calibrator_);
    } else {
      infer_builder_config_->setInt8Calibrator(nullptr);

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
    }
  }

  // If model is mixed precision, then we should cast all float output to
  // float32 precision. Otherwise, we can not confirm the output precision of
  // the trt engine.
  if (model_precision_ != phi::DataType::FLOAT32) {
    for (int i = 0; i < network()->getNbOutputs(); ++i) {
      network()->getOutput(i)->setAllowedFormats(
          static_cast<nvinfer1::TensorFormats>(
              1 << static_cast<int>(nvinfer1::TensorFormat::kLINEAR)));
      network()->getOutput(i)->setType(nvinfer1::DataType::kFLOAT);
    }
  }

  if (use_dla_) {
    if (!enable_int8 && !enable_fp16) {
      LOG(WARNING) << "TensorRT DLA must be used with int8 or fp16, but you "
                      "set float32, so DLA is not used.";
    } else if (infer_builder_->getNbDLACores() == 0) {
      LOG(WARNING)
          << "TensorRT DLA is set by config, but your device does not have "
             "DLA, so DLA is not used.";
    } else {
      if (dla_core_ < 0 || dla_core_ >= infer_builder_->getNbDLACores()) {
        dla_core_ = 0;
        LOG(WARNING) << "Invalid DLACore, must be 0 < DLACore < "
                     << infer_builder_->getNbDLACores() << ", but got "
                     << dla_core_ << ", so use use 0 as default.";
      }
      infer_builder_config_->setDefaultDeviceType(nvinfer1::DeviceType::kDLA);
      infer_builder_config_->setDLACore(dla_core_);
      infer_builder_config_->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
      LOG(INFO) << "TensorRT DLA enabled in FreezeNetwork(), DLACore "
                << dla_core_;
    }
  }

  if (with_dynamic_shape_) {
    LOG(INFO) << "Run Paddle-TRT Dynamic Shape mode.";
    for (int i = 0; i < max_profile_num_; i++) {
      for (auto &input : min_input_shape_) {
#if IS_TRT_VERSION_LT(7000)
        // trt6 will check all_of input > 0
        if (!(std::all_of(input.second.begin(),
                          input.second.end(),
                          [](int x) { return x > 0; }) &&
              std::all_of(max_input_shape_[input.first].begin(),
                          max_input_shape_[input.first].end(),
                          [](int x) { return x > 0; }) &&
              std::all_of(optim_input_shape_[input.first].begin(),
                          optim_input_shape_[input.first].end(),
                          [](int x) { return x > 0; }))) {
          continue;
        }
#endif
        VLOG(4) << "TRT dynamic_shape set " << input.first
                << " min: " << Vec2Str(input.second)
                << ", max: " << Vec2Str(max_input_shape_[input.first])
                << ", opt: " << Vec2Str(optim_input_shape_[input.first]);

        optim_profiles_[i]->setDimensions(
            input.first.c_str(),
            nvinfer1::OptProfileSelector::kMIN,
            Vec2TRT_Dims(input.second, input.first, true));
        optim_profiles_[i]->setDimensions(
            input.first.c_str(),
            nvinfer1::OptProfileSelector::kMAX,
            Vec2TRT_Dims(max_input_shape_[input.first], input.first, true));
        optim_profiles_[i]->setDimensions(
            input.first.c_str(),
            nvinfer1::OptProfileSelector::kOPT,
            Vec2TRT_Dims(optim_input_shape_[input.first], input.first, true));
      }
      infer_builder_config_->addOptimizationProfile(optim_profiles_[i]);
    }
    if (WithFp16() && disable_trt_plugin_fp16()) {
      LOG(INFO) << "NOTE: In order to achieve higher accuracy, you have "
                   "disabled the fp16 mode of TRT Plugin,\n"
                << "you can reopen it with "
                   "'config.SetDynamicShapeInfo(min_shape, max_shape, "
                   "opt_shape, false /*disable_trt_plugin_fp16*/)'";
    }
  }
#if IS_TRT_VERSION_GE(8200)
  if (use_inspector_) {
    infer_builder_config_->setProfilingVerbosity(
        nvinfer1::ProfilingVerbosity::kDETAILED);
  }
#endif

#if IS_TRT_VERSION_LT(8000)
  infer_engine_.reset(infer_builder_->buildEngineWithConfig(
      *network(), *infer_builder_config_));
#else
  infer_builder_config_->setFlag(nvinfer1::BuilderFlag::kSPARSE_WEIGHTS);
  ihost_memory_.reset(infer_builder_->buildSerializedNetwork(
      *network(), *infer_builder_config_));
  infer_ptr<nvinfer1::IRuntime> runtime(createInferRuntime(&logger_));
  infer_engine_.reset(runtime->deserializeCudaEngine(ihost_memory_->data(),
                                                     ihost_memory_->size()));
#endif

  PADDLE_ENFORCE_NOT_NULL(
      infer_engine_,
      platform::errors::Fatal(
          "Build TensorRT cuda engine failed! Please recheck "
          "you configurations related to paddle-TensorRT."));

  binding_num_ = infer_engine_->getNbBindings();
  // reset status for dynamic shape clone
  if (max_profile_num_ > 1) {
    infer_context_.clear();
    cur_profile_num_ = 0;
  }
  // for engine context memory sharing
  if (context_memory_sharing_) {
    inference::Singleton<inference::tensorrt::TRTEngineManager>::Global()
        .updateContextMemorySize(infer_engine_->getDeviceMemorySize(),
                                 predictor_id_per_thread);
  }

  GetEngineInfo();
}

nvinfer1::ITensor *TensorRTEngine::DeclareInput(const std::string &name,
                                                nvinfer1::DataType dtype,
                                                const nvinfer1::Dims &dims) {
  PADDLE_ENFORCE_EQ(network() != nullptr,
                    true,
                    platform::errors::InvalidArgument(
                        "The TRT network should be initialized first."));
  auto *input = network()->addInput(name.c_str(), dtype, dims);
  PADDLE_ENFORCE_NOT_NULL(
      input,
      platform::errors::InvalidArgument("Adding input %s failed in "
                                        "TensorRT inference network. "
                                        "Please recheck your input.",
                                        name));
  PADDLE_ENFORCE_EQ(input->isNetworkInput(),
                    true,
                    platform::errors::InvalidArgument(
                        "Input %s is not the input of TRT inference network. "
                        "Please recheck your input.",
                        name));
  TensorRTEngine::SetITensor(name, input);
  return input;
}

void TensorRTEngine::DeclareOutput(const nvinfer1::ILayer *layer,
                                   int offset,
                                   const std::string &name) {
  auto *output = layer->getOutput(offset);
  SetITensor(name, output);
  PADDLE_ENFORCE_NOT_NULL(
      output,
      platform::errors::InvalidArgument(
          "The output %s of TRT engine should not be null.", name));
  output->setName(name.c_str());
  PADDLE_ENFORCE_EQ(output->isNetworkInput(),
                    false,
                    platform::errors::InvalidArgument(
                        "The output %s of TRT engine should not be the input "
                        "of the network at the same time.",
                        name));
  network()->markOutput(*output);
  PADDLE_ENFORCE_EQ(
      output->isNetworkOutput(),
      true,
      platform::errors::InvalidArgument(
          "The output %s of TRT engine should be the output of the network.",
          name));
}

void TensorRTEngine::DeclareOutput(const std::string &name) {
  auto *output = TensorRTEngine::GetITensor(name);
  PADDLE_ENFORCE_NOT_NULL(
      output,
      platform::errors::InvalidArgument(
          "The output %s of TRT engine should not be null.", name));
  output->setName(name.c_str());
  PADDLE_ENFORCE_EQ(output->isNetworkInput(),
                    false,
                    platform::errors::InvalidArgument(
                        "The output %s of TRT engine should not be the input "
                        "of the network at the same time.",
                        name));
  network()->markOutput(*output);
}
void TensorRTEngine::DeleteITensor(const std::string &name,
                                   nvinfer1::ITensor *tensor) {
  PADDLE_ENFORCE_NOT_NULL(
      tensor,
      platform::errors::InvalidArgument(
          "Tensor named %s of TRT engine should not be null.", name));
  PADDLE_ENFORCE_EQ(
      true,
      itensor_map_.count(name),
      platform::errors::InvalidArgument(
          "Tensor named %s of TRT engine should not be null", name));
  itensor_map_.erase(name);
}

void TensorRTEngine::SetITensor(const std::string &name,
                                nvinfer1::ITensor *tensor) {
  PADDLE_ENFORCE_NOT_NULL(
      tensor,
      platform::errors::InvalidArgument(
          "Tensor named %s of TRT engine should not be null.", name));
  PADDLE_ENFORCE_EQ(
      0,
      itensor_map_.count(name),
      platform::errors::InvalidArgument(
          "Tensor named %s of TRT engine should not be duplicated", name));
  itensor_map_[name] = tensor;
}

nvinfer1::ITensor *TensorRTEngine::GetITensor(const std::string &name) {
  if (itensor_map_.count(name)) {
    return itensor_map_[name];
  } else {
    ConvertWeight2ITensor(name);
    return itensor_map_[name];
  }
}

// For cases when input is not middle-tensor , but persistable tensor
// you should call this.
nvinfer1::ITensor *TensorRTEngine::ConvertWeight2ITensor(
    const std::string &name) {
  auto *var_v = scope_->FindVar(name);
  PADDLE_ENFORCE_NOT_NULL(
      var_v,
      platform::errors::NotFound("You are converting a persistable weight to a "
                                 "tensor, but there is no "
                                 "persistable variable called %s in scope.",
                                 name));
  auto *var_t = var_v->GetMutable<framework::LoDTensor>();
  auto weight = this->GetTrtWeight(name, *var_t);

  // Now we have create weights, then we need create a itensor
  auto var_dims = var_t->dims();
  nvinfer1::Dims trt_in_shape;
  trt_in_shape.nbDims = var_t->dims().size();
  for (int64_t i = 0; i < trt_in_shape.nbDims; i++) {
    trt_in_shape.d[i] = var_dims[i];
  }
  // In fact , this is not always right, because we can't determine if the 0th
  // dimension is batch. Just for run chenqu's model
  if (!this->with_dynamic_shape()) {
    trt_in_shape.nbDims--;
    for (int i = 0; i < trt_in_shape.nbDims; i++) {
      trt_in_shape.d[i] = trt_in_shape.d[i + 1];
    }
  }
  nvinfer1::ILayer *layer =
      TRT_ENGINE_ADD_LAYER(this, Constant, trt_in_shape, weight.get());
  this->SetITensor(name, layer->getOutput(0));
  return layer->getOutput(0);
}

std::unordered_map<std::string, nvinfer1::ITensor *>
    *TensorRTEngine::GetITensorMap() {
  return &itensor_map_;
}

void TensorRTEngine::Deserialize(const std::string &engine_serialized_data) {
  freshDeviceId();
  infer_ptr<nvinfer1::IRuntime> runtime(createInferRuntime(&logger_));

  if (use_dla_) {
    if (precision_ != AnalysisConfig::Precision::kInt8 &&
        precision_ != AnalysisConfig::Precision::kHalf) {
      LOG(WARNING) << "TensorRT DLA must be used with int8 or fp16, but you "
                      "set float32, so DLA is not used.";
    } else if (runtime->getNbDLACores() == 0) {
      LOG(WARNING)
          << "TensorRT DLA is set by config, but your device does not have "
             "DLA, so DLA is not used.";
    } else {
      if (dla_core_ < 0 || dla_core_ >= runtime->getNbDLACores()) {
        dla_core_ = 0;
        LOG(WARNING) << "Invalid DLACore, must be 0 < DLACore < "
                     << runtime->getNbDLACores() << ", but got " << dla_core_
                     << ", so use use 0 as default.";
      }
      runtime->setDLACore(dla_core_);
      LOG(INFO) << "TensorRT DLA enabled in Deserialize(), DLACore "
                << dla_core_;
    }
  }

  infer_engine_.reset(runtime->deserializeCudaEngine(
      engine_serialized_data.c_str(), engine_serialized_data.size()));

  PADDLE_ENFORCE_NOT_NULL(
      infer_engine_,
      platform::errors::Fatal(
          "Building TRT cuda engine failed when deserializing engine info. "
          "Please check:\n1. Your TRT serialization is generated and loaded "
          "on the same GPU architecture;\n2. The Paddle Inference version of "
          "generating serialization file and doing inference are "
          "consistent."));

  binding_num_ = infer_engine_->getNbBindings();
  // for engine context memory sharing
  if (context_memory_sharing_) {
    inference::Singleton<inference::tensorrt::TRTEngineManager>::Global()
        .updateContextMemorySize(infer_engine_->getDeviceMemorySize(),
                                 predictor_id_per_thread);
  }

  GetEngineInfo();
}

void TensorRTEngine::SetRuntimeBatch(size_t batch_size) {
  runtime_batch_ = batch_size;
}

// Note: Only for support plugin.
TensorRTEngine::Weight TensorRTEngine::GetFp16TrtWeight(
    const std::string &name, const phi::DenseTensor &weight_tensor) {
  static int name_suffix_counter = 0;
  std::string name_suffix = std::to_string(name_suffix_counter);
  std::string splitter = "__";
  std::string name_with_suffix = name + splitter + name_suffix;
  platform::CPUPlace cpu_place;
  PADDLE_ENFORCE_EQ(weight_map.count(name_with_suffix),
                    0,
                    platform::errors::AlreadyExists(
                        "The weight named %s is set into the weight map "
                        "twice in TRT OP converter.",
                        name_with_suffix));
  weight_map[name_with_suffix].reset(new phi::DenseTensor());
  weight_map[name_with_suffix]->Resize(weight_tensor.dims());

  TensorRTEngine::Weight weight;
  weight.SetCount(weight_tensor.numel());
  weight.SetDataType(nvinfer1::DataType::kHALF);
  // weight_tensor.dims().;

  // if trt not support dtype, we need to cast to  fp16.
  if (weight_tensor.dtype() == phi::DataType::BFLOAT16) {
    phi::DenseTensor bf16_tensor;
    bf16_tensor.clear();
    paddle::framework::TensorCopySync(
        weight_tensor, platform::CPUPlace(), &bf16_tensor);
    weight_map[name_with_suffix]->set_type(
        paddle::experimental::DataType::FLOAT16);
    weight_map[name_with_suffix]->Resize(weight_tensor.dims());
    auto *fp16_data = weight_map[name_with_suffix]->mutable_data<float16>(
        platform::CPUPlace());
    auto *bf16_data = bf16_tensor.mutable_data<bfloat16>(platform::CPUPlace());
    for (int i = 0; i < weight_tensor.numel(); i++) {
      fp16_data[i] = static_cast<float16>(bf16_data[i]);
    }
  } else if (weight_tensor.dtype() == phi::DataType::FLOAT32) {
    phi::DenseTensor fp32_tensor;
    fp32_tensor.clear();
    paddle::framework::TensorCopySync(
        weight_tensor, platform::CPUPlace(), &fp32_tensor);
    weight_map[name_with_suffix]->set_type(
        paddle::experimental::DataType::FLOAT16);
    weight_map[name_with_suffix]->Resize(weight_tensor.dims());
    auto *fp16_data = weight_map[name_with_suffix]->mutable_data<float16>(
        platform::CPUPlace());
    auto *fp32_data = fp32_tensor.mutable_data<float>(platform::CPUPlace());
    for (int i = 0; i < weight_tensor.numel(); i++) {
      fp16_data[i] = static_cast<float16>(fp32_data[i]);
    }
  } else {
    paddle::framework::TensorCopySync(
        weight_tensor, cpu_place, weight_map[name_with_suffix].get());
  }
  weight.SetValues(weight_map[name_with_suffix]->data());
  name_suffix_counter += 1;
  return weight;
}

// Note: Only for support plugin.
TensorRTEngine::Weight TensorRTEngine::GetFp32TrtWeight(
    const std::string &name, const phi::DenseTensor &weight_tensor) {
  static int name_suffix_counter = 0;
  std::string name_suffix = std::to_string(name_suffix_counter);
  std::string splitter = "__";
  std::string name_with_suffix = name + splitter + name_suffix;
  platform::CPUPlace cpu_place;
  PADDLE_ENFORCE_EQ(weight_map.count(name_with_suffix),
                    0,
                    platform::errors::AlreadyExists(
                        "The weight named %s is set into the weight map "
                        "twice in TRT OP converter.",
                        name_with_suffix));
  weight_map[name_with_suffix].reset(new phi::DenseTensor());
  weight_map[name_with_suffix]->Resize(weight_tensor.dims());

  TensorRTEngine::Weight weight;
  weight.SetCount(weight_tensor.numel());
  weight.SetDataType(nvinfer1::DataType::kFLOAT);
  // weight_tensor.dims().;

  // if trt not support dtype, we need to cast to  fp32.
  if (weight_tensor.dtype() == phi::DataType::BFLOAT16) {
    phi::DenseTensor bf16_tensor;
    bf16_tensor.clear();
    paddle::framework::TensorCopySync(
        weight_tensor, platform::CPUPlace(), &bf16_tensor);
    weight_map[name_with_suffix]->set_type(
        paddle::experimental::DataType::FLOAT32);
    weight_map[name_with_suffix]->Resize(weight_tensor.dims());
    auto *fp32_data =
        weight_map[name_with_suffix]->mutable_data<float>(platform::CPUPlace());
    auto *bf16_data = bf16_tensor.mutable_data<bfloat16>(platform::CPUPlace());
    for (int i = 0; i < weight_tensor.numel(); i++) {
      fp32_data[i] = static_cast<float>(bf16_data[i]);
    }
  } else if (weight_tensor.dtype() == phi::DataType::FLOAT16) {
    phi::DenseTensor fp16_tensor;
    fp16_tensor.clear();
    paddle::framework::TensorCopySync(
        weight_tensor, platform::CPUPlace(), &fp16_tensor);
    weight_map[name_with_suffix]->set_type(
        paddle::experimental::DataType::FLOAT32);
    weight_map[name_with_suffix]->Resize(weight_tensor.dims());
    auto *fp32_data =
        weight_map[name_with_suffix]->mutable_data<float>(platform::CPUPlace());
    auto *fp16_data = fp16_tensor.mutable_data<float16>(platform::CPUPlace());
    for (int i = 0; i < weight_tensor.numel(); i++) {
      fp32_data[i] = static_cast<float>(fp16_data[i]);
    }
  } else {
    paddle::framework::TensorCopySync(
        weight_tensor, cpu_place, weight_map[name_with_suffix].get());
  }
  weight.SetValues(weight_map[name_with_suffix]->data());
  name_suffix_counter += 1;
  return weight;
}

TensorRTEngine::Weight TensorRTEngine::GetTrtWeight(
    const std::string &name, const phi::DenseTensor &weight_tensor) {
  static int name_suffix_counter = 0;
  std::string name_suffix = std::to_string(name_suffix_counter);
  std::string splitter = "__";
  std::string name_with_suffix = name + splitter + name_suffix;
  platform::CPUPlace cpu_place;
  PADDLE_ENFORCE_EQ(weight_map.count(name_with_suffix),
                    0,
                    platform::errors::AlreadyExists(
                        "The weight named %s is set into the weight map "
                        "twice in TRT OP converter.",
                        name_with_suffix));

  weight_map[name_with_suffix].reset(new phi::DenseTensor());
  weight_map[name_with_suffix]->Resize(weight_tensor.dims());

  TensorRTEngine::Weight weight;
  weight.SetCount(weight_tensor.numel());

  // if trt not support dtype, we need to cast to fp32.
  if (weight_tensor.dtype() == phi::DataType::BFLOAT16) {
    phi::DenseTensor bf16_tensor;
    bf16_tensor.clear();
    paddle::framework::TensorCopySync(
        weight_tensor, platform::CPUPlace(), &bf16_tensor);
    weight_map[name_with_suffix]->set_type(
        paddle::experimental::DataType::FLOAT32);
    auto *fp32_data =
        weight_map[name_with_suffix]->mutable_data<float>(platform::CPUPlace());
    auto *bf16_data = bf16_tensor.mutable_data<bfloat16>(platform::CPUPlace());
    for (int i = 0; i < weight_tensor.numel(); i++) {
      fp32_data[i] = static_cast<float>(bf16_data[i]);
    }
    weight.SetDataType(phi::DataType::FLOAT32);
    weight.SetValues(fp32_data);
  } else if (weight_tensor.dtype() == phi::DataType::INT64) {
    phi::DenseTensor int64_tensor;
    int64_tensor.clear();
    paddle::framework::TensorCopySync(
        weight_tensor, platform::CPUPlace(), &int64_tensor);
    weight_map[name_with_suffix]->set_type(
        paddle::experimental::DataType::INT32);
    auto *int32_data =
        weight_map[name_with_suffix]->mutable_data<int>(platform::CPUPlace());
    auto *int64_data = int64_tensor.mutable_data<int64_t>(platform::CPUPlace());
    for (int i = 0; i < weight_tensor.numel(); i++) {
      int32_data[i] = int64_data[i];
    }
    weight.SetDataType(phi::DataType::FLOAT32);
    weight.SetValues(int32_data);
  } else {
    paddle::framework::TensorCopySync(
        weight_tensor, cpu_place, weight_map[name_with_suffix].get());
    weight.SetDataType(weight_tensor.dtype());
    weight.SetValues(weight_map[name_with_suffix]->data());
  }

  name_suffix_counter += 1;
  return weight;
}

int TensorRTEngine::GetRuntimeBatch() { return runtime_batch_; }

nvinfer1::IPluginV2Layer *TensorRTEngine::AddPlugin(
    nvinfer1::ITensor *const *inputs,
    int num_inputs,
    plugin::PluginTensorRT *plugin) {
  owned_plugin_.emplace_back(plugin);
  return network()->addPluginV2(inputs, num_inputs, *plugin);
}

nvinfer1::IPluginV2Layer *TensorRTEngine::AddPluginV2Ext(
    nvinfer1::ITensor *const *inputs,
    int num_inputs,
    plugin::PluginTensorRTV2Ext *plugin) {
  owned_plugin_v2ext_.emplace_back(plugin);
  return network()->addPluginV2(inputs, num_inputs, *plugin);
}

nvinfer1::IPluginV2Layer *TensorRTEngine::AddPluginV2IOExt(
    nvinfer1::ITensor *const *inputs,
    int num_inputs,
    nvinfer1::IPluginV2IOExt *plugin) {
  owned_plugin_v2ioext_.emplace_back(plugin);
  return network()->addPluginV2(inputs, num_inputs, *plugin);
}

void TensorRTEngine::freshDeviceId() {
  int count;
  cudaGetDeviceCount(&count);
  PADDLE_ENFORCE_LT(device_id_,
                    count,
                    platform::errors::OutOfRange(
                        "Device id %d exceeds the current device count: %d.",
                        device_id_,
                        count));
  platform::SetDeviceId(device_id_);
}

void TensorRTEngine::GetEngineInfo() {
#if IS_TRT_VERSION_GE(8200)
  LOG(INFO) << "====== engine info ======";
  std::unique_ptr<nvinfer1::IEngineInspector> infer_inspector(
      infer_engine_->createEngineInspector());
  auto infer_context = infer_ptr<nvinfer1::IExecutionContext>(
      infer_engine_->createExecutionContextWithoutDeviceMemory());
  infer_inspector->setExecutionContext(infer_context.get());
  LOG(INFO) << infer_inspector->getEngineInformation(
      nvinfer1::LayerInformationFormat::kONELINE);
  LOG(INFO) << "====== engine info end ======";
#else
  LOG(INFO) << "Inspector needs TensorRT version 8.2 and after.";
#endif
}

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
