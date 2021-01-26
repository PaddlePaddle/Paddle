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

#pragma once

#include <NvInfer.h>
#include <map>
#include <memory>
#include <mutex>  // NOLINT
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/inference/api/paddle_analysis_config.h"
#include "paddle/fluid/inference/engine.h"
#include "paddle/fluid/inference/tensorrt/helper.h"
#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin.h"
#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin_factory.h"
#include "paddle/fluid/inference/tensorrt/trt_int8_calibrator.h"
#include "paddle/fluid/inference/utils/singleton.h"

namespace paddle {
namespace framework {
class Tensor;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace inference {
namespace tensorrt {

namespace plugin {
class PluginTensorRT;
}  // namespace plugin

using FluidDT = framework::proto::VarType_Type;
using TRT_DT = nvinfer1::DataType;

namespace {  // NOLINT

TRT_DT FluidDataType2TRT(FluidDT type) {
  switch (type) {
    case FluidDT::VarType_Type_FP32:
      return TRT_DT::kFLOAT;
    case FluidDT::VarType_Type_INT32:
      return TRT_DT::kINT32;
    default:
      return TRT_DT::kINT32;
  }
  PADDLE_THROW(platform::errors::InvalidArgument(
      "unknown fluid datatype in TRT op converter"));
  return TRT_DT::kINT32;
}

// The T can be int32 or int64 type.
template <typename T>
nvinfer1::Dims Vec2TRT_Dims(const std::vector<T>& shape, std::string input,
                            bool with_dynamic_shape = false) {
  PADDLE_ENFORCE_GT(shape.size(), 0UL,
                    platform::errors::InvalidArgument(
                        "TensorRT's tensor input requires at least 1 "
                        "dimensions, but input %s has %d dims.",
                        input, shape.size()));
  PADDLE_ENFORCE_LE(shape.size(), 4UL,
                    platform::errors::InvalidArgument(
                        "TensorRT's tensor input requires at most 4 "
                        "dimensions, but input %s has %d dims.",
                        input, shape.size()));
  if (!with_dynamic_shape) {
    if (shape.size() == 4UL) {
      return nvinfer1::DimsCHW(shape[1], shape[2], shape[3]);
    } else if (shape.size() == 3UL) {
      return nvinfer1::Dims2(shape[1], shape[2]);
    }
    return nvinfer1::DimsCHW(shape[1], 1, 1);
  } else {
    if (shape.size() == 4UL) {
      return nvinfer1::DimsNCHW(shape[0], shape[1], shape[2], shape[3]);
    } else if (shape.size() == 3UL) {
      return nvinfer1::Dims3(shape[0], shape[1], shape[2]);
    }
    nvinfer1::Dims dims;
    dims.nbDims = shape.size();
    for (size_t i = 0; i < shape.size(); i++) {
      dims.d[i] = shape[i];
    }
    return dims;
  }
}
}  // NOLINT

class TRTInt8Calibrator;

/*
 * TensorRT Engine.
 *
 * There are two alternative ways to use it, one is  to build from a paddle
 * protobuf model, another way is to manually construct the network.
 */
class TensorRTEngine {
  using DescType = ::paddle::framework::proto::BlockDesc;
  using ShapeMapType = std::map<std::string, std::vector<int>>;

 public:
  // Weight is model parameter.
  class Weight {
   public:
    Weight() = default;
    Weight(nvinfer1::DataType dtype, void* value, size_t num_elem) {
      w_.type = dtype;
      w_.values = value;
      w_.count = num_elem;
    }
    const nvinfer1::Weights& get() { return w_; }

    std::vector<int64_t> dims;

   private:
    nvinfer1::Weights w_;
  };

  TensorRTEngine(
      int max_batch, int max_workspace,
      AnalysisConfig::Precision precision = AnalysisConfig::Precision::kFloat32,
      TRTInt8Calibrator* calibrator = nullptr, int device_id = 0,
      const ShapeMapType min_input_shape = {},
      const ShapeMapType max_input_shape = {},
      const ShapeMapType optim_input_shape = {},
      bool disable_trt_plugin_fp16 = false,
      nvinfer1::ILogger& logger = NaiveLogger::Global())
      : max_batch_(max_batch),
        max_workspace_(max_workspace),
        precision_(precision),
        calibrator_(calibrator),
        device_id_(device_id),
        min_input_shape_(min_input_shape),
        max_input_shape_(max_input_shape),
        optim_input_shape_(optim_input_shape),
        disable_trt_plugin_fp16_(disable_trt_plugin_fp16),
        logger_(logger) {
    if (min_input_shape_.size() != 0 && max_input_shape_.size() != 0 &&
        optim_input_shape_.size() != 0) {
      PADDLE_ENFORCE_EQ(
          min_input_shape_.size(), max_input_shape_.size(),
          platform::errors::InvalidArgument(
              "The min_input_shape_'s size(%d) should be equal to the "
              "size(%d) of max_input_shape_",
              min_input_shape_.size(), max_input_shape_.size()));
      PADDLE_ENFORCE_EQ(
          min_input_shape_.size(), optim_input_shape_.size(),
          platform::errors::InvalidArgument(
              "The min_input_shape_'s size(%d) should be equal to the "
              "size(%d) of optim_input_shape_",
              min_input_shape_.size(), optim_input_shape_.size()));
#if IS_TRT_VERSION_GE(6000)
      with_dynamic_shape_ = true;
#else
      LOG(WARNING) << "Using dynamic shape of TRT need ensure that the TRT "
                      "version should be at least 6.";
#endif
    }
    dy::initLibNvInferPlugins(&logger, "");
  }

  ~TensorRTEngine() {}

  // Add an input and set its name, data type and dimension.
  nvinfer1::ITensor* DeclareInput(const std::string& name,
                                  nvinfer1::DataType dtype,
                                  const nvinfer1::Dims& dim);
  // Set the offset-th output from a layer as the network's output, and set its
  // name.
  void DeclareOutput(const nvinfer1::ILayer* layer, int offset,
                     const std::string& name);
  // Set the itensor_map_[name] as the network's output, and set its name.
  void DeclareOutput(const std::string& name);

  void SetITensor(const std::string& name, nvinfer1::ITensor* tensor);
  // Get an ITensor called name.
  nvinfer1::ITensor* GetITensor(const std::string& name);

  nvinfer1::ICudaEngine* engine() { return infer_engine_.get(); }
  nvinfer1::IExecutionContext* context() {
    std::unique_lock<std::mutex> lock(mutex_);
    const std::thread::id tid = std::this_thread::get_id();
    if (infer_context_.find(tid) == infer_context_.end()) {
      PADDLE_ENFORCE_NOT_NULL(
          infer_engine_,
          platform::errors::InvalidArgument(
              "You should build engine first and then set the context."));
      infer_context_[tid].reset(infer_engine_->createExecutionContext());
    }
    return infer_context_[tid].get();
  }

  nvinfer1::IHostMemory* Serialize() {
    PADDLE_ENFORCE_NOT_NULL(
        infer_engine_,
        platform::errors::InvalidArgument(
            "The TensorRT engine must be built first before serialization"));
    ihost_memory_.reset(infer_engine_->serialize());
    return ihost_memory_.get();
  }

  void Deserialize(const std::string& engine_serialized_data) {
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

    if (with_dynamic_shape_) {
#if IS_TRT_VERSION_GE(6000)
      infer_engine_.reset(runtime->deserializeCudaEngine(
          engine_serialized_data.c_str(), engine_serialized_data.size(),
          nullptr));
#else

      PADDLE_THROW(platform::errors::PreconditionNotMet(
          "To enable dynamic shape support, the TensorRT version should be "
          "greater than 6.0.0"));

#endif
    } else {
      infer_engine_.reset(runtime->deserializeCudaEngine(
          engine_serialized_data.c_str(), engine_serialized_data.size(),
          &inference::Singleton<plugin::PluginFactoryTensorRT>::Global()));
    }
    PADDLE_ENFORCE_NOT_NULL(
        infer_engine_,
        platform::errors::Fatal(
            "Building TRT cuda engine failed when deserializing engine info. "
            "Please check:\n1. Your TRT serialization is generated and loaded "
            "on the same GPU architecture;\n2. The Paddle Inference version of "
            "generating serialization file and doing inference are "
            "consistent."));
  }

  void SetRuntimeBatch(size_t batch_size);
  int GetRuntimeBatch();

  bool WithFp16() {
    bool enable_fp16 = (precision_ == AnalysisConfig::Precision::kHalf);
    bool support_fp16 = infer_builder_->platformHasFastFp16();
    return enable_fp16 && support_fp16;
  }

  int GetDeviceId() { return device_id_; }
  nvinfer1::IPluginLayer* AddPlugin(nvinfer1::ITensor* const* inputs,
                                    int num_inputs, plugin::PluginTensorRT*);
  void SetTensorDynamicRange(nvinfer1::ITensor* tensor, float range) {
    quant_dynamic_range_[tensor] = range;
  }

  float* GetWeightCPUData(const std::string& name,
                          framework::Tensor* weight_tensor, bool enable_int8,
                          const std::vector<float>& scale = {});

  // A pointer to CPU memory is needed of the TRT weight.
  // Before TRT runs, fluid loads weight into GPU storage.
  // so we need to copy the weights from GPU to CPU in our op converter.
  // We use a map to store these weights for the weight memory is not released
  // in advance, which affecting the construction of TRT Op.
  std::unordered_map<std::string /*name*/, std::unique_ptr<framework::Tensor>>
      weight_map;

  // When setting weight_map, a self-increasing suffix is needed for the names
  // so as to avoid repeatedly setting weights with the same name.
  void SetWeights(std::string w_name,
                  std::unique_ptr<framework::Tensor> w_tensor) {
    static int suffix_counter = 0;
    std::string suffix = std::to_string(suffix_counter);
    std::string splitter = "__";
    weight_map[w_name + splitter + suffix] = std::move(w_tensor);
    suffix_counter += 1;
  }

  void SetUseOSS(bool use_oss) { use_oss_ = use_oss; }
  void SetUseDLA(bool use_dla) { use_dla_ = use_dla; }
  void SetDLACore(int dla_core) { dla_core_ = dla_core; }
  void SetWithErnie(bool with_ernie) { with_ernie_ = with_ernie; }

  void ClearWeights() {
    for (auto& weight_pair : weight_map) {
      weight_pair.second.reset(nullptr);
    }
  }

  // NOTE: The func bellow was modified to adapt the dynamic shape.
  // Initialize the inference network, so that TensorRT layers can add to this
  // network.
  void InitNetwork();
  // After finishing adding ops, freeze this network and creates the execution
  // environment.
  void FreezeNetwork();
  void Execute(int batch_size, std::vector<void*>* buffers,
               cudaStream_t stream = nullptr);

  nvinfer1::INetworkDefinition* network() {
    if (with_dynamic_shape_) {
      return infer_networkv2_.get();
    } else {
      return infer_network_.get();
    }
  }

  ShapeMapType min_input_shape() { return min_input_shape_; }
  ShapeMapType max_input_shape() { return max_input_shape_; }
  ShapeMapType optim_input_shape() { return optim_input_shape_; }
  bool use_oss() { return use_oss_; }
  bool with_ernie() { return with_ernie_; }
  bool disable_trt_plugin_fp16() { return disable_trt_plugin_fp16_; }
  bool with_dynamic_shape() { return with_dynamic_shape_; }

#if IS_TRT_VERSION_GE(6000)
  nvinfer1::IPluginV2Layer* AddPluginV2(nvinfer1::ITensor* const* inputs,
                                        int num_inputs,
                                        plugin::DynamicPluginTensorRT* plugin) {
    owned_pluginv2_.emplace_back(plugin);
    return network()->addPluginV2(inputs, num_inputs, *plugin);
  }
#endif

 private:
  // Each ICudaEngine object is bound to a specific GPU when it is instantiated,
  // ensure that the thread is associated with the correct device by calling
  // freshDeviceId().
  void freshDeviceId();

  // the max batch size
  int max_batch_;
  // the runtime batch size
  static int runtime_batch_;
  // the max memory size the engine uses
  int max_workspace_;

  AnalysisConfig::Precision precision_;
  TRTInt8Calibrator* calibrator_;
  // batch size of the current data, will be updated each Executation.
  int batch_size_{-1};

  int device_id_;
  ShapeMapType min_input_shape_;
  ShapeMapType max_input_shape_;
  ShapeMapType optim_input_shape_;
  bool disable_trt_plugin_fp16_{false};
  bool use_oss_{false};
  bool use_dla_{false};
  int dla_core_{0};
  bool with_ernie_{false};
  nvinfer1::ILogger& logger_;

  // max data size for the buffers.
  std::unordered_map<std::string /*name*/, nvinfer1::ITensor* /*ITensor*/>
      itensor_map_;

  std::vector<std::unique_ptr<plugin::PluginTensorRT>> owned_plugin_;

  // TensorRT related internal members
  template <typename T>
  struct Destroyer {
    void operator()(T* x) {
      if (x) {
        x->destroy();
      }
    }
  };
  template <typename T>
  using infer_ptr = std::unique_ptr<T, Destroyer<T>>;
  infer_ptr<nvinfer1::IBuilder> infer_builder_;
  infer_ptr<nvinfer1::INetworkDefinition> infer_network_;
  infer_ptr<nvinfer1::ICudaEngine> infer_engine_;
  std::unordered_map<std::thread::id, infer_ptr<nvinfer1::IExecutionContext>>
      infer_context_;
  infer_ptr<nvinfer1::IHostMemory> ihost_memory_;
  std::unordered_map<nvinfer1::ITensor*, float> quant_dynamic_range_;

  // For dynamic shape
  bool with_dynamic_shape_{false};
  infer_ptr<nvinfer1::INetworkDefinition> infer_networkv2_;
#if IS_TRT_VERSION_GE(6000)
  infer_ptr<nvinfer1::IBuilderConfig> infer_builder_config_;
  nvinfer1::IOptimizationProfile* optim_profile_;
  std::vector<std::unique_ptr<plugin::DynamicPluginTensorRT>> owned_pluginv2_;
#endif
  std::mutex mutex_;
};  // class TensorRTEngine

// Add a layer__ into engine__ with args ARGS.
// For example:
//
// Reference
// https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#charRNN_define_network
//
// will add a fully connected layer into the engine.
// TensorRT has too many layers, so that is not wise to add member functions for
// them, and an macro like this is more extensible when underlying TensorRT
// library add new layer supports.
#define TRT_ENGINE_ADD_LAYER(engine__, layer__, ...) \
  engine__->network()->add##layer__(__VA_ARGS__);

class TRTEngineManager {
 public:
  bool Empty() const { return engines_.size() == 0; }
  bool Has(const std::string& name) const {
    if (engines_.count(name) == 0) return false;
    return engines_.at(name).get() != nullptr;
  }

  TensorRTEngine* Get(const std::string& name) const {
    return engines_.at(name).get();
  }

  TensorRTEngine* Create(
      std::string name, int max_batch, int max_workspace,
      AnalysisConfig::Precision precision = AnalysisConfig::Precision::kFloat32,
      TRTInt8Calibrator* calibrator = nullptr, int device_id = 0,
      const std::map<std::string, std::vector<int>> min_input_shape = {},
      const std::map<std::string, std::vector<int>> max_input_shape = {},
      const std::map<std::string, std::vector<int>> optim_input_shape = {},
      bool disable_trt_plugin_fp16 = false,
      nvinfer1::ILogger& logger = NaiveLogger::Global()) {
    auto* p =
        new TensorRTEngine(max_batch, max_workspace, precision, calibrator,
                           device_id, min_input_shape, max_input_shape,
                           optim_input_shape, disable_trt_plugin_fp16, logger);
    engines_[name].reset(p);
    return p;
  }

  void DeleteAll() {
    for (auto& item : engines_) {
      item.second.reset(nullptr);
    }
  }

 private:
  std::unordered_map<std::string, std::unique_ptr<TensorRTEngine>> engines_;
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
