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
namespace inference {
namespace tensorrt {

class TRTInt8Calibrator;
/*
 * TensorRT Engine.
 *
 * There are two alternative ways to use it, one is  to build from a paddle
 * protobuf model, another way is to manually construct the network.
 */
class TensorRTEngine {
  using DescType = ::paddle::framework::proto::BlockDesc;

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
      nvinfer1::ILogger& logger = NaiveLogger::Global())
      : max_batch_(max_batch),
        max_workspace_(max_workspace),
        precision_(precision),
        calibrator_(calibrator),
        device_id_(device_id),
        logger_(logger) {}

  ~TensorRTEngine() {}

  // TODO(Superjomn) implement it later when graph segmentation is supported.
  void Build(const DescType& paddle_model);

  void Execute(int batch_size, std::vector<void*>* buffers,
               cudaStream_t stream = nullptr);

  // Initialize the inference network, so that TensorRT layers can add to this
  // network.
  void InitNetwork() {
    freshDeviceId();
    infer_builder_.reset(createInferBuilder(&logger_));
    infer_network_.reset(infer_builder_->createNetwork());
  }
  // After finishing adding ops, freeze this network and creates the execution
  // environment.
  void FreezeNetwork();

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
  // Check if the ITensor has been declared
  bool HasDeclared(const std::string& name);

  void SetITensor(const std::string& name, nvinfer1::ITensor* tensor);
  // Get an ITensor called name.
  nvinfer1::ITensor* GetITensor(const std::string& name);

  nvinfer1::ICudaEngine* engine() { return infer_engine_.get(); }
  nvinfer1::INetworkDefinition* network() { return infer_network_.get(); }

  nvinfer1::IHostMemory* Serialize() {
    PADDLE_ENFORCE(infer_engine_ != nullptr,
                   "You should build engine first and then serialize");
    ihost_memory_.reset(infer_engine_->serialize());
    return ihost_memory_.get();
  }

  void Deserialize(const std::string& engine_serialized_data) {
    freshDeviceId();
    infer_ptr<nvinfer1::IRuntime> runtime(createInferRuntime(&logger_));
    infer_engine_.reset(runtime->deserializeCudaEngine(
        engine_serialized_data.c_str(), engine_serialized_data.size(),
        &inference::Singleton<plugin::PluginFactoryTensorRT>::Global()));
    PADDLE_ENFORCE(infer_engine_ != nullptr,
                   "build cuda engine failed when deserialize engine info.!");
  }

  void SetRuntimeBatch(size_t batch_size);
  int GetRuntimeBatch();
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

  void ClearWeights() {
    for (auto& weight_pair : weight_map) {
      weight_pair.second.reset(nullptr);
    }
  }

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
  nvinfer1::ILogger& logger_;

  // max data size for the buffers.
  std::unordered_map<std::string /*name*/, size_t /*max size*/> buffer_sizes_;
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
  std::mutex mutex_;
};  // class TensorRTEngine

#define IS_TRT_VERSION_GE(version)                       \
  ((NV_TENSORRT_MAJOR * 1000 + NV_TENSORRT_MINOR * 100 + \
    NV_TENSORRT_PATCH * 10 + NV_TENSORRT_BUILD) >= version)

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
      nvinfer1::ILogger& logger = NaiveLogger::Global()) {
    auto* p = new TensorRTEngine(max_batch, max_workspace, precision,
                                 calibrator, device_id, logger);
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
