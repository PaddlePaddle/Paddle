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

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>  // NOLINT
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "NvInferRuntimeCommon.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/inference/api/paddle_analysis_config.h"
#include "paddle/fluid/inference/engine.h"
#include "paddle/fluid/inference/tensorrt/helper.h"
#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin.h"
#include "paddle/fluid/inference/tensorrt/trt_int8_calibrator.h"
#include "paddle/fluid/inference/utils/singleton.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/stream.h"
#include "paddle/utils/any.h"

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
    case FluidDT::VarType_Type_FP16:
      return TRT_DT::kHALF;
    default:
      return TRT_DT::kINT32;
  }
  PADDLE_THROW(platform::errors::InvalidArgument(
      "unknown fluid datatype in TRT op converter"));
  return TRT_DT::kINT32;
}

// The T can be int32 or int64 type.
template <typename T>
nvinfer1::Dims Vec2TRT_Dims(const std::vector<T>& shape,
                            std::string input,
                            bool with_dynamic_shape = false) {
  PADDLE_ENFORCE_GT(shape.size(),
                    0UL,
                    platform::errors::InvalidArgument(
                        "TensorRT's tensor input requires at least 1 "
                        "dimensions, but input %s has %d dims.",
                        input,
                        shape.size()));

  auto ShapeStr = [](const std::vector<T>& shape) {
    std::ostringstream os;
    os << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
      if (i == shape.size() - 1) {
        os << shape[i];
      } else {
        os << shape[i] << ",";
      }
    }
    os << "]";
    return os.str();
  };
  if (!with_dynamic_shape) {
    if (shape.size() == 4UL) {
      if (shape[2] == -1 || shape[3] == -1) {
        PADDLE_THROW(platform::errors::InvalidArgument(
            "The input [%s] shape of trt subgraph is %s, please enable "
            "trt dynamic_shape mode by SetTRTDynamicShapeInfo.",
            input,
            ShapeStr(shape)));
      }
      return nvinfer1::Dims3(shape[1], shape[2], shape[3]);
    } else if (shape.size() == 5UL) {
      if (shape[2] == -1 || shape[3] == -1 || shape[4] == -1) {
        PADDLE_THROW(platform::errors::InvalidArgument(
            "The input [%s] shape of trt subgraph is %s, please enable "
            "trt dynamic_shape mode by SetTRTDynamicShapeInfo.",
            input,
            ShapeStr(shape)));
      }
      return nvinfer1::Dims4(shape[1], shape[2], shape[3], shape[4]);
    } else if (shape.size() == 3UL) {
      if (shape[1] == -1 || shape[2] == -1) {
        PADDLE_THROW(platform::errors::InvalidArgument(
            "The input [%s] shape of trt subgraph is %s, please enable "
            "trt dynamic_shape mode by SetTRTDynamicShapeInfo.",
            input,
            ShapeStr(shape)));
      }
      return nvinfer1::Dims2(shape[1], shape[2]);
    } else if (shape.size() == 2UL) {
      if (shape[1] == -1) {
        PADDLE_THROW(platform::errors::InvalidArgument(
            "The input [%s] shape of trt subgraph is %s, please enable "
            "trt dynamic_shape mode by SetTRTDynamicShapeInfo.",
            input,
            ShapeStr(shape)));
      }
      nvinfer1::Dims dims;
      dims.nbDims = 1;
      dims.d[0] = shape[1];
      return dims;
    }
    // static shape doesn't support 1D op so far.
    PADDLE_ENFORCE_NE(shape.size(),
                      1UL,
                      platform::errors::InvalidArgument(
                          "The input [%s] shape of trt subgraph is %s."
                          "it's not supported by trt so far",
                          input,
                          ShapeStr(shape)));

    nvinfer1::Dims dims;
    dims.nbDims = shape.size() - 1;
    for (size_t i = 1; i < shape.size(); i++) {
      dims.d[i - 1] = shape[i];
    }
    return dims;
  } else {
    if (shape.size() == 4UL) {
      return nvinfer1::Dims4(shape[0], shape[1], shape[2], shape[3]);
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
}  // namespace

class TRTInt8Calibrator;

/*
 * TensorRT Engine.
 *
 * There are two alternative ways to use it, one is to build from a paddle
 * protobuf model, another way is to manually construct the network.
 */
class TensorRTEngine {
  using DescType = ::paddle::framework::proto::BlockDesc;
  using ShapeMapType = std::map<std::string, std::vector<int>>;
  using PredictorID = int;

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

    void SetDataType(nvinfer1::DataType type) { w_.type = type; }

    void SetDataType(phi::DataType type);

    void SetValues(const void* values) { w_.values = values; }

    void SetCount(int64_t num) { w_.count = num; }

    std::vector<int64_t> dims;

   private:
    nvinfer1::Weights w_;
  };

  TensorRTEngine(
      int max_batch,
      int64_t max_workspace,
      AnalysisConfig::Precision precision = AnalysisConfig::Precision::kFloat32,
      TRTInt8Calibrator* calibrator = nullptr,
      int device_id = 0,
      const ShapeMapType min_input_shape = {},
      const ShapeMapType max_input_shape = {},
      const ShapeMapType optim_input_shape = {},
      bool disable_trt_plugin_fp16 = false,
      phi::DataType model_precision = phi::DataType::FLOAT32,
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
        model_precision_(model_precision),
        logger_(logger) {
    if (min_input_shape_.size() != 0 && max_input_shape_.size() != 0 &&
        optim_input_shape_.size() != 0) {
      PADDLE_ENFORCE_EQ(
          min_input_shape_.size(),
          max_input_shape_.size(),
          platform::errors::InvalidArgument(
              "The min_input_shape_'s size(%d) should be equal to the "
              "size(%d) of max_input_shape_",
              min_input_shape_.size(),
              max_input_shape_.size()));
      PADDLE_ENFORCE_EQ(
          min_input_shape_.size(),
          optim_input_shape_.size(),
          platform::errors::InvalidArgument(
              "The min_input_shape_'s size(%d) should be equal to the "
              "size(%d) of optim_input_shape_",
              min_input_shape_.size(),
              optim_input_shape_.size()));
#if IS_TRT_VERSION_GE(6000)
      with_dynamic_shape_ = true;
#else
      LOG(WARNING) << "Using dynamic shape of TRT need ensure that the TRT "
                      "version should be at least 6.";
#endif
    }
    dy::initLibNvInferPlugins(&logger, "");
  }

  ~TensorRTEngine() {
    for (auto& attr : attrs_) {
      if (attr_dels_.find(attr.first) != attr_dels_.end()) {
        attr_dels_[attr.first]();
      }
    }
    attrs_.clear();
    attr_dels_.clear();
  }

  // Add an input and set its name, data type and dimension.
  nvinfer1::ITensor* DeclareInput(const std::string& name,
                                  nvinfer1::DataType dtype,
                                  const nvinfer1::Dims& dim);
  // Set the offset-th output from a layer as the network's output, and set its
  // name.
  void DeclareOutput(const nvinfer1::ILayer* layer,
                     int offset,
                     const std::string& name);
  // Set the itensor_map_[name] as the network's output, and set its name.
  void DeclareOutput(const std::string& name);
  void ClearTensorMap() { itensor_map_.clear(); }

  void DeleteITensor(const std::string& name, nvinfer1::ITensor* tensor);
  void SetITensor(const std::string& name, nvinfer1::ITensor* tensor);
  // Get an ITensor called name.
  nvinfer1::ITensor* GetITensor(const std::string& name);
  nvinfer1::ITensor* ConvertWeight2ITensor(const std::string& name);
  std::unordered_map<std::string, nvinfer1::ITensor*>* GetITensorMap();

  nvinfer1::ICudaEngine* engine() { return infer_engine_.get(); }
  nvinfer1::IExecutionContext* context();

  int GetProfileIndex() {
    if (max_profile_num_ > 1) {
      std::unique_lock<std::mutex> lock(mutex_);
      return profile_index_[predictor_id_per_thread];
    } else {
      return 0;
    }
  }

  int GetBindingsOffset() {
    return (binding_num_ / max_profile_num_) * GetProfileIndex();
  }

  int GetNbBindings() { return binding_num_; }

  void ResetContext() {
    PADDLE_ENFORCE_NOT_NULL(
        infer_engine_,
        platform::errors::InvalidArgument(
            "You should build engine first and then set the context."));
    std::unique_lock<std::mutex> lock(mutex_);
    infer_context_[predictor_id_per_thread].reset(nullptr);
    infer_context_.erase(predictor_id_per_thread);
  }

  nvinfer1::IHostMemory* Serialize() {
    PADDLE_ENFORCE_NOT_NULL(
        infer_engine_,
        platform::errors::InvalidArgument(
            "The TensorRT engine must be built first before serialization"));
#if IS_TRT_VERSION_LT(8000)
    ihost_memory_.reset(infer_engine_->serialize());
#else
    PADDLE_ENFORCE_NOT_NULL(
        ihost_memory_,
        platform::errors::InvalidArgument(
            "TensorRT >= 8.0 requires that buildSerializedNetwork is called"));
#endif
    return ihost_memory_.get();
  }

  void Deserialize(const std::string& engine_serialized_data);

  void SetRuntimeBatch(size_t batch_size);
  int GetRuntimeBatch();

  bool WithFp16() {
    bool enable_fp16 = (precision_ == AnalysisConfig::Precision::kHalf);
    bool support_fp16 = infer_builder_->platformHasFastFp16();
    return enable_fp16 && support_fp16;
  }

  int GetDeviceId() { return device_id_; }

  nvinfer1::IPluginV2Layer* AddPlugin(nvinfer1::ITensor* const* inputs,
                                      int num_inputs,
                                      plugin::PluginTensorRT*);

  nvinfer1::IPluginV2Layer* AddPluginV2Ext(nvinfer1::ITensor* const* inputs,
                                           int num_inputs,
                                           plugin::PluginTensorRTV2Ext* plugin);

  nvinfer1::IPluginV2Layer* AddPluginV2IOExt(nvinfer1::ITensor* const* inputs,
                                             int num_inputs,
                                             nvinfer1::IPluginV2IOExt* plugin);

  void SetTensorDynamicRange(nvinfer1::ITensor* tensor, float range) {
    quant_dynamic_range_[tensor] = range;
  }

  // Get fp16 trt weight. If src weight is not fp16, we will cast.
  Weight GetFp16TrtWeight(const std::string& name,
                          const framework::Tensor& weight_tensor);

  // Get fp32 trt weight. If src weight is not fp32, we will cast.
  Weight GetFp32TrtWeight(const std::string& name,
                          const framework::Tensor& weight_tensor);

  // if the src weight type is fp16, then return fp16 trt weight, etc.
  Weight GetTrtWeight(const std::string& name,
                      const framework::Tensor& weight_tensor);

  float GetTensorDynamicRange(nvinfer1::ITensor* tensor) {
    return quant_dynamic_range_[tensor];
  }

  bool DynamicRangeIsSet(nvinfer1::ITensor* tensor) {
    return quant_dynamic_range_.count(tensor);
  }

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
    std::string name_with_suffix = w_name + splitter + suffix;
    PADDLE_ENFORCE_EQ(weight_map.count(name_with_suffix),
                      0,
                      platform::errors::AlreadyExists(
                          "The weight named %s is set into the weight map "
                          "twice in TRT OP converter.",
                          name_with_suffix));
    weight_map[name_with_suffix] = std::move(w_tensor);
    suffix_counter += 1;
  }

  void SetUseOSS(bool use_varseqlen) { use_varseqlen_ = use_varseqlen; }
  void SetUseDLA(bool use_dla) { use_dla_ = use_dla; }
  void SetDLACore(int dla_core) { dla_core_ = dla_core; }
  void SetWithErnie(bool with_ernie) { with_ernie_ = with_ernie; }
  void SetWithInterleaved(bool with_interleaved) {
    with_interleaved_ = with_interleaved;
  }
  void SetTransformerPosid(std::string tensorrt_transformer_posid) {
    tensorrt_transformer_posid_ = tensorrt_transformer_posid;
  }
  void SetTransformerMaskid(std::string tensorrt_transformer_maskid) {
    tensorrt_transformer_maskid_ = tensorrt_transformer_maskid;
  }
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
  void Execute(int batch_size,
               std::vector<void*>* buffers,
               cudaStream_t stream = nullptr);

  nvinfer1::INetworkDefinition* network() { return infer_network_.get(); }

  ShapeMapType min_input_shape() { return min_input_shape_; }
  ShapeMapType max_input_shape() { return max_input_shape_; }
  ShapeMapType optim_input_shape() { return optim_input_shape_; }

  bool AdjustDynamicShapeRange(const ShapeMapType& runtime_input_shape,
                               std::vector<std::string>* changed) {
    bool ret = false;
    changed->clear();
    for (const auto& it : runtime_input_shape) {
      auto name = it.first;
      auto input_shape = it.second;
      PADDLE_ENFORCE_EQ(
          min_input_shape_.count(name),
          true,
          platform::errors::InvalidArgument(
              "TRT dynamic_shape min_input_shape %s not found.", name));
      PADDLE_ENFORCE_EQ(min_input_shape_[name].size(),
                        input_shape.size(),
                        platform::errors::InvalidArgument(
                            "TRT dynamic_shape min_input_shape %s size not "
                            "equal, the min_input_shape[%s].size()=%d"
                            ", but the runtime_input_shape[%s].size()=%d.",
                            name,
                            name,
                            min_input_shape_[name].size(),
                            name,
                            input_shape.size()));
      auto bak_min_shape = min_input_shape_[name];
      auto bak_max_shape = max_input_shape_[name];
      bool min_change = false;
      bool max_change = false;
      for (size_t d = 0; d < input_shape.size(); ++d) {
        if (input_shape[d] < min_input_shape_[name][d]) {
          ret = true;
          min_change = true;
          min_input_shape_[name][d] = input_shape[d];
        }
        if (input_shape[d] > max_input_shape_[name][d]) {
          ret = true;
          max_change = true;
          max_input_shape_[name][d] = input_shape[d];
        }
      }

      if (min_change)
        LOG(INFO) << "refactor shape range: " << name << ", min_shape from "
                  << Vec2Str(bak_min_shape) << " to "
                  << Vec2Str(min_input_shape_[name]);
      if (max_change)
        LOG(INFO) << "refactor shape range: " << name << ", max_shape from "
                  << Vec2Str(bak_max_shape) << " to "
                  << Vec2Str(max_input_shape_[name]);
      if (min_change || max_change) changed->push_back(name);
    }
    return ret;
  }

  bool use_varseqlen() { return use_varseqlen_; }
  bool with_ernie() { return with_ernie_; }
  bool with_interleaved() { return with_interleaved_; }
  std::string tensorrt_transformer_posid() {
    return tensorrt_transformer_posid_;
  }
  std::string tensorrt_transformer_maskid() {
    return tensorrt_transformer_maskid_;
  }
  bool disable_trt_plugin_fp16() { return disable_trt_plugin_fp16_; }
  bool with_dynamic_shape() { return with_dynamic_shape_; }
  AnalysisConfig::Precision precision() { return precision_; }

#if IS_TRT_VERSION_GE(6000)
  nvinfer1::IPluginV2Layer* AddDynamicPlugin(
      nvinfer1::ITensor* const* inputs,
      int num_inputs,
      plugin::DynamicPluginTensorRT* plugin) {
    owned_pluginv2_.emplace_back(plugin);
    return network()->addPluginV2(inputs, num_inputs, *plugin);
  }
#endif

  bool Has(const std::string& attr_name) const {
    return attrs_.count(attr_name) > 0;
  }

  void Erase(const std::string& attr_name) {
    if (!Has(attr_name)) {
      return;
    }
    if (attr_dels_.find(attr_name) != attr_dels_.end()) {
      attr_dels_[attr_name]();
      attr_dels_.erase(attr_name);
    }
    attrs_.erase(attr_name);
  }

  // Set a pointer to the attribute. Engine takes ownership of the attribute.
  template <typename AttrType>
  void Set(const std::string& attr_name, AttrType* attr) {
    if (attrs_.count(attr_name) == 0) {
      PADDLE_ENFORCE_EQ(
          attrs_.count(attr_name),
          0,
          platform::errors::AlreadyExists(
              "Attribute %s already set in trt engine.", attr_name));
    } else {
      VLOG(3) << "Setting the attribute " << attr_name << " for trt engine "
              << this;
    }
    attrs_[attr_name] = attr;
    attr_dels_[attr_name] = [attr, attr_name]() {
      VLOG(3) << "deleting " << attr_name;
      delete attr;
    };
  }

  // Set a pointer to the attribute. Engine doesn't take ownership. Caller
  // should delete the attribute.
  template <typename AttrType>
  void SetNotOwned(const std::string& attr_name, AttrType* attr) {
    PADDLE_ENFORCE_EQ(
        attrs_.count(attr_name),
        0,
        platform::errors::AlreadyExists(
            "Attribute %s already set in trt engine.", attr_name));
    attrs_[attr_name] = attr;
  }

  // Get a reference to the attributed previously set.
  template <typename AttrType>
  AttrType& Get(const std::string& attr_name) const {
    PADDLE_ENFORCE_NE(attrs_.find(attr_name),
                      attrs_.end(),
                      platform::errors::InvalidArgument(
                          "Attribute %s not found in trt engine.", attr_name));
    try {
      return *paddle::any_cast<AttrType*>(attrs_.at(attr_name));
    } catch (paddle::bad_any_cast&) {
      auto TypeToString = [](const std::type_info& info) -> std::string {
        if (std::type_index(info) == std::type_index(typeid(bool*))) {
          return "bool";
        } else if (std::type_index(info) == std::type_index(typeid(int*))) {
          return "int";
        } else if (std::type_index(info) ==
                   std::type_index(typeid(const int*))) {
          return "const int";
        } else if (std::type_index(info) ==
                   std::type_index(typeid(std::string*))) {
          return "std::string";
        }
        return info.name();
      };

      PADDLE_THROW(platform::errors::InvalidArgument(
          "Invalid type for attritube %s, expected: %s, actual: %s.",
          attr_name,
          TypeToString(typeid(AttrType*)),
          TypeToString(attrs_.at(attr_name).type())));
    }
  }

  void SetProfileNum(int num) { max_profile_num_ = num; }

  void GetEngineInfo();

  void SetUseInspector(bool use_inspector) { use_inspector_ = use_inspector; }
  void SetScope(const framework::Scope& scope) { scope_ = &scope; }

  void SetContextMemorySharing(bool context_memory_sharing) {
    context_memory_sharing_ = context_memory_sharing;
  }

 private:
  // Each ICudaEngine object is bound to a specific GPU when it is instantiated,
  // ensure that the thread is associated with the correct device by calling
  // freshDeviceId().
  void freshDeviceId();
  // Used for convert weight into Itensor
  const framework::Scope* scope_;

  // the max batch size
  int max_batch_;
  // the runtime batch size
  static int runtime_batch_;
  // the max memory size the engine uses
  int64_t max_workspace_;

  AnalysisConfig::Precision precision_;
  TRTInt8Calibrator* calibrator_;
  // batch size of the current data, will be updated each Executation.
  int batch_size_{-1};

  // use for engine context memory sharing
  bool context_memory_sharing_{false};

  int device_id_;
  int max_profile_num_{1};
  int cur_profile_num_{0};
  std::unordered_map<PredictorID, int> profile_index_;
  ShapeMapType min_input_shape_;
  ShapeMapType max_input_shape_;
  ShapeMapType optim_input_shape_;
  bool disable_trt_plugin_fp16_{false};
  phi::DataType model_precision_{phi::DataType::FLOAT32};
  bool use_varseqlen_{false};
  bool use_dla_{false};
  int dla_core_{0};
  bool with_ernie_{false};
  bool with_interleaved_{false};
  std::string tensorrt_transformer_posid_;
  std::string tensorrt_transformer_maskid_;
  nvinfer1::ILogger& logger_;

  // max data size for the buffers.
  std::unordered_map<std::string /*name*/, nvinfer1::ITensor* /*ITensor*/>
      itensor_map_;

  std::vector<std::unique_ptr<plugin::PluginTensorRT>> owned_plugin_;
  std::vector<std::unique_ptr<plugin::PluginTensorRTV2Ext>> owned_plugin_v2ext_;
  std::vector<std::unique_ptr<nvinfer1::IPluginV2IOExt>> owned_plugin_v2ioext_;

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
  std::unordered_map<PredictorID, infer_ptr<nvinfer1::IExecutionContext>>
      infer_context_;
  infer_ptr<nvinfer1::IHostMemory> ihost_memory_;
  std::unordered_map<nvinfer1::ITensor*, float> quant_dynamic_range_;

  std::unordered_map<std::string, paddle::any> attrs_;
  std::unordered_map<std::string, std::function<void(void)>> attr_dels_;

  // For dynamic shape
  bool with_dynamic_shape_{false};
#if IS_TRT_VERSION_GE(6000)
  int binding_num_;
  infer_ptr<nvinfer1::IBuilderConfig> infer_builder_config_;
  std::vector<nvinfer1::IOptimizationProfile*> optim_profiles_;
  std::vector<std::unique_ptr<plugin::DynamicPluginTensorRT>> owned_pluginv2_;
#endif
  std::mutex mutex_;
  bool use_inspector_;

 public:
  thread_local static int predictor_id_per_thread;
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
  engine__->network()->add##layer__(__VA_ARGS__)

class TRTEngineManager {
  using PredictorID = int;
  using AllocationPtr = phi::Allocator::AllocationPtr;

 public:
  bool Empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return engines_.size() == 0;
  }

  bool Has(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (engines_.count(name) == 0) return false;
    return engines_.at(name).get() != nullptr;
  }

  TensorRTEngine* Get(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return engines_.at(name).get();
  }

  TensorRTEngine* Create(
      std::string name,
      int max_batch,
      int64_t max_workspace,
      AnalysisConfig::Precision precision = AnalysisConfig::Precision::kFloat32,
      TRTInt8Calibrator* calibrator = nullptr,
      int device_id = 0,
      const std::map<std::string, std::vector<int>> min_input_shape = {},
      const std::map<std::string, std::vector<int>> max_input_shape = {},
      const std::map<std::string, std::vector<int>> optim_input_shape = {},
      bool disable_trt_plugin_fp16 = false,
      phi::DataType model_precision = phi::DataType::FLOAT32,
      nvinfer1::ILogger& logger = NaiveLogger::Global()) {
    auto* p = new TensorRTEngine(max_batch,
                                 max_workspace,
                                 precision,
                                 calibrator,
                                 device_id,
                                 min_input_shape,
                                 max_input_shape,
                                 optim_input_shape,
                                 disable_trt_plugin_fp16,
                                 model_precision,
                                 logger);
    std::lock_guard<std::mutex> lock(mutex_);
    engines_[name].reset(p);
    return p;
  }

  void DeleteAll() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& item : engines_) {
      item.second.reset(nullptr);
    }
    engines_.clear();
  }

  void DeleteKey(const std::string& key) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto iter = engines_.find(key);
    if (iter != engines_.end()) {
      iter->second.reset(nullptr);
      engines_.erase(iter);
    }
  }

  void updateContextMemorySize(size_t mem_size, PredictorID predictor_id) {
    bool size_updated{false};

    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (max_ctx_mem_size_ < mem_size) {
        max_ctx_mem_size_ = mem_size;
        size_updated = true;
      }
    }

    if (size_updated) {
      releaseContextMemory(predictor_id);
    }
  }

  void* getContextMemory(PredictorID predictor_id,
                         const phi::GPUPlace& place,
                         const phi::Stream& stream) {
    std::lock_guard<std::mutex> lock(mutex_);
    static auto alignment = getAlignmentSize(place);
    if (context_memorys_.count(predictor_id) == 0) {
      auto context_memory =
          memory::Alloc(place, max_ctx_mem_size_ + alignment, stream);
      // context_memory_[predictor_id].reset(context_memory.release());
      context_memorys_[predictor_id] = std::move(context_memory);
    }
    return getAlignedMemory(context_memorys_[predictor_id]->ptr(), alignment);
  }

  void releaseContextMemory(PredictorID predictor_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (context_memorys_.count(predictor_id)) {
      context_memorys_[predictor_id].reset(nullptr);
      context_memorys_.erase(predictor_id);
    }
  }

 private:
  size_t getAlignmentSize(const phi::GPUPlace& place) {
    const auto& prop = platform::GetDeviceProperties(place.GetDeviceId());
    return prop.textureAlignment;
  }

  void* getAlignedMemory(void* addr, size_t alignment) {
    return reinterpret_cast<void*>(uintptr_t(addr) & (~(alignment - 1)));
  }

  mutable std::mutex mutex_;
  size_t max_ctx_mem_size_{0};
  std::unordered_map<PredictorID, AllocationPtr> context_memorys_;
  std::unordered_map<std::string, std::unique_ptr<TensorRTEngine>> engines_;
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
