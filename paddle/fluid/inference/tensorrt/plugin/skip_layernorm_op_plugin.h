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

#pragma once

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

#include "paddle/fluid/inference/tensorrt/engine.h"
#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin.h"
#include "paddle/phi/common/data_type.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

#if IS_TRT_VERSION_GE(6000)

class SkipLayerNormPluginDynamicImplBase {
 public:
  SkipLayerNormPluginDynamicImplBase() {}
  virtual ~SkipLayerNormPluginDynamicImplBase() {}

  virtual int initialize() = 0;
  virtual void terminate() = 0;
  virtual int enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
                      const nvinfer1::PluginTensorDesc* outputDesc,
                      const void* const* inputs,
                      void* const* outputs,
                      void* workspace,
                      cudaStream_t stream) = 0;
  virtual void shareGPUData(
      const SkipLayerNormPluginDynamicImplBase* anthor) = 0;
};

template <typename T>
class SkipLayerNormPluginDynamicImpl
    : public SkipLayerNormPluginDynamicImplBase {
 public:
  explicit SkipLayerNormPluginDynamicImpl(
      T* bias, T* scale, int bias_size, int scale_size, const float eps)
      : bias_(bias),
        scale_(scale),
        bias_size_(bias_size),
        scale_size_(scale_size),
        eps_(eps) {}

  ~SkipLayerNormPluginDynamicImpl() {}

  int initialize();
  void terminate();
  int enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
              const nvinfer1::PluginTensorDesc* outputDesc,
              const void* const* inputs,
              void* const* outputs,
              void* workspace,
              cudaStream_t stream) TRT_NOEXCEPT;
  void shareGPUData(const SkipLayerNormPluginDynamicImplBase* anthor);

 private:
  T* bias_{nullptr};
  T* scale_{nullptr};

  // data on devices
  T* bias_gpu_{nullptr};
  T* scale_gpu_{nullptr};

  int bias_size_;
  int scale_size_;
  float eps_;

  bool is_initialized_{false};
};

class SkipLayerNormPluginDynamic : public DynamicPluginTensorRT {
 public:
  explicit SkipLayerNormPluginDynamic(void* bias,
                                      void* scale,
                                      int bias_size,
                                      int scale_size,
                                      float eps,
                                      bool with_fp16)
      : bias_(bias),
        scale_(scale),
        bias_size_(bias_size),
        scale_size_(scale_size),
        eps_(eps),
        own_host_buff_(false) {
    with_fp16_ = with_fp16;
    if (with_fp16_) {
#ifdef TRT_PLUGIN_FP16_AVALIABLE
      VLOG(1) << "TRT Plugin DataType selected. SkipLayerNorm-->fp16";
      instantiateImpl<half>();
#else
      PADDLE_THROW(platform::errors::Fatal(
          "The Ernie(Bert) tensorRT plugin should be "
          "complied with CUDA version >= 10.0 when running with fp16. "
          "Please recomplie it or try to use fp32 by set "
          "config.EnableTensorRtEngine(1 << 30, 1, 5, "
          "AnalysisConfig::Precision::kFloat32, false, false) "));
#endif
    } else {
      VLOG(1) << "TRT Plugin DataType selected. SkipLayerNorm-->fp32";
      instantiateImpl<float>();
    }
  }

  SkipLayerNormPluginDynamic(void const* serial_data, size_t serial_length)
      : own_host_buff_(true) {
    // the first var is  with_fp16, we will use it.
    DeserializeValue(&serial_data, &serial_length, &with_fp16_);
    DeserializeValue(&serial_data, &serial_length, &bias_size_);
    DeserializeValue(&serial_data, &serial_length, &scale_size_);
    DeserializeValue(&serial_data, &serial_length, &eps_);

    if (with_fp16_) {
      if (bias_size_) {
        bias_ = new half[bias_size_];
        memcpy(bias_, serial_data, sizeof(half) * bias_size_);
      }
      reinterpret_cast<char const*&>(serial_data) += bias_size_ * sizeof(half);
      serial_length -= bias_size_ * sizeof(half);

      if (scale_size_) {
        scale_ = new half[scale_size_];
        memcpy(scale_, serial_data, sizeof(half) * scale_size_);
      }
      reinterpret_cast<char const*&>(serial_data) += scale_size_ * sizeof(half);
      serial_length -= scale_size_ * sizeof(half);
    } else {
      if (bias_size_) {
        bias_ = new float[bias_size_];
        memcpy(bias_, serial_data, sizeof(float) * bias_size_);
      }
      reinterpret_cast<char const*&>(serial_data) += bias_size_ * sizeof(float);
      serial_length -= bias_size_ * sizeof(float);

      if (scale_size_) {
        scale_ = new float[scale_size_];
        memcpy(scale_, serial_data, sizeof(float) * scale_size_);
      }
      reinterpret_cast<char const*&>(serial_data) +=
          scale_size_ * sizeof(float);
      serial_length -= scale_size_ * sizeof(float);
    }

    if (with_fp16_) {
#ifdef TRT_PLUGIN_FP16_AVALIABLE
      instantiateImpl<half>();
#else
      PADDLE_THROW(platform::errors::Fatal(
          "The Ernie(Bert) tensorRT plugin should be "
          "complied with CUDA version >= 10.0 when running with fp16. "
          "Please recomplie it or try to use fp32 by set "
          "config.EnableTensorRtEngine(1 << 30, 1, 5, "
          "AnalysisConfig::Precision::kFloat32, false, false) "));
#endif
    } else {
      instantiateImpl<float>();
    }
  }

  nvinfer1::IPluginV2DynamicExt* clone() const TRT_NOEXCEPT override {
    auto ptr = new SkipLayerNormPluginDynamic(
        bias_, scale_, bias_size_, scale_size_, eps_, with_fp16_);
    ptr->shareGPUData(this);
    return ptr;
  }

  const char* getPluginType() const TRT_NOEXCEPT override {
    return "skip_layernorm_plugin";
  }
  int getNbOutputs() const TRT_NOEXCEPT override { return 1; }
  int initialize() TRT_NOEXCEPT override;
  void terminate() TRT_NOEXCEPT override;

  size_t getSerializationSize() const TRT_NOEXCEPT override {
    size_t sum_num = 0;
    sum_num += SerializedSize(with_fp16_);

    if (with_fp16_) {
      sum_num += (bias_size_ + scale_size_) * sizeof(half);
    } else {
      sum_num += (bias_size_ + scale_size_) * sizeof(float);
    }

    sum_num += SerializedSize(bias_size_);
    sum_num += SerializedSize(scale_size_);
    sum_num += SerializedSize(eps_);

    return sum_num;
  }

  void serialize(void* buffer) const TRT_NOEXCEPT override {
    // the first var is for with_fp16, we will use it later;
    SerializeValue(&buffer, with_fp16_);
    SerializeValue(&buffer, bias_size_);
    SerializeValue(&buffer, scale_size_);
    SerializeValue(&buffer, eps_);
    if (with_fp16_) {
      for (int i = 0; i < bias_size_; ++i) {
        SerializeValue(&buffer, reinterpret_cast<half*>(bias_)[i]);
      }

      for (int i = 0; i < scale_size_; ++i) {
        SerializeValue(&buffer, reinterpret_cast<half*>(scale_)[i]);
      }
    } else {
      for (int i = 0; i < bias_size_; ++i) {
        SerializeValue(&buffer, reinterpret_cast<float*>(bias_)[i]);
      }

      for (int i = 0; i < scale_size_; ++i) {
        SerializeValue(&buffer, reinterpret_cast<float*>(scale_)[i]);
      }
    }
  }

  nvinfer1::DimsExprs getOutputDimensions(int output_index,
                                          const nvinfer1::DimsExprs* inputs,
                                          int nb_inputs,
                                          nvinfer1::IExprBuilder& expr_builder)
      TRT_NOEXCEPT override;

  bool supportsFormatCombination(int pos,
                                 const nvinfer1::PluginTensorDesc* in_out,
                                 int nb_inputs,
                                 int nb_outputs) TRT_NOEXCEPT override;

  void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in,
                       int nb_inputs,
                       const nvinfer1::DynamicPluginTensorDesc* out,
                       int nb_outputs) TRT_NOEXCEPT override {}

  size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs,
                          int nb_inputs,
                          const nvinfer1::PluginTensorDesc* outputs,
                          int nb_outputs) const TRT_NOEXCEPT override {
    return 0;
  }

  int enqueue(const nvinfer1::PluginTensorDesc* input_desc,
              const nvinfer1::PluginTensorDesc* output_desc,
              const void* const* inputs,
              void* const* outputs,
              void* workspace,
              cudaStream_t stream) TRT_NOEXCEPT override;
  nvinfer1::DataType getOutputDataType(int index,
                                       const nvinfer1::DataType* input_types,
                                       int nb_inputs) const
      TRT_NOEXCEPT override;

  void destroy() TRT_NOEXCEPT override {
    if (own_host_buff_) {
      if (with_fp16_) {
        delete[] reinterpret_cast<half*>(bias_);
        delete[] reinterpret_cast<half*>(scale_);
      } else {
        delete[] reinterpret_cast<float*>(bias_);
        delete[] reinterpret_cast<float*>(scale_);
      }
    }
    delete impl_;
    delete this;
  }

 private:
  void* bias_{nullptr};
  void* scale_{nullptr};

  int bias_size_;
  int scale_size_;
  float eps_;

  bool own_host_buff_{false};
  SkipLayerNormPluginDynamicImplBase* impl_{nullptr};

  void shareGPUData(const SkipLayerNormPluginDynamic* anthor) {
    impl_->shareGPUData(anthor->impl_);
  }

  template <typename U>
  void instantiateImpl() {
    impl_ = new SkipLayerNormPluginDynamicImpl<U>(reinterpret_cast<U*>(bias_),
                                                  reinterpret_cast<U*>(scale_),
                                                  bias_size_,
                                                  scale_size_,
                                                  eps_);
  }
};

class SkipLayerNormPluginDynamicCreator : public nvinfer1::IPluginCreator {
 public:
  SkipLayerNormPluginDynamicCreator() {}
  const char* getPluginName() const TRT_NOEXCEPT override {
    return "skip_layernorm_plugin";
  }

  const char* getPluginVersion() const TRT_NOEXCEPT override { return "1"; }

  const nvinfer1::PluginFieldCollection* getFieldNames() TRT_NOEXCEPT override {
    return &field_collection_;
  }

  nvinfer1::IPluginV2* createPlugin(const char* name,
                                    const nvinfer1::PluginFieldCollection* fc)
      TRT_NOEXCEPT override {
    return nullptr;
  }

  nvinfer1::IPluginV2* deserializePlugin(const char* name,
                                         const void* serial_data,
                                         size_t serial_length)
      TRT_NOEXCEPT override {
    return new SkipLayerNormPluginDynamic(serial_data, serial_length);
  }

  void setPluginNamespace(const char* lib_namespace) TRT_NOEXCEPT override {
    plugin_namespace_ = lib_namespace;
  }

  const char* getPluginNamespace() const TRT_NOEXCEPT override {
    return plugin_namespace_.c_str();
  }

 private:
  std::string plugin_namespace_;
  std::string plugin_name_;
  nvinfer1::PluginFieldCollection field_collection_;
  std::vector<nvinfer1::PluginField> plugin_attributes_;
};
REGISTER_TRT_PLUGIN_V2(SkipLayerNormPluginDynamicCreator);

#endif

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
