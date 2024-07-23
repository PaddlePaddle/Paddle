/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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
#include <cassert>
#include <string>
#include <vector>

#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/phi/backends/dynload/cublasLt.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

class MatmulPlugin : public nvinfer1::IPluginV2IOExt {
 public:
  MatmulPlugin(nvinfer1::Dims const& dims_x,
               nvinfer1::Dims const& dims_y,
               bool transA,
               bool transB,
               float alpha)
      : dims_x_(dims_x),
        dims_y_(dims_y),
        transB_(transA),
        transA_(transB),
        alpha_(alpha) {}

  MatmulPlugin(void const* serial_data, size_t serial_length) {
    DeserializeValue(&serial_data, &serial_length, &dims_x_);
    DeserializeValue(&serial_data, &serial_length, &dims_y_);
    DeserializeValue(&serial_data, &serial_length, &transB_);
    DeserializeValue(&serial_data, &serial_length, &transA_);
    DeserializeValue(&serial_data, &serial_length, &alpha_);
    DeserializeValue(&serial_data, &serial_length, &alpha_scale_);
    DeserializeValue(&serial_data, &serial_length, &alpha_one_);
    DeserializeValue(&serial_data, &serial_length, &alpha_zero_);
    DeserializeValue(&serial_data, &serial_length, &batch_);
    DeserializeValue(&serial_data, &serial_length, &k_);
    DeserializeValue(&serial_data, &serial_length, &m_);
    DeserializeValue(&serial_data, &serial_length, &n_);
    DeserializeValue(&serial_data, &serial_length, &cublas_);
    DeserializeValue(&serial_data, &serial_length, &type_);
    DeserializeValue(&serial_data, &serial_length, &Adesc_);
    DeserializeValue(&serial_data, &serial_length, &Bdesc_);
    DeserializeValue(&serial_data, &serial_length, &Cdesc_);
    DeserializeValue(&serial_data, &serial_length, &AtransformDesc_);
    DeserializeValue(&serial_data, &serial_length, &BtransformDesc_);
    DeserializeValue(&serial_data, &serial_length, &CtransformDesc_);
    DeserializeValue(&serial_data, &serial_length, &Atransform_);
    DeserializeValue(&serial_data, &serial_length, &Btransform_);
    DeserializeValue(&serial_data, &serial_length, &Ctransform_);
    DeserializeValue(&serial_data, &serial_length, &transformDescT_);
    DeserializeValue(&serial_data, &serial_length, &transformDescN_);
    DeserializeValue(&serial_data, &serial_length, &matmulDesc_);
  }

  virtual bool isOutputBroadcastAcrossBatch(int32_t output_index,
                                            const bool* input_is_broadcasted,
                                            int32_t nb_inputs) const
      TRT_NOEXCEPT {
    return false;
  }

  virtual bool canBroadcastInputAcrossBatch(int32_t input_index) const
      TRT_NOEXCEPT {
    return false;
  }

  const char* getPluginVersion() const TRT_NOEXCEPT override { return "1"; }

  size_t getWorkspaceSize(int) const TRT_NOEXCEPT override { return 0; }

  void setPluginNamespace(const char* plugin_namespace) TRT_NOEXCEPT override {
    name_space_ = plugin_namespace;
  }

  nvinfer1::IPluginV2IOExt* clone() const TRT_NOEXCEPT override {
    MatmulPlugin* ptr =
        new MatmulPlugin(dims_x_, dims_y_, transB_, transA_, alpha_);
    ptr->setPluginNamespace(this->getPluginNamespace());
    ptr->batch_ = batch_;
    ptr->k_ = k_;
    ptr->m_ = m_;
    ptr->n_ = n_;
    ptr->alpha_scale_ = alpha_scale_;
    ptr->alpha_one_ = alpha_one_;
    ptr->alpha_zero_ = alpha_zero_;
    ptr->cublas_ = cublas_;
    ptr->type_ = type_;
    ptr->Adesc_ = Adesc_;
    ptr->Bdesc_ = Bdesc_;
    ptr->Cdesc_ = Cdesc_;
    ptr->AtransformDesc_ = AtransformDesc_;
    ptr->BtransformDesc_ = BtransformDesc_;
    ptr->CtransformDesc_ = CtransformDesc_;
    ptr->Atransform_ = Atransform_;
    ptr->Btransform_ = Btransform_;
    ptr->Ctransform_ = Ctransform_;
    ptr->transformDescT_ = transformDescT_;
    ptr->transformDescN_ = transformDescN_;
    ptr->matmulDesc_ = matmulDesc_;
    return ptr;
  }

  const char* getPluginNamespace() const TRT_NOEXCEPT override {
    return name_space_.c_str();
  }

  const char* getPluginType() const TRT_NOEXCEPT override {
    return "matmul_int8_plugin";
  }

  nvinfer1::DataType getOutputDataType(int index,
                                       const nvinfer1::DataType* input_types,
                                       int nb_inputs) const
      TRT_NOEXCEPT override;

  int getNbOutputs() const TRT_NOEXCEPT override { return 1; }

  nvinfer1::Dims getOutputDimensions(int index,
                                     const nvinfer1::Dims* input_dims,
                                     int num_inputs) TRT_NOEXCEPT override;

  bool supportsFormatCombination(int32_t pos,
                                 nvinfer1::PluginTensorDesc const* inOut,
                                 int32_t nbInputs,
                                 int32_t nbOutputs) const TRT_NOEXCEPT override;

  void configurePlugin(const nvinfer1::PluginTensorDesc* in,
                       int32_t nbInputs,
                       const nvinfer1::PluginTensorDesc* out,
                       int32_t nbOutputs) TRT_NOEXCEPT override;

  /*
    bool supportsFormat(nvinfer1::DataType type, nvinfer1::PluginFormat format)
        const TRT_NOEXCEPT override;
  */
  int initialize() TRT_NOEXCEPT { return 0; }
  void terminate() TRT_NOEXCEPT;

#if IS_TRT_VERSION_LT(8000)
  int enqueue(int batch_size,
              const void* const* inputs,
              void** outputs,
#else
  int enqueue(int batch_size,
              const void* const* inputs,
              void* const* outputs,
#endif
              void* workspace,
              cudaStream_t stream) TRT_NOEXCEPT override;

  void destroy() TRT_NOEXCEPT override { delete this; }
  void attachToContext(cudnnContext* cudnnContext,
                       cublasContext* cublasContext,
                       nvinfer1::IGpuAllocator* gpuAllocator)
      TRT_NOEXCEPT override;
  void detachFromContext() TRT_NOEXCEPT override;

 protected:
  nvinfer1::Dims dims_x_;
  nvinfer1::Dims dims_y_;
  bool transB_;
  bool transA_;
  float alpha_;
  void *alpha_scale_{nullptr}, *alpha_one_{nullptr}, *alpha_zero_{nullptr};
  int batch_;
  uint64_t k_;
  uint64_t m_;
  uint64_t n_;
  cublasLtHandle_t cublas_{nullptr};
  nvinfer1::DataType type_;
  cublasLtMatrixLayout_t Adesc_{nullptr}, Bdesc_{nullptr}, Cdesc_{nullptr};
  cublasLtMatrixLayout_t AtransformDesc_{nullptr}, BtransformDesc_{nullptr},
      CtransformDesc_{nullptr};
  int8_t *Atransform_{nullptr}, *Btransform_{nullptr}, *Ctransform_{nullptr};
  cublasLtMatrixTransformDesc_t transformDescT_{nullptr},
      transformDescN_{nullptr};
  cublasLtMatmulDesc_t matmulDesc_{nullptr};
  std::string name_space_;

  size_t getSerializationSize() const TRT_NOEXCEPT override {
    return SerializedSize(dims_x_) + SerializedSize(dims_y_) +
           SerializedSize(transB_) + SerializedSize(transA_) +
           SerializedSize(alpha_) + SerializedSize(alpha_scale_) +
           SerializedSize(alpha_one_) + SerializedSize(alpha_zero_) +
           SerializedSize(batch_) + SerializedSize(k_) + SerializedSize(m_) +
           SerializedSize(n_) + SerializedSize(cublas_) +
           SerializedSize(type_) + SerializedSize(Adesc_) +
           SerializedSize(Bdesc_) + SerializedSize(Cdesc_) +
           SerializedSize(AtransformDesc_) + SerializedSize(BtransformDesc_) +
           SerializedSize(CtransformDesc_) + SerializedSize(Atransform_) +
           SerializedSize(Btransform_) + SerializedSize(Ctransform_) +
           SerializedSize(transformDescT_) + SerializedSize(transformDescN_) +
           SerializedSize(matmulDesc_);
  }

  void serialize(void* buffer) const TRT_NOEXCEPT override {
    SerializeValue(&buffer, dims_x_);
    SerializeValue(&buffer, dims_y_);
    SerializeValue(&buffer, transB_);
    SerializeValue(&buffer, transA_);
    SerializeValue(&buffer, alpha_);
    SerializeValue(&buffer, alpha_scale_);
    SerializeValue(&buffer, alpha_one_);
    SerializeValue(&buffer, alpha_zero_);
    SerializeValue(&buffer, batch_);
    SerializeValue(&buffer, k_);
    SerializeValue(&buffer, m_);
    SerializeValue(&buffer, n_);
    SerializeValue(&buffer, cublas_);
    SerializeValue(&buffer, type_);
    SerializeValue(&buffer, Adesc_);
    SerializeValue(&buffer, Bdesc_);
    SerializeValue(&buffer, Cdesc_);
    SerializeValue(&buffer, AtransformDesc_);
    SerializeValue(&buffer, BtransformDesc_);
    SerializeValue(&buffer, CtransformDesc_);
    SerializeValue(&buffer, Atransform_);
    SerializeValue(&buffer, Btransform_);
    SerializeValue(&buffer, Ctransform_);
    SerializeValue(&buffer, transformDescT_);
    SerializeValue(&buffer, transformDescN_);
    SerializeValue(&buffer, matmulDesc_);
  }
};

class MatmulPluginCreator : public nvinfer1::IPluginCreator {
 public:
  MatmulPluginCreator() {}
  const char* getPluginName() const TRT_NOEXCEPT override {
    return "matmul_int8_plugin";
  }
  const char* getPluginVersion() const TRT_NOEXCEPT override { return "1"; }

  const nvinfer1::PluginFieldCollection* getFieldNames() TRT_NOEXCEPT override {
    return &field_collection_;
  }

  nvinfer1::IPluginV2IOExt* createPlugin(
      const char* name,
      const nvinfer1::PluginFieldCollection* fc) TRT_NOEXCEPT override {
    return nullptr;
  }

  nvinfer1::IPluginV2IOExt* deserializePlugin(const char* name,
                                              void const* serial_data,
                                              size_t serial_length)
      TRT_NOEXCEPT override {
    MatmulPlugin* obj = new MatmulPlugin(serial_data, serial_length);
    obj->setPluginNamespace(name);
    return obj;
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
  nvinfer1::PluginFieldCollection field_collection_{0, nullptr};
  std::vector<nvinfer1::PluginField> plugin_attributes_;
};
REGISTER_TRT_PLUGIN_V2(MatmulPluginCreator);

#if IS_TRT_VERSION_GE(6000)
class MatmulPluginDynamic : public DynamicPluginTensorRT {
 public:
  MatmulPluginDynamic(bool transA, bool transB, float alpha)
      : transB_(transA), transA_(transB), alpha_(alpha) {}

  MatmulPluginDynamic(void const* serial_data, size_t serial_length) {
    DeserializeValue(&serial_data, &serial_length, &transB_);
    DeserializeValue(&serial_data, &serial_length, &transA_);
    DeserializeValue(&serial_data, &serial_length, &alpha_);
    DeserializeValue(&serial_data, &serial_length, &alpha_scale_);
    DeserializeValue(&serial_data, &serial_length, &alpha_one_);
    DeserializeValue(&serial_data, &serial_length, &alpha_zero_);
    DeserializeValue(&serial_data, &serial_length, &cublas_);
    DeserializeValue(&serial_data, &serial_length, &Atransform_);
    DeserializeValue(&serial_data, &serial_length, &Btransform_);
    DeserializeValue(&serial_data, &serial_length, &Ctransform_);
    DeserializeValue(&serial_data, &serial_length, &type_);
  }

  nvinfer1::IPluginV2DynamicExt* clone() const TRT_NOEXCEPT override {
    MatmulPluginDynamic* ptr =
        new MatmulPluginDynamic(transB_, transA_, alpha_);
    ptr->setPluginNamespace(this->getPluginNamespace());
    ptr->alpha_scale_ = alpha_scale_;
    ptr->alpha_one_ = alpha_one_;
    ptr->alpha_zero_ = alpha_zero_;
    ptr->cublas_ = cublas_;
    ptr->Atransform_ = Atransform_;
    ptr->Btransform_ = Btransform_;
    ptr->Ctransform_ = Ctransform_;
    ptr->type_ = type_;
    return ptr;
  }

  const char* getPluginType() const TRT_NOEXCEPT override {
    return "matmul_int8_dynamic_plugin";
  }

  int getNbOutputs() const TRT_NOEXCEPT override { return 1; }

  int initialize() TRT_NOEXCEPT { return 0; }
  void terminate() TRT_NOEXCEPT;
  nvinfer1::DimsExprs getOutputDimensions(
      int outputIndex,
      const nvinfer1::DimsExprs* inputs,
      int nbInputs,
      nvinfer1::IExprBuilder& exprBuilder)  // NOLINT
      TRT_NOEXCEPT override;

  bool supportsFormatCombination(int pos,
                                 const nvinfer1::PluginTensorDesc* inOut,
                                 int nbInputs,
                                 int nbOutputs) TRT_NOEXCEPT override;

  void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* inputs,
                       int nbInputs,
                       const nvinfer1::DynamicPluginTensorDesc* outputs,
                       int nbOutputs) TRT_NOEXCEPT override;

  size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs,
                          int nbInputs,
                          const nvinfer1::PluginTensorDesc* outputs,
                          int nbOutputs) const TRT_NOEXCEPT override {
    return 0;
  }

  void attachToContext(cudnnContext* cudnnContext,
                       cublasContext* cublasContext,
                       nvinfer1::IGpuAllocator* gpuAllocator)
      TRT_NOEXCEPT override;

  void detachFromContext() TRT_NOEXCEPT override;

  int enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
              const nvinfer1::PluginTensorDesc* outputDesc,
              const void* const* inputs,
              void* const* outputs,
              void* workspace,
              cudaStream_t stream) TRT_NOEXCEPT override;
  nvinfer1::DataType getOutputDataType(int index,
                                       const nvinfer1::DataType* inputTypes,
                                       int nbInputs) const
      TRT_NOEXCEPT override;

  void destroy() TRT_NOEXCEPT override { delete this; }

 protected:
  bool transB_;
  bool transA_;
  float alpha_;
  void *alpha_scale_{nullptr}, *alpha_one_{nullptr}, *alpha_zero_{nullptr};
  cublasLtHandle_t cublas_{nullptr};
  nvinfer1::DataType type_;
  int8_t *Atransform_{nullptr}, *Btransform_{nullptr}, *Ctransform_{nullptr};
  std::string name_space_;

  size_t getSerializationSize() const TRT_NOEXCEPT override {
    return SerializedSize(transB_) + SerializedSize(transA_) +
           SerializedSize(alpha_) + SerializedSize(alpha_scale_) +
           SerializedSize(alpha_one_) + SerializedSize(alpha_zero_) +
           SerializedSize(Atransform_) + SerializedSize(Btransform_) +
           SerializedSize(Ctransform_) + SerializedSize(cublas_) +
           SerializedSize(type_);
  }

  void serialize(void* buffer) const TRT_NOEXCEPT override {
    SerializeValue(&buffer, transB_);
    SerializeValue(&buffer, transA_);
    SerializeValue(&buffer, alpha_);
    SerializeValue(&buffer, alpha_scale_);
    SerializeValue(&buffer, alpha_one_);
    SerializeValue(&buffer, alpha_zero_);
    SerializeValue(&buffer, Atransform_);
    SerializeValue(&buffer, Btransform_);
    SerializeValue(&buffer, Ctransform_);
    SerializeValue(&buffer, cublas_);
    SerializeValue(&buffer, type_);
  }
};

class MatmulPluginDynamicCreator : public nvinfer1::IPluginCreator {
 public:
  MatmulPluginDynamicCreator() {}
  const char* getPluginName() const TRT_NOEXCEPT override {
    return "matmul_int8_dynamic_plugin";
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
                                         void const* serial_data,
                                         size_t serial_length)
      TRT_NOEXCEPT override {
    MatmulPluginDynamic* obj =
        new MatmulPluginDynamic(serial_data, serial_length);
    obj->setPluginNamespace(name);
    return obj;
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
  nvinfer1::PluginFieldCollection field_collection_{0, nullptr};
  std::vector<nvinfer1::PluginField> plugin_attributes_;
};
REGISTER_TRT_PLUGIN_V2(MatmulPluginDynamicCreator);
#endif
}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
