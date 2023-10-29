// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include <cmath>

#include <glog/logging.h>
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/inference/tensorrt/engine.h"
#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin.h"
#include "paddle/fluid/platform/device_context.h"


namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

#define WARP_SIZE 32
#define WARP_NUM_PRE_BLOCK 
#define SELECT_STRIDE 4


class TokenUnmergePluginDynamic : public DynamicPluginTensorRT {
 public:
  TokenUnmergePluginDynamic(bool with_fp16,
                          int bsz,
                          int token_number,
                          int final_token_number,
                          int hid_dim)
      : bsz_(bsz), token_number_(token_number), final_token_number_(final_token_number), hid_dim_(hid_dim){
    with_fp16_ = with_fp16;
    height_ = static_cast<int>(sqrt(token_number));
    width_ = static_cast<int>(sqrt(token_number));
    src_token_number_ = (token_number * 3) / 4;
    dst_token_number_ = token_number - src_token_number_;
    VLOG(3) << "createTokenUnmergePluginDynamic" <<"height_ = " << height_ << "width_ = " << width_ << ", src_token_number_ = " << src_token_number_ << ", final_token_number_ = " << final_token_number_ << "hid_dim = " << hid_dim_;
  }
  
  TokenUnmergePluginDynamic(void const* serial_data, size_t serial_length) {
    VLOG(3) << "start deserializePlugin token_unmerge in TokenUnmergePluginDynamic" << " serial_data = " << serial_data;
    DeserializeValue(&serial_data, &serial_length, &with_fp16_);
    DeserializeValue(&serial_data, &serial_length, &bsz_);
    DeserializeValue(&serial_data, &serial_length, &token_number_);
    DeserializeValue(&serial_data, &serial_length, &height_);
    DeserializeValue(&serial_data, &serial_length, &width_);
    DeserializeValue(&serial_data, &serial_length, &src_token_number_);
    DeserializeValue(&serial_data, &serial_length, &dst_token_number_);
    DeserializeValue(&serial_data, &serial_length, &final_token_number_);
    DeserializeValue(&serial_data, &serial_length, &hid_dim_);
    VLOG(3) << "finish deserializePlugin token_unmerge in TokenUnmergePluginDynamic" << " serial_data = " << serial_data;

  }
  nvinfer1::IPluginV2DynamicExt* clone() const TRT_NOEXCEPT override {
    return new TokenUnmergePluginDynamic(with_fp16_,
                                      bsz_,
                                      token_number_,
                                      final_token_number_,
                                      hid_dim_);
  }

  const char* getPluginType() const TRT_NOEXCEPT override {
    return "token_unmerge_plugin_dynamic";
  }

  int getNbOutputs() const TRT_NOEXCEPT override { return 1; }
  int initialize() TRT_NOEXCEPT override { return 0; }
  void terminate() TRT_NOEXCEPT override{};


  size_t getSerializationSize() const TRT_NOEXCEPT override {
    return SerializedSize(with_fp16_) + 
           SerializedSize(bsz_) +
           SerializedSize(token_number_) +
           SerializedSize(height_) +
           SerializedSize(width_)  +
           SerializedSize(src_token_number_) +
           SerializedSize(dst_token_number_) +
           SerializedSize(final_token_number_) + 
           SerializedSize(hid_dim_);
  }

  void serialize(void* buffer) const TRT_NOEXCEPT override {
    SerializeValue(&buffer, with_fp16_);
    SerializeValue(&buffer, bsz_);
    SerializeValue(&buffer, token_number_);
    SerializeValue(&buffer, height_);
    SerializeValue(&buffer, width_);
    SerializeValue(&buffer, src_token_number_);
    SerializeValue(&buffer, dst_token_number_);
    SerializeValue(&buffer, final_token_number_);
    SerializeValue(&buffer, hid_dim_);
  }

  nvinfer1::DimsExprs getOutputDimensions(
      int output_index,
      const nvinfer1::DimsExprs* inputs,
      int nb_inputs,
      nvinfer1::IExprBuilder& expr_builder)  // NOLINT
      TRT_NOEXCEPT override;

  bool supportsFormatCombination(int pos,
                                 const nvinfer1::PluginTensorDesc* inOut,
                                 int nbInputs,
                                 int nbOutputs) TRT_NOEXCEPT override;

  void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in,
                       int nbInputs,
                       const nvinfer1::DynamicPluginTensorDesc* out,
                       int nbOutputs) TRT_NOEXCEPT override;

  size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs,
                          int nbInputs,
                          const nvinfer1::PluginTensorDesc* outputs,
                          int nbOutputs) const TRT_NOEXCEPT override {
    return 0;
  }

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

  private:
  int bsz_;
  int token_number_;
  int height_;
  int width_;
  int src_token_number_;
  int dst_token_number_;
  int final_token_number_;
  int hid_dim_;
};

class TokenUnmergePluginDynamicCreator : public TensorRTPluginCreator {
 public:
  const char* getPluginName() const TRT_NOEXCEPT override {
    return "token_unmerge_plugin_dynamic";
  }

  const char* getPluginVersion() const TRT_NOEXCEPT override { return "1"; }

  nvinfer1::IPluginV2* deserializePlugin(const char* name,
                                         const void* serial_data,
                                         size_t serial_length)
      TRT_NOEXCEPT override {
    VLOG(3) << "start deserializePlugin token_unmerge" << "serial_data = " << serial_data;
    return new TokenUnmergePluginDynamic(serial_data, serial_length);
  }
};

REGISTER_TRT_PLUGIN_V2(TokenUnmergePluginDynamicCreator);


}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
