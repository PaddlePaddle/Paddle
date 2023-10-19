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
#include <stdio.h>

#include <cassert>
#include <string>
#include <vector>
#include <cmath>

#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin.h"


namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

#define WARP_SIZE 32
#define WARP_NUM_PRE_BLOCK 
#define SELECT_STRIDE 4


template<typename T>
int32_t tokenMerge(cudaStream_t stream,
                   int device_id,
                   nvinfer1::DataType dtype,
                   int bsz,
                   int token_number,
                   int src_token_number,
                   int dst_token_number,
                   int final_token_number,
                   int src_token_need_merged_num,
                   int hid_dim,
                   int height,
                   int width,
                   T *origined_tensor,
                   phi::DenseTensor &src_token_tensor,
                   phi::DenseTensor &dst_token_tensor,
                   phi::DenseTensor &src_L2_tensor,
                   phi::DenseTensor &dst_L2_tensor,
                   phi::DenseTensor &similarity_tensor,
                   phi::DenseTensor &max_similarity_tensor,
                   phi::DenseTensor &max_similarity_idx_tensor,
                   phi::DenseTensor &argsort_res0_tensor,
                   phi::DenseTensor &argsort_res1_tensor,
                   phi::DenseTensor &divided_rank_tensor,
                   T *merged_tensor,
                   int *rand_select_arr,
                   int *whether_to_be_merged_arr);

class TokenMergePluginDynamic : public DynamicPluginTensorRT {
 public:
  TokenMergePluginDynamic(bool with_fp16,
                          int bsz,
                          int token_number, 
                          int hid_dim, 
                          const float ratio)
      : ratio_(ratio), token_number_(token_number), 
        hid_dim_(hid_dim), 
        bsz_(bsz) {
    with_fp16_ = with_fp16;
    height_ = static_cast<int>(sqrt(num));
    width_ = static_cast<int>(sqrt(num));
    src_token_number_ = (token_number * 4) / 3;
    dst_token_number_ = token_number - src_token_number_;
    int merged_token_number = token_number * ratio > src_token_number_? src_token_number_ : token_number * ratio;
    final_token_number_ = token_number - merged_token_number;
  }
  
  TokenMergePluginDynamic(void const* serial_data, size_t serial_length) {
    deserializeBase(&serial_data, &serial_length, &ratio_);
    deserializeBase(&serial_data, &serial_length, &bsz_);
    deserializeBase(&serial_data, &serial_length, &token_number_);
    deserializeBase(&serial_data, &serial_length, &height_);
    deserializeBase(&serial_data, &serial_length, &width_);
    deserializeBase(&serial_data, &serial_length, &src_token_number_);
    deserializeBase(&serial_data, &serial_length, &dst_token_number_);
    deserializeBase(&serial_data, &serial_length, &final_token_number_);
  }


  ~TokenMergePluginDynamic() {}
  TokenMergePlugin* clone() const TRT_NOEXCEPT override {
    return new TokenMergePluginDynamic(with_fp16_,
                                      bsz_,
                                      token_number_,
                                      hid_dim_
                                      ratio_);
  }

  const char* getPluginType() const TRT_NOEXCEPT override {
    return "token_merge_plugin_dynamic";
  }

  int getNbOutputs() const TRT_NOEXCEPT override { return 3; }
  int initialize() TRT_NOEXCEPT override { return 0; }
  void terminate() TRT_NOEXCEPT override;


  size_t getSerializationSize() const TRT_NOEXCEPT override {
    return SerializedSize(ratio_) + SerializedSize(bsz_) +
           SerializedSize(token_number_) + SerializedSize(hid_dim_) +
           SerializedSize(height_) + SerializedSize(width_) +
           SerializedSize(with_fp16_) + SerializedSize(src_token_number_) + 
           SerializedSize(dst_token_number_) + SerializedSize(final_token_number_);
  }

  void serialize(void* buffer) const TRT_NOEXCEPT override {
    SerializeValue(&buffer, bias_);
    SerializeValue(&buffer, scale_);
    SerializeValue(&buffer, begin_norm_axis_);
    SerializeValue(&buffer, eps_);
    SerializeValue(&buffer, mean_shape_);
    SerializeValue(&buffer, variance_shape_);
    SerializeValue(&buffer, with_fp16_);
  }



  bool supportsFormat(nvinfer1::DataType type, nvinfer1::PluginFormat format)
      const TRT_NOEXCEPT override;
  nvinfer1::Dims getOutputDimensions(int index,
                                     const nvinfer1::Dims* inputs,
                                     int nb_input_dims) TRT_NOEXCEPT override;

  int enqueue(const nvinfer1::PluginTensorDesc* input_desc,
              const nvinfer1::PluginTensorDesc* output_desc,
              const void* const* inputs,
              void* const* outputs,
              void* workspace,
              cudaStream_t stream) TRT_NOEXCEPT override;




  size_t getSerializationSize() const TRT_NOEXCEPT override {
    return getBaseSerializationSize();
  }

  // TRT will call this func  to serialize the configuration of TRT
  // It should not be called by users.
  void serialize(void* buffer) const TRT_NOEXCEPT override {
    serializeBase(buffer);
  }
  private:
  float ratio_;
  int bsz_;
  int token_number_;
  int hid_dim_;
  int height_;
  int width_;
  int src_token_number_;
  int dst_token_number_;
  int final_token_number_;

  //need return
  // void *rand_select_arr{nullptr};
  // void *whether_to_be_merged_arr{nullptr};

  void* dst_token_gpu_{nullptr};
  void* dst_L2_gpu_{nullptr};

  void* src_token_gpu_{nullptr};
  void* src_L2_gpu_{nullptr};
  
  void* similarity_gpu_{nullptr};
  void* max_similarity_gpu_{nullptr};
  void* max_similarity_idx_gpu_{nullptr};
  void* argsort_res0_gpu_{nullptr};
  void* argsort_re1_gpu_{nullptr};
  
  void* divided_rank_gpu_{nullptr};
};

class TokenMergePluginCreator : public TensorRTPluginCreator {
 public:
  const char* getPluginName() const TRT_NOEXCEPT override {
    return "token_merge_plugin";
  }

  const char* getPluginVersion() const TRT_NOEXCEPT override { return "1"; }

  nvinfer1::IPluginV2* deserializePlugin(const char* name,
                                         const void* serial_data,
                                         size_t serial_length)
      TRT_NOEXCEPT override {
    return new TokenMergePlugin(serial_data, serial_length);
  }
};
REGISTER_TRT_PLUGIN_V2(TokenMergePluginCreator);




}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
