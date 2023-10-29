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

#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstdlib>
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

template <typename T>
static void PrintMatrix(const T* mat_d, int num, int batch_size, int hiddim, std::string name) {
    std::vector<T> tmp(num);
    cudaMemcpy(tmp.data(), mat_d, sizeof(T) * num, cudaMemcpyDeviceToHost);

    std::ofstream outfile;
    outfile.open(name+".txt", std::ios::out);
    std::stringstream ss;
    int token_id = 0;
    for (int i = 0; i < num; ++i) {
      if(i % batch_size == 0){
        ss << "batch_id = " << i / batch_size <<std::endl;
        token_id = 0;
      }
      if(i % hiddim == 0) {
          ss << "token_id = " << token_id++ <<std::endl;
      }
      if(std::is_same<T, int8_t>::value) {
        ss << static_cast<int>(tmp[i]) << std::endl;
      } else {
        ss << std::setprecision(8) << tmp[i] << std::endl;
      }
    }
    outfile << ss.str();
    outfile.close();
}




template<typename T>
struct tokenMerge{
  int32_t operator()(const phi::GPUContext &dev_ctx,
                   bool use_rand,
                   int bsz,
                   int token_number,
                   int src_token_number,
                   int dst_token_number,
                   int final_token_number,
                   int src_token_need_merged_num,
                   int hid_dim,
                   int height,
                   int width,
                   const T *origined_tensor,
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
};

class TokenMergePluginDynamic : public DynamicPluginTensorRT {
 public:
  TokenMergePluginDynamic(bool with_fp16,
                          int bsz,
                          int token_number, 
                          int hid_dim, 
                          const float ratio,
                          const bool use_rand)
      : ratio_(ratio),  bsz_(bsz),  token_number_(token_number), 
        hid_dim_(hid_dim){
    with_fp16_ = with_fp16;
    use_rand_ = use_rand;
    height_ = static_cast<int>(sqrt(token_number));
    width_ = static_cast<int>(sqrt(token_number));
    src_token_number_ = (token_number * 3) / 4;
    dst_token_number_ = token_number - src_token_number_;
    src_need_merged_number_ = token_number * ratio > src_token_number_? src_token_number_ : token_number * ratio;
    final_token_number_ = token_number - src_need_merged_number_;
    VLOG(3) << "use_rand = " << use_rand_ << ", height_ = " << height_ << "width_ = " << width_ << ", src_token_number_ = " << src_token_number_ << ", final_token_number_ = " << final_token_number_;
  }
  
  TokenMergePluginDynamic(void const* serial_data, size_t serial_length) {
    VLOG(3) << "start deserializePlugin token_merge in TokenMergePluginDynamic" << " serial_data = " << serial_data;
    DeserializeValue(&serial_data, &serial_length, &with_fp16_);
    DeserializeValue(&serial_data, &serial_length, &ratio_);
    DeserializeValue(&serial_data, &serial_length, &use_rand_);
    DeserializeValue(&serial_data, &serial_length, &bsz_);
    DeserializeValue(&serial_data, &serial_length, &token_number_);
    DeserializeValue(&serial_data, &serial_length, &hid_dim_);
    DeserializeValue(&serial_data, &serial_length, &height_);
    DeserializeValue(&serial_data, &serial_length, &width_);
    DeserializeValue(&serial_data, &serial_length, &src_token_number_);
    DeserializeValue(&serial_data, &serial_length, &dst_token_number_);
    DeserializeValue(&serial_data, &serial_length, &src_need_merged_number_);
    DeserializeValue(&serial_data, &serial_length, &final_token_number_);
    VLOG(3) << "finish deserializePlugin token_merge in TokenMergePluginDynamic" << " serial_data = " << serial_data;
  }
  nvinfer1::IPluginV2DynamicExt* clone() const TRT_NOEXCEPT override {
    return new TokenMergePluginDynamic(with_fp16_,
                                      bsz_,
                                      token_number_,
                                      hid_dim_,
                                      ratio_,
                                      use_rand_);
  }

  const char* getPluginType() const TRT_NOEXCEPT override {
    return "token_merge_plugin_dynamic";
  }

  int getNbOutputs() const TRT_NOEXCEPT override { return 3; }
  int initialize() TRT_NOEXCEPT override { return 0; }
  void terminate() TRT_NOEXCEPT override {};


  size_t getSerializationSize() const TRT_NOEXCEPT override {
    return SerializedSize(with_fp16_) +  
           SerializedSize(ratio_) + 
           SerializedSize(use_rand_) + 
           SerializedSize(bsz_) +
           SerializedSize(token_number_) +
           SerializedSize(hid_dim_) +
           SerializedSize(height_) + 
           SerializedSize(width_) +
           SerializedSize(src_token_number_) + 
           SerializedSize(dst_token_number_) + 
           SerializedSize(final_token_number_) + 
           SerializedSize(final_token_number_);
  }

  void serialize(void* buffer) const TRT_NOEXCEPT override {
    SerializeValue(&buffer, with_fp16_);
    SerializeValue(&buffer, ratio_);
    SerializeValue(&buffer, use_rand_);
    SerializeValue(&buffer, bsz_);
    SerializeValue(&buffer, token_number_);
    SerializeValue(&buffer, hid_dim_);
    SerializeValue(&buffer, height_);
    SerializeValue(&buffer, width_);
    SerializeValue(&buffer, src_token_number_);
    SerializeValue(&buffer, dst_token_number_);
    SerializeValue(&buffer, src_need_merged_number_);
    SerializeValue(&buffer, final_token_number_);
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
  float ratio_;
  bool use_rand_;
  int bsz_;
  int token_number_;
  int hid_dim_;
  int height_;
  int width_;
  int src_token_number_;
  int dst_token_number_;
  int src_need_merged_number_;
  int final_token_number_;

  //need return
  // void *rand_select_arr{nullptr};
  // void *whether_to_be_merged_arr{nullptr};
};

class TokenMergePluginDynamicCreator : public TensorRTPluginCreator {
 public:
  const char* getPluginName() const TRT_NOEXCEPT override {
    return "token_merge_plugin_dynamic";
  }

  const char* getPluginVersion() const TRT_NOEXCEPT override { return "1"; }

  nvinfer1::IPluginV2* deserializePlugin(const char* name,
                                         const void* serial_data,
                                         size_t serial_length)
      TRT_NOEXCEPT override {
    VLOG(3) << "start deserializePlugin token_merge" << " serial_data = " << serial_data;
    return new TokenMergePluginDynamic(serial_data, serial_length);
  }
};

REGISTER_TRT_PLUGIN_V2(TokenMergePluginDynamicCreator);

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
