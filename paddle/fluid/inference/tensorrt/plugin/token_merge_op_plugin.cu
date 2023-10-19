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

#include <cassert>
#include <cstring>
#include <vector>

#include "paddle/fluid/inference/tensorrt/plugin/token_merge_op_plugin.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {
  TokenMergePluginDynamic::TokenMergePluginDynamic(int bsz,int token_number, int hid_dim, const float ratio)
    : ratio_(ratio), token_number_(token_number), hid_dim_(hid_dim), bsz_(bsz);
     {
       src_token_number_ = (token_number * 4) / 3;
       dst_token_number_ = token_number - src_token_number_;
       int merged_token_number = token_number * ratio > src_token_number_? src_token_number_ : token_number * ratio;
       final_token_number_ = token_number - merged_token_number;
     }

  int TokenMergePluginDynamic::initialize() TRT_NOEXCEPT {}

  void TransLayerNormPluginDynamic::terminate() TRT_NOEXCEPT {}
  
  bool TokenMergePluginDynamic::supportsFormatCombination(
    int pos,
    const nvinfer1::PluginTensorDesc *in_out,
    int nb_inputs,
    int nb_outputs) TRT_NOEXCEPT {
  PADDLE_ENFORCE_NOT_NULL(
      in_out,
      platform::errors::InvalidArgument("The input of TokenMerge "
                                        "plugin shoule not be nullptr."));
  PADDLE_ENFORCE_LT(
      pos,
      nb_inputs + nb_outputs,
      platform::errors::InvalidArgument("The pos(%d) should be less than the "
                                        "num(%d) of the input and the output.",
                                        pos,
                                        nb_inputs + nb_outputs));
  const nvinfer1::PluginTensorDesc &in = in_out[pos];
 if (pos == 0) {
    if (with_fp16_) {
        return in.type == nvinfer1::DataType::kHALF && in.format == nvinfer1::PluginFormat::kLINEAR;
    } else {
      return in.type == nvinfer1::DataType::kFLOAT && in.format == nvinfer1::TensorFormat::kLINEAR;
    }
  }
  if (pos == 1) {
      return in.type == in_out[0].type && in.format == nvinfer1::TensorFormat::kLINEAR;
  }
  if (pos == 2) 
    return in.type == kINT32 && in.format == nvinfer1::TensorFormat::kLINEAR;

  if (pos == 3) 
      return in.type == kINT32 && in.format == nvinfer1::PluginFormat::kLINEAR;
}
  
  
  
  
  





  
  nvinfer1::DimsExprs TokenMergePluginDynamic::getOutputDimensions(
    int output_index,
    const nvinfer1::DimsExprs* inputs,
    int nb_inputs,
    nvinfer1::IExprBuilder& expr_builder) TRT_NOEXCEPT {
  PADDLE_ENFORCE_EQ(nb_inputs,
                    1,
                    platform::errors::InvalidArgument(
                        "The Split plugin should be only one input."));
  PADDLE_ENFORCE_LT(output_index,
                    3,
                    platform::errors::InvalidArgument(
                        "When GetOutputDimensions, the index(%d) should not "
                        "greater the num(%d) of the outpus.",
                        output_index,
                        3));
  int bsz = inputs[0].d[0];
  int tokenNumber = inputs[0].d[1];
  int hidDim = inputs[0].d[2];

  nvinfer1::Dims outputDims;
  if (index == 1 || index == 2) {
      outputDims.nbDims = 2;
      outputDims.d[0] = bsz;
      outputDims.d[1] = tokenNumber;
  } else if (index == 0) {
      int finalTokenNumber = tokenNumber * ratio_;
      outputDims.nbDims = 3;
      outputDims.d[0] = bsz;
      outputDims.d[1] = finalTokenNumber;
      outputDims.d[2] = hidDim;
  }
  return outputDims;
}






int TokenMergePluginDynamic::enqueue(const nvinfer1::PluginTensorDesc* input_desc,
                                const nvinfer1::PluginTensorDesc* output_desc,
                                const void* const* inputs,
                                void* const* outputs,
                                void* workspace,
                                cudaStream_t stream) TRT_NOEXCEPT {
  const auto &input_dims = input_desc[0].dims;
  auto input_type = input_desc[0].type;
  int bsz = input_dims.d[0];
  int token_numebr = input_dims.d[1];
  int hid_dim = input_dims.d[2];

  paddle::platform::DeviceContextPool &pool =
      paddle::platform::DeviceContextPool::Instance();
  platform::CUDAPlace place(platform::GetCurrentDeviceId());
  auto *device_context = static_cast<phi::GPUContext *>(pool.Get(place));
  const phi::GPUContext &dev_ctx = *device_context;


   //divied_rank
  phi::DenseTensorMeta divied_rank_meta(phi::DataType::INT32,
                               phi::make_ddim({bsz, dst_token_number_}));
  std::shared_ptr<phi::Allocation> divied_rank_alloc(new phi::Allocation(
      static_cast<void *>(const_cast<int *>(divied_rank_gpu_)),  // NOLINT
      bsz * dst_token_number_ * sizeof(int),
      place))
  auto divied_rank_tensor = phi::DenseTensor(divied_rank_alloc, divied_rank_meta);
  
  //max_similarity_idx
  phi::DenseTensorMeta max_similarity_idx_meta(phi::DataType::INT32,
                               phi::make_ddim({bsz, src_token_number_}));
  std::shared_ptr<phi::Allocation> max_similarity_idx_alloc(new phi::Allocation(
      static_cast<void *>(const_cast<int *>(max_similarity_gpu_)),  // NOLINT
      bsz * src_token_number_ * sizeof(int),
      place))
  auto max_similarity_idx_tensor = phi::DenseTensor(max_similarity_idx_alloc, max_similarity_idx_meta);


  if (input_type == nvinfer1::DataType::kFLOAT) {
    VLOG(3) << "TRT Plugin DataType selected. TokenMerge-->fp32";
    //dst_token
    phi::DenseTensorMeta dst_token_meta(phi::DataType::FLOAT32,
                                    phi::make_ddim({bsz, dst_token_number_, hid_dim}));
    std::shared_ptr<phi::Allocation> dst_token_alloc(new phi::Allocation(
        static_cast<void *>(const_cast<float *>(dst_token_gpu_)),  
        bsz * dst_token_number * hid_dim * sizeof(float),
        place));
    auto dst_token_tensor = phi::DenseTensor(dst_token_alloc, dst_token_meta);

    //src_token
    phi::DenseTensorMeta src_token_meta(phi::DataType::FLOAT32,
                                    phi::make_ddim({bsz, dst_token_number_, hid_dim}));
    std::shared_ptr<phi::Allocation> src_token_alloc(new phi::Allocation(
        static_cast<void *>(const_cast<float *>(dst_token_gpu_)),  
        bsz * src_token_number * hid_dim * sizeof(float),
        place));
    auto src_token_tensor = phi::DenseTensor(src_token_alloc, src_token_meta);
    
    
    //dst_L2
    phi::DenseTensorMeta dst_L2_meta(phi::DataType::FLOAT32,
                                    phi::make_ddim({bsz, dst_token_number_, hid_dim}));
    std::shared_ptr<phi::Allocation> dst_L2_alloc(new phi::Allocation(
        static_cast<void *>(const_cast<float *>(dst_L2_gpu_)),  
        bsz * dst_token_number * hid_dim * sizeof(float),
        place));
    auto dst_L2_tensor = phi::DenseTensor(dst_L2_alloc, dst_L2_meta);
  
    //src_L2
    phi::DenseTensorMeta src_L2_meta(phi::DataType::FLOAT32,
                                 phi::make_ddim({bsz, src_token_number_, hid_dim}));
    std::shared_ptr<phi::Allocation> src_L2_alloc(new phi::Allocation(
        static_cast<void *>(const_cast<float *>(src_L2_gpu_)),  
        bsz * src_token_number * hid_dim * sizeof(float),
        place));
    auto src_L2_tensor = phi::DenseTensor(src_L2_alloc, src_L2_meta);
    
    //similarity
    phi::DenseTensorMeta similarity_meta(phi::DataType::FLOAT32,
                                 phi::make_ddim({bsz, src_token_number_ * dst_token_number_}));
    std::shared_ptr<phi::Allocation> similarity_alloc(new phi::Allocation(
        static_cast<void *>(const_cast<float *>(similarity_gpu_)),  // NOLINT
        bsz * src_token_number_ * dst_token_number_ * sizeof(float),
        place))
    auto similarity_tensor = phi::DenseTensor(similarity_alloc, similarity_meta);


    //max_similarity and tensors for argsort
    phi::DenseTensorMeta max_similarity_meta(phi::DataType::FLOAT32,
                                 phi::make_ddim({bsz, src_token_number_}));
    std::shared_ptr<phi::Allocation> max_similarity_alloc(new phi::Allocation(
        static_cast<void *>(const_cast<float *>(max_similarity_gpu_)),  // NOLINT
        bsz * src_token_number_ * sizeof(float),
        place))
    auto max_similarity_tensor = phi::DenseTensor(max_similarity_alloc, max_similarity_meta);

    phi::DenseTensorMeta argsort_res0_meta(phi::DataType::FLOAT32,
                                 phi::make_ddim({bsz, src_token_number_}));
    std::shared_ptr<phi::Allocation> argsort_res0_alloc(new phi::Allocation(
        static_cast<void *>(const_cast<float *>(argsort_res0_gpu_)),  // NOLINT
        bsz * src_token_number_ * sizeof(float),
        place))
    auto argsort_res0_tensor = phi::DenseTensor(argsort_res0_alloc, argsort_res0_meta);

    phi::DenseTensorMeta argsort_res1_meta(phi::DataType::INT32,
                                 phi::make_ddim({bsz, src_token_number_}));
    std::shared_ptr<phi::Allocation> argsort_res1_alloc(new phi::Allocation(
        static_cast<void *>(const_cast<int *>(argsort_re1_gpu_)),  // NOLINT
        bsz * src_token_number_ * sizeof(int),
        place))
    auto argsort_res0_tensor = phi::DenseTensor(argsort_res1_alloc, argsort_res1_meta);



    float *origin_tensor = reinterpret_cast<const float *>(inputs[0]);
    float *merged_tensor = reinterpret_cast<const float *>(outputs[0]);
    int *rand_select_arr = reinterpret_cast<const int *>(outputs[1]);
    int *whether_tobe_merge = reinterpret_cast<const int *>(outputs[2]);


   
  } else if (input_type == nvinfer1::DataType::kHALF) {
    VLOG(3) << "TRT Plugin DataType selected. TokenMerge-->fp16";




    
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "The TokenMerge TRT Plugin's input type should be float or half."));
  }
  return cudaGetLastError() != cudaSuccess;                                
  



}



}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
