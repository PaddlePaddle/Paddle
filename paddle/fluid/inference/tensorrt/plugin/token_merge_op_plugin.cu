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
  nvinfer1::DimsExprs outputDims;
  if (output_index == 1 || output_index == 2) {
      outputDims.nbDims = 2;
      outputDims.d[0] = inputs[0].d[0];
      outputDims.d[1] = expr_builder.constant(token_number_);
  } else if (output_index == 0) {
      outputDims.nbDims = 3;
      outputDims.d[0] = inputs[0].d[0];
      outputDims.d[1] = expr_builder.constant(final_token_number_);
      outputDims.d[2] = expr_builder.constant(hid_dim_);
  }
  return outputDims;
}



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
  VLOG(3) << "nb_inputs" << nb_inputs << " + " << "nb_outputs" << nb_outputs;
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
    return in.type == nvinfer1::DataType::kINT32 && in.format == nvinfer1::TensorFormat::kLINEAR;

  if (pos == 3) 
      return in.type == nvinfer1::DataType::kINT32 && in.format == nvinfer1::TensorFormat::kLINEAR;
  
}



void TokenMergePluginDynamic::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc *in,
    int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc *out,
    int nbOutputs) TRT_NOEXCEPT {}

nvinfer1::DataType TokenMergePluginDynamic::getOutputDataType(
    int index,
    const nvinfer1::DataType *input_types,
    int nb_inputs) const TRT_NOEXCEPT {
  PADDLE_ENFORCE_EQ(
      nb_inputs,
      1,
      platform::errors::InvalidArgument(
          "The token_merge Plugin only has one input, so the "
          "nb_inputs value should be 1, but get %d.",
          nb_inputs));
  VLOG(3) << "exec TokenMergePluginDynamic::getOutputDataType";
  PADDLE_ENFORCE_EQ((input_types[0] == nvinfer1::DataType::kFLOAT ||
                     input_types[0] == nvinfer1::DataType::kHALF),
                    true,
                    platform::errors::InvalidArgument(
                        "The input type should be half or float"));
  if (index == 0){
    return input_types[0];
  }
  return nvinfer1::DataType::kINT32;
}



int TokenMergePluginDynamic::enqueue(
    const nvinfer1::PluginTensorDesc *input_desc,
    const nvinfer1::PluginTensorDesc *output_desc,
    const void *const *inputs,
    void *const *outputs,
    void *workspace,
    cudaStream_t stream) TRT_NOEXCEPT {
  VLOG(3) << "exec TokenMergePluginDynamic::enqueue";
  const auto &input_dims = input_desc[0].dims;
  auto input_type = input_desc[0].type;
  int bsz = input_dims.d[0];
  int token_numebr = input_dims.d[1];
  int hid_dim = input_dims.d[2];

  int device_id;
  cudaGetDevice(&device_id);
  auto *device_ctx = static_cast<phi::GPUContext *>(
        platform::DeviceContextPool::Instance().Get(
            platform::CUDAPlace(device_id)));
  const phi::GPUContext &dev_ctx = *device_ctx;

   //divied_rank
  phi::DenseTensor divied_rank_tensor;
  divied_rank_tensor.Resize({bsz, dst_token_number_});
  dev_ctx.Alloc<int>(&divied_rank_tensor, sizeof(int) * dst_token_number_ * bsz);
  
  //max_similarity_idx
  phi::DenseTensor max_similarity_idx_tensor;
  max_similarity_idx_tensor.Resize({bsz, src_token_number_});
  dev_ctx.Alloc<int>(&max_similarity_idx_tensor, sizeof(int) * src_token_number_ * bsz);
  



  if (input_type == nvinfer1::DataType::kFLOAT) {
    VLOG(3) << "TRT Plugin DataType selected. TokenMerge-->fp32";
    //dst_token
    phi::DenseTensor dst_token_tensor;
    dst_token_tensor.Resize({bsz, dst_token_number_, hid_dim});
    dev_ctx.Alloc<float>(&dst_token_tensor, sizeof(float) * bsz * dst_token_number_ * hid_dim);

    //src_token
    phi::DenseTensor src_token_tensor;
    src_token_tensor.Resize({bsz, src_token_number_, hid_dim});
    dev_ctx.Alloc<float>(&src_token_tensor, sizeof(float) * bsz * src_token_number_ * hid_dim);
    
    //dst_L2
    phi::DenseTensor dst_L2_tensor;
    dst_L2_tensor.Resize({bsz, dst_token_number_, hid_dim});
    dev_ctx.Alloc<float>(&dst_L2_tensor, sizeof(float) * bsz * dst_token_number_ * hid_dim);

    //src_L2
    phi::DenseTensor src_L2_tensor;
    src_L2_tensor.Resize({bsz, src_token_number_, hid_dim});
    dev_ctx.Alloc<float>(&src_L2_tensor, sizeof(float) * bsz * src_token_number_ * hid_dim);
    
   
    
    //similarity
    phi::DenseTensor similarity_tensor;
    similarity_tensor.Resize({bsz, src_token_number_ * dst_token_number_});
    dev_ctx.Alloc<float>(&similarity_tensor, sizeof(float) * bsz * src_token_number_ * dst_token_number_);
    


    //max_similarity and tensors for argsort
    phi::DenseTensor max_similarity_tensor;
    max_similarity_tensor.Resize({bsz, src_token_number_});
    dev_ctx.Alloc<float>(&max_similarity_tensor, sizeof(float) * bsz * src_token_number_);

    phi::DenseTensor argsort_res0_tensor;
    argsort_res0_tensor.Resize({bsz, src_token_number_});
    dev_ctx.Alloc<float>(&argsort_res0_tensor, sizeof(float) * bsz * src_token_number_);

    phi::DenseTensor argsort_res1_tensor;
    argsort_res1_tensor.Resize({bsz, src_token_number_});
    dev_ctx.Alloc<int64_t>(&argsort_res1_tensor, sizeof(int64_t) * bsz * src_token_number_);


    const float *origin_tensor = reinterpret_cast<const float *>(inputs[0]);
    float *merged_tensor = static_cast<float *>(outputs[0]);
    int *rand_select_arr = static_cast<int *>(outputs[1]);
    int *whether_tobe_merge = static_cast<int *>(outputs[2]);
    tokenMerge<float> tome;
    VLOG(3) << "start calc";
    auto ret = tome(dev_ctx,
                      use_rand_,
                      bsz_,
                      token_number_,
                      src_token_number_,
                      dst_token_number_,
                      final_token_number_,
                      src_need_merged_number_,
                      hid_dim_,
                      height_,
                      width_,
                      origin_tensor,
                      src_token_tensor,
                      dst_token_tensor,
                      src_L2_tensor,
                      dst_L2_tensor,
                      similarity_tensor,
                      max_similarity_tensor,
                      max_similarity_idx_tensor,
                      argsort_res0_tensor,
                      argsort_res1_tensor,
                      divied_rank_tensor,
                      merged_tensor,
                      rand_select_arr,
                      whether_tobe_merge);
  return ret;

   
  } else if (input_type == nvinfer1::DataType::kHALF) {
    VLOG(3) << "TRT Plugin DataType selected. TokenMerge-->fp16";
    phi::DenseTensor dst_token_tensor;
    dst_token_tensor.Resize({bsz, dst_token_number_, hid_dim});
    dev_ctx.Alloc<paddle::platform::float16>(&dst_token_tensor, sizeof(half) * bsz * dst_token_number_ * hid_dim);

    //src_token
    phi::DenseTensor src_token_tensor;
    src_token_tensor.Resize({bsz, src_token_number_, hid_dim});
    dev_ctx.Alloc<paddle::platform::float16>(&src_token_tensor, sizeof(half) * bsz * src_token_number_ * hid_dim);
    
    //dst_L2
    phi::DenseTensor dst_L2_tensor;
    dst_L2_tensor.Resize({bsz, dst_token_number_, hid_dim});
    dev_ctx.Alloc<paddle::platform::float16>(&dst_L2_tensor, sizeof(half) * bsz * dst_token_number_ * hid_dim);

    //src_L2
    phi::DenseTensor src_L2_tensor;
    src_L2_tensor.Resize({bsz, src_token_number_, hid_dim});
    dev_ctx.Alloc<paddle::platform::float16>(&src_L2_tensor, sizeof(half) * bsz * src_token_number_ * hid_dim);
    
   
    
    //similarity
    phi::DenseTensor similarity_tensor;
    similarity_tensor.Resize({bsz, src_token_number_ * dst_token_number_});
    dev_ctx.Alloc<paddle::platform::float16>(&similarity_tensor, sizeof(half) * bsz * src_token_number_ * dst_token_number_);
    


    //max_similarity and tensors for argsort
    phi::DenseTensor max_similarity_tensor;
    max_similarity_tensor.Resize({bsz, src_token_number_});
    dev_ctx.Alloc<paddle::platform::float16>(&max_similarity_tensor, sizeof(half) * bsz * src_token_number_);

    phi::DenseTensor argsort_res0_tensor;
    // argsort_res0_tensor.Resize({bsz, src_token_number_});
    // dev_ctx.Alloc<float>(&argsort_res0_tensor, sizeof(float) * bsz * src_token_number_);

    phi::DenseTensor argsort_res1_tensor;
    // argsort_res1_tensor.Resize({bsz, src_token_number_});
    // dev_ctx.Alloc<int>(&argsort_res1_tensor, sizeof(int) * bsz * src_token_number_);


    const half *origin_tensor = reinterpret_cast<const half *>(inputs[0]);
    half *merged_tensor = reinterpret_cast<half *>(outputs[0]);
    int *rand_select_arr = reinterpret_cast<int *>(outputs[1]);
    int *whether_tobe_merge = reinterpret_cast<int *>(outputs[2]);

    tokenMerge<half> tome;
    return tome(dev_ctx,
                      use_rand_,
                      bsz,
                      token_number_,
                      src_token_number_,
                      dst_token_number_,
                      final_token_number_,
                      src_need_merged_number_,
                      hid_dim_,
                      height_,
                      width_,
                      origin_tensor,
                      src_token_tensor,
                      dst_token_tensor,
                      src_L2_tensor,
                      dst_L2_tensor,
                      similarity_tensor,
                      max_similarity_tensor,
                      max_similarity_idx_tensor,
                      argsort_res0_tensor,
                      argsort_res1_tensor,
                      divied_rank_tensor,
                      merged_tensor,
                      rand_select_arr,
                      whether_tobe_merge);
    return 0;



    
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
