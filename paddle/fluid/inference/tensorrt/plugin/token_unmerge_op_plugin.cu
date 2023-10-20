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

#include "paddle/fluid/inference/tensorrt/plugin/token_unmerge_op_plugin.h"
#include "paddle/fluid/platform/float16.h"


namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

template<typename T>
__global__ void token_unmerge_kernel(const T *merged_token,
                                    const int *rand_select_arr,
                                    const int *whether_tobe_merge,
                                    int token_number,
                                    int src_token_num,
                                    int dst_token_num,
                                    int final_token_num,
                                    int width,
                                    int hid_dim,
                                    T *unmerged_token){
  int bid = gridDim.x * blockIdx.y + blockIdx.x;
  int block_num_pre_row = width / 16;
  int bsz_begin = blockIdx.z * token_number * hid_dim;
  int block_begin_index = bsz_begin;
  if(bid % block_num_pre_row ==0)
    block_begin_index = block_begin_index + (bid / block_num_pre_row) * 2 * width * hid_dim;
  else
    block_begin_index = block_begin_index + ((bid / block_num_pre_row) * 2 * width + (bid % block_num_pre_row) * 16) * hid_dim;
  int tid = blockDim.x * threadIdx.y + threadIdx.x; 
  int wrap_id_in_block = tid / WARP_SIZE; 
  int thread_id_in_wrap = threadIdx.x; 
  int gloable_wrap_id = bid * 8 + wrap_id_in_block; // for write out_put 
  int rand_select_begin = blockIdx.z * token_number + gloable_wrap_id * 4;
  int megred_tensor_batch_begin = blockIdx.z * final_token_num * hid_dim;
  //loop0
  int token0_begin = block_begin_index + wrap_id_in_block * 2 * hid_dim;
  int data_index0 = token0_begin + thread_id_in_wrap;


  for(int cnt = 0, idx = data_index0; idx < token0_begin + hid_dim; idx += WARP_SIZE, cnt++){
    if(rand_select_arr[rand_select_begin] == 0){ //是被选中的dst_token 
      unmerged_token[idx] = merged_token[megred_tensor_batch_begin + gloable_wrap_id * hid_dim + threadIdx.x + cnt * WARP_SIZE];
    }else{
      //src_id_return 在merge-split的时候 写到的src里的位置
      int src_id_return = src_token_num * blockIdx.z + gloable_wrap_id * 3 + rand_select_arr[rand_select_begin] / 2;
      //src_id_real src_token完成token_merge后 whether_to_be_merge记录的是他在unmerge的时候应该取的数据
      int src_id_real = whether_tobe_merge[src_id_return];
      unmerged_token[idx] = merged_token[megred_tensor_batch_begin + src_id_real * hid_dim + threadIdx.x + cnt * WARP_SIZE];

    }
  } 
  //loop1
  int token1_begin = token0_begin + hid_dim;
  int data_index1 = data_index0 + hid_dim; 
  for(int cnt = 0, idx = data_index1; idx < token1_begin + hid_dim; idx += WARP_SIZE, cnt++){
   if(rand_select_arr[rand_select_begin + 1] == 0){ //被选中的dst_token 
      unmerged_token[idx] = merged_token[megred_tensor_batch_begin + gloable_wrap_id * hid_dim + threadIdx.x + cnt * WARP_SIZE];
    }else{
      //src_id_return 在merge-split的时候 写到的src里的位置
      int src_id_return = src_token_num * blockIdx.z + gloable_wrap_id * 3 + rand_select_arr[rand_select_begin + 1] / 2;
      //src_id_real src_token完成token_merge后 whether_to_be_merge记录的是他在unmerge的时候应该取的数据
      int src_id_real = whether_tobe_merge[src_id_return];
      unmerged_token[idx] = merged_token[megred_tensor_batch_begin + src_id_real * hid_dim + threadIdx.x + cnt * WARP_SIZE];
    }
  }

  //loop2
  int token2_begin = token0_begin + hid_dim * width;
  int data_index2 =  data_index0 + hid_dim * width;
  for(int cnt = 0, idx = data_index2; idx < token2_begin + hid_dim; idx += WARP_SIZE, cnt++){
    if(rand_select_arr[rand_select_begin + 2] == 0){ //被选中的dst_token 
      unmerged_token[idx] = merged_token[megred_tensor_batch_begin + gloable_wrap_id * hid_dim + threadIdx.x + cnt * WARP_SIZE];
    }else{
      //src_id_return 在merge-split的时候 写到的src里的位置
      int src_id_return = src_token_num * blockIdx.z + gloable_wrap_id * 3 + rand_select_arr[rand_select_begin + 2] / 2;
      //src_id_real src_token完成token_merge后 whether_to_be_merge记录的是他在unmerge的时候应该取的数据
      int src_id_real = whether_tobe_merge[src_id_return];
      unmerged_token[idx] = merged_token[megred_tensor_batch_begin + src_id_real * hid_dim + threadIdx.x + cnt * WARP_SIZE];

    }
  }

  //loop3
  int token3_begin = token2_begin + hid_dim;
  int data_index3 = data_index2 + hid_dim;
  for(int cnt = 0, idx = data_index3; idx < token3_begin + hid_dim; idx += WARP_SIZE, cnt++){
    if(rand_select_arr[rand_select_begin + 3] == 0){ //被选中的dst_token 
      unmerged_token[idx] = merged_token[megred_tensor_batch_begin + gloable_wrap_id * hid_dim + threadIdx.x + cnt * WARP_SIZE];
    }else{
      //src_id_return 在merge-split的时候 写到的src里的位置
      int src_id_return = src_token_num * blockIdx.z + gloable_wrap_id * 3 + rand_select_arr[rand_select_begin + 3] / 2;
      //src_id_real src_token完成token_merge后 whether_to_be_merge记录的是他在unmerge的时候应该取的数据
      int src_id_real = whether_tobe_merge[src_id_return];
      unmerged_token[idx] = merged_token[megred_tensor_batch_begin + src_id_real * hid_dim + threadIdx.x + cnt * WARP_SIZE];

    }
  }
}



nvinfer1::DimsExprs TokenUnmergePluginDynamic::getOutputDimensions(
  int output_index,
  const nvinfer1::DimsExprs* inputs,
  int nb_inputs,
  nvinfer1::IExprBuilder& expr_builder) TRT_NOEXCEPT {
  PADDLE_ENFORCE_EQ(nb_inputs,
                   3,
                   platform::errors::InvalidArgument(
                       "The TokenUnmerge plugin should be three input."));
  PADDLE_ENFORCE_LT(output_index,
                   1,
                   platform::errors::InvalidArgument(
                      "When GetOutputDimensions, the index(%d) should not "
                      "greater the num(%d) of the outpus.",
                      output_index,1));
  nvinfer1::DimsExprs outputDims;
  outputDims.d[0] = inputs[0].d[0];
  outputDims.d[1] = expr_builder.constant(token_number_);
  outputDims.d[2] = expr_builder.constant(hid_dim_);
  return outputDims;
}



bool TokenUnmergePluginDynamic::supportsFormatCombination(
    int pos,
    const nvinfer1::PluginTensorDesc *in_out,
    int nb_inputs,
    int nb_outputs) TRT_NOEXCEPT {
  PADDLE_ENFORCE_NOT_NULL(
      in_out,
      platform::errors::InvalidArgument("The input of TokenUnmerge "
                                        "plugin shoule not be nullptr."));
  PADDLE_ENFORCE_LT(
      pos,
      nb_inputs + nb_outputs,
      platform::errors::InvalidArgument("The pos(%d) should be less than the "
                                        "num(%d) of the input and the output.",
                                        pos,
                                        nb_inputs + nb_outputs));
  const nvinfer1::PluginTensorDesc &in = in_out[pos];
 if (pos == 0 || pos == 3) {
    if (with_fp16_) {
        return in.type == nvinfer1::DataType::kHALF && in.format == nvinfer1::PluginFormat::kLINEAR;
    } else {
      return in.type == nvinfer1::DataType::kFLOAT && in.format == nvinfer1::TensorFormat::kLINEAR;
    }
  }
  if (pos == 1 || pos ==2) {
      return in.type == nvinfer1::DataType::kINT32 && in.format == nvinfer1::TensorFormat::kLINEAR;
  }
}



void TokenUnmergePluginDynamic::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc *in,
    int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc *out,
    int nbOutputs) TRT_NOEXCEPT {}

nvinfer1::DataType TokenUnmergePluginDynamic::getOutputDataType(
    int index,
    const nvinfer1::DataType *input_types,
    int nb_inputs) const TRT_NOEXCEPT {
  PADDLE_ENFORCE_EQ(
      nb_inputs,
      3,
      platform::errors::InvalidArgument(
          "The token_unmerge Plugin has thress input, so the "
          "nb_inputs value should be 3, but get %d.",
          nb_inputs));
  PADDLE_ENFORCE_EQ((input_types[0] == nvinfer1::DataType::kFLOAT ||
                     input_types[0] == nvinfer1::DataType::kHALF),
                    true,
                    platform::errors::InvalidArgument(
                        "The input type should be half or float"));
  return input_types[0];
}



int TokenUnmergePluginDynamic::enqueue(
    const nvinfer1::PluginTensorDesc *input_desc,
    const nvinfer1::PluginTensorDesc *output_desc,
    const void *const *inputs,
    void *const *outputs,
    void *workspace,
    cudaStream_t stream) TRT_NOEXCEPT {
  int device_id;
  cudaGetDevice(&device_id);
  auto *device_ctx = static_cast<phi::GPUContext *>(
        platform::DeviceContextPool::Instance().Get(
            platform::CUDAPlace(device_id)));
  const phi::GPUContext &dev_ctx = *device_ctx;
  auto input_type = input_desc[0].type;
  if (input_type == nvinfer1::DataType::kFLOAT) {
    VLOG(3) << "TRT Plugin DataType selected. TokenMerge-->fp32";
    const float *merged_token = reinterpret_cast<const float *>(inputs[0]);
    const int *rand_select_tensor = reinterpret_cast<const int *>(inputs[1]);
    const int *whether_tobe_merge_tensor = reinterpret_cast<const int *>(inputs[2]);
    float *unmerged_tensor = reinterpret_cast<float *>(outputs[0]);
    dim3 grid_dim_for_unmerge(height_ / 16, width_ / 2, bsz_);
    dim3 block_dim_for_unmerge(32, 8);
    token_unmerge_kernel<float><<<grid_dim_for_unmerge, block_dim_for_unmerge>>>(
                          merged_token,
                          rand_select_tensor,
                          whether_tobe_merge_tensor,
                          token_number_,
                          src_token_number_,
                          dst_token_number_,
                          final_token_number_,
                          width_,
                          hid_dim_,
                          unmerged_tensor);

  } else if (input_type == nvinfer1::DataType::kHALF) {
    VLOG(3) << "TRT Plugin DataType selected. TokenUnmerge-->fp16";
    const half *merged_token = reinterpret_cast<const half *>(inputs[0]);
    const int *rand_select_tensor = reinterpret_cast<const int *>(inputs[1]);
    const int *whether_tobe_merge_tensor = reinterpret_cast<const int *>(inputs[2]);
    half *unmerged_tensor = reinterpret_cast<half *>(outputs[0]);
    dim3 grid_dim_for_unmerge(height_ / 16, width_ / 2, bsz_);
    dim3 block_dim_for_unmerge(32, 8);
    token_unmerge_kernel<half><<<grid_dim_for_unmerge, block_dim_for_unmerge>>>(
                          merged_token,
                          rand_select_tensor,
                          whether_tobe_merge_tensor,
                          token_number_,
                          src_token_number_,
                          dst_token_number_,
                          final_token_number_,
                          width_,
                          hid_dim_,
                          unmerged_tensor);
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "The TokenUmerge TRT Plugin's input type should be float or half."));
  }
  return cudaGetLastError() != cudaSuccess;
}



}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
