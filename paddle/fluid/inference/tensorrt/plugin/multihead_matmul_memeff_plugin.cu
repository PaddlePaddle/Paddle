// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
// Copyright (c) 2020-2021, NVIDIA CORPORATION. All Rights Reserved.
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
#include "multihead_matmul_memeff_plugin.h"


namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

// Dynamic Plugin below.
#if IS_TRT_VERSION_GE(6000)
int QkvToContextPluginDynamic::initialize() TRT_NOEXCEPT { return 0; }

nvinfer1::DimsExprs MultiheadMatmulMemEffPluginDynamic::getOutputDimensions(
    int output_index,
    const nvinfer1::DimsExprs *inputs,
    int nb_inputs,
    nvinfer1::IExprBuilder &expr_builder) TRT_NOEXCEPT {
  // input[0], (B, S, 3 * N * H, 1, 1)
  // if has_biasqk_mask_
  // output, (B, seq_len, hidden)

  PADDLE_ENFORCE_EQ(
      nb_inputs,
      1,
      platform::errors::InvalidArgument(
          "The Input of the qkv_to_context_plugin should be 1, but we found "
          "it has (%d) inputs",
          nb_inputs));
  
  nvinfer1::DimsExprs ret;
  ret.nbDims = 3;
  ret.d[0] = inputs[0].d[0];
  ret.d[1] = inputs[0].d[1];
  ret.d[2] = expr_builder.constant(head_size_ * head_number_);
  return ret;
}
void MultiheadMatmulMemEffPluginDynamic::allocateSeqlens(int32_t max_batchsize){
  if(!cu_seqlen_ && max_batchsize){
    cudaMalloc(&cu_seqlen_, sizeof(int32)*(max_batchsize+1));
  }
  max_batchsize_=max_batchsize;
}
void MultiheadMatmulMemEffPluginDynamic::initializeSeqlens(
  int32_t b, int32_t s, void* cu_seqlens_gpu, cudaStream_t stream)
{
    if (!b || !s)
    {
        return;
    }

    std::vector<int32_t> cu_seqLens(b + 1, 0);
    // Compute the prefix sum of the seqlen
    for (int32_t it = 0; it < b; it++)
    {
        cu_seqLens[it + 1] = cu_seqLens[it] + s;
    }

    cudaMemcpyAsync(
        cu_seqlens_gpu, cu_seqLens.data(), sizeof(int32_t) * cu_seqLens.size(), cudaMemcpyHostToDevice, stream));
    opt_batchsize_ = b;
    opt_seqlen_ = s;
}

void MultiheadMatmulMemEffPluginDynamic::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc *in,
    int nb_inputs,
    const nvinfer1::DynamicPluginTensorDesc *out,
    int nb_outputs) TRT_NOEXCEPT {
  int32_t const batchsize = in[0].max.d[0];
  int32_t const seqlen = in[0].max.d[1];
  allocateSeqlens(batchSize)
  if(batchsize!=opt_batchsize_ || seqlen!=opt_seqlen_){
    initializeSeqlens(batchsize,seqlen,cu_seqlen_);
  }
}

bool MultiheadMatmulMemEffPluginDynamic::supportsFormatCombination(
    int pos,
    const nvinfer1::PluginTensorDesc *in_out,
    int nb_inputs,
    int nb_outputs) TRT_NOEXCEPT {
  PADDLE_ENFORCE_NOT_NULL(
      in_out,
      platform::errors::InvalidArgument(
          "The input of swish plugin shoule not be nullptr."));

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
#ifdef TRT_PLUGIN_FP16_AVALIABLE
      return (in.type == nvinfer1::DataType::kHALF) &&
             (in.format == nvinfer1::TensorFormat::kLINEAR);
#else
      return (in.type == nvinfer1::DataType::kFLOAT) &&
             (in.format == nvinfer1::TensorFormat::kLINEAR);
#endif
    } else {
      return (in.type == nvinfer1::DataType::kFLOAT) &&
             (in.format == nvinfer1::TensorFormat::kLINEAR);
    }
  }

  const nvinfer1::PluginTensorDesc &prev = in_out[pos - 1];

  if (pos == 1) {
    return in.type == prev.type && in.format == prev.format;
  }

  // output
  return in.type == prev.type && in.format == prev.format;
}

nvinfer1::DataType MultiheadMatmulMemEffPluginDynamic::getOutputDataType(
    int index,
    const nvinfer1::DataType *input_types,
    int nb_inputs) const TRT_NOEXCEPT {
  PADDLE_ENFORCE_EQ(
      index,
      0,
      platform::errors::InvalidArgument(
          "The multihead_matmul_memeff Plugin only has one input, so the "
          "index value should be 0, but get %d.",
          index));
  return input_types[0];
}

int MultiheadMatmulMemEffPluginDynamic::enqueue(
    const nvinfer1::PluginTensorDesc *input_desc,
    const nvinfer1::PluginTensorDesc *output_desc,
    const void *const *inputs,
    void *const *outputs,
    void *workspace,
    cudaStream_t stream) TRT_NOEXCEPT {
  auto input_type = input_desc[0].type;
  auto input_dims = input_desc[0].dims;
  int input_num = ProductDim(input_dims);

  // input[0], (B, S, 3 * N * H, 1, 1)
  int batch = input_dims.d[0];
  int seq_len = input_dims.d[1];
  int device_id=0;
  cudaGetDevice(&device_id);

  if (input_type == nvinfer1::DataType::kFLOAT) {

  }else if (input_type == nvinfer1::DataType::kHALF){
    const int sm = platform::GetGPUComputeCapability(device_id);
    auto *device_ctx = static_cast<phi::GPUContext *>(
      platform::DeviceContextPool::Instance().Get(
        platform::CUDAPlace(device_id)));
    const phi::GPUContext &dev_ctx = *device_ctx;

    const float16 *input0_data =
        static_cast<const float16 *>(inputs[0]);  // qkv

  }
}


#endif

}
}
}
}