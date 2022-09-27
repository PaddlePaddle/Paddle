// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
// Copyright (c) 2019-2022, NVIDIA CORPORATION.  All rights reserved.
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
#include <vector>
#include "paddle/fluid/inference/tensorrt/plugin/reverse_roll_op_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

/*******************  invokeReverseRoll  ***********************/
// src is [batch*window_num, window_len, dim]
// dst is [batch, H, W, dim] + rolled
// grid(W, H, batch)
// block(min(1024, dim))

template<typename T>
__global__ void reverse_roll(T*        dst,
                             const T*  src,
                             const int batch,
                             const int window_num,
                             const int window_len,
                             const int window_size,
                             const int H,
                             const int W,
                             const int shift_size,
                             const int dim)
{
    const int batch_idx     = blockIdx.z;
    const int H_idx_shifted = (blockIdx.y + shift_size) % H;
    const int W_idx_shifted = (blockIdx.x + shift_size) % W;
    const int H_idx         = blockIdx.y;
    const int W_idx         = blockIdx.x;
    const int window_idx    = H_idx / window_size * (W / window_size) + W_idx / window_size;
    const int idx_in_window = (H_idx % window_size) * window_size + (W_idx % window_size);
    const int input_offset  = (batch_idx * window_num + window_idx) * window_len + idx_in_window;
    const int output_offset = (batch_idx * H + H_idx_shifted) * W + W_idx_shifted;
    __shared__ T shift_array[1025];
    for (int tid = threadIdx.x; tid < dim; tid += blockDim.x) {
        shift_array[tid] = src[input_offset * dim + tid];
    }
    __syncthreads();
    for (int tid = threadIdx.x; tid < dim; tid += blockDim.x) {
        dst[output_offset * dim + tid] = shift_array[tid];
    }

    // for (int tid = threadIdx.x; tid < dim; tid += blockDim.x) {
    //     dst[output_offset * dim + tid] = src[input_offset * dim + tid];
    // }
}

// src is [batch*window_num, window_len, dim]
// dst is [batch, H, W, dim] + rolled
// grid(W, H, batch)
// block(min(1024, dim))
template<typename T>
void invokeReverseRoll(T*           dst,
                       const T*     src,
                       int          batch,
                       int          window_num,
                       int          window_len,
                       int          window_size,
                       int          H,
                       int          W,
                       int          dim,
                       int          shift_size,
                       cudaStream_t stream)
{
    dim3 grid(W, H, batch);
    int  blockSize = dim;
    if (std::is_same<T, half>::value && (dim % 2 == 0)) {
        blockSize = dim / 2;
        if (blockSize > 1024) {
            blockSize = 1024;
        }
        using T2 = half2;
        reverse_roll<<<grid, blockSize, 0, stream>>>(
            (T2*)dst, (const T2*)src, batch, window_num, window_len, window_size, H, W, shift_size, dim / 2);
    }
    else {
        if (blockSize > 1024) {
            blockSize = 1024;
        }
        reverse_roll<<<grid, blockSize, 0, stream>>>(
            dst, src, batch, window_num, window_len, window_size, H, W, shift_size, dim);
    }
}

template void invokeReverseRoll(float*       dst,
                                const float* src,
                                int          batch,
                                int          window_num,
                                int          window_len,
                                int          window_size,
                                int          H,
                                int          W,
                                int          dim,
                                int          shift_size,
                                cudaStream_t stream);

template void invokeReverseRoll(half*        dst,
                                const half*  src,
                                int          batch,
                                int          window_num,
                                int          window_len,
                                int          window_size,
                                int          H,
                                int          W,
                                int          dim,
                                int          shift_size,
                                cudaStream_t stream);

void ReverseRollPluginDynamic::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc *in,
    int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc *out,
    int nbOutputs) TRT_NOEXCEPT {}
bool ReverseRollPluginDynamic::supportsFormatCombination(
    int pos,
    const nvinfer1::PluginTensorDesc *in_out,
    int nb_inputs,
    int nb_outputs) TRT_NOEXCEPT {
  PADDLE_ENFORCE_NOT_NULL(
      in_out,
      platform::errors::InvalidArgument("The input of ReverseRoll "
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
      return in.type == nvinfer1::DataType::kHALF &&
             in.format == nvinfer1::TensorFormat::kLINEAR;
    } else {
      return in.type == nvinfer1::DataType::kFLOAT &&
             in.format == nvinfer1::TensorFormat::kLINEAR;
    }
  }
  const nvinfer1::PluginTensorDesc &prev = in_out[pos - 1];
  // output
  return in.type == prev.type && in.format == prev.format;
}

nvinfer1::DataType ReverseRollPluginDynamic::getOutputDataType(
    int index,
    const nvinfer1::DataType *input_types,
    int nb_inputs) const TRT_NOEXCEPT {
  PADDLE_ENFORCE_EQ(
      index,
      0,
      platform::errors::InvalidArgument(
          "The ReverseRoll only has one input, so the "
          "index value should be 0, but get %d.",
          index));
  return input_types[0];
}

nvinfer1::DimsExprs ReverseRollPluginDynamic::getOutputDimensions(
    int output_index,
    const nvinfer1::DimsExprs *inputs,
    int nb_inputs,
    nvinfer1::IExprBuilder &expr_builder) TRT_NOEXCEPT {
  PADDLE_ENFORCE_EQ(
      output_index,
      0,
      platform::errors::InvalidArgument(
          "There is only one output of the ReverseRoll, "
          "so the index should be zero,"
          "but it's (%d)",
          output_index));
  PADDLE_ENFORCE_EQ(
      nb_inputs,
      1,
      platform::errors::InvalidArgument(
          "The Input of the ReverseRoll should be 1, but we found "
          "it has (%d) inputs",
          nb_inputs));

  nvinfer1::DimsExprs ret;
  ret.nbDims = 3;
  ret.d[0] = expr_builder.operation(
      nvinfer1::DimensionOperation::kFLOOR_DIV,
      *inputs[0].d[0],
      *expr_builder.constant(window_num_));
  ret.d[1] = expr_builder.operation(
                nvinfer1::DimensionOperation::kPROD,
                *inputs[0].d[1],
                *expr_builder.constant(window_num_));
  ret.d[2] = inputs[0].d[2];
  return ret;
}
int ReverseRollPluginDynamic::enqueue(
    const nvinfer1::PluginTensorDesc *input_desc,
    const nvinfer1::PluginTensorDesc *output_desc,
    const void *const *inputs,
    void *const *outputs,
    void *workspace,
    cudaStream_t stream) TRT_NOEXCEPT {
  const auto &input_dims = input_desc[0].dims;
  auto input_type = input_desc[0].type;
  int batch = input_dims.d[0]/window_num_;
  int dim = input_dims.d[2];
  if (input_type == nvinfer1::DataType::kFLOAT) {
    VLOG(3) << "TRT Plugin DataType selected. ReverseRoll-->fp32";
    invokeReverseRoll(
        reinterpret_cast<float *>(outputs[0]),
        reinterpret_cast<const float *>(inputs[0]),
        batch,
        window_num_,
        window_len_,
        window_size_,
        input_resolution_,
        input_resolution_,
        dim,
        shift_size_,
        stream);
  } else if (input_type == nvinfer1::DataType::kHALF) {
    VLOG(3) << "TRT Plugin DataType selected. ReverseRoll-->fp16";
    invokeReverseRoll(
        reinterpret_cast<half *>(outputs[0]),
        reinterpret_cast<const half *>(inputs[0]),
        batch,
        window_num_,
        window_len_,
        window_size_,
        input_resolution_,
        input_resolution_,
        dim,
        shift_size_,
        stream);
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "The ReverseRoll TRT Plugin's input type should be float or half."));
  }
  return cudaGetLastError() != cudaSuccess;

}

} // plugin
} // tensorrt
} // inference
} // paddle
