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

#include <stdio.h>

#include <cassert>
#include <cub/cub.cuh>  // NOLINT
#include <vector>

#include "glog/logging.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/inference/tensorrt/plugin/qkv_to_context_plugin.h"
#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin_utils.h"
#include "paddle/fluid/operators/math/bert_encoder_functor.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

// Dynamic Plugin below.
#if IS_TRT_VERSION_GE(6000)

template <typename T>
__global__ void transpose(T *src,
                          T *dst,
                          const int batch_size,
                          const int seq_len,
                          const int head_num,
                          const int size_per_head) {
  int batch_id = blockIdx.x / (head_num * seq_len);
  int seq_id = blockIdx.x % seq_len;
  int head_id = (blockIdx.x % (head_num * seq_len)) / seq_len;
  dst[batch_id * (head_num * seq_len * size_per_head) +
      seq_id * head_num * size_per_head + head_id * size_per_head +
      threadIdx.x] = src[blockIdx.x * size_per_head + threadIdx.x];
}

template <typename T>
__global__ void TransposeQkvKernel(const int H, const T *input, T *output) {
  // Input: BxSx3xNxH
  // Bias: 3xSxB
  // Output: 3xBxNxSxH
  int n = threadIdx.y;
  int s = blockIdx.x;
  int b = blockIdx.y;
  int m = blockIdx.z;

  const int N = blockDim.y;
  const int S = gridDim.x;
  const int B = gridDim.y;

  const int NH = N * H;
  const int NHS = NH * S;
  const int in_offset = n * H + m * NH + s * 3 * NH + b * NHS * 3;
  const int out_offset = s * H + n * S * H + b * NHS + m * NHS * B;

  const int i = threadIdx.x;
  output[out_offset + i] = input[in_offset + i];
}

inline void TransposeQKV(const int batch,
                         const int seq_len,
                         const int head_size,
                         const int head_num,
                         const float *input,
                         float *output,
                         cudaStream_t stream) {
  int scratch_size = batch * head_num * seq_len * seq_len;
  const dim3 grid(seq_len, batch, 3);
  if (head_size % 4 == 0 && scratch_size % 4 == 0) {
    const int h = head_size / 4;
    const float4 *input4 = reinterpret_cast<const float4 *>(input);
    float4 *output4 = reinterpret_cast<float4 *>(output);
    const dim3 block(h, head_num, 1);
    // limit h * head_num to max block size(1024).
    PADDLE_ENFORCE_LE(h * head_num,
                      1024,
                      platform::errors::InvalidArgument(
                          "head_num (%d) * head_size (%d) should <= %d",
                          head_num,
                          head_size,
                          1024 * 4));
    TransposeQkvKernel<float4><<<grid, block, 0, stream>>>(h, input4, output4);
  } else if (head_size % 2 == 0 && scratch_size % 2 == 0) {
    const int h = head_size / 2;
    const float2 *input2 = reinterpret_cast<const float2 *>(input);
    float2 *output2 = reinterpret_cast<float2 *>(output);
    const dim3 block(h, head_num, 1);
    // limit h * head_num to max block size(1024).
    PADDLE_ENFORCE_LE(h * head_num,
                      1024,
                      platform::errors::InvalidArgument(
                          "head_num (%d) * head_size (%d) should <= %d",
                          head_num,
                          head_size,
                          1024 * 2));
    TransposeQkvKernel<float2><<<grid, block, 0, stream>>>(h, input2, output2);
  } else {
    const dim3 block(head_size, head_num, 1);
    // limit head_size * head_num to max block size(1024).
    PADDLE_ENFORCE_LE(head_size * head_num,
                      1024,
                      platform::errors::InvalidArgument(
                          "head_num (%d) * head_size (%d) should <= %d",
                          head_num,
                          head_size,
                          1024));
    TransposeQkvKernel<float>
        <<<grid, block, 0, stream>>>(head_size, input, output);
  }
}

inline void TransposeQKV(const int batch,
                         const int seq_len,
                         const int head_size,
                         const int head_num,
                         const half *input,
                         half *output,
                         cudaStream_t stream) {
  int scratch_size = batch * head_num * seq_len * seq_len;
  const dim3 grid(seq_len, batch, 3);
  if (head_size % 8 == 0 && scratch_size % 8 == 0) {
    int h = head_size / 8;
    const int4 *input4 = reinterpret_cast<const int4 *>(input);
    int4 *output4 = reinterpret_cast<int4 *>(output);
    dim3 block(h, head_num, 1);
    // limit h * head_num to max block size(1024).
    PADDLE_ENFORCE_LE(h * head_num,
                      1024,
                      platform::errors::InvalidArgument(
                          "head_num (%d) * head_size (%d) should <= %d",
                          head_num,
                          head_size,
                          1024 * 8));
    TransposeQkvKernel<int4><<<grid, block, 0, stream>>>(h, input4, output4);
  } else if (head_size % 2 == 0 && scratch_size % 2 == 0) {
    const int h = head_size / 2;
    const half2 *input2 = reinterpret_cast<const half2 *>(input);
    half2 *output2 = reinterpret_cast<half2 *>(output);
    const dim3 block(h, head_num, 1);
    // limit h * head_num to max block size(1024).
    PADDLE_ENFORCE_LE(h * head_num,
                      1024,
                      platform::errors::InvalidArgument(
                          "head_num (%d) * head_size (%d) should <= %d",
                          head_num,
                          head_size,
                          1024 * 2));
    TransposeQkvKernel<half2><<<grid, block, 0, stream>>>(h, input2, output2);
  } else {
    const dim3 block(head_size, head_num, 1);
    // limit head_size * head_num to max block size(1024).
    PADDLE_ENFORCE_LE(head_size * head_num,
                      1024,
                      platform::errors::InvalidArgument(
                          "head_num (%d) * head_size (%d) should <= %d",
                          head_num,
                          head_size,
                          1024));
    TransposeQkvKernel<half>
        <<<grid, block, 0, stream>>>(head_size, input, output);
  }
}

int QkvToContextPluginDynamic::initialize() TRT_NOEXCEPT { return 0; }

nvinfer1::DimsExprs QkvToContextPluginDynamic::getOutputDimensions(
    int output_index,
    const nvinfer1::DimsExprs *inputs,
    int nb_inputs,
    nvinfer1::IExprBuilder &expr_builder) TRT_NOEXCEPT {
  // input[0], (B, S, 3 * N * H, 1, 1)
  // input[1], (B, head_num, seq_len, seq_len)
  // output, (B, seq_len, hidden)
  PADDLE_ENFORCE_EQ(output_index,
                    0,
                    platform::errors::InvalidArgument(
                        "There is only one output of the EmbEltwiseLayernorm, "
                        "so the index should be zero,"
                        "but it's (%d)",
                        output_index));
  PADDLE_ENFORCE_EQ(
      nb_inputs,
      2,
      platform::errors::InvalidArgument(
          "The Input of the EmbEltwiseLayernorm should be 3, but we found "
          "it has (%d) inputs",
          nb_inputs));
  nvinfer1::DimsExprs ret;
  ret.nbDims = 3;
  ret.d[0] = inputs[0].d[0];
  ret.d[1] = inputs[0].d[1];
  ret.d[2] = expr_builder.constant(head_size_ * head_number_);
  return ret;
}

bool QkvToContextPluginDynamic::supportsFormatCombination(
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
      return (in.type == nvinfer1::DataType::kFLOAT ||
              in.type == nvinfer1::DataType::kHALF) &&
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

nvinfer1::DataType QkvToContextPluginDynamic::getOutputDataType(
    int index,
    const nvinfer1::DataType *input_types,
    int nb_inputs) const TRT_NOEXCEPT {
  PADDLE_ENFORCE_EQ(
      index,
      0,
      platform::errors::InvalidArgument(
          "The EmbEltwiseLayernorm Plugin only has one input, so the "
          "index value should be 0, but get %d.",
          index));
  return input_types[0];
}

template <typename T>
__global__ void apply_scale(T *data, T scale, int n) {
#if CUDA_ARCH_FP16_SUPPORTED(__CUDA_ARCH__)
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) {
    data[tid] = data[tid] * scale;
  }
#endif
}

inline int round_up(int seq_len, int multiple = 32) {
  PADDLE_ENFORCE_GT(
      multiple,
      0,
      platform::errors::InvalidArgument(
          "multiple should be a positive number，but it's (%d)", multiple));
  return ((seq_len + multiple - 1) / multiple) * multiple;
}

template <typename T>
__global__ void broadcast(const T *src,
                          T *dst,
                          const int seq_len,
                          const int head_num) {
  int batch_id = blockIdx.x / (head_num * seq_len);
  int dst_offset = blockIdx.x * seq_len;
  if (threadIdx.x < seq_len) {
    dst[threadIdx.x + dst_offset] = src[threadIdx.x + batch_id * seq_len];
  }
}

template <typename T>
__global__ void reset_qk_bias(T *input, int real_seq_len, int seq_len) {
  if (threadIdx.x < seq_len) {
    int id = threadIdx.x + blockIdx.x * seq_len;
    input[id] = threadIdx.x >= real_seq_len ? (T)-1e20f : (T)0.0f;
  }
}

template <typename T>
__global__ void rebuild_sequence_length_padding(const T *src,
                                                T *dst,
                                                const int *padding_offset,
                                                const int n) {
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int dst_seq_id = bid + padding_offset[bid];
  const int src_seq_id = bid;

  for (int i = tid; i < n; i += blockDim.x) {
    dst[dst_seq_id * n + i] = src[src_seq_id * n + i];
  }
}

__global__ void set_padding_offset(int *padding_offset,
                                   int real_seq,
                                   const int batch_size,
                                   const int vir_seq) {
  // do cumulated sum
  int cum_offset = 0;
  int index = 0;
  for (int i = 0; i < batch_size; i++) {
    for (int j = 0; j < real_seq; j++) {
      padding_offset[index] = cum_offset;
      index++;
    }
    cum_offset += vir_seq - real_seq;
  }
}

int QkvToContextPluginDynamic::enqueue(
    const nvinfer1::PluginTensorDesc *input_desc,
    const nvinfer1::PluginTensorDesc *output_desc,
    const void *const *inputs,
    void *const *outputs,
    void *workspace,
    cudaStream_t stream) TRT_NOEXCEPT {
  auto input_dims = input_desc[0].dims;
  int input_num = ProductDim(input_dims);
  // input[0], (B, S, 3 * N * H, 1, 1)
  int batch = input_dims.d[0];
  int seq_len = input_dims.d[1];
  int real_seq_len = seq_len;
  if (input_desc[0].type == nvinfer1::DataType::kHALF) {
    seq_len = (seq_len + 7) / 8 * 8;
    input_num = batch * seq_len * 3 * head_number_ * head_size_;
  }
  framework::Tensor multihead_temp_tensor;
  int scratch_size = batch * head_number_ * seq_len * seq_len * 1;

  int device_id;
  cudaGetDevice(&device_id);
  multihead_temp_tensor.Resize({scratch_size + input_num});

  auto input_type = input_desc[0].type;
  if (input_type == nvinfer1::DataType::kFLOAT) {
    VLOG(1) << "TRT Plugin DataType selected. QkvToContext-->fp32";
    auto *multihead_temp_data = multihead_temp_tensor.mutable_data<float>(
        platform::CUDAPlace(device_id));
    auto *qkptr = multihead_temp_data;
    auto *tptr = multihead_temp_data + scratch_size;

    const float *input0_data = static_cast<const float *>(inputs[0]);
    // fit to [batch, head_num, length, length] + [batch, 1, 1, length]
    framework::Tensor temp_qk_bias_tensor;
    float *qk_bias = const_cast<float *>(static_cast<const float *>(inputs[1]));
    if (ProductDim(input_desc[1].dims) == (batch * seq_len)) {
      temp_qk_bias_tensor.Resize({batch, head_number_, seq_len, seq_len});
      auto *temp_qk_bias = temp_qk_bias_tensor.mutable_data<float>(
          platform::CUDAPlace(device_id));
      int grid = batch * head_number_ * seq_len;
      int block = round_up(seq_len);
      broadcast<<<grid, block, 0, stream>>>(
          static_cast<const float *>(inputs[1]),
          temp_qk_bias,
          seq_len,
          head_number_);
      qk_bias = temp_qk_bias;
    }
    // fake qk_bias
    if (ProductDim(input_desc[1].dims) == ProductDim(input_desc[0].dims)) {
      qk_bias = reinterpret_cast<float *>(workspace);
      auto size = batch * head_number_ * seq_len * seq_len;
      cudaMemset(qk_bias, 0, sizeof(float) * size);
    }
    const float *input1_data = static_cast<const float *>(qk_bias);
    // BxSx3xNxH => tptr: 3xBxNxSxH.
    TransposeQKV(
        batch, seq_len, head_size_, head_number_, input0_data, tptr, stream);

    auto *device_ctx = static_cast<phi::GPUContext *>(
        platform::DeviceContextPool::Instance().Get(
            platform::CUDAPlace(device_id)));

    const phi::GPUContext &dev_ctx = *device_ctx;
    operators::math::MultiHeadGPUComputeFunctor<float> multihead_compute_func;
    multihead_compute_func(dev_ctx,
                           batch,
                           seq_len,
                           head_number_,
                           head_size_,
                           qkptr,
                           input1_data,
                           tptr,
                           scale_,
                           static_cast<float>(0.0));

    int grid = batch * head_number_ * seq_len;
    int block = head_size_;
    float *output = static_cast<float *>(outputs[0]);
    transpose<float><<<grid, block, 0, stream>>>(
        tptr, output, batch, seq_len, head_number_, head_size_);

  } else if (input_type == nvinfer1::DataType::kHALF) {
#ifdef TRT_PLUGIN_FP16_AVALIABLE
    VLOG(1) << "TRT Plugin DataType selected. QkvToContext-->fp16";
    int *padding_offset = nullptr;
    half *padding_input = nullptr;
    framework::Tensor padding_offset_tensor;
    framework::Tensor padding_input_tensor;
    if (real_seq_len != seq_len) {
      padding_offset_tensor.Resize({batch, real_seq_len});
      padding_offset = padding_offset_tensor.mutable_data<int>(
          platform::CUDAPlace(device_id));
      cudaMemset(padding_offset, 0, sizeof(int) * batch * real_seq_len);

      padding_input_tensor.Resize(
          {batch, seq_len, 3, head_number_, head_size_});  // BxSx3xNxH
      padding_input =
          reinterpret_cast<half *>(padding_input_tensor.mutable_data<int16_t>(
              platform::CUDAPlace(device_id)));
      cudaMemset(
          padding_input,
          0,
          sizeof(half) * batch * seq_len * 3 * head_number_ * head_size_);

      set_padding_offset<<<1, 1, 0, stream>>>(
          padding_offset, real_seq_len, batch, seq_len);
      int m = batch * real_seq_len;
      rebuild_sequence_length_padding<<<m, 256, 0, stream>>>(
          static_cast<const half *>(inputs[0]),
          padding_input,
          padding_offset,
          head_number_ * head_size_ * 3);
    }
    auto *multihead_temp_data =
        multihead_temp_tensor.mutable_data<int16_t>(  // NOLINT
            platform::CUDAPlace(device_id));

    half *qkptr = reinterpret_cast<half *>(multihead_temp_data);
    half *tptr = qkptr + scratch_size;

    const half *input0_data = static_cast<const half *>(inputs[0]);
    // fit to [batch, head_num, length, length] + [batch, 1, 1, length]
    framework::Tensor temp_qk_bias_tensor;
    half *qk_bias = const_cast<half *>(static_cast<const half *>(inputs[1]));
    if (ProductDim(input_desc[1].dims) == (batch * seq_len)) {
      temp_qk_bias_tensor.Resize({batch, head_number_, seq_len, seq_len});
      auto *temp_qk_bias =
          reinterpret_cast<half *>(temp_qk_bias_tensor.mutable_data<int16_t>(
              platform::CUDAPlace(device_id)));
      int grid = batch * head_number_ * seq_len;
      int block = round_up(seq_len);
      broadcast<<<grid, block, 0, stream>>>(
          static_cast<const half *>(inputs[1]),
          temp_qk_bias,
          seq_len,
          head_number_);
      qk_bias = temp_qk_bias;
    }
    if (ProductDim(input_desc[1].dims) == ProductDim(input_desc[0].dims)) {
      qk_bias = reinterpret_cast<half *>(workspace);
      auto size = batch * head_number_ * seq_len * seq_len;
      cudaMemset(qk_bias, 0, sizeof(float) * size);
    }
    if (real_seq_len != seq_len) {
      int blocks = batch * head_number_ * seq_len;
      reset_qk_bias<<<blocks, 1024, 0, stream>>>(
          qk_bias, real_seq_len, seq_len);
    }
    const half *input1_data = static_cast<const half *>(qk_bias);
    // BxSx3xNxH => tptr: 3xBxNxSxH.
    if (real_seq_len != seq_len) {
      TransposeQKV(batch,
                   seq_len,
                   head_size_,
                   head_number_,
                   padding_input,
                   tptr,
                   stream);
    } else {
      TransposeQKV(
          batch, seq_len, head_size_, head_number_, input0_data, tptr, stream);
    }

    auto *device_ctx = static_cast<phi::GPUContext *>(
        platform::DeviceContextPool::Instance().Get(
            platform::CUDAPlace(device_id)));

    int n_q = seq_len * head_number_ * head_size_ * batch;
    constexpr int threads = 128;
    int blocks = (n_q + threads - 1) / threads;

    apply_scale<<<blocks, threads, 0, stream>>>(
        tptr, static_cast<half>(scale_), n_q);

    const phi::GPUContext &dev_ctx = *device_ctx;
    operators::math::MultiHeadGPUComputeFunctor<half> multihead_compute_func;
    multihead_compute_func(dev_ctx,
                           batch,
                           seq_len,
                           head_number_,
                           head_size_,
                           qkptr,
                           input1_data,
                           tptr,
                           half(1.),
                           half(0.0));

    int grid = batch * head_number_ * seq_len;
    int block = head_size_;
    half *output = static_cast<half *>(outputs[0]);
    transpose<half><<<grid, block, 0, stream>>>(
        tptr, output, batch, seq_len, head_number_, head_size_);
#else
    PADDLE_THROW(platform::errors::Fatal(
        "The Ernie(Bert) TensorRT Plugin should be "
        "complied with CUDA version >= 10.0 when running with fp16. "
        "Please recomplie it or try to use fp32 by set "
        "config.SetTRTDynamicShapeInfo(min_input_shape, "
        "max_input_shape, opt_input_shape, true"));
#endif
  } else {
    PADDLE_THROW(platform::errors::Fatal(
        "The QKV TRT Plugin's input type should be float or half."));
  }
  return cudaGetLastError() != cudaSuccess;
}
#endif

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
