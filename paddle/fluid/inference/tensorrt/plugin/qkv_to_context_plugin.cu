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
#include "paddle/fluid/inference/tensorrt/plugin/common/common.cuh"
#include "paddle/fluid/inference/tensorrt/plugin/qkv_to_context_plugin.h"
#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin_utils.h"
#include "paddle/phi/core/platform/device_context.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/multihead_matmul_functor.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

// Dynamic Plugin below.
#if IS_TRT_VERSION_GE(6000)

inline int round_up(int seq_len, int multiple = 32) {
  PADDLE_ENFORCE_GT(
      multiple,
      0,
      common::errors::InvalidArgument(
          "multiple should be a positive number, but it's (%d)", multiple));
  return ((seq_len + multiple - 1) / multiple) * multiple;
}

template <typename T>
__global__ void reset_qk_bias(T *input, int real_seq_len, int seq_len) {
  if (threadIdx.x < seq_len) {
    int id = threadIdx.x + blockIdx.x * seq_len;
    input[id] = threadIdx.x >= real_seq_len ? (T)0.0f : (T)1.0f;
  }
}

template <typename T>
__global__ void transpose_qkv_padding(
    const T *src,  // (Batch, real_seq_len, 3 , head_num * size_per_head)
    T *dst,        // (3 * batch * head_num * seq_len * size_per_head)
    const int batch_size,
    const int seq_len,
    const int head_num,
    const int size_per_head,
    const int real_seq_len) {
  // const dim3 grid(seq_len, batch, 3);
  // const dim3 block(head_size, head_num, 1);
  int qkv_id = blockIdx.z;
  int batch_id = blockIdx.y;
  int seq_id = blockIdx.x;
  int head_id = threadIdx.y;
  const int dst_offset =
      qkv_id * batch_size * head_num * seq_len * size_per_head +
      batch_id * head_num * seq_len * size_per_head +
      head_id * seq_len * size_per_head + seq_id * size_per_head;
  const int src_offset =
      batch_id * real_seq_len * 3 * head_num * size_per_head +
      seq_id * 3 * head_num * size_per_head +
      qkv_id * head_num * size_per_head + head_id * size_per_head;
  if (seq_id < real_seq_len) {
    dst[threadIdx.x + dst_offset] = src[threadIdx.x + src_offset];
  }
}

template <typename T>
__global__ void transpose_qkv_unpadding(const T *src,
                                        T *dst,
                                        const int batch_size,
                                        const int seq_len,
                                        const int head_num,
                                        const int size_per_head,
                                        const int real_seq_len) {
  int batch_id = blockIdx.y;
  int seq_id = blockIdx.x;
  int head_id = threadIdx.y;
  const int src_offset = batch_id * head_num * seq_len * size_per_head +
                         head_id * seq_len * size_per_head +
                         seq_id * size_per_head;
  const int dst_offset = batch_id * real_seq_len * head_num * size_per_head +
                         seq_id * head_num * size_per_head +
                         head_id * size_per_head;

  dst[threadIdx.x + dst_offset] = src[threadIdx.x + src_offset];
}

#define LAUNCH_TRANSPOSE_KERNEL(TYPE, VECTOR_SIZE, PAD_TYPE)                \
  do {                                                                      \
    int h = head_size / VECTOR_SIZE;                                        \
    const TYPE *input##VECTOR_SIZE = reinterpret_cast<const TYPE *>(input); \
    TYPE *output##VECTOR_SIZE = reinterpret_cast<TYPE *>(output);           \
    dim3 block(h, head_num, 1);                                             \
    transpose_qkv_##PAD_TYPE<TYPE>                                          \
        <<<grid, block, 0, stream>>>(input##VECTOR_SIZE,                    \
                                     output##VECTOR_SIZE,                   \
                                     batch,                                 \
                                     seq_len,                               \
                                     head_num,                              \
                                     h,                                     \
                                     real_seq_len);                         \
  } while (0)

inline void TransposePadding(const half *input,
                             half *output,
                             const int batch,
                             const int seq_len,
                             const int head_num,
                             const int head_size,
                             const int real_seq_len,
                             cudaStream_t stream) {
  const dim3 grid(seq_len, batch, 3);
  if (head_size % 8 == 0) {
    LAUNCH_TRANSPOSE_KERNEL(int4, 8, padding);
  } else if (head_size % 2 == 0) {
    LAUNCH_TRANSPOSE_KERNEL(half2, 2, padding);
  } else {
    LAUNCH_TRANSPOSE_KERNEL(half, 1, padding);
  }
}

inline void TransposeUnPadding(const half *input,
                               half *output,
                               const int batch,
                               const int seq_len,
                               const int head_num,
                               const int head_size,
                               const int real_seq_len,
                               cudaStream_t stream) {
  const dim3 grid(real_seq_len, batch);
  if (head_size % 8 == 0) {
    LAUNCH_TRANSPOSE_KERNEL(int4, 8, unpadding);
  } else if (head_size % 2 == 0) {
    LAUNCH_TRANSPOSE_KERNEL(half2, 2, unpadding);
  } else {
    LAUNCH_TRANSPOSE_KERNEL(half, 1, unpadding);
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
                    common::errors::InvalidArgument(
                        "There is only one output of the EmbEltwiseLayernorm, "
                        "so the index should be zero,"
                        "but it's (%d)",
                        output_index));
  PADDLE_ENFORCE_EQ(
      nb_inputs,
      2,
      common::errors::InvalidArgument(
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

void QkvToContextPluginDynamic::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc *in,
    int nb_inputs,
    const nvinfer1::DynamicPluginTensorDesc *out,
    int nb_outputs) TRT_NOEXCEPT {
  auto input_dims = in[0].desc.dims;
  int batch = input_dims.d[0];
  int real_seq_len = input_dims.d[1];
  int seq_len = round_up(real_seq_len, 8);
  if (batch != -1 && real_seq_len != -1) {
    int device_id = 0;
    cudaGetDevice(&device_id);
    auto *device_ctx = static_cast<phi::GPUContext *>(
        phi::DeviceContextPool::Instance().Get(phi::GPUPlace(device_id)));
    const phi::GPUContext &dev_ctx = *device_ctx;
    auto stream = dev_ctx.stream();
    tensor_.Resize({batch, seq_len, seq_len, head_number_});
    if (in[0].desc.type == nvinfer1::DataType::kHALF) {
      tensor_.Resize({batch, seq_len, seq_len, 1});
      int blocks = batch * 1 * seq_len;
      mask_half_ = reinterpret_cast<half *>(
          tensor_.mutable_data<int16_t>(phi::GPUPlace(device_id)));
      reset_qk_bias<<<blocks, 1024, 0, stream>>>(
          mask_half_, real_seq_len, seq_len);
    } else if (in[0].desc.type == nvinfer1::DataType::kFLOAT) {
      fake_qk_bias_ = reinterpret_cast<float *>(
          tensor_.mutable_data<int32_t>(phi::GPUPlace(device_id)));
      int64_t size = sizeof(int32_t) * batch * seq_len * seq_len * head_number_;
#ifdef PADDLE_WITH_HIP
      PADDLE_ENFORCE_GPU_SUCCESS(
          hipMemsetAsync(fake_qk_bias_, 0, size, dev_ctx.stream()));
#else
      PADDLE_ENFORCE_GPU_SUCCESS(
          cudaMemsetAsync(fake_qk_bias_, 0, size, dev_ctx.stream()));
#endif
    } else {
      PADDLE_THROW(common::errors::Fatal(
          "The QKV TRT Plugin's input type should be float or half."));
    }
  }
}

bool QkvToContextPluginDynamic::supportsFormatCombination(
    int pos,
    const nvinfer1::PluginTensorDesc *in_out,
    int nb_inputs,
    int nb_outputs) TRT_NOEXCEPT {
  PADDLE_ENFORCE_NOT_NULL(
      in_out,
      common::errors::InvalidArgument(
          "The input of swish plugin should not be nullptr."));

  PADDLE_ENFORCE_LT(
      pos,
      nb_inputs + nb_outputs,
      common::errors::InvalidArgument("The pos(%d) should be less than the "
                                      "num(%d) of the input and the output.",
                                      pos,
                                      nb_inputs + nb_outputs));

  const nvinfer1::PluginTensorDesc &in = in_out[pos];
  if (pos == 0) {
    if (with_fp16_) {
#ifdef TRT_PLUGIN_FP16_AVAILABLE
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
      common::errors::InvalidArgument(
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
__global__ void broadcast_batch_head_number(const T *src,
                                            T *dst,
                                            const int batch_size,
                                            const int seq_len,
                                            const int head_num) {
  int batch_id = blockIdx.x % seq_len;
  int dst_offset = blockIdx.x * seq_len;
  if (threadIdx.x < seq_len) {
    dst[threadIdx.x + dst_offset] = src[threadIdx.x + batch_id * seq_len];
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
  phi::DenseTensor multihead_temp_tensor;
  int scratch_size = batch * head_number_ * seq_len * seq_len * 1;

  int device_id;
  cudaGetDevice(&device_id);
  multihead_temp_tensor.Resize({scratch_size + input_num});

  auto input_type = input_desc[0].type;
  if (input_type == nvinfer1::DataType::kFLOAT) {
    VLOG(1) << "TRT Plugin DataType selected. QkvToContext-->fp32";
    auto *multihead_temp_data =
        multihead_temp_tensor.mutable_data<float>(phi::GPUPlace(device_id));
    auto *qkptr = multihead_temp_data;
    auto *tptr = multihead_temp_data + scratch_size;

    const float *input0_data = static_cast<const float *>(inputs[0]);
    // fit to [batch, head_num, length, length] + [batch, 1, 1, length]
    phi::DenseTensor temp_qk_bias_tensor;
    float *qk_bias = const_cast<float *>(static_cast<const float *>(inputs[1]));
    if (ProductDim(input_desc[1].dims) == (batch * seq_len)) {
      temp_qk_bias_tensor.Resize({batch, head_number_, seq_len, seq_len});
      auto *temp_qk_bias =
          temp_qk_bias_tensor.mutable_data<float>(phi::GPUPlace(device_id));
      int grid = batch * head_number_ * seq_len;
      int block = round_up(seq_len);
      broadcast<<<grid, block, 0, stream>>>(
          static_cast<const float *>(inputs[1]),
          temp_qk_bias,
          seq_len,
          head_number_);
      qk_bias = temp_qk_bias;
    }
    // fit to [batch, head_num, length, length] + [1, 1, length, length]
    if (ProductDim(input_desc[1].dims) == (seq_len * seq_len)) {
      temp_qk_bias_tensor.Resize({batch, head_number_, seq_len, seq_len});
      auto *temp_qk_bias = reinterpret_cast<float *>(
          temp_qk_bias_tensor.mutable_data<float>(phi::GPUPlace(device_id)));
      int grid = batch * head_number_ * seq_len;
      int block = round_up(seq_len);
      broadcast_batch_head_number<<<grid, block, 0, stream>>>(
          static_cast<const float *>(inputs[1]),
          temp_qk_bias,
          batch,
          seq_len,
          head_number_);
      qk_bias = temp_qk_bias;
    }
    // fake qk_bias
    if (ProductDim(input_desc[1].dims) == ProductDim(input_desc[0].dims)) {
      qk_bias = fake_qk_bias_;
    }
    const float *input1_data = static_cast<const float *>(qk_bias);
    // BxSx3xNxH => tptr: 3xBxNxSxH.
    TransposeQKV(
        batch, seq_len, head_size_, head_number_, input0_data, tptr, stream);

    auto *device_ctx = static_cast<phi::GPUContext *>(
        phi::DeviceContextPool::Instance().Get(phi::GPUPlace(device_id)));

    const phi::GPUContext &dev_ctx = *device_ctx;
    phi::funcs::MultiheadGPUComputeFunctor<float> multihead_compute_func;
    multihead_compute_func(dev_ctx,
                           batch,
                           seq_len,
                           head_number_,
                           head_size_,
                           qkptr,
                           input1_data,
                           false,
                           tptr,
                           scale_,
                           static_cast<float>(0.0));

    int grid = batch * head_number_ * seq_len;
    int block = head_size_;
    float *output = static_cast<float *>(outputs[0]);
    transpose<float><<<grid, block, 0, stream>>>(
        tptr, output, batch, seq_len, head_number_, head_size_);

  } else if (input_type == nvinfer1::DataType::kHALF) {
#ifdef TRT_PLUGIN_FP16_AVAILABLE
    VLOG(1) << "TRT Plugin DataType selected. QkvToContext-->fp16";
    int real_seq_len = seq_len;
    int need_padding = false;
    // fake qk_bias
    if (ProductDim(input_desc[1].dims) == ProductDim(input_desc[0].dims)) {
      seq_len = round_up(real_seq_len, 8);
      scratch_size = batch * head_number_ * seq_len * seq_len * 1;
      input_num = batch * seq_len * 3 * head_number_ * head_size_;
      multihead_temp_tensor.Resize({scratch_size + input_num});
      need_padding = (real_seq_len != seq_len) ? true : false;
    }
    auto *multihead_temp_data =
        multihead_temp_tensor.mutable_data<int16_t>(  // NOLINT
            phi::GPUPlace(device_id));

    half *qkptr = reinterpret_cast<half *>(multihead_temp_data);
    half *tptr = qkptr + scratch_size;

    const half *input0_data = static_cast<const half *>(inputs[0]);
    // fit to [batch, head_num, length, length] + [batch, 1, 1, length]
    phi::DenseTensor temp_qk_bias_tensor;
    half *qk_bias = const_cast<half *>(static_cast<const half *>(inputs[1]));
    if (ProductDim(input_desc[1].dims) == (batch * seq_len)) {
      temp_qk_bias_tensor.Resize({batch, head_number_, seq_len, seq_len});
      auto *temp_qk_bias = reinterpret_cast<half *>(
          temp_qk_bias_tensor.mutable_data<int16_t>(phi::GPUPlace(device_id)));
      int grid = batch * head_number_ * seq_len;
      int block = round_up(seq_len);
      broadcast<<<grid, block, 0, stream>>>(
          static_cast<const half *>(inputs[1]),
          temp_qk_bias,
          seq_len,
          head_number_);
      qk_bias = temp_qk_bias;
    }
    // fit to [batch, head_num, length, length] + [1, 1, length, length]
    if (ProductDim(input_desc[1].dims) == (seq_len * seq_len)) {
      temp_qk_bias_tensor.Resize({batch, head_number_, seq_len, seq_len});
      auto *temp_qk_bias = reinterpret_cast<half *>(
          temp_qk_bias_tensor.mutable_data<int16_t>(phi::GPUPlace(device_id)));
      int grid = batch * head_number_ * seq_len;
      int block = round_up(seq_len);
      broadcast_batch_head_number<<<grid, block, 0, stream>>>(
          static_cast<const half *>(inputs[1]),
          temp_qk_bias,
          batch,
          seq_len,
          head_number_);
      qk_bias = temp_qk_bias;
    }
    // padding:    mask_half_ = [1.0,....1.0...1.0....,0.0f]
    // no_padding: mask_half_ = [1.0,....1.0,.........,1.0f]
    bool bias_is_mask = false;
    if (ProductDim(input_desc[1].dims) == ProductDim(input_desc[0].dims)) {
      qk_bias = mask_half_;
      bias_is_mask = true;
    }
    const half *input1_data = static_cast<const half *>(qk_bias);
    // BxSx3xNxH => tptr: 3xBxNxSxH.
    if (need_padding) {
      PADDLE_ENFORCE_GPU_SUCCESS(
          cudaMemsetAsync(tptr, 0, sizeof(half) * input_num, stream));
      TransposePadding(input0_data,
                       tptr,
                       batch,
                       seq_len,
                       head_number_,
                       head_size_,
                       real_seq_len,
                       stream);
    } else {
      TransposeQKV(
          batch, seq_len, head_size_, head_number_, input0_data, tptr, stream);
    }

    auto *device_ctx = static_cast<phi::GPUContext *>(
        phi::DeviceContextPool::Instance().Get(phi::GPUPlace(device_id)));

    int n_q = seq_len * head_number_ * head_size_ * batch;
    constexpr int threads = 128;
    int blocks = (n_q + threads - 1) / threads;

    apply_scale<<<blocks, threads, 0, stream>>>(
        tptr, static_cast<half>(scale_), n_q);

    const phi::GPUContext &dev_ctx = *device_ctx;
    phi::funcs::MultiheadGPUComputeFunctor<half> multihead_compute_func;
    multihead_compute_func(dev_ctx,
                           batch,
                           seq_len,
                           head_number_,
                           head_size_,
                           qkptr,
                           input1_data,
                           bias_is_mask,
                           tptr,
                           half(1.),
                           half(0.0));

    int grid = batch * head_number_ * seq_len;
    int block = head_size_;
    half *output = static_cast<half *>(outputs[0]);
    if (need_padding) {
      TransposeUnPadding(tptr,
                         output,
                         batch,
                         seq_len,
                         head_number_,
                         head_size_,
                         real_seq_len,
                         stream);
    } else {
      transpose<half><<<grid, block, 0, stream>>>(
          tptr, output, batch, seq_len, head_number_, head_size_);
    }
#else
    PADDLE_THROW(common::errors::Fatal(
        "The Ernie(Bert) TensorRT Plugin should be "
        "complied with CUDA version >= 10.0 when running with fp16. "
        "Please recompile it or try to use fp32 by set "
        "config.SetTRTDynamicShapeInfo(min_input_shape, "
        "max_input_shape, opt_input_shape, true"));
#endif
  } else {
    PADDLE_THROW(common::errors::Fatal(
        "The QKV TRT Plugin's input type should be float or half."));
  }
  return cudaGetLastError() != cudaSuccess;
}
#endif

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
