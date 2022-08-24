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
__global__ void transpose_qkv_unpadding(const T *src,
                                        T *dst,
                                        const int batch_size,
                                        const int seq_len,
                                        const int head_num,
                                        const int size_per_head,
                                        const int real_seq_len){
  int batch_id = blockIdx.x / (head_num*real_seq_len);
  int seq_id = blockIdx.x % real_seq_len;
  int head_id = blockIdx.x % (head_num * real_seq_len) / real_seq_len;
  dst[batch_id * head_num * real_seq_len *size_per_head +
      seq_id   * head_num * size_per_head +
      head_id  * size_per_head +
      threadIdx.x] = src[batch_id * head_num * seq_len * size_per_head +
                         head_id  * seq_len * size_per_head +
                         seq_id   * size_per_head +
                         threadIdx.x];
}

template <typename T>
__global__ void transpose_qkv_padding(const T *src, // (Batch, real_seq_len, 3 , head_num * size_per_head)
                                      T *dst,       // (3 * batch * head_num * seq_len * size_per_head)
                                      const int batch_size,
                                      const int seq_len,
                                      const int head_num,
                                      const int size_per_head,
                                      const int real_seq_len){
  //const dim3 grid(seq_len, batch, 3);
  //const dim3 block(head_size, head_num, 1);
  int qkv_id = blockIdx.z;
  int batch_id = blockIdx.y;
  int seq_id = blockIdx.x;
  int head_id = threadIdx.y;
  const int dst_offset = qkv_id * batch_size * head_num * seq_len * size_per_head +
                         batch_id * head_num * seq_len * size_per_head +
                         head_id * seq_len * size_per_head +
                         seq_id * size_per_head;
  const int src_offset = batch_id * real_seq_len * 3 * head_num * size_per_head +
                         seq_id * 3 * head_num * size_per_head +
                         qkv_id * head_num * size_per_head +
                         head_id * size_per_head;
  if(seq_id<real_seq_len){
    dst[threadIdx.x + dst_offset] = src[threadIdx.x + src_offset];
  } else if (seq_id < seq_len) {
    dst[threadIdx.x + dst_offset] = 0;
  }
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
  // printf("@#@@ in TransposeQKV, half. batch: %d, seq_len: %d, head_size: %d, head_num: %d \r\n",
        // batch,seq_len,head_size,head_num);

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
// printf("@#@@ in TransposeQKV, half. batch: %d, seq_len: %d, head_size: %d, head_num: %d \r\n",
        // batch,seq_len,head_size,head_num);
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
      return (in.type == nvinfer1::DataType::kHALF) &&
             (in.format == nvinfer1::TensorFormat::kLINEAR);


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
__global__ void broadcast_batch(const T *src,
                          T *dst,
                          const int seq_len,
                          const int head_num,
                          const int window_num) {
  int WindownumHeadSeqlen_id = blockIdx.x % (window_num * head_num * seq_len);
  int dst_offset = blockIdx.x * seq_len;
  if (threadIdx.x < seq_len) {
    dst[threadIdx.x + dst_offset] = src[threadIdx.x+WindownumHeadSeqlen_id*seq_len];
  }
}


template <typename T>
__global__ void broadcast_batch(const T *src,
                          T *dst,
                          const int seq_len,
                          const int head_num,
                          const int window_num,
                          const int real_seq_len // unpadding seq_len
                          ) {

  int WindownumHeadSeqlen_id = blockIdx.x % (window_num * head_num * seq_len);
  int seqlen_id = blockIdx.x % seq_len;
  int src_offset = WindownumHeadSeqlen_id / seq_len * real_seq_len * real_seq_len + WindownumHeadSeqlen_id % seq_len * real_seq_len;
  int dst_offset = blockIdx.x * seq_len;
  if (threadIdx.x < real_seq_len && seqlen_id < real_seq_len) {
    dst[threadIdx.x + dst_offset] = src[threadIdx.x+src_offset];
  } else if(threadIdx.x<seq_len){
    dst[threadIdx.x + dst_offset] = -1e5;
  }
}

// template <typename T>
// __global__ void padding_k8_rowMajor(const T *src,
//                           T * dst,
//                           const int64_t batch,
//                           const int64_t row,
//                           const int64_t col,
//                           const int64_t paddingRow
//                           const int64_t paddingCol){
//   int rowId=blockIdx.x % (paddingRow)
//   int dst_offset = blockIdx.x * paddingCol;
//   if (rowId<row){
//     int src_offset = blockIdx.x * col;
//     if(threadIdx.x<paddingCol){
//       dst[threadIdx.x+dst_offset]=src[threadIdx.x+src_offset];
//     } else {
//       dst[threadIdx.x+dst_offset]=0;
//     }
//   } else {
//     if(threadIdx.x<paddingCol){
//       dst[threadIdx+dst_offset]=0;
//     }
//   }
// }

// TODO wangbojun for debug
template<typename T>
__global__ void print_float(const T *src, int start_index, int end_index, int numPerRow=49, int stride=1){
  printf("start print float \r\n");
  for (int i=start_index;i<end_index;i+=stride){
    printf("%f, ",static_cast<double>(src[i]));
    if((i-start_index)/stride%numPerRow==numPerRow-1){
      printf("\r\n");
    }
  }
}

void QkvToContextPluginDynamic::attachToContext(
    cudnnContext* cudnnContext, cublasContext* cublasContext,
    nvinfer1::IGpuAllocator* gpuAllocator) TRT_NOEXCEPT {
  platform::dynload::cublasLtCreate(&cublas_);
}

void QkvToContextPluginDynamic::detachFromContext() TRT_NOEXCEPT {
  platform::dynload::cublasLtDestroy(cublas_);
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
  const int paddingNum=8;
  int batch = input_dims.d[0];
  int seq_len = input_dims.d[1];

  // TODO, padding from very beginning
  int real_seq_len = seq_len;
  if (input_desc[0].type == nvinfer1::DataType::kHALF) {
    seq_len = (seq_len + paddingNum - 1) / paddingNum * paddingNum;
    input_num = batch * seq_len * 3 * head_number_ * head_size_;
  }
  

  auto input_type = input_desc[0].type;
  auto biasqk_type = input_desc[1].type;
  auto biasqk_dims = input_desc[1].dims;
  // VLOG(1)<<"@@@ input type: "<<static_cast<int>(input_type)<<"biasqk type: "<<static_cast<int>(biasqk_type);
  // VLOG(1)<<"@@@ biasqk_dims: "<<biasqk_dims.d[0]<<" "<<biasqk_dims.d[1]<<" "<<biasqk_dims.d[2]<<" "<<biasqk_dims.d[3];
  if (input_type == nvinfer1::DataType::kFLOAT) {
    VLOG(1) << "TRT Plugin DataType selected. QkvToContext-->fp32";
    framework::Tensor multihead_temp_tensor;
    int scratch_size = batch * head_number_ * seq_len * seq_len * 1;
    int device_id;
    cudaGetDevice(&device_id);
    multihead_temp_tensor.Resize({scratch_size + input_num});

    auto *multihead_temp_data = multihead_temp_tensor.mutable_data<float>(
        platform::CUDAPlace(device_id));
    auto *qkptr = multihead_temp_data;
    auto *tptr = multihead_temp_data + scratch_size;

    // cudaDeviceSynchronize();
    const float *input0_data = static_cast<const float *>(inputs[0]);

    // printf("@#@@@ input0 data \r\n");
    // print_float<float><<<1,1>>>(input0_data,0,input_num);
    // cudaDeviceSynchronize();
    // printf("\r\n");

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
        //TODO broadcast kernel need update with padding information
      qk_bias = temp_qk_bias;
    }

    // if bias_qk is [window_num,head_number,seq_len,seq_len]
    // in swin SW-MSA block dim[0] of input is batch_number*windows_number 
    // therefore, we broadcast bias_qk to [Batch_num*window_num, head_number, seq_len, seq_len]
    int window_num=input_desc[1].dims.d[0];
    if(ProductDim(input_desc[1].dims)==window_num*head_number_*seq_len*seq_len){
      temp_qk_bias_tensor.Resize({batch, head_number_, seq_len, seq_len});
      auto *temp_qk_bias = temp_qk_bias_tensor.mutable_data<float>(
          platform::CUDAPlace(device_id));
      int grid = batch * head_number_ * seq_len;
      int block = round_up(seq_len);
      broadcast_batch<float><<<grid, block, 0, stream>>>(
          static_cast<const float *>(inputs[1]), 
          temp_qk_bias, 
          seq_len, 
          head_number_, 
          window_num,
          real_seq_len);
      qk_bias = temp_qk_bias;
    }

    // printf("@#@@ input_desc[0] shape: %d, %d, %d \r\n",input_desc[0].dims.d[0],input_desc[0].dims.d[1],input_desc[0].dims.d[2]);
    // printf("@#@@ input_desc[1] shape: %d, %d, %d, %d \r\n",input_desc[1].dims.d[0],input_desc[1].dims.d[1],input_desc[1].dims.d[2],input_desc[1].dims.d[3]);
    // printf("\r\n");

    const float *input1_data = static_cast<const float *>(qk_bias);
    // printf("@#@@ in fp32 plugin biasqk 0 1 2 3: \r\n");
    // print_float<float><<<1,1,0,stream>>>(input1_data,0,batch*head_number_*seq_len*seq_len,seq_len);
    // cudaDeviceSynchronize();
    // printf("\r\n");

    // BxSx3xNxH => tptr: 3xBxNxSxH.
    TransposeQKV(
        batch, seq_len, head_size_, head_number_, input0_data, tptr, stream);
    // cudaDeviceSynchronize();
    // printf("@@@@ tptr data \r\n");
    // print_float<float><<<1,1>>>(tptr,0,3*batch*head_number_*seq_len*head_size_);
    // cudaDeviceSynchronize();
    // printf("\r\n");


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
    int device_id;
    cudaGetDevice(&device_id);
    const half *input0_data = static_cast<const half *>(inputs[0]);
    // input[0], (B, S, 3 * N * H, 1, 1)

    // cudaDeviceSynchronize();
    // printf("@#@@@ input0 data before padding \r\n");
    // if(batch==64){
    //   print_float<half><<<1,1>>>(input0_data,
    //                             0,
    //                             batch*real_seq_len * 3 * head_number_ * head_size_,
    //                             3 * head_number_ * head_size_,
    //                             1);
    // }
    // cudaDeviceSynchronize();

    // fengshuai's padding
    //     int *padding_offset = nullptr;
    // half *padding_input = nullptr;

    // framework::Tensor padding_offset_tensor;
    // framework::Tensor padding_input_tensor;
    //  if (real_seq_len != seq_len) {
    //    padding_offset_tensor.Resize({batch, real_seq_len});
    //    padding_offset = padding_offset_tensor.mutable_data<int>(
    //        platform::CUDAPlace(device_id));
    //    cudaMemset(padding_offset, 0, sizeof(int) * batch * real_seq_len);

    //    padding_input_tensor.Resize(
    //        {batch* seq_len* 3* head_number_* head_size_});  // BxSx3xNxH
    //    padding_input =
    //        reinterpret_cast<half *>(padding_input_tensor.mutable_data<int16_t>(
    //            platform::CUDAPlace(device_id)));
    //   // printf("@@@ padding_input_size : %d, padding input begin: %X, end: %X \r\n",
    //   //  batch* seq_len* 3* head_number_* head_size_,padding_input,padding_input+batch* seq_len* 3* head_number_* head_size_);
    //    cudaMemset(
    //        padding_input,
    //        0,
    //        sizeof(half) * batch * seq_len * 3 * head_number_ * head_size_);
    //    set_padding_offset<<<1, 1, 0, stream>>>(
    //        padding_offset, real_seq_len, batch, seq_len);
    //    int m = batch * real_seq_len;
    //    rebuild_sequence_length_padding<<<m, 256, 0, stream>>>(
    //        static_cast<const half *>(inputs[0]),
    //        padding_input,
    //        padding_offset,
    //        head_number_ * head_size_ * 3);
    //    input0_data = padding_input;
    //  }
    // fengshuai's padding

    // cudaDeviceSynchronize();
    // printf("@#@@@ input0 data after padding \r\n");
    // print_float<half><<<1,1>>>(input0_data,
    //                            0,
    //                            56*3*head_number_*head_size_,
    //                            56,
    //                            3*head_number_*head_size_);
    // cudaDeviceSynchronize();
    // printf("\r\n");
    // print_float<half><<<1,1>>>(input0_data,
    //                            56*3*head_number_*head_size_,
    //                            2*56*3*head_number_*head_size_,
    //                            56,
    //                            3*head_number_*head_size_);
    // cudaDeviceSynchronize();
    // printf("\r\n");
    framework::Tensor multihead_temp_tensor;
    int scratch_size = batch * head_number_ * seq_len * seq_len * 1;
    multihead_temp_tensor.Resize({scratch_size + input_num});

    auto *multihead_temp_data =
        multihead_temp_tensor.mutable_data<int16_t>(  // NOLINT
            platform::CUDAPlace(device_id));
    
    half *qkptr = reinterpret_cast<half *>(multihead_temp_data);
    half *tptr = qkptr + scratch_size;

    // cudaDeviceSynchronize();
    // printf("@#@@@ input0 data \r\n");
    // print_float<half><<<1,1>>>(input0_data,0,input_num);
    // cudaDeviceSynchronize();
    // printf("\r\n");

    // fit to [batch, head_num, length, length] + [batch, 1, 1, length]
    framework::Tensor temp_qk_bias_tensor;
    
    half *qk_bias = const_cast<half *>(static_cast<const half *>(inputs[1]));
    // printf("@#@@ in origin plugin biasqk 0 1 2 3 fp16, before type check: \r\n");
    // print_float<half><<<1,1,0,stream>>>(qk_bias,0,head_number_*seq_len*seq_len);
    // cudaDeviceSynchronize();
    // printf("\r\n");

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
    
    // if bias_qk is [window_num,head_number,seq_len,seq_len]
    // in swin SW-MSA block dim[0] of input is batch_number*windows_number 
    // therefore, we broadcast bias_qk to [Batch_num*window_num, head_number, seq_len, seq_len]
    int window_num=input_desc[1].dims.d[0];

    // printf("@@@ window_num %d head_number_ %d real_seq_len %d real_seq_len %d \r\n",
    //       window_num,head_number_,real_seq_len,real_seq_len);
    
    const size_t swin_qk_bias_size=window_num*head_number_*real_seq_len*real_seq_len;
    if(ProductDim(input_desc[1].dims)==swin_qk_bias_size){
      temp_qk_bias_tensor.Resize({batch, head_number_, seq_len, seq_len});
      auto *temp_qk_bias =
          reinterpret_cast<half *>(temp_qk_bias_tensor.mutable_data<int16_t>(
              platform::CUDAPlace(device_id)));
      int grid = batch * head_number_ * seq_len;
      int block = round_up(seq_len);

      // printf("@@@ seq_len %d, real_seq_len %d \r\n",
      //       seq_len, real_seq_len);
      // printf("@@@ temp qk bias address %X \r\n",temp_qk_bias);
      
      broadcast_batch<half><<<grid, block, 0, stream>>>(
          static_cast<const half *>(inputs[1]), 
          temp_qk_bias, 
          seq_len, 
          head_number_, 
          window_num,
          real_seq_len);
      qk_bias = temp_qk_bias;
    }
    
    const half *input1_data = static_cast<const half *>(qk_bias);
    // cudaDeviceSynchronize();
    // printf("@#@@ in bocasted plugin biasqk: \r\n");
    // if(batch==64){
    //   print_float<half><<<1,1,0,stream>>>(input1_data,0,batch*head_number_*seq_len*seq_len,seq_len,1);
    // }
    // cudaDeviceSynchronize();
    // printf("\r\n");

    // cudaDeviceSynchronize();
    // printf("@#@@@ input0 data after padding \r\n");
    // printf("@@@ input0 data address %X \r\n",input0_data);

    // if(batch==64){
    //   print_float<half><<<1,1>>>(input0_data,
    //                               0,
    //                               batch * seq_len * 3 * head_number_ * head_size_,
    //                               3 * head_number_ * head_size_,
    //                               1);
    // }
    // cudaDeviceSynchronize();

    // BxSx3xNxH => tptr: 3xBxNxSxH.
    // TransposeQKV(
    //     batch, seq_len, head_size_, head_number_, input0_data, tptr, stream);
    const dim3 grid_trans_qkv_padding(seq_len, batch, 3);
    const dim3 block_trans_qkv_padding(head_size_, head_number_, 1);
    transpose_qkv_padding<<<grid_trans_qkv_padding,
                            block_trans_qkv_padding,0,
                            stream>>>(input0_data,
                                      tptr,batch,
                                      seq_len,
                                      head_number_,
                                      head_size_,
                                      real_seq_len);

    // cudaDeviceSynchronize();
    // if(batch==64){
    // printf("@#@@  TransposeQKV before mha functor result: \r\n");
    // print_float<half><<<1,1,0,stream>>>(tptr,
    //                                     0,
    //                                     batch * seq_len * 3 * head_number_ * head_size_,
    //                                     head_size_,1);
    // }
    // cudaDeviceSynchronize();
    // printf("\r\n");
    // cudaDeviceSynchronize();
    // printf("@@@@ tptr data \r\n");
    // print_float<half><<<1,1>>>(tptr,0,3*batch*head_size_*seq_len*head_size_);
    // cudaDeviceSynchronize();
    // printf("\r\n");

    auto *device_ctx = static_cast<phi::GPUContext *>(
        platform::DeviceContextPool::Instance().Get(
            platform::CUDAPlace(device_id)));

    // int n_q = seq_len * head_number_ * head_size_ * batch;
    // constexpr int threads = 128;
    // int blocks = (n_q + threads - 1) / threads;

    // apply_scale<<<blocks, threads, 0, stream>>>(
    //     tptr, static_cast<half>(scale_), n_q);

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
                           static_cast<half>(scale_),
                           half(0.0));
    // printf("@@ multihead_compute_func done \r\n");

    // cudaDeviceSynchronize();
    // if(batch==64){
    // printf("@#@@ multihead_compute_func qkv result: \r\n");
    // print_float<half><<<1,1,0,stream>>>(qkptr,0,(batch*head_number_)*seq_len*head_size_,head_size_,1);
    // }
    // // cudaDeviceSynchronize();
    // // print_float<half><<<1,1,0,stream>>>(tptr,
    // //                                     (batch*window_num-2)*seq_len*head_size_,
    // //                                     (batch*window_num)*seq_len*head_size_,head_size_,1);
    // cudaDeviceSynchronize();
    // printf("\r\n");

    half *output = static_cast<half *>(outputs[0]);
    // int grid = batch * head_number_ * seq_len;
    // int block = head_size_;
    // transpose<half><<<grid, block, 0, stream>>>(
    //     tptr, output, batch, seq_len, head_number_, head_size_);
    int grid = batch * head_number_ * real_seq_len;
    int block = head_size_;
    transpose_qkv_unpadding<half><<<grid, block, 0, stream>>>(
        tptr, output, batch, seq_len, head_number_, head_size_, real_seq_len);

    // cudaDeviceSynchronize();
    // printf("@#@@ multihead_compute_func qkv^T result: \r\n");
    // print_float<half><<<1,1,0,stream>>>(output,0,2*seq_len*head_size_,head_size_,1);
    // cudaDeviceSynchronize();
    // print_float<half><<<1,1,0,stream>>>(output,
    //                                     (batch*head_number_-2)*seq_len*head_size_,
    //                                     (batch*head_number_)*seq_len*head_size_,head_size_,1);
    // cudaDeviceSynchronize();

    // printf("\r\n");
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
