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

// todo ,wangbojun for testing
#ifdef TRT_FT_WINDOWS_ATTENTION
#include "3rdparty/trt_fused_multihead_attention/qkvToContext.h"
namespace fastertransformer {

/*******************  invokeTransformMask  ***********************/

// transform mask [B, S, S](half) into [B, S2*S2/64, 64](half), S2 is the actural  seqlen used in fmha row-major
// in one MMA (16*16 elements calculated by a warp), each thread calculates 8 elements
// the offsets of elements calculated by each thread are : for n, +0 +1 +8 +9; for m, +0 +8 (M_XMMAS*N_XMMAS times)
// in transformed_mask, the masks of one warp are stored in 4 continuous rows ([4, 64]), with two elements of one thread
// stored in 2 continuous halfs. one cta calculates warps_m*warps_n mma == 16*warps_m*16*warps_n elements grid(B,
// S2*S2/64) block(32)
__global__ void transform_mask_kernel(half2*         tranformed_mask,
                                      const half2*   mask,
                                      const uint32_t warps_m,
                                      const uint32_t warps_n,
                                      const uint32_t B,
                                      const uint32_t S,
                                      const uint32_t S2)
{
    const int bi = blockIdx.x;
    const int r  = blockIdx.y;

    const int    N_per_XMMAS       = warps_n << 4;
    const int    M_per_XMMAS       = warps_m << 4;
    const int    N_XMMAS           = (S2 + N_per_XMMAS - 1) / (N_per_XMMAS);
    const int    warps_in_XMMAS    = warps_m * warps_n;
    const half2* mask_b            = mask + ((bi * S * S) >> 1);
    half2*       tranformed_mask_b = tranformed_mask + (bi * gridDim.y << 5);  //((bi * gridDim.y << 6) >> 1);

    half2 tmp = {half(-30000.0f), half(-30000.0f)};

    int c               = threadIdx.x * 2;
    int elt_offset      = c % 2;
    int warp_id         = r / 4;
    int elt_in_thread   = (r % 4) * 2 + elt_offset;
    int noffset_in_warp = (((elt_in_thread & 3) >> 1) << 3) + (elt_in_thread & 1);
    int moffset_in_warp = ((elt_in_thread >> 2) & 1) << 3;

    int XMMAS_mi         = warp_id / (N_XMMAS * warps_in_XMMAS);
    int XMMAS_ni         = warp_id % (N_XMMAS * warps_in_XMMAS) / warps_in_XMMAS;
    int warp_id_in_XMMAS = warp_id - (XMMAS_mi * N_XMMAS + XMMAS_ni) * warps_in_XMMAS;
    int warp_mi          = warp_id_in_XMMAS % warps_m;
    int warp_ni          = warp_id_in_XMMAS / warps_m;
    int noffset          = XMMAS_ni * N_per_XMMAS + (warp_ni << 4) + noffset_in_warp;
    int moffset          = XMMAS_mi * M_per_XMMAS + (warp_mi << 4) + moffset_in_warp;

    int mi = moffset + (c >> 3);
    int ni = noffset + (((c >> 1) & 3) << 1);

    if (mi < S && ni < S) {
        tmp = __ldg(mask_b + ((mi * S + ni) >> 1));
    }

    tranformed_mask_b[(r << 5) + threadIdx.x] = tmp;
}

// transform mask [B, S, S](half) into [B, S2*S2/64, 64](half), S2 is the actural  seqlen used in fmha row-major
// in one MMA (16*16 elements calculated by a warp), each thread calculates 8 elements
// the offsets of elements calculated by each thread are : for n, +0 +1 +8 +9; for m, +0 +8 (M_XMMAS*N_XMMAS times)
// in transformed_mask, the masks of one warp are stored in 4 continuous rows ([4, 64]), with two elements of one thread
// stored in 2 continuous halfs. one cta calculates warps_m*warps_n mma == 16*warps_m*16*warps_n elements grid(B,
// S2*S2/64) block(32)
__global__ void transform_mask_kernel(half*          tranformed_mask,
                                      const half*    mask,
                                      const uint32_t warps_m,
                                      const uint32_t warps_n,
                                      const uint32_t B,
                                      const uint32_t S,
                                      const uint32_t S2)
{
    const int bi = blockIdx.x;
    const int r  = blockIdx.y;

    const int N_per_XMMAS       = warps_n << 4;
    const int M_per_XMMAS       = warps_m << 4;
    const int N_XMMAS           = (S2 + N_per_XMMAS - 1) / (N_per_XMMAS);
    const int warps_in_XMMAS    = warps_m * warps_n;
    half2*    tranformed_mask_b = (half2*)(tranformed_mask + (bi * gridDim.y << 6));

    half2 tmp = {half(-30000.0f), half(-30000.0f)};

    int c               = threadIdx.x * 2;
    int elt_offset      = c % 2;
    int warp_id         = r / 4;
    int elt_in_thread   = (r % 4) * 2 + elt_offset;
    int noffset_in_warp = (((elt_in_thread & 3) >> 1) << 3) + (elt_in_thread & 1);
    int moffset_in_warp = ((elt_in_thread >> 2) & 1) << 3;

    int XMMAS_mi         = warp_id / (N_XMMAS * warps_in_XMMAS);
    int XMMAS_ni         = warp_id % (N_XMMAS * warps_in_XMMAS) / warps_in_XMMAS;
    int warp_id_in_XMMAS = warp_id - (XMMAS_mi * N_XMMAS + XMMAS_ni) * warps_in_XMMAS;
    int warp_mi          = warp_id_in_XMMAS % warps_m;
    int warp_ni          = warp_id_in_XMMAS / warps_m;
    int noffset          = XMMAS_ni * N_per_XMMAS + (warp_ni << 4) + noffset_in_warp;
    int moffset          = XMMAS_mi * M_per_XMMAS + (warp_mi << 4) + moffset_in_warp;

    int mi = moffset + (c >> 3);
    int ni = noffset + (((c >> 1) & 3) << 1);

    if (mi < S) {
        mask += bi * S * S;
        int idx = mi * S + ni;
        if (ni < S) {
            tmp.x = __ldg(mask + idx);
        }
        if (ni + 1 < S) {
            tmp.y = __ldg(mask + idx + 1);
        }
    }

    tranformed_mask_b[(r << 5) + threadIdx.x] = tmp;
}

void invokeTransformMask(
    half* tranformed_mask, const half* mask, const uint32_t B, const uint32_t S, cudaStream_t stream)
{
    uint32_t S2;
    uint32_t warps_m = 2, warps_n = 2;
    if (S <= 64) {
        S2 = 64;
    }
    else if (S <= 128) {
        S2 = 128;
    }
    else if (S <= 256) {
        S2      = 256;
        warps_m = 1;
        warps_n = 4;
    }
    else if (S <= 384) {
        S2      = 384;
        warps_m = 1;
        warps_n = 8;
    }
    else {
        printf("[ERROR][invokeTransformMask]unsupported seq_len %d\n", S);
        exit(-1);
    }
    assert(S2 * S2 % 64 == 0);
    dim3 grid(B, S2 * S2 / 64);
    dim3 block(32);
    if (S % 2 == 0) {
        transform_mask_kernel<<<grid, block, 0, stream>>>(
            (half2*)tranformed_mask, (const half2*)mask, warps_m, warps_n, B, S, S2);
    }
    else {
        transform_mask_kernel<<<grid, block, 0, stream>>>(tranformed_mask, mask, warps_m, warps_n, B, S, S2);
    }
}

}  // namespace fastertransformer

#endif

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
  // input[1], (B, head_num, seq_len, seq_len) / (1,head_num, seq_len, seq_len)
  // if has_biasqk_mask_
  // input[2], (window_number, seq_len, seq_len)
  // output, (B, seq_len, hidden)
  PADDLE_ENFORCE_EQ(output_index,
                    0,
                    platform::errors::InvalidArgument(
                        "There is only one output of the EmbEltwiseLayernorm, "
                        "so the index should be zero,"
                        "but it's (%d)",
                        output_index));
  if(!has_biasqk_mask_){
  PADDLE_ENFORCE_EQ(
      nb_inputs,
      2,
      platform::errors::InvalidArgument(
          "The Input of the EmbEltwiseLayernorm should be 3, but we found "
          "it has (%d) inputs",
          nb_inputs));
  } else {
  PADDLE_ENFORCE_EQ(
      nb_inputs,
      3,
      platform::errors::InvalidArgument(
          "The Input of the EmbEltwiseLayernorm should be 3, but we found "
          "it has (%d) inputs",
          nb_inputs));
  }
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
    dst[threadIdx.x + dst_offset] =
        src[threadIdx.x + WindownumHeadSeqlen_id * seq_len];
  }
}


// TODO wangbojun for debug
template<typename T>
__global__ void print_float(const T *src, int start_index, int end_index, int numPerRow=49, int stride=1){
  printf("start print float \r\n");
  for (int i=start_index;i<end_index;i+=stride){
    printf("%.1e, ",static_cast<double>(src[i]));
    if((i-start_index)/stride%numPerRow==numPerRow-1){
      printf("\r\n");
    }
  }
}

template <typename T>
__global__ void transpose_qkv_for_ftmha(const T *src, // (Batch, real_seq_len, 3 , head_num * size_per_head)
                                         T *dst,       
                                      const int batch_size,
                                      const int seq_len,
                                      const int head_num,
                                      const int size_per_head){
  //const dim3 grid(seq_len, batch, 3);
  //const dim3 block(head_size, head_num, 1);
  int qkv_id = blockIdx.z;
  int batch_id = blockIdx.y;
  int seq_id = blockIdx.x;
  int head_id = threadIdx.y;
  // (batch * seq_len * head_num * 3(qkv) * size_per_head)
  const int dst_offset = batch_id * seq_len * 3 * head_num * size_per_head +
                         seq_id * head_num * 3 * size_per_head+
                         head_id * 3 * size_per_head +
                         qkv_id * size_per_head;
  const int src_offset = batch_id * seq_len * 3 * head_num * size_per_head +
                         seq_id * 3 * head_num * size_per_head +
                         qkv_id * head_num * size_per_head +
                         head_id * size_per_head;
  if(seq_id<seq_len){
    dst[threadIdx.x + dst_offset] = src[threadIdx.x + src_offset];
  };
}

template <typename T>
__global__ void transpose_qkv_for_ftmha_shared(const T *src, // (Batch, real_seq_len, 3 , head_num * size_per_head)
                                        T *dst,       
                                        const int batch_size,
                                        const int seq_len,
                                        const int head_num,
                                        const int size_per_head){
  // const dim3 grid_t_ftmha_shared(seq_len, batch, head_num_in_grid);
  // const dim3 block_t_ftmha_shared(head_size_, 3, head_num_in_block);
  
  const int seq_id                   = blockIdx.x;
  const int batch_id                 = blockIdx.y;
  const int head_num_in_grid_id      = blockIdx.z;
  const int size_head_num_in_block   = blockDim.z;
  const int size_per_head_id         = threadIdx.x;
  const int qkv_id                   = threadIdx.y;
  const int head_num_in_block_id     = threadIdx.z;
  const int head_id                  = head_num_in_grid_id * size_head_num_in_block + head_num_in_block_id;

  // to (batch * seq_len * head_num * 3(qkv) * size_per_head)
  const int dst_offset = batch_id * seq_len * 3 * head_num * size_per_head +
                         seq_id * head_num * 3 * size_per_head+
                         head_id * 3 * size_per_head +
                         qkv_id * size_per_head;
  const int src_offset = batch_id * seq_len * 3 * head_num * size_per_head +
                         seq_id * 3 * head_num * size_per_head +
                         qkv_id * head_num * size_per_head +
                         head_id * size_per_head;
  __shared__ T smem_matrix[1024];   
  // if( blockIdx.x==0 && blockIdx.y==0 && threadIdx.x==0 && threadIdx.y==0 ){
  //   printf("@@@ in trans kernel shared matrix, head_id_grid:%d, head_id_block:%d [%d][%d][%d][%d][%d] -> [%d] [%d] [%d] \r\n",
  //                 head_num_in_grid_id,head_num_in_block_id, 
  //                 batch_id, seq_id, qkv_id, head_id, size_per_head_id, 
  //                 qkv_id,head_num_in_block_id,size_per_head_id);
  // }
  if(head_id<head_num){
    smem_matrix[qkv_id * size_head_num_in_block * size_per_head + 
                head_num_in_block_id * size_per_head + 
                size_per_head_id                                  ] = src[size_per_head_id + src_offset];
  }
  __syncthreads();
  // if( blockIdx.x==0 && blockIdx.y==0 && threadIdx.x==0 && threadIdx.y==0 ){
  // printf("@@@ in trans kernel do trans, head_id_grid:%d, head_id_block:%d [%d][%d][%d][%d][%d] <- [%d] [%d] [%d] \r\n",
  //                 head_num_in_grid_id,head_num_in_block_id, 
  //                 batch_id, seq_id, head_id, qkv_id, size_per_head_id, 
  //                 qkv_id,head_num_in_block_id,size_per_head_id);
  // }
  if(head_id<head_num){
    dst[size_per_head_id + dst_offset] = smem_matrix[qkv_id * size_head_num_in_block * size_per_head + 
                                                     head_num_in_block_id * size_per_head + 
                                                     size_per_head_id                                  ];
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
  framework::Tensor multihead_temp_tensor;
  int scratch_size = batch * head_number_ * seq_len * seq_len * 1;

  int device_id;
  cudaGetDevice(&device_id);
  multihead_temp_tensor.Resize({scratch_size + input_num});

  auto input_type = input_desc[0].type;
  auto biasqk_type = input_desc[1].type;
  auto biasqk_dims = input_desc[1].dims;
  if (input_type == nvinfer1::DataType::kFLOAT) {
    VLOG(1) << "TRT Plugin DataType selected. QkvToContext-->fp32";
    operators::math::MultiHeadGPUComputeFunctor<float> multihead_compute_func;
    auto *multihead_temp_data = multihead_temp_tensor.mutable_data<float>(
        platform::CUDAPlace(device_id));
    auto *qkptr = multihead_temp_data;
    auto *tptr = multihead_temp_data + scratch_size;

    const float *input0_data = static_cast<const float *>(inputs[0]);

    float *qk_bias = const_cast<float *>(static_cast<const float *>(inputs[1]));
    framework::Tensor temp_qk_bias_tensor;

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
    // if bias_qk is [window_num,head_number,seq_len,seq_len]
    // in swin SW-MSA block dim[0] of input is batch_number*windows_number
    // therefore, we broadcast bias_qk to [Batch_num*window_num, head_number,
    // seq_len, seq_len]
    int window_num = input_desc[1].dims.d[0];
    if (ProductDim(input_desc[1].dims) ==
        window_num * head_number_ * seq_len * seq_len) {
      temp_qk_bias_tensor.Resize({batch, head_number_, seq_len, seq_len});
      auto *temp_qk_bias = temp_qk_bias_tensor.mutable_data<float>(
          platform::CUDAPlace(device_id));
      int grid = batch * head_number_ * seq_len;
      int block = round_up(seq_len);
      // origin batch_num=1, batch==window_num, no need for broadcast
      if (batch != window_num) {
        broadcast_batch<float>
            <<<grid, block, 0, stream>>>(static_cast<const float *>(inputs[1]),
                                         temp_qk_bias,
                                         seq_len,
                                         head_number_,
                                         window_num);
      }
      qk_bias = temp_qk_bias;
    }

    const float *input1_data = static_cast<const float *>(qk_bias);

    // BxSx3xNxH => tptr: 3xBxNxSxH.
    TransposeQKV(
        batch, seq_len, head_size_, head_number_, input0_data, tptr, stream);

    auto *device_ctx = static_cast<phi::GPUContext *>(
        platform::DeviceContextPool::Instance().Get(
            platform::CUDAPlace(device_id)));

    const phi::GPUContext &dev_ctx = *device_ctx;

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
#ifndef TRT_FT_WINDOWS_ATTENTION
    VLOG(1) << "TRT Plugin DataType selected. QkvToContext-->fp16";
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

    // if bias_qk is [window_num,head_number,seq_len,seq_len]
    // in swin SW-MSA block dim[0] of input is batch_number*windows_number
    // therefore, we broadcast bias_qk to [Batch_num*window_num, head_number,
    // seq_len, seq_len]
    int window_num = input_desc[1].dims.d[0];
    const size_t swin_qk_bias_size =
        window_num * head_number_ * seq_len * seq_len;
    if (ProductDim(input_desc[1].dims) == swin_qk_bias_size) {
      temp_qk_bias_tensor.Resize({batch, head_number_, seq_len, seq_len});
      auto *temp_qk_bias =
          reinterpret_cast<half *>(temp_qk_bias_tensor.mutable_data<int16_t>(
              platform::CUDAPlace(device_id)));
      int grid = batch * head_number_ * seq_len;
      int block = round_up(seq_len);
      if (batch != window_num) {
        broadcast_batch<half>
            <<<grid, block, 0, stream>>>(static_cast<const half *>(inputs[1]),
                                         temp_qk_bias,
                                         seq_len,
                                         head_number_,
                                         window_num);
      }
      qk_bias = temp_qk_bias;
    }

    const half *input1_data = static_cast<const half *>(qk_bias);

    // BxSx3xNxH => tptr: 3xBxNxSxH.
    TransposeQKV(
        batch, seq_len, head_size_, head_number_, input0_data, tptr, stream);

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

    int grid = batch * head_number_ * seq_len;
    int block = head_size_;
    half *output = static_cast<half *>(outputs[0]);
    transpose<half><<<grid, block, 0, stream>>>(
        tptr, output, batch, seq_len, head_number_, head_size_);
#else //if define TRT_FT_WINDOWS_ATTENTION
    VLOG(1)<<"@@@ use faster transformer trt fused multihead matmul kernel";
    // printf("@@@ use faster transformer trt fused multihead matmul kernel\r\n");
    auto *multihead_temp_data =
        multihead_temp_tensor.mutable_data<int16_t>(  // NOLINT
            platform::CUDAPlace(device_id));

    half *qkptr = reinterpret_cast<half *>(multihead_temp_data);
    half *tptr = qkptr + scratch_size;
    const int sm = 86; // TODO for A10, sm is 86
    if (ft_dispatcher_fp16_.get() && head_number_ == ft_dispatcher_fp16_num_head_) {}
    else {
      // printf("@@@ ft_dispatcher_fp16_.reset head_number_:%d, head_size_:%d \r\n",head_number_, head_size_);
      ft_dispatcher_fp16_.reset(new fastertransformer::FusedMHARunnerFP16v2(head_number_, head_size_, sm, 1.0f));
      ft_dispatcher_fp16_num_head_ = head_number_;
    }
    int S;
    S = ft_dispatcher_fp16_->getSFromMaxSeqLen(seq_len);
    // printf("@@@ ft S %d \r\n",S);
    framework::Tensor temp_qk_bias_tensor;
    temp_qk_bias_tensor.Resize({head_number_,S*S/64,64});
    auto * temp_qk_bias_data = reinterpret_cast<half *>(temp_qk_bias_tensor.mutable_data<int16_t>(
                                                              platform::CUDAPlace(device_id)));
    int window_num = input_desc[0].dims.d[0];
    framework::Tensor temp_qk_bias_mask_tensor;

    // BxSx3xNxH 
    const half *input0_data = static_cast<const half *>(inputs[0]); //qkv

    // const dim3 grid_t_ftmha(seq_len, batch, 3);
    // const dim3 block_t_ftmha(head_size_, head_number_, 1);
    // transpose_qkv_for_ftmha<half><<<grid_t_ftmha,block_t_ftmha,0,stream>>>(
    //   input0_data,
    //   tptr,
    //   batch,
    //   seq_len,
    //   head_number_,
    //   head_size_
    // );

    // shared transpose
    int head_num_in_block=std::min(head_number_, static_cast<int>(std::floor(1024.0/head_size_/3)));
    int head_num_in_grid = static_cast<int>(std::ceil((float)head_number_/(float)head_num_in_block));
    head_num_in_block=std::ceil((float)head_number_/head_num_in_grid);
    const dim3 grid_t_ftmha_shared(seq_len, batch, head_num_in_grid);
    const dim3 block_t_ftmha_shared(head_size_, 3, head_num_in_block);
    // printf("@@@@ shared head number: %d, in grid: %d, in block: %d 1024/headnum/3 : %d \r\n", head_number_ ,head_num_in_grid,head_num_in_block,static_cast<int>(std::floor(1024.0/head_size_/3)));
    // printf("@@@ grid_t_ftmha_shared %d, %d, %d \r\n", seq_len, batch,head_num_in_grid);
    // printf("@@@ block_t_ftmha_shared %d, %d, %d \r\n",head_size_,3,head_num_in_block);
    transpose_qkv_for_ftmha_shared<half><<<grid_t_ftmha_shared, block_t_ftmha_shared, 0 , stream>>>(
      input0_data,
      tptr,
      batch,
      seq_len,
      head_number_,
      head_size_
    );
    // cudaDeviceSynchronize();
    // if(window_num==64){
    //   cudaDeviceSynchronize();
    //   print_float<half><<<1,1>>>(input0_data,0,2*seq_len*3*head_number_*head_size_,3*head_number_*head_size_,1);
    //   cudaDeviceSynchronize();
    // }

    const half *input1_data = static_cast<const half *>(inputs[1]); //relative pos
    VLOG(1)<<"@@@ invokeTransformMask(temp_qk_bias_data,input1_data ";
    fastertransformer::invokeTransformMask(temp_qk_bias_data,input1_data,head_number_,seq_len,stream);

    const half *input2_data = nullptr;
    half * temp_qk_bias_mask_data = nullptr;
    if (has_biasqk_mask_){
      // printf("@@@ has biasqk mask \r\n");
      VLOG(1)<<"@@@ invokeTransformMask(temp_qk_bias_data,input2_data";
      window_num = input_desc[2].dims.d[0];
      input2_data = static_cast<const half *>(inputs[2]); //mask
      temp_qk_bias_mask_tensor.Resize({window_num,S*S/64,64});
      temp_qk_bias_mask_data = reinterpret_cast<half *>(
          temp_qk_bias_mask_tensor.mutable_data<int16_t>(platform::CUDAPlace(device_id)));
      // printf("@@@ input 2 (qkbias mask) \r\n");
      // if(window_num==64){
      //   cudaDeviceSynchronize();
      //   print_float<half><<<1,1>>>(input2_data,
      //   0,S*S,
      //   64,
      //   1);
      //   cudaDeviceSynchronize();
      //   print_float<half><<<1,1>>>(input2_data,
      //   S*S,2*S*S,
      //   64,
      //   1);
      // }
      fastertransformer::invokeTransformMask(temp_qk_bias_mask_data,input2_data,window_num,seq_len,stream);
      // printf("@@@ temp_qk_bias_mask_data \r\n");
      // if(window_num==64){
      //   cudaDeviceSynchronize();
      //   print_float<half><<<1,1>>>(temp_qk_bias_mask_data,
      //       0,
      //       S*S,
      //       64,
      //       1);
      //   cudaDeviceSynchronize();
      //   print_float<half><<<1,1>>>(temp_qk_bias_mask_data,
      //       S*S,
      //       2*S*S,
      //       64,
      //       1);
      // }
    }
    // printf("@@@ ft_dispatcher_fp16_ setup, S:%d, Batch: %d, window_num: %d \r\n",
      // S,batch,window_num);
    ft_dispatcher_fp16_->setup(S,batch,window_num);
    half *output = static_cast<half *>(outputs[0]);

    // if(window_num == 64){
    //     printf("@before run \r\n");
    //     printf("@ q_buf \r\n");
    //     cudaDeviceSynchronize();
    //     print_float<half><<<1,1>>>(tptr, 0, batch*seq_len*3*head_number_*head_size_,head_size_,1);
    //     cudaDeviceSynchronize();
    //     // if(temp_qk_bias_mask_data!=nullptr){
    //     //     printf("@ trt_attention_mask \r\n");
    //     //     print_float<half><<<1,1>>>(temp_qk_bias_mask_data,0,window_num*seq_len*seq_len,seq_len,1);
    //     //     cudaDeviceSynchronize();
    //     // }
    //     // printf("@ trt_relative_position_bias_ \r\n");
    //     // print_float<half><<<1,1>>>(temp_qk_bias_data,0,head_number_*seq_len*seq_len,seq_len,1);
    //     // cudaDeviceSynchronize();
    // }


    ft_dispatcher_fp16_->run(
      tptr, 
      temp_qk_bias_mask_data,
      temp_qk_bias_data,
      seq_len,
      nullptr,
      output,
      stream);
    // printf("@@@ output after run \r\n");
    // cudaDeviceSynchronize();
    // if(window_num==64){
    //   print_float<half><<<1,1>>>(output,0,seq_len*head_size_,head_size_,1);
    // }
    // cudaDeviceSynchronize();
    int grid = batch * head_number_ * seq_len;
    int block = head_size_;

    // transpose<half><<<grid, block, 0, stream>>>(
    //     qkptr, output, batch, seq_len, head_number_, head_size_);

#endif //TRT_FT_WINDOWS_ATTENTION
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

