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
#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin_factory.h"
#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin_utils.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

#define FINAL_MASK 0xffffffff
#define HALF_WARP 16
#define WARP_SIZE 32

template <typename T>
__inline__ __device__ T warpReduceSum(T val, unsigned lane_mask) {
  for (int mask = HALF_WARP; mask > 0; mask >>= 1)
#if __CUDA_ARCH__ >= 350 && CUDA_VERSION >= 9000
    val += __shfl_xor_sync(lane_mask, val, mask, warpSize);
#else
    val += __shfl_xor(val, mask, warpSize);
#endif
  return val;
}

/* Calculate the sum of all elements in a block */
template <typename T>
__inline__ __device__ T blockReduceSum(T val, unsigned mask) {
  static __shared__ T shared[WARP_SIZE];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  val = warpReduceSum<T>(val, mask);

  if (lane == 0) shared[wid] = val;

  __syncthreads();

  // align block_span to warpSize
  int block_span = (blockDim.x + warpSize - 1) >> 5;
  val = (threadIdx.x < block_span) ? shared[lane] : static_cast<T>(0.0f);
  val = warpReduceSum<T>(val, mask);

  return val;
}

template <typename T>
__inline__ __device__ T warpReduceMax(T val, unsigned lane_mask) {
  for (int mask = HALF_WARP; mask > 0; mask >>= 1)
#if __CUDA_ARCH__ >= 350 && CUDA_VERSION >= 9000
    val = max(val, __shfl_xor_sync(lane_mask, val, mask, warpSize));
#else
    val = max(val, __shfl_xor(val, mask, warpSize));
#endif
  return val;
}

/* Calculate the maximum of all elements in a block */
template <typename T>
__inline__ __device__ T blockReduceMax(T val, unsigned mask) {
  static __shared__ T shared[WARP_SIZE];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  val = warpReduceMax(val, mask);

  if (lane == 0) shared[wid] = val;

  __syncthreads();

  // align block_span to warpSize
  int block_span = (blockDim.x + warpSize - 1) >> 5;
  val = (threadIdx.x < block_span) ? shared[lane] : -1e10f;
  val = warpReduceMax(val, mask);

  return val;
}

template <typename T>
__global__ void softmax_kernel_with_eltadd(T *qk_buf_, const T *bias_qk_,
                                           const int batch_size,
                                           const int head_num,
                                           const int seq_len,
                                           const unsigned mask) {
  int qk_offset = blockIdx.x * seq_len;
  assert(blockDim.x % 32 == 0);

  __shared__ float s_sum, s_max;

  float qk = threadIdx.x < seq_len
                 ? static_cast<float>((qk_buf_[threadIdx.x + qk_offset] +
                                       bias_qk_[threadIdx.x + qk_offset]))
                 : 0.0f;
  float tmp = threadIdx.x < seq_len ? static_cast<float>(qk) : -1e20f;

  float max_val = blockReduceMax<float>(tmp, mask);

  if (threadIdx.x == 0) s_max = max_val;
  __syncthreads();

  float qk_tmp =
      threadIdx.x < seq_len ? __expf(static_cast<float>(tmp - s_max)) : 0.0f;
  float sum_val = blockReduceSum<float>(qk_tmp, mask);

  if (threadIdx.x == 0) {
    s_sum = sum_val + 1e-6f;
  }
  __syncthreads();

  if (threadIdx.x < seq_len)
    qk_buf_[threadIdx.x + qk_offset] = (T)(qk_tmp / s_sum);
}

// Dynamic Plugin below.
#if IS_TRT_VERSION_GE(6000)

template <typename T>
__device__ T exp_func(T a);

template <>
__device__ float exp_func<float>(float a) {
  return expf(a);
}

#if __CUDA_ARCH__ >= 600
template <>
__device__ half exp_func<half>(half a) {
  return hexp(a);
}
#endif

template <typename T>
__global__ void transpose(T *src, T *dst, const int batch_size,
                          const int seq_len, const int head_num,
                          const int size_per_head) {
  int batch_id = blockIdx.x / (head_num * seq_len);
  int seq_id = blockIdx.x % seq_len;
  int head_id = (blockIdx.x % (head_num * seq_len)) / seq_len;
  dst[batch_id * (head_num * seq_len * size_per_head) +
      seq_id * head_num * size_per_head + head_id * size_per_head +
      threadIdx.x] = src[blockIdx.x * size_per_head + threadIdx.x];
}

template <typename T, unsigned BlockDim>
__global__ void softmax_kernel_with_eltadd_kernel(const int seq_len, T *qk_buf_,
                                                  const T *bias_qk_) {
  int qk_offset = blockIdx.x * seq_len;
  using BlockReduce = cub::BlockReduce<T, BlockDim>;
  __shared__ typename BlockReduce::TempStorage tmp_storage;
  __shared__ T block_sum;
  cub::Sum sum;

  T s_sum = T(0.0);
  for (int it = threadIdx.x; it < seq_len; it += BlockDim) {
    qk_buf_[qk_offset + it] =
        exp_func(qk_buf_[qk_offset + it] + bias_qk_[qk_offset + it]);
    s_sum += qk_buf_[qk_offset + it];
  }

  const auto tmp_sum = BlockReduce(tmp_storage).Reduce(s_sum, sum);
  if (threadIdx.x == 0) {
    block_sum = tmp_sum + T(1e-6);
  }
  __syncthreads();

  for (int it = threadIdx.x; it < seq_len; it += BlockDim) {
    qk_buf_[qk_offset + it] = qk_buf_[qk_offset + it] / block_sum;
  }
}

template <typename T>
__global__ void transpose_qkv_kernel(const int H, const T *input, T *output) {
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

inline void TransposeQKV(const int batch, const int seq_len,
                         const int head_size, const int head_num,
                         const float *input, float *output,
                         cudaStream_t stream) {
  const dim3 grid(seq_len, batch, 3);
  if (head_size % 4 == 0) {
    const int h = head_size / 4;
    const float4 *input4 = reinterpret_cast<const float4 *>(input);
    float4 *output4 = reinterpret_cast<float4 *>(output);
    const dim3 block(h, head_num, 1);
    // limit h * head_num to max block size(1024).
    PADDLE_ENFORCE_LE(h * head_num, 1024,
                      platform::errors::InvalidArgument(
                          "head_num (%d) * head_size (%d) should <= %d",
                          head_num, head_size, 1024 * 4));
    transpose_qkv_kernel<float4><<<grid, block, 0, stream>>>(h, input4,
                                                             output4);
  } else if (head_size % 2 == 0) {
    const int h = head_size / 2;
    const float2 *input2 = reinterpret_cast<const float2 *>(input);
    float2 *output2 = reinterpret_cast<float2 *>(output);
    const dim3 block(h, head_num, 1);
    // limit h * head_num to max block size(1024).
    PADDLE_ENFORCE_LE(h * head_num, 1024,
                      platform::errors::InvalidArgument(
                          "head_num (%d) * head_size (%d) should <= %d",
                          head_num, head_size, 1024 * 2));
    transpose_qkv_kernel<float2><<<grid, block, 0, stream>>>(h, input2,
                                                             output2);
  } else {
    const dim3 block(head_size, head_num, 1);
    // limit head_size * head_num to max block size(1024).
    PADDLE_ENFORCE_LE(head_size * head_num, 1024,
                      platform::errors::InvalidArgument(
                          "head_num (%d) * head_size (%d) should <= %d",
                          head_num, head_size, 1024));
    transpose_qkv_kernel<float><<<grid, block, 0, stream>>>(head_size, input,
                                                            output);
  }
}

inline void TransposeQKV(const int batch, const int seq_len,
                         const int head_size, const int head_num,
                         const half *input, half *output, cudaStream_t stream) {
  const dim3 grid(seq_len, batch, 3);
  if (head_size % 2 == 0) {
    const int h = head_size / 2;
    const half2 *input2 = reinterpret_cast<const half2 *>(input);
    half2 *output2 = reinterpret_cast<half2 *>(output);
    const dim3 block(h, head_num, 1);
    // limit h * head_num to max block size(1024).
    PADDLE_ENFORCE_LE(h * head_num, 1024,
                      platform::errors::InvalidArgument(
                          "head_num (%d) * head_size (%d) should <= %d",
                          head_num, head_size, 1024 * 2));
    transpose_qkv_kernel<half2><<<grid, block, 0, stream>>>(h, input2, output2);
  } else {
    const dim3 block(head_size, head_num, 1);
    // limit head_size * head_num to max block size(1024).
    PADDLE_ENFORCE_LE(head_size * head_num, 1024,
                      platform::errors::InvalidArgument(
                          "head_num (%d) * head_size (%d) should <= %d",
                          head_num, head_size, 1024));
    transpose_qkv_kernel<half><<<grid, block, 0, stream>>>(head_size, input,
                                                           output);
  }
}

// Compute Q*K->softmax->eltadd
template <typename T>
inline void MatMulWithHeadQK(const platform::CUDADeviceContext &context,
                             int head_num, int seq_len, int size_per_head,
                             int batch_size, bool q_trans, bool k_trans,
                             T *q_buf_, T *k_buf_, T *qk_buf_, const T *bias_qk,
                             T alpha, T beta) {
  CBLAS_TRANSPOSE transA = !q_trans ? CblasNoTrans : CblasTrans;
  CBLAS_TRANSPOSE transB = !k_trans ? CblasNoTrans : CblasTrans;

  typedef typename PluginTypeTraits<T>::TYPE run_type;
  auto blas =
      operators::math::GetBlas<platform::CUDADeviceContext, run_type>(context);
  auto stream = context.stream();

  blas.BatchedGEMM(
      transA, transB, seq_len, seq_len, size_per_head,
      static_cast<run_type>(alpha), reinterpret_cast<run_type *>(q_buf_),
      reinterpret_cast<run_type *>(k_buf_), static_cast<run_type>(beta),
      reinterpret_cast<run_type *>(qk_buf_), batch_size * head_num,
      seq_len * size_per_head, seq_len * size_per_head);

  int grid = batch_size * head_num * seq_len;
  int block = seq_len;

  // Align block to 32, also limit seq_len to max block size.
  PADDLE_ENFORCE_LE(seq_len, 1024, platform::errors::InvalidArgument(
                                       "seq_len should <= 1024, "
                                       "but received seq_len is:%d",
                                       seq_len));
  if (seq_len <= 32)
    block = 32;
  else if (seq_len > 32 && seq_len <= 64)
    block = 64;
  else if (seq_len > 64 && seq_len <= 128)
    block = 128;
  else if (seq_len > 128 && seq_len <= 256)
    block = 256;
  else if (seq_len > 256 && seq_len <= 512)
    block = 512;
  else
    block = 1024;

  const int threads = 256;
  softmax_kernel_with_eltadd<T><<<grid, block, 0, stream>>>(
      qk_buf_, bias_qk, batch_size, head_num, seq_len, FINAL_MASK);
}

// Compute QK*V->transpose
template <typename T>
inline void MatMulWithHeadQKV(const platform::CUDADeviceContext &context,
                              int head_num, int seq_len, int size_per_head,
                              int batch_size, bool qk_trans, bool v_trans,
                              T *v_buf_, const T *qk_buf_, T *dst, T alpha,
                              T beta) {
  int m = batch_size * seq_len;
  int k = head_num * size_per_head;

  typedef typename PluginTypeTraits<T>::TYPE run_type;
  auto blas =
      operators::math::GetBlas<platform::CUDADeviceContext, run_type>(context);
  auto stream = context.stream();
  CBLAS_TRANSPOSE transA = !qk_trans ? CblasNoTrans : CblasTrans;
  CBLAS_TRANSPOSE transB = !v_trans ? CblasNoTrans : CblasTrans;

  blas.BatchedGEMM(
      transA, transB, seq_len, size_per_head, seq_len,
      static_cast<run_type>(alpha), reinterpret_cast<const run_type *>(qk_buf_),
      reinterpret_cast<run_type *>(v_buf_), static_cast<run_type>(beta),
      reinterpret_cast<run_type *>(dst), batch_size * head_num,
      seq_len * seq_len, seq_len * size_per_head);
}

template <typename T>
inline void MultiHeadGPUComputeV2(const platform::CUDADeviceContext &dev_ctx,
                                  int batch, int seq_len, int head_num,
                                  int head_size, T *qkptr, const T *bias_qk_ptr,
                                  T *tptr, T alpha, T beta) {
  auto stream = dev_ctx.stream();
  const int tsize = batch * head_num * seq_len * head_size;

  T *qptr = tptr;
  T *kptr = qptr + tsize;
  T *vptr = kptr + tsize;
  // batch gemm stride, softmaxwithscale.
  MatMulWithHeadQK<T>(dev_ctx, head_num, seq_len, head_size, batch, false, true,
                      qptr, kptr, qkptr, bias_qk_ptr, alpha, beta);
  // batch gemm stride, transpose.
  MatMulWithHeadQKV<T>(dev_ctx, head_num, seq_len, head_size, batch, false,
                       false, vptr, qkptr, tptr, T(1.0), beta);
}

int QkvToContextPluginDynamic::initialize() { return 0; }

size_t QkvToContextPluginDynamic::getSerializationSize() const { return 0; }

void QkvToContextPluginDynamic::serialize(void *buffer) const {}

nvinfer1::DimsExprs QkvToContextPluginDynamic::getOutputDimensions(
    int output_index, const nvinfer1::DimsExprs *inputs, int nb_inputs,
    nvinfer1::IExprBuilder &expr_builder) {
  // input[0], (B, S, 3 * N * H, 1, 1)
  // input[1], (B, head_num, seq_len, seq_len)
  // output, (B, seq_len, hidden)
  PADDLE_ENFORCE_EQ(output_index, 0,
                    platform::errors::InvalidArgument(
                        "There is only one output of the EmbEltwiseLayernorm, "
                        "so the index should be zero,"
                        "but it's (%d)",
                        output_index));
  PADDLE_ENFORCE_EQ(
      nb_inputs, 2,
      platform::errors::InvalidArgument(
          "The Input of the EmbEltwiseLayernorm should be 3, but we found "
          "it has (%d) inputs",
          nb_inputs));
  nvinfer1::DimsExprs ret;
  ret.nbDims = 5;
  ret.d[0] = inputs[0].d[0];
  ret.d[1] = inputs[0].d[1];
  ret.d[2] = expr_builder.constant(hidden_);
  ret.d[3] = expr_builder.constant(1);
  ret.d[4] = expr_builder.constant(1);
  return ret;
}

bool QkvToContextPluginDynamic::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc *in_out, int nb_inputs,
    int nb_outputs) {
  PADDLE_ENFORCE_NOT_NULL(
      in_out, platform::errors::InvalidArgument(
                  "The input of swish plugin shoule not be nullptr."));

  PADDLE_ENFORCE_LT(
      pos, nb_inputs + nb_outputs,
      platform::errors::InvalidArgument("The pos(%d) should be less than the "
                                        "num(%d) of the input and the output.",
                                        pos, nb_inputs + nb_outputs));

  const nvinfer1::PluginTensorDesc &in = in_out[pos];
  if (pos == 0) {
#ifdef SUPPORT_CUDA_FP16
    return (in.type == nvinfer1::DataType::kFLOAT ||
            in.type == nvinfer1::DataType::kHALF) &&
           (in.format == nvinfer1::TensorFormat::kLINEAR);
#else
    return (in.type == nvinfer1::DataType::kFLOAT) &&
           (in.format == nvinfer1::TensorFormat::kLINEAR);
#endif
  }
  const nvinfer1::PluginTensorDesc &prev = in_out[pos - 1];

  if (pos == 1) {
    return in.type == prev.type && in.format == prev.format;
  }

  // output
  return in.type == prev.type && in.format == prev.format;
}

nvinfer1::DataType QkvToContextPluginDynamic::getOutputDataType(
    int index, const nvinfer1::DataType *input_types, int nb_inputs) const {
  PADDLE_ENFORCE_EQ(
      index, 0, platform::errors::InvalidArgument(
                    "The EmbEltwiseLayernorm Plugin only has one input, so the "
                    "index value should be 0, but get %d.",
                    index));
  return input_types[0];
}

int QkvToContextPluginDynamic::enqueue(
    const nvinfer1::PluginTensorDesc *input_desc,
    const nvinfer1::PluginTensorDesc *output_desc, const void *const *inputs,
    void *const *outputs, void *workspace, cudaStream_t stream) {
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
  if (input_type == nvinfer1::DataType::kFLOAT) {
    auto *multihead_temp_data = multihead_temp_tensor.mutable_data<float>(
        platform::CUDAPlace(device_id));
    auto *qkptr = multihead_temp_data;
    auto *tptr = multihead_temp_data + scratch_size;

    const float *input0_data = static_cast<const float *>(inputs[0]);
    const float *input1_data = static_cast<const float *>(inputs[1]);
    // BxSx3xNxH => tptr: 3xBxNxSxH.
    TransposeQKV(batch, seq_len, head_size_, head_number_, input0_data, tptr,
                 stream);

    auto *device_ctx = static_cast<platform::CUDADeviceContext *>(
        platform::DeviceContextPool::Instance().Get(
            platform::CUDAPlace(device_id)));

    const platform::CUDADeviceContext &dev_ctx = *device_ctx;
    MultiHeadGPUComputeV2<float>(dev_ctx, batch, seq_len, head_number_,
                                 head_size_, qkptr, input1_data, tptr, scale_,
                                 static_cast<float>(0.0));

    int grid = batch * head_number_ * seq_len;
    int block = head_size_;
    float *output = static_cast<float *>(outputs[0]);
    transpose<float><<<grid, block, 0, stream>>>(tptr, output, batch, seq_len,
                                                 head_number_, head_size_);

  } else if (input_type == nvinfer1::DataType::kHALF) {
#ifdef SUPPORT_CUDA_FP16
    auto *multihead_temp_data =
        multihead_temp_tensor.mutable_data<int16_t>(  // NOLINT
            platform::CUDAPlace(device_id));

    half *qkptr = reinterpret_cast<half *>(multihead_temp_data);
    half *tptr = qkptr + scratch_size;

    const half *input0_data = static_cast<const half *>(inputs[0]);
    const half *input1_data = static_cast<const half *>(inputs[1]);
    // BxSx3xNxH => tptr: 3xBxNxSxH.
    TransposeQKV(batch, seq_len, head_size_, head_number_, input0_data, tptr,
                 stream);

    auto *device_ctx = static_cast<platform::CUDADeviceContext *>(
        platform::DeviceContextPool::Instance().Get(
            platform::CUDAPlace(device_id)));

    const platform::CUDADeviceContext &dev_ctx = *device_ctx;
    MultiHeadGPUComputeV2<half>(dev_ctx, batch, seq_len, head_number_,
                                head_size_, qkptr, input1_data, tptr,
                                half(scale_), half(0.0));

    int grid = batch * head_number_ * seq_len;
    int block = head_size_;
    half *output = static_cast<half *>(outputs[0]);
    transpose<half><<<grid, block, 0, stream>>>(tptr, output, batch, seq_len,
                                                head_number_, head_size_);
#else
    PADDLE_THROW("The cuda arch must greater than 600.");
#endif
  } else {
    PADDLE_THROW("The QKV TRT Plugin's input type should be float or half.");
  }
  return cudaGetLastError() != cudaSuccess;
}
#endif

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
