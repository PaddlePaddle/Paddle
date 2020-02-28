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
#include <vector>
#include "glog/logging.h"
#include "paddle/fluid/inference/tensorrt/plugin/multihead_matmul_op_plugin.h"
#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin_factory.h"
#include "paddle/fluid/operators/math/blas.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

#define FINAL_MASK 0xffffffff
#define HALF_WARP 16
#define WARP_SIZE 32

MultiheadMatmulPlugin *CreateMultiheadMatmulPluginDeserialize(
    const void *buffer, size_t length) {
  return new MultiheadMatmulPlugin(buffer, length);
}
REGISTER_TRT_PLUGIN("multihead_matmul_plugin",
                    CreateMultiheadMatmulPluginDeserialize);

int MultiheadMatmulPlugin::initialize() { return 0; }

nvinfer1::Dims MultiheadMatmulPlugin::getOutputDimensions(
    int index, const nvinfer1::Dims *inputDims, int nbInputs) {
  assert(index < this->getNbOutputs());
  nvinfer1::Dims const &input_dims = inputDims[0];
  nvinfer1::Dims output_dims = input_dims;
  return output_dims;
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

void TransQKVWithBias(const int batch, const int seq_len, const int head_size,
                      const int head_num, const float *input, float *output,
                      cudaStream_t stream) {
  // BxSx3xNxH + 3xNxH -> 3xBxNxSxH
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
__global__ void add_QKV(const T *Q, const T *K, const T *V, T *q_buf_,
                        T *k_buf_, T *v_buf_, const T *bias_q, const T *bias_k,
                        const T *bias_v, int batch_size, int seq_len,
                        int head_num, int size_per_head) {
  const T *data_ptr_q, *data_ptr_k, *data_ptr_v;
  const T *bias_ptr_q, *bias_ptr_k, *bias_ptr_v;

  int m = batch_size * seq_len;
  int n = head_num * size_per_head;

  int row_offset = (blockIdx.x % m) * n;

  data_ptr_q = Q + row_offset;
  data_ptr_k = K + row_offset;
  data_ptr_v = V + row_offset;
  // bias ptr
  bias_ptr_q = bias_q;
  bias_ptr_k = bias_k;
  bias_ptr_v = bias_v;

  int batch_id = (blockIdx.x % m) / seq_len;
  int head_id = threadIdx.x / size_per_head;
  int id_in_head = threadIdx.x % size_per_head;
  int word_start_id = (blockIdx.x) % seq_len;

#if __CUDA_ARCH__ >= 350
  T tmp_q = __ldg(&data_ptr_q[threadIdx.x]) + __ldg(&bias_ptr_q[threadIdx.x]);
  T tmp_k = __ldg(&data_ptr_k[threadIdx.x]) + __ldg(&bias_ptr_k[threadIdx.x]);
  T tmp_v = __ldg(&data_ptr_v[threadIdx.x]) + __ldg(&bias_ptr_v[threadIdx.x]);
#else
  T tmp_q = data_ptr_q[threadIdx.x] + bias_ptr_q[threadIdx.x];
  T tmp_k = data_ptr_k[threadIdx.x] + bias_ptr_k[threadIdx.x];
  T tmp_v = data_ptr_v[threadIdx.x] + bias_ptr_v[threadIdx.x];
#endif

  int target_id = batch_id * (seq_len * head_num * size_per_head) +
                  head_id * seq_len * size_per_head +
                  word_start_id * size_per_head + id_in_head;

  q_buf_[target_id] = tmp_q;
  k_buf_[target_id] = tmp_k;
  v_buf_[target_id] = tmp_v;
}

// Keep to compare performance
template <typename T>
__global__ void add_QKV_V2(const T *Q, const T *K, const T *V, T *q_buf_,
                           T *k_buf_, T *v_buf_, const T *bias_Q,
                           const T *bias_K, const T *bias_V, int batch_size,
                           int seq_len, int head_num, int size_per_head,
                           const int word_per_block) {
  const T *data_ptr;
  T *buf_ptr;
  const T *bias_ptr;

  int m = batch_size * seq_len;
  int n = head_num * size_per_head;

  int qkv_id = blockIdx.x * word_per_block / m;
  int row_offset = (blockIdx.x * word_per_block % m) * n;

  if (qkv_id == 0) {
    data_ptr = Q + row_offset;
    buf_ptr = q_buf_;
    bias_ptr = bias_Q;
  } else if (qkv_id == 1) {
    data_ptr = K + row_offset;
    buf_ptr = k_buf_;
    bias_ptr = bias_K;
  } else {
    data_ptr = V + row_offset;
    buf_ptr = v_buf_;
    bias_ptr = bias_V;
  }

  int batch_id = (blockIdx.x * word_per_block % m) / seq_len;
  int head_id = threadIdx.x / size_per_head;
  int id_in_head = threadIdx.x % size_per_head;
  int word_start_id = (blockIdx.x * word_per_block) % seq_len;

#if __CUDA_ARCH__ >= 350
  T bias = __ldg(&bias_ptr[threadIdx.x]);
#else
  T bias = bias_ptr[threadIdx.x];
#endif

  for (int i = word_start_id; i < word_start_id + word_per_block; ++i) {
    T tmp = data_ptr[threadIdx.x] + bias;

    int target_id = batch_id * (seq_len * head_num * size_per_head) +
                    head_id * seq_len * size_per_head + i * size_per_head +
                    id_in_head;

    buf_ptr[target_id] = tmp;
    data_ptr += n;
  }
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

// For verify result
template <typename T>
__global__ void elt_qk_add(const T *bias_qk, T *qk_buf, int head_num,
                           int seq_len, int size_per_head, int batch_size) {
  int m = batch_size * head_num * seq_len;
  int row_id = blockIdx.x % m;
  int dst_id = row_id * seq_len + threadIdx.x;
  const T *bias_ptr = bias_qk;
#if __CUDA_ARCH__ >= 350
  int tmp_bias = __ldg(&bias_ptr[dst_id]);
#else
  int tmp_bias = bias_ptr[dst_id];
#endif

  qk_buf[dst_id] += tmp_bias;
}

// Compute Q*K->softmax->eltadd
template <typename T>
void MatMulWithHeadQK(const platform::CUDADeviceContext &context, int head_num,
                      int seq_len, int size_per_head, int batch_size,
                      bool q_trans, bool k_trans, T *q_buf_, T *k_buf_,
                      T *qk_buf_, const T *bias_qk, T alpha, T beta) {
  CBLAS_TRANSPOSE transA = !q_trans ? CblasNoTrans : CblasTrans;
  CBLAS_TRANSPOSE transB = !k_trans ? CblasNoTrans : CblasTrans;

  auto blas =
      paddle::operators::math::GetBlas<platform::CUDADeviceContext, T>(context);
  auto stream = context.stream();

  blas.BatchedGEMM(transA, transB, seq_len, seq_len, size_per_head, alpha,
                   q_buf_, k_buf_, beta, qk_buf_, batch_size * head_num,
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

  softmax_kernel_with_eltadd<T><<<grid, block, 0, stream>>>(
      qk_buf_, bias_qk, batch_size, head_num, seq_len, FINAL_MASK);
}

// Compute QK*V->transpose
template <typename T>
void MatMulWithHeadQKV(const platform::CUDADeviceContext &context, int head_num,
                       int seq_len, int size_per_head, int batch_size,
                       bool qk_trans, bool v_trans, T *v_buf_, const T *qk_buf_,
                       T *dst, T alpha, T beta) {
  int m = batch_size * seq_len;
  int k = head_num * size_per_head;

  auto blas =
      paddle::operators::math::GetBlas<platform::CUDADeviceContext, T>(context);
  auto stream = context.stream();
  CBLAS_TRANSPOSE transA = !qk_trans ? CblasNoTrans : CblasTrans;
  CBLAS_TRANSPOSE transB = !v_trans ? CblasNoTrans : CblasTrans;

  blas.BatchedGEMM(transA, transB, seq_len, size_per_head, seq_len, alpha,
                   qk_buf_, v_buf_, beta, dst, batch_size * head_num,
                   seq_len * seq_len, seq_len * size_per_head);
}

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

template <typename T>
void MultiHeadGPUComputeV2(int batch, int seq_len, int head_num, int head_size,
                           T *qkptr, const T *bias_qk_ptr, T *tptr, T alpha,
                           T beta) {
  const int tsize = batch * head_num * seq_len * head_size;

  T *qptr = tptr;
  T *kptr = qptr + tsize;
  T *vptr = kptr + tsize;

  int device_id;
  cudaGetDevice(&device_id);
  auto gpu_place = new platform::CUDAPlace(device_id);
  platform::CUDADeviceContext gpu_ctx(*gpu_place);

  // batch gemm stride, softmaxwithscale.
  MatMulWithHeadQK<T>(gpu_ctx, head_num, seq_len, head_size, batch, false, true,
                      qptr, kptr, qkptr, bias_qk_ptr, alpha, beta);
  // batch gemm stride, transpose.
  MatMulWithHeadQKV<T>(gpu_ctx, head_num, seq_len, head_size, batch, false,
                       false, vptr, qkptr, tptr, T(1.0), beta);
}

int MultiheadMatmulPlugin::enqueue(int batch_size, const void *const *inputs,
                                   void **outputs, void *workspace,
                                   cudaStream_t stream) {
  const auto &input_dims = this->getInputDims(0);
  const float *input = reinterpret_cast<const float *>(inputs[0]);
  const float *bias_qk = reinterpret_cast<const float *>(inputs[1]);
  float *output = reinterpret_cast<float **>(outputs)[0];
  typedef float T;

  framework::Tensor multihead_temp_tensor;
  int scratch_size = batch_size * head_number_ * seq_len_ * seq_len_ * 1;
  int input_size = batch_size * seq_len_ * 3 * head_number_ * size_per_head_;
  multihead_temp_tensor.Resize({scratch_size + input_size});
  int device_id;
  cudaGetDevice(&device_id);
  auto *multihead_temp_data =
      multihead_temp_tensor.mutable_data<T>(platform::CUDAPlace(device_id));
  auto *qkptr = multihead_temp_data;
  auto *tptr = multihead_temp_data + scratch_size;

  TransQKVWithBias(batch_size, seq_len_, size_per_head_, head_number_, input,
                   tptr, stream);

  MultiHeadGPUComputeV2<T>(batch_size, seq_len_, head_number_, size_per_head_,
                           qkptr, bias_qk, tptr, alpha_, T(0.0));

  int grid = batch_size * head_number_ * seq_len_;
  int block = size_per_head_;
  transpose<T><<<grid, block, 0, stream>>>(tptr, output, batch_size, seq_len_,
                                           head_number_, size_per_head_);
}

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
