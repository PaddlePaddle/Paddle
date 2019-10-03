// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include <cuda_runtime.h>
#include <paddle/fluid/platform/device_context.h>
#include <algorithm>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/memory/malloc.h"
#include "paddle/fluid/operators/detail/safe_ref.h"
#include "paddle/fluid/operators/math/blas.h"

namespace paddle {
namespace operators {

#define FINAL_MASK 0xffffffff
#define HALF_WARP 16
#define WARP_SIZE 32

template <typename T>
__inline__ __device__ T warpReduceSum(T val) {
  for (int mask = HALF_WARP; mask > 0; mask >>= 1)
#if __CUDA_ARCH__ >= 350 && CUDA_VERSION >= 9000
    val += __shfl_xor_sync(FINAL_MASK, val, mask, warpSize);
#else
    val += __shfl_xor(val, mask, warpSize);
#endif
  return val;
}

/* Calculate the sum of all elements in a block */
template <typename T>
__inline__ __device__ T blockReduceSum(T val) {
  static __shared__ T shared[WARP_SIZE];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  val = warpReduceSum<T>(val);

  if (lane == 0) shared[wid] = val;

  __syncthreads();

  val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : (T)(0.0f);
  val = warpReduceSum<T>(val);

  return val;
}

template <typename T>
__inline__ __device__ T warpReduceMax(T val) {
  for (int mask = HALF_WARP; mask > 0; mask >>= 1)
#if __CUDA_ARCH__ >= 350 && CUDA_VERSION >= 9000
    val = max(val, __shfl_xor_sync(FINAL_MASK, val, mask, warpSize));
#else
    val = max(val, __shfl_xor(val, mask, warpSize));
#endif
  return val;
}

/* Calculate the maximum of all elements in a block */
template <typename T>
__inline__ __device__ T blockReduceMax(T val) {
  static __shared__ T shared[WARP_SIZE];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  val = warpReduceMax(val);

  if (lane == 0) shared[wid] = val;

  __syncthreads();

  val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : -1e10f;
  val = warpReduceMax(val);

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
                                           const int seq_len) {
  int seq_id = blockIdx.x % seq_len;
  int qk_offset = blockIdx.x * seq_len;
  int bias_offset = blockIdx.x % (head_num * seq_len) * seq_len;

  __shared__ float s_sum, s_max;

  float qk = threadIdx.x < seq_len
                 ? static_cast<float>((qk_buf_[threadIdx.x + qk_offset] +
                                       bias_qk_[threadIdx.x + bias_offset]))
                 : 0.0f;
  float tmp = threadIdx.x < seq_len ? static_cast<float>(qk) : -1e20f;
  float max_val = blockReduceMax<float>(tmp);
  if (threadIdx.x == 0) s_max = max_val;
  __syncthreads();

  float qk_tmp =
      threadIdx.x < seq_len ? __expf(static_cast<float>(tmp - s_max)) : 0.0f;
  float sum_val = blockReduceSum<float>(qk_tmp);

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

  auto blas = math::GetBlas<platform::CUDADeviceContext, T>(context);
  auto stream = context.stream();

  blas.BatchedGEMM(transA, transB, seq_len, seq_len, size_per_head, alpha,
                   q_buf_, k_buf_, beta, qk_buf_, batch_size * head_num,
                   seq_len * size_per_head, seq_len * size_per_head);

  int m = batch_size * head_num * seq_len;
  int k = seq_len;

  int grid = m;
  int block = k;

  softmax_kernel_with_eltadd<T><<<grid, block, 0, stream>>>(
      qk_buf_, bias_qk, batch_size, head_num, seq_len);
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

// Compute QK*V->transpose
template <typename T>
void MatMulWithHeadQKV(const platform::CUDADeviceContext &context, int head_num,
                       int seq_len, int size_per_head, int batch_size,
                       bool qk_trans, bool v_trans, T *v_buf_, const T *qk_buf_,
                       T *dst, T *out, T alpha, T beta) {
  int m = batch_size * seq_len;
  int k = head_num * size_per_head;

  auto blas = math::GetBlas<platform::CUDADeviceContext, T>(context);
  auto stream = context.stream();
  CBLAS_TRANSPOSE transA = !qk_trans ? CblasNoTrans : CblasTrans;
  CBLAS_TRANSPOSE transB = !v_trans ? CblasNoTrans : CblasTrans;

  blas.BatchedGEMM(transA, transB, seq_len, size_per_head, seq_len, alpha,
                   qk_buf_, v_buf_, beta, dst, batch_size * head_num,
                   seq_len * seq_len, seq_len * size_per_head);

  int grid = batch_size * head_num * seq_len;
  int block = size_per_head;
  transpose<T><<<grid, block, 0, stream>>>(dst, out, batch_size, seq_len,
                                           head_num, size_per_head);
}

template <typename T>
void MultiHeadGPUCompute(const platform::CUDADeviceContext &dev_ctx,
                         int head_num, const framework::DDim &mat_q,
                         const framework::DDim &mat_k,
                         const framework::DDim &mat_v, const T *Q, const T *K,
                         const T *V, const T *bias_q, const T *bias_k,
                         const T *bias_v, const T *bias_qk, T *out, T alpha,
                         T beta, bool trans_q, bool trans_k, bool trans_v) {
  int seq_len = mat_q[1];
  int size_per_head = (mat_q[2] / head_num);
  int batch_size = mat_q[0];
  int buf_size = batch_size * head_num * seq_len * size_per_head;
  int qk_buf_size = batch_size * head_num * seq_len * seq_len;

  auto alloc_buf =
      memory::Alloc(dev_ctx, (buf_size * 4 + qk_buf_size) * sizeof(T));

  T *buf = reinterpret_cast<T *>(alloc_buf->ptr());
  T *q_buf = buf;
  T *k_buf = buf + buf_size;
  T *v_buf = buf + 2 * buf_size;
  T *qk_buf = buf + 3 * buf_size;
  T *dst_buf = buf + 3 * buf_size + qk_buf_size;

  int m = batch_size * seq_len;
  int k = head_num * size_per_head;

  // Each block process head*size-per_head element,
  // have m lines. bias is m lines
  auto blas = math::GetBlas<platform::CUDADeviceContext, T>(dev_ctx);
  auto stream = dev_ctx.stream();

  int grid = m;
  PADDLE_ENFORCE_LT(k, 1024,
                    "Input head_number * size_per_head should <= 1024");
  int block = k <= 1024 ? k : 1024;
  add_QKV<T><<<grid, block, 0, stream>>>(Q, K, V, q_buf, k_buf, v_buf, bias_q,
                                         bias_k, bias_v, batch_size, seq_len,
                                         head_num, size_per_head);

  MatMulWithHeadQK<T>(dev_ctx, head_num, seq_len, size_per_head, batch_size,
                      trans_q, trans_k, q_buf, k_buf, qk_buf, bias_qk, alpha,
                      beta);
  MatMulWithHeadQKV<T>(dev_ctx, head_num, seq_len, size_per_head, batch_size,
                       false, trans_v, v_buf, qk_buf, dst_buf, out, T(1.0),
                       beta);
}

template <typename DeviceContext, typename T>
class MultiHeadMatMulKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto *q = context.Input<framework::Tensor>("Q");
    auto *k = context.Input<framework::Tensor>("K");
    auto *v = context.Input<framework::Tensor>("V");

    auto &bias_q = detail::Ref(context.Input<framework::Tensor>("BiasQ"),
                               "Cannot find BiasQ");
    auto &bias_k = detail::Ref(context.Input<framework::Tensor>("BiasK"),
                               "Cannot find BiasK");
    auto &bias_v = detail::Ref(context.Input<framework::Tensor>("BiasV"),
                               "Cannot find BiasV");

    auto &bias_qk = detail::Ref(context.Input<framework::Tensor>("BiasQK"),
                                "Cannot find QK");

    auto *out = context.Output<framework::Tensor>("Out");
    out->mutable_data<T>(context.GetPlace());

    T scale = static_cast<T>(context.Attr<float>("alpha"));
    bool transpose_q = context.Attr<bool>("transpose_Q");
    bool transpose_k = context.Attr<bool>("transpose_K");
    bool transpose_v = context.Attr<bool>("transpose_V");

    int head_number = context.Attr<int>("head_number");
    // compute q*k with eltadd
    auto &device_ctx = context.template device_context<DeviceContext>();

    MultiHeadGPUCompute<T>(device_ctx, head_number, q->dims(), k->dims(),
                           v->dims(), q->data<T>(), k->data<T>(), v->data<T>(),
                           bias_q.data<T>(), bias_k.data<T>(), bias_v.data<T>(),
                           bias_qk.data<T>(), out->data<T>(), scale, T(0.0),
                           transpose_q, transpose_k, transpose_v);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    multihead_matmul,
    ops::MultiHeadMatMulKernel<paddle::platform::CUDADeviceContext, float>,
    ops::MultiHeadMatMulKernel<paddle::platform::CUDADeviceContext, double>);
