/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <cuda_runtime.h>
#include <algorithm>
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/operators/math/bert_encoder_functor.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/math/math_cuda_utils.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace operators {
namespace math {

template <typename T, int TPB>
__device__ inline void LayerNormSmall(T val, const kvp<T> &thread_data,
                                      const int ld, const int idx,
                                      const float *bias, const float *scale,
                                      T *output, T eps) {
  using BlockReduce = cub::BlockReduce<kvp<T>, TPB>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ T mu;      // mean
  __shared__ T rsigma;  // 1 / std.dev.

  const auto sum_kv = BlockReduce(temp_storage).Reduce(thread_data, cub::Sum());

  if (threadIdx.x == 0) {
    mu = sum_kv.key;
    rsigma = rsqrt(sum_kv.value - mu * mu + eps);
  }
  __syncthreads();

  if (threadIdx.x < ld) {
    const T g(scale[threadIdx.x]);
    const T b(bias[threadIdx.x]);
    output[idx] = g * (val - mu) * rsigma + b;
  }
}

template <typename T, int TPB>
__device__ inline void LayerNorm(const kvp<T> &thread_data, const int ld,
                                 const int offset, const float *bias,
                                 const float *scale, T *output, T eps) {
  using BlockReduce = cub::BlockReduce<kvp<T>, TPB>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ T mu;      // mean
  __shared__ T rsigma;  // 1 / std.dev.

  const auto sum_kv = BlockReduce(temp_storage).Reduce(thread_data, cub::Sum());

  if (threadIdx.x == 0) {
    mu = sum_kv.key;
    rsigma = rsqrt(sum_kv.value - mu * mu + eps);
  }
  __syncthreads();

  for (int i = threadIdx.x; i < ld; i += TPB) {
    const int idx = offset + i;
    const T val = output[idx];
    const T g(scale[i]);
    const T b(bias[i]);
    output[idx] = g * (val - mu) * rsigma + b;
  }
}

template <typename T, typename T2, int TPB>
__device__ inline void LayerNorm2(const kvp<T> &thread_data, const int ld,
                                  const int offset, const float2 *bias,
                                  const float2 *scale, T2 *output, T eps) {
  using BlockReduce = cub::BlockReduce<kvp<T>, TPB>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ T mu;      // mean
  __shared__ T rsigma;  // 1 / std.dev.

  const auto sum_kv = BlockReduce(temp_storage).Reduce(thread_data, cub::Sum());

  if (threadIdx.x == 0) {
    mu = sum_kv.key;
    rsigma = rsqrt(sum_kv.value - mu * mu + eps);
  }
  __syncthreads();

  for (int i = threadIdx.x; i < ld; i += TPB) {
    const int idx = offset + i;
    T2 val = output[idx];
    const float2 g = scale[i];
    const float2 b = bias[i];
    val.x = T(g.x) * (val.x - mu) * rsigma + T(b.x);
    val.y = T(g.y) * (val.y - mu) * rsigma + T(b.y);
    output[idx] = val;
  }
}

template <typename T, unsigned TPB>
__global__ void EmbEltwiseLayernormKernel(int hidden, const int64_t *ids,
                                          const float *scale, const float *bias,
                                          const int64_t *embs, T *output,
                                          float eps, int input_num) {
  cub::Sum pair_sum;
  // blockIdx.x: position in the sequence
  // blockIdx.y: batch
  // gridDim.x: Seq
  // gridDim.y: Batch

  extern __shared__ int64_t array_id[];

  const T rhidden = T(1.f) / T(hidden);
  const int64_t seq_pos = blockIdx.y + blockIdx.x * gridDim.y;
  if (threadIdx.x == 0) {
    for (int i = 0; i < input_num; ++i) {
      const int64_t *ids_p = reinterpret_cast<const int64_t *>(ids[i]);
      array_id[i] = ids_p[seq_pos];
    }
  }
  __syncthreads();

  const int64_t out_offset = seq_pos * hidden;

  kvp<T> thread_data(0, 0);

#pragma unroll
  for (int it = threadIdx.x; it < hidden; it += TPB) {
    T val = 0;
    for (int i = 0; i < input_num; ++i) {
      val += reinterpret_cast<const T *>(embs[i])[array_id[i] * hidden + it];
    }

    output[out_offset + it] = val;
    const T rhiddenval = rhidden * val;
    thread_data = pair_sum(thread_data, kvp<T>(rhiddenval, rhiddenval * val));
  }
  LayerNorm<T, TPB>(thread_data, hidden, out_offset, bias, scale, output, eps);
}

template <typename T>
void EmbEltwiseLayerNormFunctor<T>::operator()(
    int batch, int seq_len, int hidden, const int64_t *ids, const float *scale,
    const float *bias, const int64_t *embs, T *output, float eps, int input_num,
    cudaStream_t stream) {
  const unsigned tpb = 256;
  const dim3 grid(seq_len, batch, 1);
  const dim3 block(tpb, 1, 1);
  int shared_bytes = input_num * sizeof(int64_t);
  EmbEltwiseLayernormKernel<T, tpb><<<grid, block, shared_bytes, stream>>>(
      hidden, ids, scale, bias, embs, output, eps, input_num);
}

template class EmbEltwiseLayerNormFunctor<float>;

#ifdef SUPPORTS_CUDA_FP16
template class EmbEltwiseLayerNormFunctor<half>;
#endif

template <typename T>
__global__ void SoftmaxKernelWithEltadd(T *qk_buf_, const T *bias_qk_,
                                        const int batch_size,
                                        const int head_num, const int seq_len,
                                        const unsigned mask) {
  int qk_offset = blockIdx.x * seq_len;
  assert(blockDim.x % 32 == 0);

  float tmp = threadIdx.x < seq_len
                  ? static_cast<float>(qk_buf_[threadIdx.x + qk_offset] +
                                       bias_qk_[threadIdx.x + qk_offset])
                  : -1e20f;
  float max_val = blockReduceMax<float>(tmp, mask);

  float qk_tmp = threadIdx.x < seq_len ? __expf(tmp - max_val) : 0.0f;
  float sum_val = blockReduceSum<float>(qk_tmp, mask);

  if (threadIdx.x < seq_len)
    qk_buf_[threadIdx.x + qk_offset] = (T)(qk_tmp / sum_val);
}

template <typename T>
__global__ void SoftmaxKernelWithEltadd2(T *qk_buf_, const T *bias_qk_,
                                         const int batch_size,
                                         const int head_num, const int seq_len,
                                         const unsigned mask) {
  int qk_offset = blockIdx.x * seq_len;
  int idx = threadIdx.x;
  assert(blockDim.x % 32 == 0);

  float2 tmp =
      idx < seq_len
          ? ToFloat2<T>(qk_buf_[idx + qk_offset] + bias_qk_[idx + qk_offset])
          : make_float2(-1e20f, -1e20f);
  float max_val = blockReduceMax<float>(max(tmp.x, tmp.y), mask);
  float2 qk_tmp = idx < seq_len ? make_float2(__expf(tmp.x - max_val),
                                              __expf(tmp.y - max_val))
                                : make_float2(0.f, 0.f);
  float sum_val = blockReduceSum<float>(qk_tmp.x + qk_tmp.y, mask) + 1e-6f;

  if (idx < seq_len) {
    qk_buf_[idx + qk_offset] =
        FloatsToPair<T>(qk_tmp.x / sum_val, qk_tmp.y / sum_val);
  }
}

template <typename T>
inline void MatMulWithHeadQK(const platform::CUDADeviceContext &context,
                             int head_num, int seq_len, int size_per_head,
                             int batch_size, bool q_trans, bool k_trans,
                             T *q_buf_, T *k_buf_, T *qk_buf_, const T *bias_qk,
                             T alpha, T beta) {
  CBLAS_TRANSPOSE transA = !q_trans ? CblasNoTrans : CblasTrans;
  CBLAS_TRANSPOSE transB = !k_trans ? CblasNoTrans : CblasTrans;

  typedef typename CUDATypeTraits<T>::TYPE run_type;
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
  if (seq_len % 2 == 0) {
    block = (seq_len <= 64) ? 32 : ((seq_len + 63) / 64) * 32;
#ifdef SUPPORTS_CUDA_FP16
    if (std::is_same<T, float>::value) {
#endif
      SoftmaxKernelWithEltadd2<float2><<<grid, block, 0, stream>>>(
          reinterpret_cast<float2 *>(qk_buf_),
          reinterpret_cast<const float2 *>(bias_qk), batch_size, head_num,
          seq_len / 2, FINAL_MASK);
#ifdef SUPPORTS_CUDA_FP16
    } else {
      SoftmaxKernelWithEltadd2<__half2><<<grid, block, 0, stream>>>(
          reinterpret_cast<__half2 *>(qk_buf_),
          reinterpret_cast<const __half2 *>(bias_qk), batch_size, head_num,
          seq_len / 2, FINAL_MASK);
    }
#endif
  } else {
    block = (seq_len <= 32) ? 32 : ((seq_len + 31) / 32) * 32;
    SoftmaxKernelWithEltadd<T><<<grid, block, 0, stream>>>(
        qk_buf_, bias_qk, batch_size, head_num, seq_len, FINAL_MASK);
  }
}

template <typename T>
inline void MatMulWithHeadQKV(const platform::CUDADeviceContext &context,
                              int head_num, int seq_len, int size_per_head,
                              int batch_size, bool qk_trans, bool v_trans,
                              T *v_buf_, const T *qk_buf_, T *dst, T alpha,
                              T beta) {
  int m = batch_size * seq_len;
  int k = head_num * size_per_head;

  typedef typename CUDATypeTraits<T>::TYPE run_type;
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
void MultiHeadGPUComputeFunctor<T>::operator()(
    const platform::CUDADeviceContext &dev_ctx, int batch, int seq_len,
    int head_num, int head_size, T *qkptr, const T *bias_qk_ptr, T *tptr,
    T alpha, T beta) {
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

template class MultiHeadGPUComputeFunctor<float>;

#ifdef SUPPORTS_CUDA_FP16
template class MultiHeadGPUComputeFunctor<half>;
#endif

template <typename T, unsigned TPB>
__global__ void SkipLayerNormSmallKernel(int num, int hidden, const T *input1,
                                         const T *input2, T *output,
                                         const float *scale, const float *bias,
                                         float eps) {
  const T rld = T(1) / T(hidden);
  const int offset = blockIdx.x * hidden;
  cub::Sum pair_sum;
  kvp<T> thread_data(0, 0);
  const int idx = offset + threadIdx.x;
  T val = 0;
  if (threadIdx.x < hidden) {
    val = input1[idx] + input2[idx];
    const T rldval = rld * val;
    thread_data = pair_sum(thread_data, kvp<T>(rldval, rldval * val));
  }
  LayerNormSmall<T, TPB>(val, thread_data, hidden, idx, bias, scale, output,
                         eps);
}

template <typename T, unsigned TPB>
__global__ void SkipLayerNormKernel(int num, int hidden, const T *input1,
                                    const T *input2, T *output,
                                    const float *scale, const float *bias,
                                    float eps) {
  const T rld = T(1) / T(hidden);
  const int offset = blockIdx.x * hidden;
  cub::Sum pair_sum;
  kvp<T> thread_data(0, 0);

  for (int it = threadIdx.x; it < hidden; it += TPB) {
    const int idx = offset + it;
    const T val = input1[idx] + input2[idx];
    const T rldval = rld * val;
    thread_data = pair_sum(thread_data, kvp<T>(rldval, rldval * val));
    output[idx] = val;
  }
  LayerNorm<T, TPB>(thread_data, hidden, offset, bias, scale, output, eps);
}

template <typename T, typename T2, unsigned TPB>
__global__ void SkipLayerNormKernel2(int num, int hidden, const T2 *input1,
                                     const T2 *input2, T2 *output,
                                     const float2 *scale, const float2 *bias,
                                     float eps) {
  const T rld = T(0.5f / hidden);  // because hidden is hidden/2
  const int offset = blockIdx.x * hidden;
  cub::Sum pair_sum;
  kvp<T> thread_data(0, 0);

  for (int it = threadIdx.x; it < hidden; it += TPB) {
    const int idx = offset + it;
    const T2 val2 = input1[idx] + input2[idx];
    thread_data = pair_sum(
        thread_data, kvp<T>(rld * (val2.x + val2.y),
                            rld * val2.x * val2.x + rld * val2.y * val2.y));
    output[idx] = val2;
  }
  LayerNorm2<T, T2, TPB>(thread_data, hidden, offset, bias, scale, output, eps);
}

template <typename T>
void SkipLayerNormFunctor<T>::operator()(const int num, const int hidden,
                                         const T *input1, const T *input2,
                                         const float *scale, const float *bias,
                                         T *output, T eps,
                                         cudaStream_t stream) {
  int block = num / hidden;
  if (hidden <= 32) {
    const int threads = 32;
    SkipLayerNormSmallKernel<T, threads><<<block, threads, 0, stream>>>(
        num, hidden, input1, input2, output, scale, bias, eps);
  } else if (hidden <= 128) {
    const int threads = 128;
    SkipLayerNormSmallKernel<T, threads><<<block, threads, 0, stream>>>(
        num, hidden, input1, input2, output, scale, bias, eps);
  } else if (hidden == 384) {
    const int threads = 384;
    SkipLayerNormSmallKernel<T, threads><<<block, threads, 0, stream>>>(
        num, hidden, input1, input2, output, scale, bias, eps);
  } else {
    const int threads = 256;
    if (hidden % 2 == 0) {
#ifdef SUPPORTS_CUDA_FP16
      if (std::is_same<T, float>::value) {
#endif
        SkipLayerNormKernel2<float, float2,
                             threads><<<block, threads, 0, stream>>>(
            num, hidden / 2, reinterpret_cast<const float2 *>(input1),
            reinterpret_cast<const float2 *>(input2),
            reinterpret_cast<float2 *>(output),
            reinterpret_cast<const float2 *>(scale),
            reinterpret_cast<const float2 *>(bias), eps);
#ifdef SUPPORTS_CUDA_FP16
      } else if (std::is_same<T, __half>::value) {
        SkipLayerNormKernel2<__half, __half2,
                             threads><<<block, threads, 0, stream>>>(
            num, hidden / 2, reinterpret_cast<const __half2 *>(input1),
            reinterpret_cast<const __half2 *>(input2),
            reinterpret_cast<__half2 *>(output),
            reinterpret_cast<const float2 *>(scale),
            reinterpret_cast<const float2 *>(bias), eps);
      } else {
        assert(false);
        // should not be here
      }
#endif
    } else {
      SkipLayerNormKernel<T, threads><<<block, threads, 0, stream>>>(
          num, hidden, input1, input2, output, scale, bias, eps);
    }
  }
}

template class SkipLayerNormFunctor<float>;

#ifdef SUPPORTS_CUDA_FP16
template class SkipLayerNormFunctor<half>;
#endif

}  // namespace math
}  // namespace operators
}  // namespace paddle
