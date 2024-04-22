// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include <immintrin.h>
#include <omp.h>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <new>
#include <string>
#include "glog/logging.h"

#ifdef PADDLE_WITH_DNNL
#include "dnnl.hpp"  //NOLINT
#endif

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"

namespace phi {
namespace fusion {

template <typename T, typename Tt>
void arraycpy(T* dst, const Tt* src, int n) {
#ifdef PADDLE_WITH_MKLML
#pragma omp simd
#endif
  for (int i = 0; i < n; i++) {
    dst[i] = static_cast<T>(src[i]);
  }
}

// batches x tokens x 3 x head x heads ->  3 x batches x head x tokens x heads
// (2 0 3 1 4)
template <typename T, typename Tt>
void transpose_before_bmm1(const T* qkvBuffer,
                           Tt* qkvTransBuffer,
                           int batchSize,
                           int tokenSize,
                           int headNum,
                           int headSize) {
  int hiddenSize = headNum * headSize;
  int blocksize = tokenSize * hiddenSize;  // dst buffer stride in each batch

  const T* qBuffer = qkvBuffer;
  const T* kBuffer = qkvBuffer + hiddenSize;
  const T* vBuffer = qkvBuffer + hiddenSize * 2;

  Tt* q_buffer = qkvTransBuffer;
  Tt* k_buffer = qkvTransBuffer + batchSize * blocksize;
  Tt* v_buffer = qkvTransBuffer + batchSize * blocksize * 2;

  int bmHead = headNum;
  int cols_per_bmHead = hiddenSize / headNum;  // 768/12 = 64

#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for collapse(3)
#endif
  for (int i = 0; i < batchSize; i++) {
    for (int k = 0; k < bmHead; k++) {
      for (int j = 0; j < tokenSize; j++) {
        const T* q_src_each_batch =
            reinterpret_cast<const T*>(qBuffer) + blocksize * 3 * i;
        const T* k_src_each_batch =
            reinterpret_cast<const T*>(kBuffer) + blocksize * 3 * i;
        const T* v_src_each_batch =
            reinterpret_cast<const T*>(vBuffer) + blocksize * 3 * i;

        int dst_offset_each_bmHead = k * tokenSize * cols_per_bmHead;
        int src_offset_each_line = k * cols_per_bmHead;

        int dst_offset_each_line = j * cols_per_bmHead;
        int src_offset_each_bmHead = j * hiddenSize * 3;

        Tt* q_dst_each_line = q_buffer + i * blocksize +
                              dst_offset_each_bmHead + dst_offset_each_line;
        const T* q_src_each_line =
            q_src_each_batch + src_offset_each_bmHead + src_offset_each_line;

        Tt* k_dst_each_line = k_buffer + i * blocksize +
                              dst_offset_each_bmHead + dst_offset_each_line;
        const T* k_src_each_line =
            k_src_each_batch + src_offset_each_bmHead + src_offset_each_line;

        Tt* v_dst_each_line = v_buffer + i * blocksize +
                              dst_offset_each_bmHead + dst_offset_each_line;
        const T* v_src_each_line =
            v_src_each_batch + src_offset_each_bmHead + src_offset_each_line;
        arraycpy<Tt, T>(q_dst_each_line, q_src_each_line, cols_per_bmHead);
        arraycpy<Tt, T>(k_dst_each_line, k_src_each_line, cols_per_bmHead);
        arraycpy<Tt, T>(v_dst_each_line, v_src_each_line, cols_per_bmHead);
      }
    }
  }
}

// batches x head x tokens x heads -> batches x tokens x head x heads (0 2 1 3)
template <typename T, typename Tt>
void transpose_after_bmm2(T* Buffer,
                          Tt* TransBuffer,
                          int batchSize,
                          int tokenSize,
                          int headNum,
                          int headSize) {
  int hiddenSize = headNum * headSize;
  int blocksize = tokenSize * hiddenSize;  // dst buffer stride in each batch

  int bmHead = headNum;
  int cols_per_bmHead = hiddenSize / headNum;  // 768/12 = 64

#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for collapse(2)
#endif
  for (int i = 0; i < batchSize; i++) {
    for (int k = 0; k < tokenSize; k++) {
      int src_offset_each_head = k * cols_per_bmHead;
      int dst_offset_each_line = k * hiddenSize;

      for (int j = 0; j < bmHead; j++) {
        int src_offset_each_line = j * tokenSize * cols_per_bmHead;
        int dst_offset_each_head = j * cols_per_bmHead;

        Tt* q_dst_each_line = TransBuffer + dst_offset_each_head +
                              dst_offset_each_line + i * blocksize;
        const T* q_src_each_line = Buffer + src_offset_each_line +
                                   src_offset_each_head + i * blocksize;

        arraycpy<Tt, T>(q_dst_each_line, q_src_each_line, cols_per_bmHead);
      }
    }
  }
}

// C = A * B
// bTranspose: B need to be transposed or not
void sgemm(const float* A,
           const float* B,
           float* C,
           int m,
           int n,
           int k,
           bool transa,
           bool transb) {
#ifdef PADDLE_WITH_DNNL
  int lda = (transa ? m : k);
  int ldb = (transb ? k : n);
  int ldc = n;
  float alpha = 1;
  float beta = 0;
  std::array<char, 2> ta = {"N"};
  std::array<char, 2> tb = {"N"};
  if (transa) ta[0] = 'T';
  if (transb) tb[0] = 'T';

  dnnl_sgemm(ta[0], tb[0], m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#else
  LOG(ERROR) << "scaled_dp_atten not supported without WITH_MKL!";
#endif
}

// exp based-on jit code
static inline __m512 vexp(const __m512& _x) {
  __m512 p16f_1 = _mm512_set1_ps(1.0f);
  __m512 p16f_half = _mm512_set1_ps(0.5f);
  __m512 p16f_127 = _mm512_set1_ps(127.f);
  __m512 p16f_exp_hi = _mm512_set1_ps(88.3762626647950f);
  __m512 p16f_exp_lo = _mm512_set1_ps(-88.3762626647949f);

  __m512 p16f_cephes_LOG2EF = _mm512_set1_ps(1.44269504088896341f);

  __m512 p16f_cephes_exp_p0 = _mm512_set1_ps(1.9875691500E-4f);
  __m512 p16f_cephes_exp_p1 = _mm512_set1_ps(1.3981999507E-3f);
  __m512 p16f_cephes_exp_p2 = _mm512_set1_ps(8.3334519073E-3f);
  __m512 p16f_cephes_exp_p3 = _mm512_set1_ps(4.1665795894E-2f);
  __m512 p16f_cephes_exp_p4 = _mm512_set1_ps(1.6666665459E-1f);
  __m512 p16f_cephes_exp_p5 = _mm512_set1_ps(5.0000001201E-1f);

  // Clamp x.
  __m512 x = _mm512_max_ps(_mm512_min_ps(_x, p16f_exp_hi), p16f_exp_lo);

  // Express exp(x) as exp(m*ln(2) + r), start by extracting
  // m = floor(x/ln(2) + 0.5).
  __m512 m = _mm512_floor_ps(_mm512_fmadd_ps(x, p16f_cephes_LOG2EF, p16f_half));

  // Get r = x - m*ln(2).
  __m512 p16f_nln2 = _mm512_set1_ps(-0.6931471805599453f);
  __m512 r = _mm512_fmadd_ps(m, p16f_nln2, x);

  __m512 r2 = _mm512_mul_ps(r, r);

  __m512 y = p16f_cephes_exp_p0;
  y = _mm512_fmadd_ps(y, r, p16f_cephes_exp_p1);
  y = _mm512_fmadd_ps(y, r, p16f_cephes_exp_p2);
  y = _mm512_fmadd_ps(y, r, p16f_cephes_exp_p3);
  y = _mm512_fmadd_ps(y, r, p16f_cephes_exp_p4);
  y = _mm512_fmadd_ps(y, r, p16f_cephes_exp_p5);
  y = _mm512_fmadd_ps(y, r2, r);
  y = _mm512_add_ps(y, p16f_1);

  // Build emm0 = 2^m.
  __m512i emm0 = _mm512_cvttps_epi32(_mm512_add_ps(m, p16f_127));
  emm0 = _mm512_slli_epi32(emm0, 23);

  // Return 2^m * exp(r).
  return _mm512_max_ps(_mm512_mul_ps(y, _mm512_castsi512_ps(emm0)), _x);
}

// need to do for res.
void softmax_sum_max(float* AB,
                     float* sum,
                     float* max,
                     float* pre_sum,
                     float* pre_max,
                     float refac,
                     int m,
                     int k) {
  float max_val = std::numeric_limits<float>::lowest();
  __m512 vrefac = _mm512_set1_ps(refac);
  for (int i = 0; i < m; ++i) {
    float* buf = AB + i * k;
    // max val for avoiding inf and nan
    __m512 vmax = _mm512_set1_ps(max_val);
    for (int off = 0; off < k; off += 16) {
      int remain = k - off;
      __mmask16 mask = (remain >= 16 ? 0xffff : (1 << remain) - 1);
      __m512 vx = _mm512_maskz_loadu_ps(mask, buf + off);

      vmax = _mm512_mask_max_ps(vmax, mask, vmax, vx);
    }
    float _max = _mm512_reduce_max_ps(vmax);

    _max *= refac;
    _max = _max > max[i] ? _max : max[i];
    __m512 merr = _mm512_set1_ps(max[i] - _max);
    merr = vexp(merr);
    max[i] = _max;

    // exp and get sum
    __m512 vsum = _mm512_set1_ps(0);
    vmax = _mm512_set1_ps(_max);
    for (int off = 0; off < k; off += 16) {
      int remain = k - off;
      __mmask16 mask = (remain >= 16 ? 0xffff : (1 << remain) - 1);

      __m512 vx = _mm512_maskz_loadu_ps(mask, buf + off);
      vx = _mm512_mask_mul_ps(vx, mask, vx, vrefac);
      vx = _mm512_mask_sub_ps(vx, mask, vx, vmax);
      vx = vexp(vx);

      _mm512_mask_storeu_ps(buf + off, mask, vx);

      vsum = _mm512_mask_add_ps(vsum, mask, vsum, vx);
    }
    float _sum = _mm512_reduce_add_ps(vsum);
    float fac = _mm512_cvtss_f32(merr);
    sum[i] = sum[i] * fac + _sum;
    _sum = sum[i];

    // Compute exp/sum(exp) and store
    __m512 vrsum = _mm512_set1_ps(1.0f / _sum);
    for (int off = 0; off < k; off += 16) {
      int remain = k - off;
      __mmask16 mask = (remain >= 16 ? 0xffff : (1 << remain) - 1);

      __m512 vx = _mm512_maskz_loadu_ps(mask, buf + off);
      vx = _mm512_mask_mul_ps(vx, mask, vx, vrsum);
      _mm512_mask_storeu_ps(buf + off, mask, vx);
    }
  }
}

void update_out_blk(float* output,
                    const float* exp_ABC,
                    float* pre_sum,
                    float* sum,
                    float* pre_max,
                    float* max,
                    int m,
                    int n) {
  for (int i = 0; i < m; ++i) {
    const float* buf = exp_ABC + i * n;
    float* outbuf = output + i * n;
    __m512 merr = _mm512_set1_ps(pre_max[i] - max[i]);
    merr = vexp(merr);
    __m512 vfac = _mm512_set1_ps(pre_sum[i] / sum[i]);
    for (int off = 0; off < n; off += 16) {
      int remain = n - off;
      __mmask16 mask = (remain >= 16 ? 0xffff : (1 << remain) - 1);
      __m512 vout = _mm512_maskz_loadu_ps(mask, outbuf + off);
      __m512 vabc = _mm512_maskz_loadu_ps(mask, buf + off);
      vout = _mm512_mask_mul_ps(vout, mask, vout, merr);
      vout = _mm512_mask_mul_ps(vout, mask, vout, vfac);
      __m512 vupt = _mm512_set1_ps(0.0f);
      vupt = _mm512_mask_add_ps(vupt, mask, vout, vabc);
      _mm512_mask_storeu_ps(outbuf + off, mask, vupt);
    }
    pre_sum[i] = sum[i];
    pre_max[i] = max[i];
  }
}

// hard code: axis = 1
// sum += sum(exp(A[i]))
// output = output * pre_sum / sum + (exp(A) / sum) x B
// pre_sum = sum
void incremental_tile_attention(const float* A,
                                const float* B,
                                const float* C,
                                int m,
                                int n,
                                int k,
                                float* pre_sum,
                                float* sum,
                                float* pre_max,
                                float* max,
                                float refac,
                                float* AB,
                                float* exp_ABC,
                                float* output) {
  sgemm(A, B, AB, m, k, n, false, true);
  softmax_sum_max(AB, sum, max, pre_sum, pre_max, refac, m, k);
  sgemm(AB, C, exp_ABC, m, n, k, false, false);
  update_out_blk(output, exp_ABC, pre_sum, sum, pre_max, max, m, n);
}

// scaled dot-product attention: bmm1 + softmax + bmm2
void scaled_dp_attention(const float* query,
                         const float* key,
                         const float* value,
                         float scale,
                         int batch_size,
                         int itsize,
                         int otsize,
                         int num_head,
                         int head_size,
                         float* output) {
  // output = trans(softmax(query * trans(key)) * value)
  int iblk = std::min(512, itsize / 1);
  int oblk = std::min(512, otsize / 1);
  float refac = scale;

#ifdef PADDLE_WITH_MKLML
  int nth = omp_get_max_threads();
#else
  int nth = 1;
#endif

  float** pre_sum;
  float** sum;
  float** pre_max;
  float** max;
  float** qk_arr;
  float** exp_qkv_arr;
  pre_sum = new float*[nth];
  sum = new float*[nth];
  pre_max = new float*[nth];
  max = new float*[nth];
  qk_arr = new float*[nth];
  exp_qkv_arr = new float*[nth];
  for (int i = 0; i < nth; ++i) {
    pre_sum[i] = new float[iblk];
    sum[i] = new float[iblk];
    pre_max[i] = new float[iblk];
    max[i] = new float[iblk];
    qk_arr[i] = new float[iblk * oblk];
    exp_qkv_arr[i] = new float[iblk * head_size];
  }

#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for collapse(3)
#endif
  for (int i = 0; i < batch_size; ++i) {
    for (int j = 0; j < num_head; ++j) {
      for (int m = 0; m < itsize; m += iblk) {
#ifdef PADDLE_WITH_MKLML
        int tid = omp_get_thread_num();
#else
        int tid = 0;
#endif
        int ooffset =
            i * num_head * otsize * head_size + j * otsize * head_size;
        const float* k = key + ooffset;
        const float* v = value + ooffset;

        int q_rblk = std::min(iblk, itsize - m);
        int ioffset =
            i * num_head * otsize * head_size + j * otsize * head_size;
        const float* q = query + ioffset + m * head_size;
        float* out = output + ioffset + m * head_size;

        // reset out
        for (int ii = 0; ii < q_rblk; ++ii) {
#ifdef PADDLE_WITH_MKLML
#pragma omp simd
#endif
          for (int jj = 0; jj < head_size; ++jj) {
            out[ii * head_size + jj] = 0;  // reset output
          }
        }
        // reset sum
#ifdef PADDLE_WITH_MKLML
#pragma omp simd
#endif
        for (int ii = 0; ii < q_rblk; ++ii) {
          pre_sum[tid][ii] = 0;
          sum[tid][ii] = 0;
          pre_max[tid][ii] = std::numeric_limits<float>::lowest();
          max[tid][ii] = std::numeric_limits<float>::lowest();
        }
        //
        for (int b = 0; b < otsize; b += oblk) {
          int kv_rblk = std::min(oblk, otsize - b);
          const float* blk_k = k + b * head_size;
          const float* blk_v = v + b * head_size;

          incremental_tile_attention(q,
                                     blk_k,
                                     blk_v,
                                     q_rblk,
                                     head_size,
                                     kv_rblk,
                                     pre_sum[tid],
                                     sum[tid],
                                     pre_max[tid],
                                     max[tid],
                                     refac,
                                     qk_arr[tid],
                                     exp_qkv_arr[tid],
                                     out);
        }
      }
    }
  }

  for (int i = 0; i < nth; ++i) {
    delete[] pre_sum[i];
    delete[] sum[i];
    delete[] pre_max[i];
    delete[] max[i];
    delete[] qk_arr[i];
    delete[] exp_qkv_arr[i];
  }
  delete[] pre_sum;
  delete[] sum;
  delete[] pre_max;
  delete[] max;
  delete[] qk_arr;
  delete[] exp_qkv_arr;

  return;
}

template <typename T, typename Context>
void SelfDPAttenKernel(const Context& dev_ctx,
                       const DenseTensor& x,
                       const float alpha,
                       const int head_number,
                       DenseTensor* out) {
  auto* input_d = x.data<T>();
  auto* output_d = dev_ctx.template Alloc<T>(out);
  float scale = static_cast<float>(alpha);
  auto input_dims = x.dims();
  // in shouble be (batch * seq * 3 * head_num * head_size)
  // out shouble be (batch * seq * head_num * head_size)
  int batch_size = static_cast<int>(input_dims[0]);
  int seq_len = static_cast<int>(input_dims[1]);
  int head_size = static_cast<int>(input_dims[4]);

  DenseTensor temp1, temp2;
  temp1.Resize(input_dims);
  float* trans_input = dev_ctx.template Alloc<float>(&temp1);
  temp2.Resize(input_dims);
  float* trans_output = dev_ctx.template Alloc<float>(&temp2);

  transpose_before_bmm1<T, float>(
      input_d, trans_input, batch_size, seq_len, head_number, head_size);
  float* query = trans_input;
  float* key = trans_input + batch_size * head_number * seq_len * head_size;
  float* value =
      trans_input + batch_size * head_number * seq_len * head_size * 2;

  scaled_dp_attention(query,
                      key,
                      value,
                      scale,
                      batch_size,
                      seq_len,
                      seq_len,
                      head_number,
                      head_size,
                      trans_output);
  transpose_after_bmm2<float, T>(
      trans_output, output_d, batch_size, seq_len, head_number, head_size);
}

}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(self_dp_attention,
                   CPU,
                   ALL_LAYOUT,
                   phi::fusion::SelfDPAttenKernel,
                   float,
                   double) {}
