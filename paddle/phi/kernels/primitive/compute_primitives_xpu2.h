// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once
#include "paddle/phi/common/float16.h"
#include "xpu/kernel/cluster_header.h"
#include "xpu/kernel/debug.h"
#include "xpu/kernel/math.h"
#include "xpu/kernel/simd_header.h"

#include "paddle/phi/kernels/primitive/functor_primitives_xpu2.h"
#include <type_traits>

namespace phi {
namespace kps {
namespace details {

// kGlobalMode: block reduce, each block gets an output;
// kLocalMode: thread reduce, each thread gets an output;
enum ReduceMode { kGlobalMode, kLocalMode };

template <typename T>
class MPTypeTrait {
 public:
  using Type = T;
};

template <>
class MPTypeTrait<phi::dtype::float16> {
 public:
  using Type = float;
};

static inline __device__ void sync_all() {
  __asm__ __volatile__(
      "sync_local\t\n"
      "csr_set csr3, %0\t\n"
      "sync_group csr3" ::"r"(-1));
}

static __device__ inline float32x16_t vload_sm_float32x16(_shared_ptr_ const float* src_ptr) {
    float32x16_t ret;
    __asm__ __volatile__("vloads_mask16.mz %0{mr1}, 0(%1)":"=v"(ret):"r"(src_ptr));
    return ret;
}

static __device__ inline float loada_float(_shared_ptr_ const float* ptr) {
    float ret;
    __asm__ __volatile__("loada.w %0,%1":"=r"(ret):"r"(ptr));
    return ret;
}

static __device__ inline bool storea_float(_shared_ptr_ float* ptr, float value) {
    bool ret;
    __asm__ __volatile__("storea.w %0,%1,%2":"=r"(ret):"r"(value), "r"(ptr));
    return ret;
}

static __device__ void atomic_mul(_shared_ptr_ float* ptr, float value) {
    bool fail = true;
    while (fail) {
        float a = SM2REG_atomic(ptr);
        a = a * value;
        fail = REG2SM_atomic(ptr, a);
    }
}

static __device__ void atomic_add(_shared_ptr_ float* ptr, float value) {
    bool fail = true;
    while (fail) {
        float a = loada_float(ptr);
        a += value;
        fail = storea_float(ptr, a);
    }
}

static __device__ void atomic_max(_shared_ptr_ float* ptr, float value) {
    bool fail = true;
    while (fail) {
        float a = loada_float(ptr);
        a = fmax(a, value);
        fail = storea_float(ptr, a);
    }
}

static __device__ void atomic_min(_shared_ptr_ float* ptr, float value) {
    bool fail = true;
    while (fail) {
        float a = loada_float(ptr);
        a = fmin(a, value);
        fail = storea_float(ptr, a);
    }
}

static __device__ float do_mul(float* lmptr, int size) {
    while ((size % 16) != 0) {
        lmptr[size] = 1.0f;
        size++;
    }
    mfence();
    __simd__ float acc_buf[16];
    int offset_last = size - 16;
    float32x16_t v_last = vload_lm_float32x16(lmptr + offset_last);
    for (int i = 0; i < offset_last; i += 16) {
        float32x16_t v0 = vload_lm_float32x16(lmptr + i);
        v_last = vvmul_float32x16(v_last, v0);
    }
    vstore_lm_float32x16(acc_buf, v_last);
    mfence();
    float res = 1.0f;
    for (int i = 0; i < 16; i++) {
        res = res * acc_buf[i];
    }
    return res;
}

static __device__ float do_max(float* lmptr, int size) {
    while ((size % 16) != 0) {
        lmptr[size] = -1e30;
        size++;
    }
    mfence();
    __simd__ float acc_buf[16];
    int offset_last = size - 16;
    float32x16_t v_last = vload_lm_float32x16(lmptr + offset_last);
    for (int i = 0; i < offset_last; i += 16) {
        float32x16_t v0 = vload_lm_float32x16(lmptr + i);
        v_last = vvmax_float32x16(v_last, v0);
    }
    vstore_lm_float32x16(acc_buf, v_last);
    mfence();
    float res = -1e30;
    for (int i = 0; i < 16; i++) {
        res = fmax(res, acc_buf[i]);
    }
    return res;
}

static __device__ float do_min(float* lmptr, int size) {
    while ((size % 16) != 0) {
        lmptr[size] = 1e30;
        size++;
    }
    mfence();
    __simd__ float acc_buf[16];
    int offset_last = size - 16;
    float32x16_t v_last = vload_lm_float32x16(lmptr + offset_last);
    for (int i = 0; i < offset_last; i += 16) {
        float32x16_t v0 = vload_lm_float32x16(lmptr + i);
        v_last = vvmin_float32x16(v_last, v0);
    }
    vstore_lm_float32x16(acc_buf, v_last);
    mfence();
    float res = 1e30;
    for (int i = 0; i < 16; i++) {
        res = fmin(res, acc_buf[i]);
    }
    return res;
}

static __device__ float do_add(float* lmptr, int size) {
    while ((size % 16) != 0) {
        lmptr[size] = 0.0f;
        size++;
    }
    mfence();
    __simd__ float acc_buf[16];
    int offset_last = size - 16;
    float32x16_t v_last = vload_lm_float32x16(lmptr + offset_last);
    for (int i = 0; i < offset_last; i += 16) {
        float32x16_t v0 = vload_lm_float32x16(lmptr + i);
        v_last = vvadd_float32x16(v_last, v0);
    }
    vstore_lm_float32x16(acc_buf, v_last);
    mfence();
    float res = 0.0f;
    for (int i = 0; i < 16; i++) {
        res = res + acc_buf[i];
    }
    return res;
}

static inline __device__ void do_max_2d(float* buffer, int tt, int nn) {
    float* lm_out = buffer;
    for (int n_iter = 0; n_iter < nn; n_iter += 16) {
        float32x16_t v0 = vload_lm_float32x16(lm_out);
        float* lm_in = lm_out + nn;
        for (int curr_t = 0; curr_t < tt; curr_t++) {
            float32x16_t v1 = vload_lm_float32x16(lm_in);
            v0 = vvmax_float32x16(v0, v1);
            lm_in += nn;
        }
        vstore_lm_float32x16(lm_out, v0);
        lm_out = lm_out + 16;
    }
    mfence();
}

static inline __device__ void do_min_2d(float* buffer, int tt, int nn) {
    float* lm_out = buffer;
    for (int n_iter = 0; n_iter < nn; n_iter += 16) {
        float32x16_t v0 = vload_lm_float32x16(lm_out);
        float* lm_in = lm_out + nn;
        for (int curr_t = 0; curr_t < tt; curr_t++) {
            float32x16_t v1 = vload_lm_float32x16(lm_in);
            v0 = vvmin_float32x16(v0, v1);
            lm_in += nn;
        }
        vstore_lm_float32x16(lm_out, v0);
        lm_out = lm_out + 16;
    }
    mfence();
}

static inline __device__ void do_add_2d(float* buffer, int tt, int nn) {
    float* lm_out = buffer;
    for (int n_iter = 0; n_iter < nn; n_iter += 16) {
        float32x16_t v0 = vload_lm_float32x16(lm_out);
        float* lm_in = lm_out + nn;
        for (int curr_t = 0; curr_t < tt; curr_t++) {
            float32x16_t v1 = vload_lm_float32x16(lm_in);
            v0 = vvadd_float32x16(v0, v1);
            lm_in += nn;
        }
        vstore_lm_float32x16(lm_out, v0);
        lm_out = lm_out + 16;
    }
    mfence();
}

static inline __device__ void mean_apply(float* buffer, int t, int nn) {
    float scale = 1.0f / t;
    float* lm_out = buffer;
    for (int n_iter = 0; n_iter < nn; n_iter++) {
        buffer[n_iter] = buffer[n_iter] * scale;
    }
    mfence();
}

static inline __device__ void mfence_sm() {
    __asm__("mfence {sm}\n\t");
}


template<typename MPType>
__device__ void initializeSharedMemory(int sm_start, int sm_end, _shared_ptr_ float* sm_buffer, MPType init) {
    int cid = core_id();
    int ncores = core_num();
    for (int i = cid; i < sm_end - sm_start; i += ncores) {
        sm_buffer[i] = (float)init;
    }
    mfence();
    sync_all();
}

template<class T>
__device__ void ReduceLastDim(_shared_ptr_ float* sm_buffer, float* buffer, int curr_tt, int op) {
    if (op == 0 || op == 1) {
        float res = do_add(buffer, curr_tt);
        atomic_add(sm_buffer, res);
    }
    if (op == 2) {
        float res = do_max(buffer, curr_tt);
        atomic_max(sm_buffer, res);
    }
    if (op == 3) {
        float res = do_min(buffer, curr_tt);
        atomic_min(sm_buffer, res);
    }
    if (op == 4) {
        float res = do_mul(buffer, curr_tt);
        atomic_mul(sm_buffer, res);
    }
}

static inline __device__ void mean_apply_2d(_shared_ptr_ float* buffer, int mm, int t) {
    float scale = 1.0f / t;
    for (int m_iter = 0; m_iter < mm; m_iter++) {
        buffer[m_iter] = buffer[m_iter] * scale;
    }
    mfence();
}

// template<class T>
__device__ void ReduceLastDimPost(_shared_ptr_ float* sm_buffer, int m_start, int m_end, int t, int op) {
    int cid = core_id();
    if (cid == 0 && op == 1) {
        mean_apply(sm_buffer, m_end - m_start, t);
    }
}

template<class Ty>
__device__ void WriteDataReduce(_shared_ptr_ float* sm_buffer, _global_ptr_ float* y, int m_start, int m_end) {
    int cid = core_id();
    if (cid == 0) {
        SM2GM((_shared_ptr_ Ty*)sm_buffer, y + m_start, (m_end - m_start) * sizeof(Ty));
    }
}

static inline __device__ void do_mul_2d(float* lm_out, int tt, int nn) {
    for (int n_iter = 0; n_iter < nn; n_iter += 16) {
        float32x16_t v0 = vload_lm_float32x16(lm_out);
        float* lm_in = lm_out + nn;
        for (int curr_t = 0; curr_t < tt; curr_t++) {
            float32x16_t v1 = vload_lm_float32x16(lm_in);
            v0 = vvmul_float32x16(v0, v1);
            lm_in += nn;
        }
        vstore_lm_float32x16(lm_out, v0);
        lm_out = lm_out + 16;
    }
    mfence();
}

template<typename Tx, typename Ty>
__global__ void ReduceOneDim(const Tx* x, Ty* y, phi::kps::MulFunctor<float> reducer, int m, int t, int n) {
    int ncores = core_num();
    int cid = core_id();
    if (cid >= ncores) {
        return;
    }
    __simd__ float buffer[1024];
    int max_nn = roundup32(min(256, n));
    int max_tt = 1024 / max_nn - 1;
    int total_block = m * roundup_div(n, max_nn);
    int thread_id = cluster_num() * cid + cluster_id();
    int total_thread = cluster_num() * ncores;
    for (int b = thread_id; b < total_block; b += total_thread) {
        // reduce [curr_m, 0:t, n_start:n_end] to [curr_m, n_start:n_end]
        int curr_m = b % m;
        int n_start = b / m * max_nn;
        int n_end = min(n, n_start + max_nn);
        int nn = n_end - n_start;
        __global_ptr__ const Tx* curr_x = x + (curr_m * t * n + n_start);
        GM2LM(curr_x, (Tx*)buffer, nn * sizeof(Tx));
        curr_x += n;
        for (int t_start = 1; t_start < t; t_start += max_tt) {
            int t_end = min(t, t_start + max_tt);
            int tt = t_end - t_start;
            Tx* curr_lm = (Tx*)(buffer + max_nn);
            if (max_nn == n) {
                GM2LM_ASYNC(curr_x, curr_lm, tt * n * sizeof(Tx));
                curr_x += tt * n;
            } else {
                for (int curr_t = t_start; curr_t < t_end; curr_t++) {
                    GM2LM_ASYNC(curr_x, curr_lm, nn * sizeof(Tx));
                    curr_x += n;
                    curr_lm += max_nn;
                }
            }
            mfence();
            do_mul_2d(buffer, tt, max_nn);
        }
        LM2GM((Ty*)buffer, y + (curr_m * n + n_start), nn * sizeof(Ty));
    }
}

#define ncores 64
template <typename T, typename OpFunc, int VecSize>
__device__ void BlockXReduce(T* out, const T* data, OpFunc reducer) {
  __shared__ T sum_array[ncores * VecSize];
  int core_idx = core_id() * VecSize;
  mfence();
  sync_all();

#pragma unroll
  for (int i = 0; i < VecSize; i++) {
    mfence();
    sum_array[i * ncores + core_idx] = data[i];
    mfence();
  }
  sync_all();
#pragma unroll
  for (int i = 0; i < VecSize; i++) {
    T start = data[i * ncores];
#pragma unroll
    for (int j = 1; j < ncores; j++) {
      mfence();
      T tmp = sum_array[i * ncores + j];
      mfence();
      start = reducer(start, tmp);
      mfence();
    }
    out[i] = start;
  }
  sync_all();
}
#undef ncores

}  // namespace details

/**
 * @brief Perform unary calculation according to OpFunc. Shape of input and
 * output are the same.
 *
 * @template paraments
 * InT: The data type of in.
 * OutT: The data type of out.
 * NX: The number of data columns loaded by each thread.
 * NY: The number of data rows loaded by each thread.
 * BlockSize: Identifies the current device thread index method. For xpu,
 * core_id() is used as the index.
 * OpFunc: Compute functor which has an operator() as following:
 *     template <typename InT, typename OutT>
 *     struct XxxFunctor {
 *       HOSTDEVICE OutT operator()(const InT& a) const {
 *         return ...;
 *       }
 *     };
 *
 * @param：
 * out: The register pointer of out, the size is NX * NY.
 * in: The register pointer of in, the size is NX * NY.
 * compute: Compute function which was declared like OpFunc<InT, OutT>().
 */
template <typename InT,
          typename OutT,
          int NX,
          int NY,
          int BlockSize,
          class OpFunc>
__device__ __forceinline__ void ElementwiseUnary(OutT* out,
                                                 const InT* in,
                                                 OpFunc compute) {
#pragma unroll
  for (int idx = 0; idx < NX * NY; idx++) {
    out[idx] = static_cast<OutT>(compute(in[idx]));
  }
}

/**
 * @brief Binary calculation according to OpFunc. Shape of The input and output
 * are the same.
 *
 * @template paraments
 * InT: The data type of in1 and in2.
 * OutT: The data type of out.
 * NX: The number of data columns computed by each thread.
 * NY: The number of data rows computed by each thread.
 * BlockSize: Identifies the current device thread index method. For xpu,
 * core_id() is used as the index.
 * OpFunc: Compute functor which has an operator() as following:
 *     template <typename InT>
 *     struct XxxFunctor {
 *       HOSTDEVICE InT operator()(const InT& a, const InT& b) const {
 *         return ...;
 *       }
 *     };
 *
 * @param：
 * out: The register pointer of out, the size is NX * NY.
 * in1: The register pointer of fist input, size is NX * NY.
 * in2: The register pointer of second input, size is NX * NY.
 * compute: Compute function which was declared like OpFunc<InT>().
 */
template <typename InT,
          typename OutT,
          int NX,
          int NY,
          int BlockSize,
          class OpFunc>
__device__ __forceinline__ void ElementwiseBinary(OutT* out,
                                                  const InT* in1,
                                                  const InT* in2,
                                                  OpFunc compute) {
#pragma unroll
  for (int idx = 0; idx < NX * NY; ++idx) {
    out[idx] = static_cast<OutT>(compute(in1[idx], in2[idx]));
  }
}

template <typename InT,
          typename OutT,
          int NX,
          int NY,
          int BlockSize,
          class OpFunc>
__device__ __forceinline__ void ElementwiseBinary(
    OutT* out, const InT* in1, const InT* in2, OpFunc compute, int read_lens) {
  for (int idx = 0; idx < read_lens; ++idx) {
    out[idx] = static_cast<OutT>(compute(in1[idx], in2[idx]));
  }
}

/**
 * @brief Ternary calculation according to OpFunc. Shape of input and output
 * are the same.
 *
 * @template paraments
 * InT: The data type of in1 and in2.
 * OutT: The data type of out.
 * NX: The number of data columns loaded by each thread.
 * NY: The number of data rows loaded by each thread.
 * BlockSize: Identifies the current device thread index method. For xpu,
 * core_id() is used as the index.
 * OpFunc: Compute functor which has an operator() as following
 *     template <typename InT>
 *     struct XxxFunctor {
 *       HOSTDEVICE InT operator()(const InT& a, const InT& b, const InT& c)
 * const {
 *         return ...;
 *       }
 *     };
 *
 * @param
 * out: The register pointer of out, the size is NX * NY.
 * in1: The register pointer of fist input, size is NX * NY.
 * in2: The register pointer of second input, size is NX * NY.
 * in3: The register pointer of third input, size is NX * NY.
 * compute: Compute function which was declared like OpFunc<InT>().
 */
template <typename InT,
          typename OutT,
          int NX,
          int NY,
          int BlockSize,
          class OpFunc>
__device__ __forceinline__ void ElementwiseTernary(
    OutT* out, const InT* in1, const InT* in2, const InT* in3, OpFunc compute) {
#pragma unroll
  for (int idx = 0; idx < NX * NY; ++idx) {
    out[idx] = static_cast<OutT>(compute(in1[idx], in2[idx], in3[idx]));
  }
}

/**
 * @brief Multivariate calculation according to OpFunc. Shape of inputs and
 * output are the same.
 *
 * @template paraments
 * InT: The data type of in1, in2 and in3.
 * OutT: The data type of out.
 * NX: The number of data columns loaded by each thread.
 * NY: The number of data rows loaded by each thread.
 * BlockSize: Identifies the current device thread index method. For xpu,
 * core_id() is used as the index.
 * Arity: The size of ins
 * OpFunc: Compute functor which has an operator() as following:
 *     template <typename InT>
 *     struct XxxFunctor {
 *       HOSTDEVICE InT operator()(const InT* args) const {
 *         return ...;
 *       }
 *     };
 *
 * @param
 * out: The register pointer of out, the size is NX * NY.
 * ins: A pointers of array consisting of multiple inputs.
 * compute: Compute function which was declared like OpFunc<InT>().
 */
template <typename InT,
          typename OutT,
          int NX,
          int NY,
          int BlockSize,
          int Arity,
          class OpFunc>
__device__ __forceinline__ void ElementwiseAny(OutT* out,
                                               InT (*ins)[NX * NY],
                                               OpFunc compute) {
  __local__ InT args[Arity];
#pragma unroll
  for (int idx = 0; idx < NX * NY; ++idx) {
#pragma unroll
    for (int j = 0; j < Arity; ++j) {
      args[j] = ins[j][idx];
    }
    out[idx] = static_cast<OutT>(compute(args));
  }
}

/**
 * @brief Binary calculation according to OpFunc. The shape of in1 and in2 are
 * different. When in1's shape is [1, NX], in2's shape is [NY, NX], then
 * output's shape is [NY, NX].
 *
 * @template paraments
 * InT: The data type of in1 and in2.
 * OutT: The data type of out.
 * NX: The number of data columns loaded by each thread.
 * NY: The number of data rows loaded by each thread.
 * BlockSize: Identifies the current device thread index method. For xpu,
 * core_id() is used as the index.
 * OpFunc: Compute functor which has an operator() as following
 *     template <typename InT, typename OutT>
 *     struct XxxFunctor {
 *       HOSTDEVICE OutT operator()(const InT& a, const InT& b) const {
 *         return ...;
 *       }
 *     };
 *
 * @param
 * out: The register pointer of out, the size is NX * NY.
 * in1: The register pointer of fist input, size is NX * 1.
 * in2: The register pointer of second input, size is NX * NY.
 * compute: Compute function which was declared like OpFunc<InT, OutT>().
 */
template <typename InT,
          typename OutT,
          int NX,
          int NY,
          int BlockSize,
          class OpFunc>
__device__ __forceinline__ void CycleBinary(OutT* out,
                                            const InT* in1,
                                            const InT* in2,
                                            OpFunc compute) {
#pragma unroll
  for (int idx = 0; idx < NX; idx++) {
#pragma unroll
    for (int idy = 0; idy < NY; idy++) {
      out[idx + idy * NX] =
          static_cast<OutT>(compute(in1[idx], in2[idx + idy * NX]));
    }
  }
}

/**
 * @brief The Reduce provides collective methods for computing a parallel
 * reduction of items partitioned across a CUDA block and intra thread. When
 * ReduceMode == kLocalMode, thread reduce along nx. When ReduceMode ==
 * kGlobalMode, use shared memory to reduce between threads.
 *
 * @template paraments
 * T: The type of data.
 * NX: The number of data continuously loaded by each thread.
 * NY: The number of data rows loaded by each thread, only NY = 1 was supported.
 * BlockSize: Identifies the current device thread index method. For xpu,
 * core_id() is used as the index.
 * ReduceFunctor: Compute functor which has an operator() as following
 *     template <typename InT>
 *     struct ReduceFunctor {
 *       HOSTDEVICE InT operator()(const InT& a, const InT& b) const {
 *         return ...;
 *       }
 *     };
 * ReduceMode: Reduce mode, can be kLocalMode, kGlobalMode.
 *
 * @param
 * out: The register pointer of out, the size is NX * NY.
 * in: The register pointer of in, the size is NX * NY.
 * reducer: Compute function which was declared like ReduceFunctor<InT>().
 * reduce_last_dim: if the last dim gets involved in reduction.
 */

template <typename T,
          int NX,
          int NY,
          int BlockSize,
          class ReduceFunctor,
          details::ReduceMode Mode>
__device__ __forceinline__ void Reduce(T* out,
                                       const T* in,
                                       ReduceFunctor reducer,
                                       bool reduce_last_dim) {
  if (Mode == details::kGlobalMode) {
    if (reduce_last_dim) {
#pragma unroll
      for (int i = 0; i < NY * NX; i++) {  // reduce along blockDim.x
        details::BlockXReduce<T, ReduceFunctor, 1>(&out[i], &in[i], reducer);
      }
    }
  } else {  // else  kLocalMode
#pragma unroll
    for (int i = 0; i < NY; ++i) {
#pragma unroll
      for (int j = 0; j < NX; ++j) {
        out[i] = reducer(out[i], in[i * NX + j]);
      }
    }
  }
}

/*
 * @brief Fill register with a constant according to OpFunc
 *
 * @template paraments
 * InT: The data type of in1 and in2.
 * OutT: The data type of out.
 * NX: The number of data columns loaded by each thread.
 * NY: The number of data rows loaded by each thread.
 * BlockSize: Identifies the current device thread index method. For xpu,
 * core_id() is used as the index.
 * OpFunc: Compute functor which has an operator() as following
 *     template <typename InT>
 *     struct XxxFunctor {
 *       HOSTDEVICE InT operator()()
 * const {
 *         return a;
 *       }
 *     };
 *
 * @param
 * out: The register pointer of out, the size is NX * NY.
 * compute: Compute function which was declared like OpFunc<InT>().
 */
template <typename InT,
          typename OutT,
          int NX,
          int NY,
          int BlockSize,
          class OpFunc>
__device__ __forceinline__ void ElementwiseConstant(OutT* out, OpFunc compute) {
#pragma unroll
  for (int idx = 0; idx < NX * NY; idx++) {
    out[idx] = static_cast<OutT>(compute());
  }
}

}  // namespace kps
}  // namespace phi
