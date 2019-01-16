/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. */

#pragma once

#include <cmath>
#include <limits>
#include "paddle/fluid/operators/jit/helper.h"
#include "paddle/fluid/operators/jit/kernel_base.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace operators {
namespace jit {
namespace refer {

// Refer code only focus on correctness
template <typename T>
void VMul(const T* x, const T* y, T* z, int n) {
  for (int i = 0; i < n; ++i) {
    z[i] = x[i] * y[i];
  }
}

template <typename T>
void VAdd(const T* x, const T* y, T* z, int n) {
  for (int i = 0; i < n; ++i) {
    z[i] = x[i] + y[i];
  }
}

template <typename T>
void VAddRelu(const T* x, const T* y, T* z, int n) {
  for (int i = 0; i < n; ++i) {
    z[i] = x[i] + y[i];
    z[i] = z[i] > 0 ? z[i] : 0;
  }
}

template <typename T>
void VSub(const T* x, const T* y, T* z, int n) {
  for (int i = 0; i < n; ++i) {
    z[i] = x[i] - y[i];
  }
}

template <typename T>
void VScal(const T* a, const T* x, T* y, int n) {
  for (int i = 0; i < n; ++i) {
    y[i] = a[0] * x[i];
  }
}

template <typename T>
void VAddBias(const T* a, const T* x, T* y, int n) {
  for (int i = 0; i < n; ++i) {
    y[i] = a[0] + x[i];
  }
}

template <typename T>
void VRelu(const T* x, T* y, int n) {
  for (int i = 0; i < n; ++i) {
    y[i] = x[i] > 0 ? x[i] : 0;
  }
}

template <typename T>
inline void VIdentity(const T* x, T* y, int n) {
  for (int i = 0; i < n; ++i) {
    y[i] = x[i];
  }
}

template <typename T>
inline void VSquare(const T* x, T* y, int n) {
  for (int i = 0; i < n; ++i) {
    y[i] = x[i] * x[i];
  }
}

template <typename T>
void VExp(const T* x, T* y, int n) {
  for (int i = 0; i < n; ++i) {
    y[i] = std::exp(x[i]);
  }
}

template <typename T>
void VSigmoid(const T* x, T* y, int n) {
  // y = 1 / (1 + e^-x)
  const T min = SIGMOID_THRESHOLD_MIN;
  const T max = SIGMOID_THRESHOLD_MAX;
  for (int i = 0; i < n; ++i) {
    T tmp = (x[i] < min) ? min : ((x[i] > max) ? max : x[i]);
    y[i] = static_cast<T>(1) / (static_cast<T>(1) + std::exp(-tmp));
  }
}

template <typename T>
void VTanh(const T* x, T* y, int n) {
  // y = 2 * sigmoid(2x) - 1
  for (int i = 0; i < n; ++i) {
    y[i] = static_cast<T>(2) * x[i];
  }
  VSigmoid(y, y, n);
  for (int i = 0; i < n; ++i) {
    y[i] = static_cast<T>(2) * y[i] - static_cast<T>(1);
  }
}

template <typename T>
void (*getActFunc(KernelType type))(const T*, T*, int) {  // NOLINT
  if (type == kVSigmoid) {
    return VSigmoid<T>;
  } else if (type == kVRelu) {
    return VRelu<T>;
  } else if (type == kVTanh) {
    return VTanh<T>;
  } else if (type == kVIdentity) {
    return VIdentity<T>;
  }
  PADDLE_THROW("Not support type: %s", type);
  return nullptr;
}

// TODO(TJ): add refer gemm and make LSTM kernels combine as same GRU kernels

// compute ct and ht
template <typename T>
void LSTMCtHt(lstm_t* step, const lstm_attr_t* attr) {
  T* gates = reinterpret_cast<T*>(step->gates);
  const T* ct_1 = reinterpret_cast<const T*>(step->ct_1);
  T* ct = reinterpret_cast<T*>(step->ct);
  T* ht = reinterpret_cast<T*>(step->ht);
  const T* wp = reinterpret_cast<const T*>(step->wp);
  T* checked = reinterpret_cast<T*>(step->checked);
  auto act_gate = getActFunc<T>(attr->act_gate);
  auto act_cand = getActFunc<T>(attr->act_cand);
  auto act_cell = getActFunc<T>(attr->act_cell);
  int d = attr->d;
  int d2 = d * 2;
  int d3 = d * 3;
  // gates: W_ch, W_ih, W_fh, W_oh
  if (attr->use_peephole) {
    VMul(wp, ct_1, checked, d);
    VMul(wp + d, ct_1, checked + d, d);
    VAdd(checked, gates + d, gates + d, d2);
    act_gate(gates + d, gates + d, d2);
  } else {
    act_gate(gates + d, gates + d, d3);
  }

  // C_t = C_t-1 * fgated + cand_gated * igated
  act_cand(gates, gates, d);
  VMul(gates, gates + d, gates + d, d);
  VMul(ct_1, gates + d2, gates + d2, d);
  VAdd(gates + d, gates + d2, ct, d);

  if (attr->use_peephole) {
    // get ogated
    VMul(wp + d2, ct, gates + d, d);
    VAdd(gates + d, gates + d3, gates + d3, d);
    act_gate(gates + d3, gates + d3, d);
  }
  // H_t = act_cell(C_t) * ogated
  act_cell(ct, gates + d2, d);
  VMul(gates + d2, gates + d3, ht, d);
}

// compute c1 and h1 without c0 or h0
template <typename T>
void LSTMC1H1(lstm_t* step, const lstm_attr_t* attr) {
  T* gates = reinterpret_cast<T*>(step->gates);
  T* ct = reinterpret_cast<T*>(step->ct);
  T* ht = reinterpret_cast<T*>(step->ht);
  auto act_gate = getActFunc<T>(attr->act_gate);
  auto act_cand = getActFunc<T>(attr->act_cand);
  auto act_cell = getActFunc<T>(attr->act_cell);
  int d = attr->d;
  int d2 = d * 2;
  int d3 = d * 3;
  /* C_t = igated * cgated*/
  act_gate(gates + d, gates + d, d);
  act_cand(gates, gates, d);
  VMul(gates, gates + d, ct, d);
  if (attr->use_peephole) {
    // get outgated, put W_oc * C_t on igated
    const T* wp = reinterpret_cast<const T*>(step->wp);
    VMul(wp + d2, ct, gates + d, d);
    VAdd(gates + d, gates + d3, gates + d3, d);
  }
  /* H_t = act_cell(C_t) * ogated */
  act_gate(gates + d3, gates + d3, d);
  act_cell(ct, gates + d2, d);
  VMul(gates + d2, gates + d3, ht, d);
}

// compute h1 without h0
template <typename T>
void GRUH1(gru_t* step, const gru_attr_t* attr) {
  T* gates = reinterpret_cast<T*>(step->gates);
  T* ht = reinterpret_cast<T*>(step->ht);
  auto act_gate = getActFunc<T>(attr->act_gate);
  auto act_cand = getActFunc<T>(attr->act_cand);
  int d = attr->d;
  int d2 = d * 2;
  act_gate(gates, gates, d);
  act_cand(gates + d2, gates + d2, d);
  VMul(gates, gates + d2, ht, d);
}

// compute the first part of GRU: ht = act_gate(r) * ht_1
template <typename T>
void GRUHtPart1(gru_t* step, const gru_attr_t* attr) {
  // W: {W_update, W_reset; W_state}
  T* gates = reinterpret_cast<T*>(step->gates);
  T* ht = reinterpret_cast<T*>(step->ht);
  const T* ht_1 = reinterpret_cast<const T*>(step->ht_1);
  auto act_gate = getActFunc<T>(attr->act_gate);
  act_gate(gates + attr->d, gates + attr->d, attr->d);
  VMul(ht_1, gates + attr->d, ht, attr->d);
}

// compute the second part of GRU:
// ht = act_gate(u) * act_cand(s) + (1-act_gate(u)) * ht_1
template <typename T>
void GRUHtPart2(gru_t* step, const gru_attr_t* attr) {
  T* gates = reinterpret_cast<T*>(step->gates);
  T* ht = reinterpret_cast<T*>(step->ht);
  const T* ht_1 = reinterpret_cast<const T*>(step->ht_1);
  auto act_gate = getActFunc<T>(attr->act_gate);
  auto act_cand = getActFunc<T>(attr->act_cand);
  int d = attr->d;
  T* y = gates + d * 2;
  act_gate(gates, gates, d);
  act_cand(y, y, d);
  // out = zt*ht~ + (1-zt)*ht_1
  for (int i = 0; i < d; ++i) {
    ht[i] = gates[i] * y[i] + (static_cast<T>(1) - gates[i]) * ht_1[i];
  }
}

template <typename T>
void CRFDecoding(const int seq_len, const T* x, const T* w, T* alpha,
                 int* track, int right) {
  constexpr int state_trans_base_idx = 2;
  for (int i = 0; i < right; ++i) {
    alpha[i] = w[i] + x[i];
  }
  for (int k = 1; k < seq_len; ++k) {
    for (int i = 0; i < right; ++i) {
      T max_score = -std::numeric_limits<T>::max();
      int max_j = 0;
      for (int j = 0; j < right; ++j) {
        T score = alpha[(k - 1) * right + j] +
                  w[(j + state_trans_base_idx) * right + i];
        if (score > max_score) {
          max_score = score;
          max_j = j;
        }
      }
      alpha[k * right + i] = max_score + x[k * right + i];
      track[k * right + i] = max_j;
    }
  }
}

template <typename T>
void LayerNorm(T* x, T* out, T* mean, T* var, const T* scale, const T* bias,
               int height, const float epsilon, int right) {
  // get mean
  for (int i = 0; i < height; i++) {
    T sum = 0.0;
    int offset = i * right;
    for (int j = 0; j < right; j++) {
      sum += x[offset + j];
    }
    mean[i] = sum / right;
  }

  // get variance
  for (int i = 0; i < height; i++) {
    T sum = 0.0;
    int offset = i * right;
    for (int j = 0; j < right; j++) {
      sum += (x[offset + j] - mean[i]) * (x[offset + j] - mean[i]);
    }
    var[i] = sum / right;
  }

  for (int i = 0; i < height; i++) {
    int offset = i * right;
    T sqrt_var = std::sqrt(var[i] + (T)epsilon);
    for (int j = 0; j < right; j++) {
      out[offset + j] = (x[offset + j] - mean[i]) / sqrt_var;
    }
  }
  if (scale) {
    for (int i = 0; i < height; i++) {
      int offset = i * right;
      for (int j = 0; j < right; j++) {
        out[offset + j] *= scale[j];
      }
    }
  }

  if (bias) {
    for (int i = 0; i < height; i++) {
      int offset = i * right;
      for (int j = 0; j < right; j++) {
        out[offset + j] += bias[j];
      }
    }
  }
}

template <typename T>
void NCHW16CMulNC(const T* x, const T* y, T* z, int height, int width) {
  int offset = 0;
  for (int h = 0; h < height; ++h) {
    for (int w = 0; w < width; ++w) {
      for (int i = 0; i < 16; ++i) {
        z[i + offset] = y[i] * x[i + offset];
      }
      offset += ZMM_FLOAT_BLOCK;
    }
  }
}

template <typename T>
void SeqPool(const T* x, T* y, const seq_pool_attr_t* attr) {
  for (int w = 0; w < attr->w; ++w) {
    const T* src = x + w;
    T* dst = y + w;
    *dst = static_cast<T>(0);
    for (int h = 0; h < attr->h; ++h) {
      *dst = *dst + *src;
      src += attr->w;
    }
  }
  if (attr->type == SeqPoolType::kAvg || attr->type == SeqPoolType::kSqrt) {
    T scalar = static_cast<T>(1);
    if (attr->type == SeqPoolType::kAvg) {
      scalar = scalar / static_cast<T>(attr->h);
    } else {
      scalar = scalar / std::sqrt(static_cast<T>(attr->h));
    }
    VScal<T>(&scalar, y, y, attr->w);
  }
}

// A(M,K) * B(K,N) = C(M,N)
template <typename T>
void MatMul(const T* A, const T* B, T* C, int M, int N, int K) {
  for (int m = 0; m < M; ++m) {
    const T* pa = A + m * K;
    T* pc = C + m * N;
    for (int n = 0; n < N; ++n) {
      const T* pb = B + n;
      T sum = static_cast<T>(0);
      for (int k = 0; k < K; ++k) {
        sum += (pa[k] * pb[k * N]);
      }
      *(pc + n) = sum;
    }
  }
}

#define DECLARE_REFER_KERNEL(name, tuples)             \
  template <typename T>                                \
  class name##Kernel : public ReferKernel<tuples<T>> { \
   public:                                             \
    name##Kernel() { this->func = name<T>; }           \
  }

// const T* x, const T* y, T* z, int n
DECLARE_REFER_KERNEL(VMul, XYZNTuples);
DECLARE_REFER_KERNEL(VAdd, XYZNTuples);
DECLARE_REFER_KERNEL(VAddRelu, XYZNTuples);
DECLARE_REFER_KERNEL(VSub, XYZNTuples);

// const T* a, const T* x, T* y, int n
DECLARE_REFER_KERNEL(VScal, AXYNTuples);
DECLARE_REFER_KERNEL(VAddBias, AXYNTuples);

// const T* x, T* y, int n
DECLARE_REFER_KERNEL(VRelu, XYNTuples);
DECLARE_REFER_KERNEL(VIdentity, XYNTuples);
DECLARE_REFER_KERNEL(VExp, XYNTuples);
DECLARE_REFER_KERNEL(VSigmoid, XYNTuples);
DECLARE_REFER_KERNEL(VTanh, XYNTuples);
DECLARE_REFER_KERNEL(VSquare, XYNTuples);

// lstm_t*, const lstm_attr_t*
DECLARE_REFER_KERNEL(LSTMCtHt, LSTMTuples);
DECLARE_REFER_KERNEL(LSTMC1H1, LSTMTuples);

// gru_t*, const gru_attr_t*
DECLARE_REFER_KERNEL(GRUH1, GRUTuples);
DECLARE_REFER_KERNEL(GRUHtPart1, GRUTuples);
DECLARE_REFER_KERNEL(GRUHtPart2, GRUTuples);

DECLARE_REFER_KERNEL(CRFDecoding, CRFDecodingTuples);
DECLARE_REFER_KERNEL(LayerNorm, LayerNormTuples);

DECLARE_REFER_KERNEL(NCHW16CMulNC, NCHW16CMulNCTuples);

DECLARE_REFER_KERNEL(SeqPool, SeqPoolTuples);

DECLARE_REFER_KERNEL(MatMul, MatMulTuples);

#undef DECLARE_REFER_KERNEL

}  // namespace refer
}  // namespace jit
}  // namespace operators
}  // namespace paddle
