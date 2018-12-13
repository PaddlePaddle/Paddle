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
  if (type == vsigmoid) {
    return VSigmoid<T>;
  } else if (type == vrelu) {
    return VRelu<T>;
  } else if (type == vtanh) {
    return VTanh<T>;
  } else if (type == videntity) {
    return VIdentity<T>;
  }
  PADDLE_THROW("Not support type: %s", type);
  return nullptr;
}

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

// lstm_t* , const lstm_attr_t*
DECLARE_REFER_KERNEL(LSTMCtHt, LSTMTuples);
DECLARE_REFER_KERNEL(LSTMC1H1, LSTMTuples);

#undef DECLARE_REFER_KERNEL

}  // namespace refer
}  // namespace jit
}  // namespace operators
}  // namespace paddle
