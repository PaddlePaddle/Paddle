/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/funcs/jit/more/mix/mix.h"

#include "paddle/phi/kernels/funcs/jit/kernels.h"
#include "paddle/phi/kernels/funcs/jit/registry.h"

namespace phi::jit::more::mix {

using CPUPlace = phi::CPUPlace;

void VSigmoid(const T* x, T* y, int n) {
  const float min = SIGMOID_THRESHOLD_MIN;
  const float max = SIGMOID_THRESHOLD_MAX;
  for (int i = 0; i < n; ++i) {
    y[i] = (x[i] < min) ? min : ((x[i] > max) ? max : x[i]);
    y[i] = static_cast<T>(0) - y[i];
  }
  auto compute = KernelFuncs<VExpTuple<T>, CPUPlace>::Cache().At(n);
  compute(y, y, n);
  for (int i = 0; i < n; ++i) {
    y[i] = static_cast<T>(1) / (static_cast<T>(1) + y[i]);
  }
}

void VTanh(const T* x, T* y, int n) {
  const T a = 2, b = -1;
  auto compute_scal = KernelFuncs<VScalTuple<T>, CPUPlace>::Cache().At(n);
  auto compute_addbias = KernelFuncs<VAddBiasTuple<T>, CPUPlace>::Cache().At(n);
  auto compute_sigmoid = KernelFuncs<VSigmoidTuple<T>, CPUPlace>::Cache().At(n);
  compute_scal(&a, x, y, n);
  compute_sigmoid(y, y, n);
  compute_scal(&a, y, y, n);
  compute_addbias(&b, y, y, n);
}

void (*getActFunc(KernelType type, int d))(const T*, T*, int) {  // NOLINT
  if (type == kVSigmoid) {
    return KernelFuncs<VSigmoidTuple<T>, CPUPlace>::Cache().At(d);
  } else if (type == kVRelu) {
    return KernelFuncs<VReluTuple<T>, CPUPlace>::Cache().At(d);
  } else if (type == kVTanh) {
    return KernelFuncs<VTanhTuple<T>, CPUPlace>::Cache().At(d);
  } else if (type == kVIdentity) {
    return KernelFuncs<VIdentityTuple<T>, CPUPlace>::Cache().At(d);
  }
  PADDLE_THROW(common::errors::Unimplemented(
      "Act JIT kernel do not support type: %s", type));
  return nullptr;
}

void LSTMCtHt(lstm_t* step, const lstm_attr_t* attr) {
  T* gates = reinterpret_cast<T*>(step->gates);
  const T* ct_1 = reinterpret_cast<const T*>(step->ct_1);
  T* ct = reinterpret_cast<T*>(step->ct);
  T* ht = reinterpret_cast<T*>(step->ht);
  const T* wp = reinterpret_cast<const T*>(step->wp);
  T* checked = reinterpret_cast<T*>(step->checked);
  const int d = attr->d;
  const int d2 = d * 2;
  const int d3 = d * 3;
  auto vmul_d = KernelFuncs<VMulTuple<T>, CPUPlace>::Cache().At(d);
  auto vadd_d = KernelFuncs<VAddTuple<T>, CPUPlace>::Cache().At(d);
  auto vadd_d2 = KernelFuncs<VAddTuple<T>, CPUPlace>::Cache().At(d2);
  auto act_gate_d = getActFunc(attr->act_gate, d);
  auto act_gate_d2 = getActFunc(attr->act_gate, d2);
  auto act_gate_d3 = getActFunc(attr->act_gate, d3);
  auto act_cand_d = getActFunc(attr->act_cand, d);
  auto act_cell_d = getActFunc(attr->act_cell, d);

  if (attr->use_peephole) {
    vmul_d(wp, ct_1, checked, d);
    vmul_d(wp + d, ct_1, checked + d, d);
    vadd_d2(checked, gates + d, gates + d, d2);
    act_gate_d2(gates + d, gates + d, d2);
  } else {
    act_gate_d3(gates + d, gates + d, d3);
  }

  // C_t = C_t-1 * fgated + cand_gated * igated
  act_cand_d(gates, gates, d);
  vmul_d(gates, gates + d, gates + d, d);
  vmul_d(ct_1, gates + d2, gates + d2, d);
  vadd_d(gates + d, gates + d2, ct, d);

  if (attr->use_peephole) {
    // get ogated
    vmul_d(wp + d2, ct, gates + d, d);
    vadd_d(gates + d, gates + d3, gates + d3, d);
    act_gate_d(gates + d3, gates + d3, d);
  }
  // H_t = act_cell(C_t) * ogated
  act_cell_d(ct, gates + d2, d);
  vmul_d(gates + d2, gates + d3, ht, d);
}

void LSTMC1H1(lstm_t* step, const lstm_attr_t* attr) {
  T* gates = reinterpret_cast<T*>(step->gates);
  T* ct = reinterpret_cast<T*>(step->ct);
  T* ht = reinterpret_cast<T*>(step->ht);
  int d = attr->d;
  int d2 = d * 2;
  int d3 = d * 3;
  auto vmul_d = KernelFuncs<VMulTuple<T>, CPUPlace>::Cache().At(d);
  auto vadd_d = KernelFuncs<VAddTuple<T>, CPUPlace>::Cache().At(d);
  auto act_gate_d = getActFunc(attr->act_gate, d);
  auto act_cand_d = getActFunc(attr->act_cand, d);
  auto act_cell_d = getActFunc(attr->act_cell, d);
  /* C_t = igated * cgated*/
  act_gate_d(gates + d, gates + d, d);
  act_cand_d(gates, gates, d);
  vmul_d(gates, gates + d, ct, d);
  if (attr->use_peephole) {
    // get outgated, put W_oc * C_t on igated
    const T* wp = reinterpret_cast<const T*>(step->wp);
    vmul_d(wp + d2, ct, gates + d, d);
    vadd_d(gates + d, gates + d3, gates + d3, d);
  }
  /* H_t = act_cell(C_t) * ogated */
  act_gate_d(gates + d3, gates + d3, d);
  act_cell_d(ct, gates + d2, d);
  vmul_d(gates + d2, gates + d3, ht, d);
}

// compute h1 without h0
void GRUH1(gru_t* step, const gru_attr_t* attr) {
  T* gates = reinterpret_cast<T*>(step->gates);
  T* ht = reinterpret_cast<T*>(step->ht);
  int d = attr->d;
  int d2 = d * 2;
  auto act_gate = getActFunc(attr->act_gate, d);
  auto act_cand = getActFunc(attr->act_cand, d);
  auto vmul_d = KernelFuncs<VMulTuple<T>, CPUPlace>::Cache().At(d);
  act_gate(gates, gates, d);
  act_cand(gates + d2, gates + d2, d);
  vmul_d(gates, gates + d2, ht, d);
}

// compute the first part of GRU: ht = act_gate(r) * ht_1
void GRUHtPart1(gru_t* step, const gru_attr_t* attr) {
  // W: {W_update, W_reset; W_state}
  T* gates = reinterpret_cast<T*>(step->gates);
  T* ht = reinterpret_cast<T*>(step->ht);
  const T* ht_1 = reinterpret_cast<const T*>(step->ht_1);
  auto act_gate = getActFunc(attr->act_gate, attr->d);
  auto vmul_d = KernelFuncs<VMulTuple<T>, CPUPlace>::Cache().At(attr->d);
  act_gate(gates + attr->d, gates + attr->d, attr->d);
  vmul_d(ht_1, gates + attr->d, ht, attr->d);
}

// compute the second part of GRU:
// ht = act_gate(u) * act_cand(s) + (1-act_gate(u)) * ht_1
void GRUHtPart2(gru_t* step, const gru_attr_t* attr) {
  T* gates = reinterpret_cast<T*>(step->gates);
  T* ht = reinterpret_cast<T*>(step->ht);
  const T* ht_1 = reinterpret_cast<const T*>(step->ht_1);
  int d = attr->d;
  auto act_gate = getActFunc(attr->act_gate, d);
  auto act_cand = getActFunc(attr->act_cand, d);
  T* y = gates + d * 2;
  act_gate(gates, gates, d);
  act_cand(y, y, d);
  // out = zt*ht~ + (1-zt)*ht_1
  for (int i = 0; i < d; ++i) {
    ht[i] = gates[i] * y[i] + (static_cast<T>(1) - gates[i]) * ht_1[i];
  }
}

// TODO(TJ): tuning me
bool VSigmoidKernel::CanBeUsed(const int& d) const { return true; }

bool VTanhKernel::CanBeUsed(const int& d) const { return true; }

bool LSTMCtHtKernel::CanBeUsed(const lstm_attr_t& attr) const { return true; }

bool LSTMC1H1Kernel::CanBeUsed(const lstm_attr_t& attr) const { return true; }

bool GRUH1Kernel::CanBeUsed(const gru_attr_t& attr) const { return true; }

bool GRUHtPart1Kernel::CanBeUsed(const gru_attr_t& attr) const { return true; }

bool GRUHtPart2Kernel::CanBeUsed(const gru_attr_t& attr) const { return true; }

}  // namespace phi::jit::more::mix

namespace mix = phi::jit::more::mix;

#define REGISTER_MORE_KERNEL(func) \
  REGISTER_JITKERNEL_MORE(k##func, mix, mix::func##Kernel)

REGISTER_MORE_KERNEL(VSigmoid);
REGISTER_MORE_KERNEL(VTanh);
REGISTER_MORE_KERNEL(LSTMCtHt);
REGISTER_MORE_KERNEL(LSTMC1H1);
REGISTER_MORE_KERNEL(GRUH1);
REGISTER_MORE_KERNEL(GRUHtPart1);
REGISTER_MORE_KERNEL(GRUHtPart2);

#undef REGISTER_MORE_KERNEL
