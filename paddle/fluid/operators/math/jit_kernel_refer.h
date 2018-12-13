/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
#include <cmath>
#include <string>
#include "paddle/fluid/operators/math/jit_kernel_impl.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace operators {
namespace math {
namespace jitkernel {
namespace refer {

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

}  // namespace refer
}  // namespace jitkernel
}  // namespace math
}  // namespace operators
}  // namespace paddle
