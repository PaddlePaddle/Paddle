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

#include "paddle/fluid/operators/jit/kernel_key.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace operators {
namespace jit {

template <>
size_t JitCodeKey<int>(const int& d) {
  return d;
}

// TODO(TJ): refine and benchmark JitCodeKey generatation
constexpr int act_type_shift = 3;  // suppot 2^3 act types
static inline int act_type_convert(KernelType type) {
  if (type == kVIdentity) {
    return 0;
  } else if (type == kVExp) {
    return 1;
  } else if (type == kVRelu) {
    return 2;
  } else if (type == kVSigmoid) {
    return 3;
  } else if (type == kVTanh) {
    return 4;
  }
  PADDLE_THROW("Unsupported act type %d", type);
  return 0;
}

template <>
size_t JitCodeKey<lstm_attr_t>(const lstm_attr_t& attr) {
  size_t key = attr.d;
  int gate_key = act_type_convert(attr.act_gate) << 1;
  int cand_key = act_type_convert(attr.act_cand) << (1 + act_type_shift);
  int cell_key = act_type_convert(attr.act_cell) << (1 + act_type_shift * 2);
  return (key << (1 + act_type_shift * 3)) + gate_key + cand_key + cell_key +
         attr.use_peephole;
}

template <>
size_t JitCodeKey<gru_attr_t>(const gru_attr_t& attr) {
  size_t key = attr.d;
  return (key << (act_type_shift * 2)) + act_type_convert(attr.act_gate) +
         (act_type_convert(attr.act_cand) << act_type_shift);
}

template <>
size_t JitCodeKey<seq_pool_attr_t>(const seq_pool_attr_t& attr) {
  size_t key = attr.w;
  constexpr int pool_type_shift = 3;
  return (key << pool_type_shift) + static_cast<int>(attr.type);
}

template <>
size_t JitCodeKey<matmul_attr_t>(const matmul_attr_t& attr) {
  size_t key = attr.m;
  constexpr int shift = 21;
  return (key << shift * 2) + ((static_cast<size_t>(attr.n)) << shift) + attr.k;
}

template <>
size_t JitCodeKey<emb_seq_pool_attr_t>(const emb_seq_pool_attr_t& attr) {
  return attr.table_width;
}

template <>
size_t JitCodeKey<sgd_attr_t>(const sgd_attr_t& attr) {
  return attr.grad_width;
}

}  // namespace jit
}  // namespace operators
}  // namespace paddle
