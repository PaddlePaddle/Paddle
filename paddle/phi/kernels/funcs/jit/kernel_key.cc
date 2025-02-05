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

#include "paddle/phi/kernels/funcs/jit/kernel_key.h"

#include <xxhash.h>  // XXH64: 13.8 GB/s
#include <array>

namespace phi::jit {

template <>
int64_t JitCodeKey<int>(const int& d) {
  return d;
}

template <>
int64_t JitCodeKey<int64_t>(const int64_t& d) {
  return d;
}

template <>
int64_t JitCodeKey<gru_attr_t>(const gru_attr_t& attr) {
  return static_cast<int64_t>(XXH64(&attr, sizeof(gru_attr_t), 0));
}

template <>
int64_t JitCodeKey<lstm_attr_t>(const lstm_attr_t& attr) {
  std::array<int, 5> keys = {attr.d,
                             static_cast<int>(attr.act_gate),
                             static_cast<int>(attr.act_cand),
                             static_cast<int>(attr.act_cell),
                             static_cast<int>(attr.use_peephole)};
  return static_cast<int64_t>(XXH64(keys.data(), sizeof(int) * 5, 0));
}

template <>
int64_t JitCodeKey<seq_pool_attr_t>(const seq_pool_attr_t& attr) {
  std::array<int, 2> keys = {attr.w, static_cast<int>(attr.type)};
  return static_cast<int64_t>(XXH64(keys.data(), sizeof(int) * 2, 0));
}

template <>
int64_t JitCodeKey<matmul_attr_t>(const matmul_attr_t& attr) {
  return static_cast<int64_t>(XXH64(&attr, sizeof(int) * 3, 0));  // m, n, k
}

template <>
int64_t JitCodeKey<emb_seq_pool_attr_t>(const emb_seq_pool_attr_t& attr) {
  return attr.table_width;
}

template <>
int64_t JitCodeKey<sgd_attr_t>(const sgd_attr_t& attr) {
  return attr.grad_width;
}

template <>
int64_t JitCodeKey<adam_attr_t>(const adam_attr_t& attr) {
  // if use amsgrad, we add `10` for hashcode
  return static_cast<int64_t>(attr.beta1 + attr.beta2 +
                              (attr.amsgrad ? 10 : 0));
}

template <>
int64_t JitCodeKey<adamw_attr_t>(const adamw_attr_t& attr) {
  // if use amsgrad, we add `10` for hashcode
  return static_cast<int64_t>(attr.beta1 + attr.beta2 + attr.coeff +
                              (attr.amsgrad ? 10 : 0));
}

}  // namespace phi::jit
