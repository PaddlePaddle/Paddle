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

#include "paddle/phi/kernels/funcs/jit/helper.h"

#include <numeric>

#include "paddle/phi/core/enforce.h"

namespace phi::jit {

std::map<size_t, std::shared_ptr<void>>& GetFuncCacheMap() {
  static thread_local std::map<size_t, std::shared_ptr<void>> g_func_cache_map;
  return g_func_cache_map;
}

#define ONE_CASE(key) \
  case key:           \
    return #key

const char* to_string(KernelType kt) {
  switch (kt) {
    ONE_CASE(kNone);
    ONE_CASE(kVMul);
    ONE_CASE(kVAdd);
    ONE_CASE(kVAddRelu);
    ONE_CASE(kVSub);
    ONE_CASE(kVScal);
    ONE_CASE(kVAddBias);
    ONE_CASE(kVRelu);
    ONE_CASE(kVBroadcast);
    ONE_CASE(kVCopy);
    ONE_CASE(kVIdentity);
    ONE_CASE(kVExp);
    ONE_CASE(kVSquare);
    ONE_CASE(kVSigmoid);
    ONE_CASE(kVTanh);
    ONE_CASE(kLSTMCtHt);
    ONE_CASE(kLSTMC1H1);
    ONE_CASE(kGRUH1);
    ONE_CASE(kGRUHtPart1);
    ONE_CASE(kGRUHtPart2);
    ONE_CASE(kCRFDecoding);
    ONE_CASE(kLayerNorm);
    ONE_CASE(kSeqPool);
    ONE_CASE(kMatMul);
    ONE_CASE(kAdam);
    ONE_CASE(kAdamW);
    ONE_CASE(kEmbSeqPool);
    ONE_CASE(kSgd);
    default:
      PADDLE_THROW(common::errors::Unimplemented(
          "JIT kernel do not support type: %d.", kt));
      return "NOT JITKernel";
  }
  return nullptr;
}

const char* to_string(SeqPoolType tp) {
  switch (tp) {
    ONE_CASE(kNonePoolType);
    ONE_CASE(kSum);
    ONE_CASE(kAvg);
    ONE_CASE(kSqrt);
    default:
      PADDLE_THROW(common::errors::Unimplemented(
          "SeqPool JIT kernel do not support type: %d.", tp));
      return "NOT PoolType";
  }
  return nullptr;
}
#undef ONE_CASE

KernelType to_kerneltype(const std::string& act) {
  std::string lower = act;
  std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
  if (lower == "relu" || lower == "vrelu") {
    return kVRelu;
  } else if (lower == "identity" || lower == "videntity" || lower.empty()) {
    return kVIdentity;
  } else if (lower == "exp" || lower == "vexp") {
    return kVExp;
  } else if (lower == "sigmoid" || lower == "vsigmoid") {
    return kVSigmoid;
  } else if (lower == "tanh" || lower == "vtanh") {
    return kVTanh;
  }
  PADDLE_THROW(common::errors::Unimplemented(
      "Act JIT kernel do not support type: %s.", act));
  return kNone;
}

template <>
void pack_weights<float>(const float* src, float* dst, int n, int k) {
  int block = 0, rest = 0;
  const auto groups = packed_groups(n, k, &block, &rest);
  std::for_each(groups.begin(), groups.end(), [&](int i) {
    PADDLE_ENFORCE_GT(i,
                      0,
                      common::errors::InvalidArgument(
                          "Each element of groups should be larger than "
                          "0. However the element: %d doesn't satisfy.",
                          i));
  });
  int sum = std::accumulate(groups.begin(), groups.end(), 0);
  std::memset(dst, 0, k * sum * block * sizeof(float));
  PADDLE_ENFORCE_GE(sum * block,
                    n,
                    common::errors::InvalidArgument(
                        "The packed n (sum * block) should be equal to or "
                        "larger than n (matmul row size). "
                        "However, the packed n is %d and n is %d.",
                        sum * block,
                        n));

  const int block_len = static_cast<int>(sizeof(float)) * block;
  int n_offset = 0;

  for (size_t g = 0; g < groups.size(); ++g) {
    const float* from = src + n_offset;
    for (int j = 0; j < k; ++j) {
      size_t copy_sz = groups[g] * block_len;
      if (g == groups.size() - 1 && rest != 0) {
        copy_sz = (groups[g] - 1) * block_len + rest * sizeof(float);
      }
      std::memcpy(dst, from + j * n, copy_sz);
      dst += groups[g] * block;
    }
    n_offset += groups[g] * block;
  }
}

template <typename T>
typename std::enable_if<!std::is_same<T, float>::value>::type pack_weights(
    const T* src, T* dst, int n, int k) {
  PADDLE_THROW(common::errors::Unimplemented(
      "Only supports pack weights with float type."));
}

}  // namespace phi::jit
