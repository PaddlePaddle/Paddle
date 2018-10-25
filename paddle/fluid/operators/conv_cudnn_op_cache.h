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

#pragma once

#include <functional>
#include <unordered_map>
#include <vector>

namespace paddle {
namespace operators {

template <typename TAlgorithm>
class AlgorithmsCache {
 public:
  // Caches the best algorithm for a given
  // combination of tensor dimensions & compute data type.
  TAlgorithm GetAlgorithm(
      const std::vector<int64_t>& dims1, const std::vector<int64_t>& dims2,
      const std::vector<int>& strides, const std::vector<int>& paddings,
      const std::vector<int>& dilations,
      int algorithmFlags,  // can set for different data type
      std::function<TAlgorithm()> gen_func);

 private:
  std::unordered_map<int64_t, TAlgorithm> hash_;
  std::mutex mutex_;
};

template <typename TAlgorithm>
TAlgorithm AlgorithmsCache<TAlgorithm>::GetAlgorithm(
    const std::vector<int64_t>& dims1, const std::vector<int64_t>& dims2,
    const std::vector<int>& strides, const std::vector<int>& paddings,
    const std::vector<int>& dilations, int algorithmFlags,
    std::function<TAlgorithm()> gen_func) {
  std::lock_guard<std::mutex> lock(mutex_);
  int64_t seed = 0;
  // Hash all of the inputs, use to try and look up a previously
  // discovered algorithm, or fall back to generating a new one.
  std::hash<int64_t> hashFn;
  // do hash like boost
  // https://stackoverflow.com/questions/2590677/how-do-i-combine-hash-values-in-c0x
  for (const auto num : dims1) {
    seed ^= hashFn(num) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  }

  for (const auto num : dims2) {
    seed ^= hashFn(num) + 0x9e3779b9 + (seed << 6) + (seed >> 2) + 1;
  }

  for (const auto num : strides) {
    seed ^= hashFn(static_cast<int64_t>(num)) + 0x9e3779b9 + (seed << 6) +
            (seed >> 2) + 2;
  }

  for (const auto num : paddings) {
    seed ^= hashFn(static_cast<int64_t>(num)) + 0x9e3779b9 + (seed << 6) +
            (seed >> 2) + 3;
  }

  for (const auto num : dilations) {
    seed ^= hashFn(static_cast<int64_t>(num)) + 0x9e3779b9 + (seed << 6) +
            (seed >> 2) + 4;
  }

  seed ^= hashFn(static_cast<int64_t>(algorithmFlags)) + 0x9e3779b9 +
          (seed << 6) + (seed >> 2) + 5;

  if (seed == 0) return gen_func();

  if (hash_.find(seed) == hash_.end()) {
    TAlgorithm value = gen_func();
    hash_[seed] = value;
  }
  return hash_[seed];
}

}  // namespace operators
}  // namespace paddle
