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
  //
  TAlgorithm getAlgorithm(
      const std::vector<int64_t>& tensorDimensions1,
      const std::vector<int64_t>& tensorDimensions2,
      const std::vector<int>& strides, const std::vector<int>& paddings,
      const std::vector<int>& dilations,
      int algorithmFlags,  // Differentiate between algorithms with different
                           // parameters in a generic way
      std::function<TAlgorithm()> generatingFunc);

 private:
  std::unordered_map<int64_t, TAlgorithm> hash_;
};

template <typename TAlgorithm>
TAlgorithm AlgorithmsCache<TAlgorithm>::getAlgorithm(
    const std::vector<int64_t>& tensorDimensions1,
    const std::vector<int64_t>& tensorDimensions2,
    const std::vector<int>& strides, const std::vector<int>& paddings,
    const std::vector<int>& dilations, int algorithmFlags,
    std::function<TAlgorithm()> generatingFunc) {
  int64_t seed = 0;
  // Hash all of the inputs, which we wiill then use to try and look up
  // a previously discovered algorithm, or fall back to generating a new one.
  std::hash<int64_t> hashFn;
  for (const auto num : tensorDimensions1) {
    // Copied from boost::hash_combine.
    seed ^= hashFn(num) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  }

  for (const auto num : tensorDimensions2) {
    // Copied from boost::hash_combine.
    // Adding 1 to differentiate between first and second vector.
    seed ^= hashFn(num) + 0x9e3779b9 + (seed << 6) + (seed >> 2) + 1;
  }

  for (const auto num : strides) {
    // Copied from boost::hash_combine.
    // Adding 1 to differentiate between first and second vector.
    seed ^= hashFn(static_cast<int64_t>(num)) + 0x9e3779b9 + (seed << 6) +
            (seed >> 2) + 2;
  }

  for (const auto num : paddings) {
    // Copied from boost::hash_combine.
    // Adding 1 to differentiate between first and second vector.
    seed ^= hashFn(static_cast<int64_t>(num)) + 0x9e3779b9 + (seed << 6) +
            (seed >> 2) + 3;
  }

  for (const auto num : dilations) {
    // Copied from boost::hash_combine.
    // Adding 1 to differentiate between first and second vector.
    seed ^= hashFn(static_cast<int64_t>(num)) + 0x9e3779b9 + (seed << 6) +
            (seed >> 2) + 4;
  }

  // Adding 2 to differentiate from previous vectors
  seed ^= hashFn(static_cast<int64_t>(algorithmFlags)) + 0x9e3779b9 +
          (seed << 6) + (seed >> 2) + 5;

  if (seed == 0) {
    LOG(ERROR) << "==== get func====";
    return generatingFunc();
  }

  if (hash_.find(seed) == hash_.end()) {
    LOG(ERROR) << "==== get func====";
    TAlgorithm value = generatingFunc();
    hash_[seed] = value;
  } else {
    LOG(ERROR) << "=== use cache func====";
  }
  return hash_[seed];
}

}  // namespace operators
}  // namespace paddle
