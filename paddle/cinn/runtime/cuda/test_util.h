// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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
#include <numeric>
#include <vector>

#include "paddle/cinn/backends/cuda_util.h"

namespace cinn {
namespace runtime {
namespace cuda {
namespace util {

template <typename T>
class Vector {
 public:
  explicit Vector(const std::vector<T>& other) : size_{other.size()} {
    size_t bytes = sizeof(T) * size_;
    CUDA_CALL(cudaMalloc(&ptr_, bytes));
    CUDA_CALL(cudaMemcpy(ptr_, other.data(), bytes, cudaMemcpyHostToDevice));
  }
  explicit Vector(size_t size) : size_{size} {
    size_t bytes = sizeof(T) * size_;
    CUDA_CALL(cudaMalloc(&ptr_, bytes));
    CUDA_CALL(cudaMemset(ptr_, 0, bytes));
  }
  std::vector<T> to_host() const {
    std::vector<T> ret(size_);
    size_t bytes = sizeof(T) * size_;
    CUDA_CALL(cudaMemcpy(ret.data(), ptr_, bytes, cudaMemcpyDeviceToHost));
    return ret;
  }
  ~Vector() { CUDA_CALL(cudaFree(ptr_)); }
  size_t size() const { return size_; }
  T* data() const { return ptr_; }

 private:
  size_t size_{};
  T* ptr_{};
};

}  // namespace util
}  // namespace cuda
}  // namespace runtime
}  // namespace cinn
