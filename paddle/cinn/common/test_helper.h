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

#include <string>
#include <vector>

#include "paddle/cinn/backends/llvm/codegen_llvm.h"
#include "paddle/cinn/backends/llvm/simple_jit.h"
#include "paddle/cinn/cinn.h"
#include "paddle/common/enforce.h"
namespace cinn {
namespace common {

/**
 * Create buffer for test.
 *
 * usage:
 *
 * auto* buf = BufferBuilder(Float(32), {20, 20}).set_random().Build();
 */
struct BufferBuilder {
  enum class InitType {
    kRandom = 0,
    kZero = 1,
    kSetValue = 2,
  };
  explicit BufferBuilder(Type type, const std::vector<int>& shape)
      : type_(type), shape_(shape) {}

  BufferBuilder& set_random() {
    init_type_ = InitType::kRandom;
    return *this;
  }

  BufferBuilder& set_zero() {
    init_type_ = InitType::kZero;
    return *this;
  }

  BufferBuilder& set_val(float x) {
    init_type_ = InitType::kSetValue;
    init_val_ = x;
    return *this;
  }

  BufferBuilder& set_align(int align) {
    align_ = align;
    return *this;
  }

  cinn_buffer_t* Build();

 private:
  template <typename T>
  void RandomFloat(void* arr, uint64_t len) {
    auto* data = static_cast<T*>(arr);
    for (uint64_t i = 0; i < len; i++) {
      data[i] = static_cast<T>(rand()) / RAND_MAX;  // NOLINT
    }
  }

  template <typename T>
  void RandomInt(void* arr, int len) {
    auto* data = static_cast<T*>(arr);
    for (int i = 0; i < len; i++) {
      data[i] =
          static_cast<T>(rand() % std::numeric_limits<T>::max());  // NOLINT
    }
  }

  template <typename T>
  void SetVal(void* arr, int len, T x) {
    auto* data = static_cast<T*>(arr);
    for (int i = 0; i < len; i++) {
      data[i] = x;
    }
  }

 private:
  std::vector<int> shape_;
  InitType init_type_{InitType::kZero};
  float init_val_{};
  int align_{};
  Type type_;
};

struct ArgsBuilder {
  template <typename T>
  ArgsBuilder& Add(T x) {
    data_.emplace_back(x);
    return *this;
  }

  std::vector<cinn_pod_value_t> Build() {
    PADDLE_ENFORCE_EQ(!data_.empty(),
                      true,
                      ::common::errors::InvalidArgument(
                          "The data_ container is empty. Please "
                          "ensure it contains valid data."));
    return data_;
  }

 private:
  std::vector<cinn_pod_value_t> data_;
};

}  // namespace common
}  // namespace cinn
