// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/platform/macros.h"

namespace paddle {
namespace distributed {

enum DistModelDataType { FLOAT16, FLOAT32, INT64, INT32, INT8 };

template <typename T>
constexpr DistModelDataType DistModelGetDtype();

template <>
constexpr DistModelDataType DistModelGetDtype<int32_t>() {
  return DistModelDataType::INT32;
}

template <>
constexpr DistModelDataType DistModelGetDtype<int64_t>() {
  return DistModelDataType::INT64;
}

template <>
constexpr DistModelDataType DistModelGetDtype<float>() {
  return DistModelDataType::FLOAT32;
}

class DistModelDataBuf {
 public:
  explicit DistModelDataBuf(size_t length)
      : data_(new char[length]), length_(length), memory_owned_(true) {}
  DistModelDataBuf(void* data, size_t length)
      : data_(data), length_(length), memory_owned_(false) {}
  void Reset(void* data, size_t length);
  size_t length() const { return length_; }
  void* data() const { return data_; }
  ~DistModelDataBuf() { Free(); }
  DistModelDataBuf() = default;
  void Resize(size_t length);

  DistModelDataBuf& operator=(const DistModelDataBuf& other);
  DistModelDataBuf& operator=(DistModelDataBuf&& other);
  DistModelDataBuf(DistModelDataBuf&& other);
  DistModelDataBuf(const DistModelDataBuf& other);

 private:
  void Free();
  void* data_{nullptr};
  size_t length_{0};
  bool memory_owned_{true};
};

struct DistModelTensor {
  std::string name;
  std::vector<int> shape;
  DistModelDataBuf data;
  DistModelDataType dtype;
  std::vector<std::vector<size_t>> lod;
};

}  // namespace distributed
}  // namespace paddle
