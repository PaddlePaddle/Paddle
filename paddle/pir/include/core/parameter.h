// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include <cstring>

#include "paddle/pir/include/core/type.h"

namespace pir {
///
/// \brief Parameter represents the weight in the calculation graph.
///
class IR_API Parameter {
 public:
  Parameter(void* data, size_t size, pir::Type type) {
    data_ = malloc(size);
    memcpy(data_, data, size);
    size_ = size;
    type_ = type;
  }

  Parameter(const Parameter& param) {
    data_ = malloc(param.size_);
    memcpy(data_, param.data_, param.size_);
    size_ = param.size_;
    type_ = param.type_;
  }

  Parameter& operator=(const Parameter& param) {
    data_ = malloc(param.size_);
    memcpy(data_, param.data_, param.size_);
    size_ = param.size_;
    type_ = param.type_;
    return *this;
  }

  ~Parameter() { free(data_); }

  Type type() const { return type_; }

  void* data() const { return data_; }

  bool is_mutable() const { return is_mutable_; }

  void set_mutable() { is_mutable_ = true; }

 private:
  void* data_;

  ///
  /// \brief Number of bytes held in data_.
  ///
  size_t size_;

  Type type_;

  bool is_mutable_ = false;
};

}  // namespace pir
