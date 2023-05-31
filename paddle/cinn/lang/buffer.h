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

#include "paddle/cinn/ir/buffer.h"

namespace cinn {
namespace lang {

/**
 * This is a DSL wrapper for ir::Buffer.
 */
class Buffer {
 public:
  explicit Buffer(Type type, const std::string& name = "");
  explicit Buffer(const ir::Buffer& x) : buffer_(x) {}

  ir::_Buffer_* operator->() { return buffer_.As<ir::_Buffer_>(); }
  const ir::_Buffer_* operator->() const { return buffer_.As<ir::_Buffer_>(); }

  ir::_Buffer_* self() { return buffer_.As<ir::_Buffer_>(); }

  ir::Buffer buffer() const { return buffer_; }

 private:
  ir::Buffer buffer_;
};

}  // namespace lang
}  // namespace cinn
