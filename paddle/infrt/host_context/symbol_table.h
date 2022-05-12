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

#include <unordered_map>

#include <memory>

#include "paddle/infrt/host_context/value.h"

namespace infrt {
namespace host_context {

/**
 * SymbolTable holds all the states of the kernel graph in the runtime.
 */
class SymbolTable {
 public:
  SymbolTable();

  /**
   * Register a state called \p key.
   */
  Value* Register(const std::string& key);

  Value* Register(const std::string& key, ValueRef value);

  /**
   * Register a state and set value.
   */
  template <typename T>
  Value* Register(const std::string& key, T&& v);

  size_t size() const;

  /**
   * Get a state called \p key.
   */
  Value* GetValue(const std::string& key) const;

  template <typename T>
  T Get(const std::string& key);

  ~SymbolTable();

 private:
  class Impl;

  std::unique_ptr<Impl> impl_;
};

}  // namespace host_context
}  // namespace infrt
