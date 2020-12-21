// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <memory>
#include <mutex>  // NOLINT
#include <unordered_set>
#include "paddle/fluid/framework/scope.h"

namespace paddle {
namespace framework {

class Scope;

class ScopePool {
 public:
  static ScopePool &Instance();  // NOLINT

  void Insert(std::unique_ptr<Scope> &&s);

  void Remove(Scope *s);

  void Clear();

  ~ScopePool();

 private:
  ScopePool() = default;

  static void DeleteScope(Scope *scope);

  std::unordered_set<Scope *> scopes_;
  std::mutex mtx_;
};

}  // namespace framework
}  // namespace paddle
