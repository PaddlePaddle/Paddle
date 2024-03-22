// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
#include <unordered_map>
#include "paddle/cinn/backends/compiler.h"
#include "paddle/cinn/common/macros.h"
#include "paddle/cinn/common/target.h"
#include "paddle/cinn/hlir/framework/pir/group.h"
#include "paddle/cinn/hlir/framework/pir/utils.h"

namespace cinn::hlir::framework {

class BackendResource final {
 public:
  explicit BackendResource(const Target& target) {
    backend_compiler = std::make_shared<backends::Compiler>(target);
  }
  void* GetHostFuncPtr() const;
  void* GetInferFuncPtr() const;
  pir::CINNKernelInfo GernerateKernelInfo(const std::shared_ptr<Group>& group);

 private:
  friend class CompilationTask;
  CINN_DISALLOW_COPY_AND_ASSIGN(CompilationResult);

  std::string host_fn_name_;
  std::string infer_fn_name_;
  // std::string host_code_;
  // std::vector<std::string> device_code_;
  std::shared_ptr<backends::Compiler> backend_compiler_;
};

class CompilationResult final {
 public:
  explicit CompilationResult(const Target& target)
      : target_(target), backend_resource_(target) {}

  BackendResource& GetBackendResource() { return backend_resource_; }
  const BackendResource& GetBackendResource() const {
    return backend_resource_;
  }
  pir::CINNKernelInfo GetKernelInfo(const std::shared_ptr<Group>& group) {
    return backend_resource_.GernerateKernelInfo(group);
  }

 private:
  CINN_DISALLOW_COPY_AND_ASSIGN(CompilationResult);
  Target target_;
  BackendResource backend_resource_;
};

class CompilationCache {
 public:
  using CacheKey = std::shared_ptr<Group>;
  using CacheValue = std::shared_ptr<CompilationResult>;

  static CompilationCache& Instance() {
    static CompilationCache instance;
    return instance;
  }

  bool Has(const CacheKey& key) const;
  const CacheValue& Get(const CacheKey& key) const;
  pir::CINNKernelInfo GetKernelInfo(const CacheKey& key) const;
  bool Insert(const CacheKey& key, const CacheValue& value);
  void Clear();
  int64_t KeyHash(const CacheKey& key) const;

 private:
  CompilationCache() = default;
  std::unordered_map<size_t, CacheValue> cache_;
};

}  // namespace cinn::hlir::framework
