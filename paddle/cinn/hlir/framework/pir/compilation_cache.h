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
#include "paddle/cinn/hlir/framework/pir/utils.h"

namespace cinn::hlir::framework {

namespace pir {
class OpLoweringGroup;
class BackendResource final {
 public:
  BackendResource(const Target& target) {
    backend_compiler_ = backends::Compiler::Create(target);
  }

  BackendResource(const Target& target,
                  const std::string& host_fn_name,
                  const std::string& infer_fn_name)
      : host_fn_name_(host_fn_name), infer_fn_name_(infer_fn_name) {
    backend_compiler_ = backends::Compiler::Create(target);
  }

  void* GetHostFuncPtr() const;
  void* GetInferFuncPtr() const;
  pir::CINNKernelInfo GernerateKernelInfo(
      const std::shared_ptr<pir::OpLoweringGroup>& group) const;
  std::shared_ptr<backends::Compiler>& GetBackendCompiler();
  const std::shared_ptr<backends::Compiler>& GetBackendCompiler() const;
  void SetHostFnName(const std::string& name);
  void SetInferFnName(const std::string& name);

 private:
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

  BackendResource& MutableBackendResource() { return backend_resource_; }
  const BackendResource& GetBackendResource() const {
    return backend_resource_;
  }
  pir::CINNKernelInfo GetKernelInfo(
      const std::shared_ptr<pir::OpLoweringGroup>& group) {
    return backend_resource_.GernerateKernelInfo(group);
  }

 private:
  Target target_;
  BackendResource backend_resource_;
};
}  // namespace pir

class CompilationCache {
 public:
  using CacheKey = std::shared_ptr<pir::OpLoweringGroup>;
  using CacheValue = std::shared_ptr<pir::CompilationResult>;

  static CompilationCache& Instance() {
    static CompilationCache instance;
    return instance;
  }

  bool Has(const CacheKey& key) const;
  const CacheValue& Get(const CacheKey& key) const;
  pir::CINNKernelInfo GetKernelInfo(const CacheKey& key) const;
  void Insert(const CacheKey& key, const CacheValue& value);
  void Clear();
  size_t KeyHash(const CacheKey& key) const;

 private:
  CompilationCache() = default;
  CINN_DISALLOW_COPY_AND_ASSIGN(CompilationCache);

  std::unordered_map<size_t, CacheValue> cache_;
};

}  // namespace cinn::hlir::framework
