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
#include "paddle/cinn/hlir/framework/pir/fusion_info.h"
#include "paddle/cinn/hlir/framework/pir/utils.h"
#include "paddle/common/enforce.h"

namespace cinn::hlir::framework {

namespace pir {
class OpLoweringGroup;
class BackendResource final {
 public:
  BackendResource(const Target& target,
                  const std::string& host_fn_name,
                  const std::string& infer_fn_name,
                  const std::map<int, CINNKernelInfo::ArgDimIdx>& int_args_map)
      : host_fn_name_(host_fn_name),
        infer_fn_name_(infer_fn_name),
        int_args_map_(int_args_map) {
    backend_compiler_ = backends::Compiler::Create(target);
  }

  void* GetHostFuncPtr() const;
  void* GetInferFuncPtr() const;
  void* GetCX86HostFuncPtr() const;
  const std::map<int, CINNKernelInfo::ArgDimIdx>& GetIntArgsMap() const {
    return int_args_map_;
  }
  const std::shared_ptr<backends::Compiler>& GetBackendCompiler() const {
    return backend_compiler_;
  }
  pir::CINNKernelInfo GenerateKernelInfo() const;

 private:
  std::string host_fn_name_;
  std::string infer_fn_name_;
  std::map<int, CINNKernelInfo::ArgDimIdx> int_args_map_;

  std::shared_ptr<backends::Compiler> backend_compiler_{nullptr};
};

class CompilationResult final {
 public:
  explicit CompilationResult(const Target& target) : target_(target) {}
  const std::shared_ptr<BackendResource>& GetBackendResource() const {
    return backend_resource_;
  }

  void SetBackendResource(const std::shared_ptr<BackendResource>& other) {
    backend_resource_ = other;
  }

  pir::CINNKernelInfo GetKernelInfo() {
    PADDLE_ENFORCE_NOT_NULL(backend_resource_,
                            ::common::errors::PreconditionNotMet(
                                "Found backend_resource_ is nullptr, please "
                                "call SetBackendResource first."));
    return backend_resource_->GenerateKernelInfo();
  }

 private:
  Target target_;
  std::shared_ptr<BackendResource> backend_resource_{nullptr};
};

}  // namespace pir

class CompilationCache {
 public:
  using CacheKey = pir::FusionInfo;
  using CacheValue = std::shared_ptr<pir::CompilationResult>;

  static CompilationCache& Instance() {
    thread_local static CompilationCache instance;
    return instance;
  }

  bool Has(const CacheKey& key) const;
  const CacheValue& Get(const CacheKey& key) const;
  void Insert(const CacheKey& key, const CacheValue& value);
  void Clear();
  size_t Size() const { return cache_.size(); }

  pir::CINNKernelInfo GetKernelInfo(const CacheKey& key) const;

 private:
  CompilationCache() = default;
  CINN_DISALLOW_COPY_AND_ASSIGN(CompilationCache);

  std::unordered_map<CacheKey, CacheValue> cache_;
};

}  // namespace cinn::hlir::framework
