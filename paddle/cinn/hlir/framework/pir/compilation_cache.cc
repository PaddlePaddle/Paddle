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

#include "paddle/cinn/hlir/framework/pir/compilation_cache.h"

#include "paddle/common/enforce.h"

namespace cinn::hlir::framework {

void* BackendResource::GetHostFuncPtr() const {
  VLOG(4) << "Lookup kernel name: " << host_fn_name;
  void* ptr = backend_compiler->Lookup(host_fn_name);
  PADDLE_ENFORCE_NOT_NULL(ptr, "Can't find kernel function %s", host_fn_name);
  return ptr;
}

void* BackendResource::GetInferFuncPtr() const {
  VLOG(4) << "Lookup infer shape fn name: " << infer_fn_name;
  void* ptr = backend_compiler->Lookup(infer_fn_name);
  PADDLE_ENFORCE_NOT_NULL(
      ptr, "Can't find infer shape function %s", infer_fn_name);
  return ptr;
}

pir::CINNKernelInfo BackendResource::GernerateKernelInfo(
    const std::shared_ptr<Group>& group) const {
  pir::CINNKernelInfo kernel_info;
  kernel_info.fn_name = host_fn_name;
  kernel_info.fn_ptr = GetHostFuncPtr();
  kernel_info.infer_shape_fn_ptr = GetInferFuncPtr();
  kernel_info.int_args_map = group->int_args_map;
  return kernel_info;
}

bool CompilationCache::Has(const CacheKey& key) const {
  return cache_.find(KeyHash(key)) != cache_.end();
}

const CompilationCache::CacheValue& CompilationCache::Get(
    const CacheKey& key) const {
  return cache_.at(KeyHash(key));
}

pir::CINNKernelInfo CompilationCache::GetKernelInfo(const CacheKey& key) const {
  return Get(key)->GetKernelInfo(key);
}

void CompilationCache::Insert(const CacheKey& key, const CacheValue& value) {
  cache_.insert({KeyHash(key), value});
}

void CompilationCache::Clear() { cache_.clear(); }

struct StringsHash {
  size_t operator()(const& std::vector<std::string> values) const {
    size_t seed = values.size();
    for (auto& value : values) {
      seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
  }
};

size_t CompilationCache::KeyHash(const CacheKey& key) const {
  // TODO(Aurelius84): use a better hash function
  return pir::Group::SharedGroupHasher()(CacheKey);
}

}  // namespace cinn::hlir::framework
