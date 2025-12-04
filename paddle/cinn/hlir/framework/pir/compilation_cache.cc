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
#include "paddle/cinn/hlir/framework/pir/op_lowering_group.h"

namespace cinn::hlir::framework {

namespace pir {
void* BackendResource::GetHostFuncPtr() const {
  VLOG(4) << "Lookup kernel name: " << host_fn_name_;
  void* ptr = backend_compiler_->Lookup(host_fn_name_);
  PADDLE_ENFORCE_NOT_NULL(ptr,
                          ::common::errors::InvalidArgument(
                              "Can't find kernel function %s", host_fn_name_));
  return ptr;
}

void* BackendResource::GetInferFuncPtr() const {
  VLOG(4) << "Lookup infer shape fn name: " << infer_fn_name_;
  void* ptr = backend_compiler_->Lookup(infer_fn_name_);
  PADDLE_ENFORCE_NOT_NULL(
      ptr,
      ::common::errors::InvalidArgument("Can't find infer shape function %s",
                                        infer_fn_name_));
  return ptr;
}

void* BackendResource::GetCX86HostFuncPtr() const {
  VLOG(4) << "Lookup kernel name: " << host_fn_name_ + "_CX86";
  void* ptr = backend_compiler_->Lookup(host_fn_name_ + "_CX86");
  PADDLE_ENFORCE_NOT_NULL(
      ptr,
      ::common::errors::InvalidArgument("Can't find kernel function %s",
                                        host_fn_name_ + "_CX86"));
  return ptr;
}

pir::CINNKernelInfo BackendResource::GenerateKernelInfo(
    bool need_x86_kernel) const {
  pir::CINNKernelInfo kernel_info;
  kernel_info.fn_name = host_fn_name_;
  kernel_info.fn_ptr = GetHostFuncPtr();
  kernel_info.infer_shape_fn_ptr = GetInferFuncPtr();
  if (need_x86_kernel) {
    kernel_info.CX86_fn_ptr = GetCX86HostFuncPtr();
  }
  kernel_info.symbol_args_map = GetSymbolArgsMap();
  kernel_info.temp_space_sizes = GetTempSpaceSizes();
  return kernel_info;
}
}  // namespace pir

bool CompilationCache::Has(const CacheKey& key) const {
  const bool has_existed = cache_.find(key) != cache_.end();
  VLOG(6) << "Check IsExisted in CompilationCache: " << has_existed << " - "
          << key;
  return has_existed;
}

const CompilationCache::CacheValue& CompilationCache::Get(
    const CacheKey& key) const {
  PADDLE_ENFORCE_EQ(
      Has(key),
      true,
      ::common::errors::NotFound("%s is not in CompliatonCache.", key));
  return cache_.at(key);
}

pir::CINNKernelInfo CompilationCache::GetKernelInfo(const CacheKey& key) const {
  return Get(key)->GetKernelInfo();
}

void CompilationCache::Insert(const CacheKey& key, const CacheValue& value) {
  VLOG(6) << "Insert CompilationCache for: " << key;
  PADDLE_ENFORCE_EQ(Has(key),
                    false,
                    ::common::errors::PreconditionNotMet(
                        "%s is already in CompliatonCache while calling "
                        "CompilationCache::Insert().",
                        key));
  cache_.insert({key, value});
}

void CompilationCache::Clear() { cache_.clear(); }

}  // namespace cinn::hlir::framework
