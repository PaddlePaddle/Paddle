#pragma once

#include <thread>
#include <unordered_map>
#include "paddle/fluid/framework/op_kernel_type.h"
#include "paddle/fluid/framework/scope.h"

namespace paddle {
namespace framework {

static std::unordered_map<size_t, Scope*>& global_transfer_scope_cache() {
  thread_local auto* x = new std::unordered_map<size_t, Scope*>;
  return *x;
}

// Combine two hash values to a single hash.
static size_t CombineHash(size_t seed, size_t a) {
  return (seed ^ a) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

static Scope* TryCreateTransferScope(OpKernelType type0, OpKernelType type1,
                                     const Scope* scope) {
  Scope* new_scope{nullptr};
  size_t infer_cache_key =
      CombineHash(OpKernelType::Hash()(type0), OpKernelType::Hash()(type1));
  infer_cache_key =
      CombineHash(infer_cache_key, std::hash<const Scope*>()(scope));

  auto it = global_transfer_scope_cache().find(infer_cache_key);
  if (it != global_transfer_scope_cache().end()) {
    new_scope = global_transfer_scope_cache()[infer_cache_key];
  } else {
    new_scope = &scope->NewScope();
    global_transfer_scope_cache()[infer_cache_key] = new_scope;
  }
  return new_scope;
}

}  // namespace framework
}  // namespace paddle
