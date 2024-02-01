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

#include <thread>  // NOLINT
#include <unordered_map>
#include <unordered_set>

#include "paddle/fluid/framework/op_kernel_type.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/utils/test_macros.h"

namespace paddle {
namespace framework {

class OpKernelType;
class Scope;

TEST_API std::unordered_map<size_t, Scope*>& global_transfer_data_cache();

TEST_API std::unordered_set<Scope*>& global_transfer_scope_cache();

std::unordered_map<const Scope*, std::unordered_set<size_t>>&
global_transfer_scope_key();

// Combine two hash values to a single hash.
static size_t CombineHash(size_t seed, size_t a) {
  return (seed ^ a) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

Scope* TryCreateTransferScope(const phi::KernelKey& type0,
                              const phi::KernelKey& type1,
                              const Scope* scope);

}  // namespace framework
}  // namespace paddle
