// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/lite/core/op_registry.h"

namespace paddle {
namespace lite {

std::unique_ptr<KernelBase> KernelRegistry::Create(const std::string &op_type,
                                                   TargetType target,
                                                   PrecisionType precision) {
#define CREATE_KERNEL(target__)                                    \
  switch (precision) {                                             \
    case PRECISION(kFloat):                                        \
      return Create<TARGET(target__), PRECISION(kFloat)>(op_type); \
    default:                                                       \
      CHECK(false) << "not supported kernel place yet";            \
  }

  switch (target) {
    case TARGET(kHost): {
      CREATE_KERNEL(kHost);
    } break;
    case TARGET(kX86): {
      CREATE_KERNEL(kX86);
    } break;
    case TARGET(kCUDA): {
      CREATE_KERNEL(kCUDA);
    } break;
    default:
      CHECK(false) << "not supported kernel place";
  }

#undef CREATE_KERNEL
  return nullptr;
}

KernelRegistry::KernelRegistry() {
#define INIT_FOR(target__, precision__)                                      \
  registries_[KernelRegistry::GetKernelOffset<TARGET(target__),              \
                                              PRECISION(precision__)>()]     \
      .set<KernelRegistryForTarget<TARGET(target__), PRECISION(precision__)> \
               *>(&KernelRegistryForTarget<TARGET(target__),                 \
                                           PRECISION(precision__)>::Global());
  // Currently, just register 2 kernel targets.
  INIT_FOR(kHost, kFloat);
#undef INIT_FOR
}

KernelRegistry &KernelRegistry::Global() {
  static auto *x = new KernelRegistry;
  return *x;
}

}  // namespace lite
}  // namespace paddle