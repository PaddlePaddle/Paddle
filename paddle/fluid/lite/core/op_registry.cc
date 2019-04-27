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

std::list<std::unique_ptr<KernelBase>> KernelRegistry::Create(
    const std::string &op_type, TargetType target, PrecisionType precision,
    DataLayoutType layout) {
  Place place{target, precision, layout};
  LOG(INFO) << "creating " << op_type << " kernel for " << place;
#define CREATE_KERNEL1(target__, precision__)                                \
  switch (layout) {                                                          \
    case DATALAYOUT(kNCHW):                                                  \
      return Create<TARGET(target__), PRECISION(precision__),                \
                    DATALAYOUT(kNCHW)>(op_type);                             \
    case DATALAYOUT(kAny):                                                   \
      return Create<TARGET(target__), PRECISION(precision__),                \
                    DATALAYOUT(kAny)>(op_type);                              \
    default:                                                                 \
      LOG(FATAL) << "unsupported kernel layout " << DataLayoutToStr(layout); \
  }

#define CREATE_KERNEL(target__)                         \
  switch (precision) {                                  \
    case PRECISION(kFloat):                             \
      CREATE_KERNEL1(target__, kFloat);                 \
    case PRECISION(kInt8):                              \
      CREATE_KERNEL1(target__, kInt8);                  \
    case PRECISION(kAny):                               \
      CREATE_KERNEL1(target__, kAny);                   \
    default:                                            \
      CHECK(false) << "not supported kernel precision " \
                   << PrecisionToStr(precision);        \
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
      CHECK(false) << "not supported kernel target " << TargetToStr(target);
  }

#undef CREATE_KERNEL
  return std::list<std::unique_ptr<KernelBase>>();
}

KernelRegistry::KernelRegistry() {
#define INIT_FOR(target__, precision__, layout__)                            \
  registries_[KernelRegistry::GetKernelOffset<TARGET(target__),              \
                                              PRECISION(precision__),        \
                                              DATALAYOUT(layout__)>()]       \
      .set<KernelRegistryForTarget<TARGET(target__), PRECISION(precision__), \
                                   DATALAYOUT(layout__)> *>(                 \
          &KernelRegistryForTarget<TARGET(target__), PRECISION(precision__), \
                                   DATALAYOUT(layout__)>::Global());
  // Currently, just register 2 kernel targets.
  INIT_FOR(kCUDA, kFloat, kNCHW);
  INIT_FOR(kCUDA, kAny, kNCHW);
  INIT_FOR(kHost, kFloat, kNCHW);
  INIT_FOR(kHost, kAny, kNCHW);
  INIT_FOR(kHost, kAny, kAny);
  INIT_FOR(kCUDA, kAny, kAny);
#undef INIT_FOR
}

KernelRegistry &KernelRegistry::Global() {
  static auto *x = new KernelRegistry;
  return *x;
}

}  // namespace lite
}  // namespace paddle