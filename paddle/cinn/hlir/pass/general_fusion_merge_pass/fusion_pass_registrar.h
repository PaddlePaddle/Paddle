// Copyright (c) 2023 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/common/macros.h"
#include "paddle/cinn/hlir/pass/general_fusion_merge_pass/fusion_pass_map.h"

namespace cinn {
namespace hlir {
namespace pass {

class Registrar {
 public:
  // In our design, various kinds of classes, e.g., operators and kernels,
  // have their corresponding registry and registrar. The action of
  // registration is in the constructor of a global registrar variable, which
  // are not used in the code that calls package framework, and would
  // be removed from the generated binary file by the linker. To avoid such
  // removal, we add Touch to all registrar classes and make USE_OP macros to
  // call this method. So, as long as the callee code calls USE_OP, the global
  // registrar variable won't be removed by the linker.
  void Touch() {}
};

template <typename PassClassT>
class FusionPassRegistrar final : public Registrar {
 public:
  explicit FusionPassRegistrar(const std::string& pass_name) {
    FusionPassMap::Instance().Insert(
        pass_name, std::shared_ptr<PassClassT>(new PassClassT()));
  }
};

}  // namespace pass
}  // namespace hlir
}  // namespace cinn

#define CINN_REGISTER_FUSION_PASS(pass_name, pass_class)               \
  STATIC_ASSERT_GLOBAL_NAMESPACE(                                      \
      __reg_pass__##pass_name,                                         \
      "CINN_REGISTER_FUSION_PASS must be called in global namespace"); \
  static ::cinn::hlir::pass::FusionPassRegistrar<pass_class>           \
      __pass_registrar_##pass_name##__(#pass_name);                    \
  int TouchFusionPassRegistrar_##pass_name() {                         \
    __pass_registrar_##pass_name##__.Touch();                          \
    return 0;                                                          \
  }
