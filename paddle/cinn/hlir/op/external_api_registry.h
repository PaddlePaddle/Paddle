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
#include <sstream>

#include "paddle/cinn/common/target.h"
#include "paddle/cinn/hlir/framework/node.h"
#include "paddle/cinn/utils/registry.h"

#define CINN_OP_REGISTER_EXTERNAL_API(Name, Target)          \
  static ::cinn::hlir::op::ExternalApiInfo& CINN_STR_CONCAT( \
      __make_##ExternalApiInfo##_##Name##__, __COUNTER__) =  \
      ::cinn::hlir::op::ExternalApiRegistry::Global()->Register(#Name, Target)

namespace cinn {
namespace hlir {
namespace op {

using OpNodeTransToExternalApiFunction =
    std::function<std::string(const framework::Node* op_node)>;

// This class contains detail external api information of a specified Operator.
// To provide the external api name, we can directly set it through
// `set_api_name` or set a transform function wth `set_trans_func` that return a
// api name finally
struct ExternalApiInfo {
  std::string name;
  std::string api_name;
  OpNodeTransToExternalApiFunction trans_func;

  inline ExternalApiInfo& set_api_name(const std::string& name) {
    this->api_name = name;
    return *this;
  }

  inline ExternalApiInfo& set_trans_func(
      OpNodeTransToExternalApiFunction func) {
    this->trans_func = func;
    return *this;
  }
};

// A registry that stores external api for ops supported by vendor library
class ExternalApiRegistry : public Registry<ExternalApiInfo> {
 public:
  static ExternalApiRegistry* Global() {
    static ExternalApiRegistry x;
    return &x;
  }

  ExternalApiInfo& Register(const std::string& op_name,
                            const common::Target& target);

  bool Has(const std::string& op_name, const common::Target& target) {
    return nullptr != Registry<ExternalApiInfo>::Find(GenKey(op_name, target));
  }

  // return the api name on the specified target
  std::string GetExternalApi(const framework::Node* op_node,
                             const common::Target& target);

 private:
  ExternalApiRegistry() = default;
  CINN_DISALLOW_COPY_AND_ASSIGN(ExternalApiRegistry);

  // the registered key consist of the name of op and the specified target
  std::string GenKey(const std::string& op_name, const common::Target& target);
};

}  // namespace op
}  // namespace hlir
}  // namespace cinn
