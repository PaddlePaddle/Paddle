/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/pten/core/arg_map_context.h"

#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/string/string_helper.h"

namespace pten {

OpArgumentMappingFnMap& OpArgumentMappingFnMap::Instance() {
  static OpArgumentMappingFnMap g_op_arg_mapping_fn_map;
  return g_op_arg_mapping_fn_map;
}

bool OpArgumentMappingFnMap::Has(const std::string& op_type) const {
  return fn_map_.find(op_type) != fn_map_.end();
}

const ArgumentMappingFn& OpArgumentMappingFnMap::Get(
    const std::string& op_type) const {
  auto it = fn_map_.find(op_type);
  PADDLE_ENFORCE_NE(
      it,
      fn_map_.end(),
      paddle::platform::errors::NotFound(
          "Operator `%s`'s argument mapping funciton is not registered.",
          op_type));
  return it->second;
}

void OpArgumentMappingFnMap::Emplace(const std::string& op_type,
                                     const std::string api_name,
                                     ArgumentMappingFn fn) {
  name_map_.emplace(op_type, api_name);
  fn_map_.emplace(op_type, fn);
}

std::ostream& operator<<(std::ostream& os, KernelSignature signature) {
  os << "Kernel Signature - name: " << signature.name << "; inputs: "
     << paddle::string::join_strings(std::get<0>(signature.args), ", ")
     << "; attributes: "
     << paddle::string::join_strings(std::get<1>(signature.args), ", ")
     << "; outputs: "
     << paddle::string::join_strings(std::get<2>(signature.args), ", ");
  return os;
}

}  // namespace pten
