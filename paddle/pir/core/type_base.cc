// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/pir/core/type_base.h"
#include "glog/logging.h"
#include "paddle/pir/core/ir_context.h"

namespace pir {

void *AbstractType::GetInterfaceImpl(TypeId interface_id) const {
  if (interface_map_.empty()) {
    VLOG(6) << "Interface map is empty!";
    return nullptr;
  } else {
    for (size_t i = 0; i < interface_map_.size(); ++i) {
      if (interface_map_[i].type_id() == interface_id)
        return interface_map_[i].model();
    }
    VLOG(6) << "Find no interface!";
    return nullptr;
  }
  // TODO(zhangbo63): Add LookUp method like:
  // return ir::detail::LookUp<AbstractType>(
  //     interface_id, num_interfaces_, num_traits_, this);
}

}  // namespace pir
