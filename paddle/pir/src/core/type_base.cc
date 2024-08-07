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

#include "paddle/pir/include/core/type_base.h"
#include "glog/logging.h"
#include "paddle/pir/include/core/ir_context.h"

namespace pir {

void *AbstractType::GetInterfaceImpl(TypeId interface_id) const {
  auto iter = interface_set_.find(interface_id);
  return iter == interface_set_.end() ? nullptr : iter->model();
}

}  // namespace pir
