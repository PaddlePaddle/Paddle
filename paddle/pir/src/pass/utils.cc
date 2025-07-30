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

#include "paddle/pir/include/pass/utils.h"

#include "paddle/pir/include/core/builtin_type.h"
#include "paddle/pir/include/core/ir_context.h"

namespace pir {

void SetNewLayoutForValue(pir::Value value,
                          common::DataLayout new_layout,
                          TransLayoutCallbackFn callback) {
  if (!value || !value.type()) {
    return;
  }
  auto tensor_type = value.type().dyn_cast<pir::DenseTensorType>();
  if (!tensor_type) {
    return;
  }
  auto new_tensor_type = pir::DenseTensorType::get(pir::IrContext::Instance(),
                                                   tensor_type.dtype(),
                                                   tensor_type.dims(),
                                                   new_layout,
                                                   tensor_type.lod(),
                                                   tensor_type.offset());
  value.set_type(new_tensor_type);
  if (callback) {
    callback(value, new_layout);
  }
}

}  // namespace pir
