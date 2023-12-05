// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/lang/buffer.h"

#include "paddle/cinn/ir/buffer.h"

namespace cinn {
namespace lang {

using ir::_Buffer_;

Buffer::Buffer(Type type, const std::string& name) {
  buffer_ = _Buffer_::Make();
  buffer_->dtype = type;
  buffer_->set_type(type_of<cinn_buffer_t*>());
  buffer_->elem_offset = Expr(0);
  if (!name.empty()) {
    buffer_->name = name;
  }
  buffer_->target = cinn::common::DefaultHostTarget();
}

}  // namespace lang
}  // namespace cinn
