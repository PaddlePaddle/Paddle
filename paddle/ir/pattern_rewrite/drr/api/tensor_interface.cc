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

#include "paddle/ir/pattern_rewrite/drr/api/tensor_interface.h"
#include "paddle/ir/pattern_rewrite/drr/ir_value.h"

namespace ir {
namespace drr {

bool ShapeInterface::operator==(const ShapeInterface& other) const {
  return *shape_ == *other.shape_;
}

bool DtypeInterface::operator==(const DtypeInterface& other) const {
  return *dtype_ == *other.dtype_;
}

}  // namespace drr
}  // namespace ir
