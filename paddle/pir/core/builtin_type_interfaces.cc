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

#include "paddle/pir/core/builtin_type_interfaces.h"
#include "paddle/pir/core/type_id.h"

namespace pir {

Type ShapedTypeInterface::GetElementType() const {
  return impl_->get_element_type(*this);
}

std::vector<int64_t> ShapedTypeInterface::GetDyShape() const {
  if (dy_shape_.size() == 0) {
    auto ddim_vec = common::vectorize(impl_->get_shape(*this));
    dy_shape_ = ddim_vec;
    std::replace(dy_shape_.begin(),
                 dy_shape_.end(),
                 (int64_t)-1,
                 ShapedTypeInterface::kDynamic);
  }
  return dy_shape_;
}

}  // namespace pir
IR_DEFINE_EXPLICIT_TYPE_ID(pir::ShapedTypeInterface)
