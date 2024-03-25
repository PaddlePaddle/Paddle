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
#pragma once

#include "paddle/pir/include/core/cast_utils.h"
#include "paddle/pir/include/core/dll_decl.h"
#include "paddle/pir/include/core/type.h"

namespace paddle {
namespace dialect {

class IR_API DistTypeInterface
    : public pir::TypeInterfaceBase<DistTypeInterface> {
 public:
  struct Concept {
    /// Defined these methods with the interface.
    explicit Concept(pir::Type (*local_type)(pir::Type))
        : local_type(local_type) {}
    pir::Type (*local_type)(pir::Type);
  };

  template <class ConcreteType>
  struct Model : public Concept {
    static Type local_type(Type type) {
      return pir::cast<ConcreteType>(type).local_type();
    }
    Model() : Concept(local_type) {}
  };

  DistTypeInterface(pir::Type type, Concept *impl)
      : pir::TypeInterfaceBase<DistTypeInterface>(type), impl_(impl) {}

  pir::Type local_type() { return impl_->local_type(*this); }

 private:
  Concept *impl_;
};

}  // namespace dialect
}  // namespace paddle

IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::DistTypeInterface)
