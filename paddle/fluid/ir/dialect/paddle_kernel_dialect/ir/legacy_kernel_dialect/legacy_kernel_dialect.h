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

#pragma once

#include "paddle/ir/core/dialect.h"

namespace paddle {
namespace dialect {

class LegacyKernelDialect : public ir::Dialect {
 public:
  explicit LegacyKernelDialect(ir::IrContext* context);

  static const char* name() { return "legacy_kernel"; }

  void PrintType(ir::Type type, std::ostream& os) const override;

  void PrintAttribute(ir::Attribute attr, std::ostream& os) const override;

 private:
  void initialize();
};

}  // namespace dialect
}  // namespace paddle

IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::LegacyKernelDialect)
