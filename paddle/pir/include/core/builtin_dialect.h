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

#include "paddle/pir/include/core/dialect.h"

namespace pir {
///
/// \brief Built-in Dialect: automatically registered into global IrContext,
/// all built-in types defined in builtin_type.h will be registered in this
/// Dialect.
///
class IR_API BuiltinDialect : public pir::Dialect {
 public:
  explicit BuiltinDialect(pir::IrContext* context);
  ///
  /// \brief Each Dialect needs to provide a name function to return the name of
  /// the Dialect.
  ///
  /// \return The name of this Dialect.
  ///
  static const char* name() { return "builtin"; }

  pir::Type ParseType(pir::IrParser& parser) override;  // NOLINT
  void PrintType(pir::Type type, std::ostream& os) const override;

 private:
  void initialize();
};

}  // namespace pir

IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::BuiltinDialect)
