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

#include "paddle/pir/include/core/builtin_dialect.h"

#include "paddle/common/ddim.h"
#include "paddle/common/layout.h"
#include "paddle/pir/include/core/builtin_attribute.h"
#include "paddle/pir/include/core/builtin_op.h"
#include "paddle/pir/include/core/builtin_type.h"

namespace pir {
BuiltinDialect::BuiltinDialect(IrContext* context)
    : Dialect(name(), context, TypeId::get<BuiltinDialect>()) {
  initialize();
}

void BuiltinDialect::initialize() {
  // Register all built-in types defined in builtin_type.h.
  RegisterTypes<UndefinedType,
                BFloat16Type,
                Float16Type,
                Float32Type,
                Float64Type,
                Int8Type,
                UInt8Type,
                Int16Type,
                Int32Type,
                Int64Type,
                IndexType,
                BoolType,
                Complex64Type,
                Complex128Type,
                Float8E4M3FNType,
                Float8E5M2Type,
                VectorType,
                DenseTensorType>();

  RegisterAttributes<StrAttribute,
                     BoolAttribute,
                     FloatAttribute,
                     DoubleAttribute,
                     PointerAttribute,
                     Int32Attribute,
                     IndexAttribute,
                     Int64Attribute,
                     ArrayAttribute,
                     TypeAttribute,
                     TensorNameAttribute,
                     Complex64Attribute,
                     Complex128Attribute>();

  RegisterOps<ModuleOp,
              ParameterOp,
              SetParameterOp,
              ShadowOutputOp,
              CombineOp,
              SliceOp,
              SplitOp,
              ConstantOp,
              GroupOp>();
}

void BuiltinDialect::PrintType(pir::Type type, std::ostream& os) const {
  if (auto tensor_type = type.dyn_cast<DenseTensorType>()) {
    os << "tensor<";
    for (auto d : common::vectorize(tensor_type.dims())) {
      os << d;
      os << "x";
    }
    tensor_type.dtype().Print(os);
    os << ">";
  }
}

}  // namespace pir

IR_DEFINE_EXPLICIT_TYPE_ID(pir::BuiltinDialect)
