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
#include "paddle/pir/include/core/parser/ir_parser.h"

namespace pir {
BuiltinDialect::BuiltinDialect(IrContext* context)
    : Dialect(name(), context, TypeId::get<BuiltinDialect>()) {
  initialize();
}

void BuiltinDialect::initialize() {
  // Register all built-in types defined in builtin_type.h.
  RegisterTypes<BFloat16Type,
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
              ConstantOp>();
}

pir::Type BuiltinDialect::ParseType(pir::IrParser& parser) {  // NOLINT
  parser.ConsumeAToken("builtin.tensor");
  parser.ConsumeAToken("<");
  std::vector<int> dim{};
  Token dim_token = parser.PeekToken();
  while (dim_token.token_type_ == DIGIT) {
    dim_token = parser.ConsumeToken();
    dim.push_back(atoi(dim_token.val_.c_str()));
    std::string peek_token_val = parser.PeekToken().val_;
    if (peek_token_val[0] != 'x') {
      break;
    }
    parser.ConsumeToken();
    parser.lexer->Unget(static_cast<int>(peek_token_val.size() - 1));
    if (parser.PeekToken().token_type_ != DIGIT) {
      break;
    }
  }
  pir::DDim ddim = common::make_ddim(dim);
  pir::Type dtype = parser.ParseType();
  std::vector<std::vector<size_t>> lod;
  std::vector<size_t> lodv;
  lodv.push_back(0);
  lod.push_back(lodv);
  parser.ConsumeAToken(">");
  return DenseTensorType::get(
      parser.ctx, dtype, ddim, pir::DataLayout::UNDEFINED, lod, 0);
}

void BuiltinDialect::PrintType(pir::Type type, std::ostream& os) const {
  os << type.dialect().name();
  os << '.';
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
