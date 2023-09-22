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

#include "paddle/pir/core/parser/ir_parser.h"

#include "paddle/pir/core/builtin_dialect.h"
#include "paddle/pir/core/builtin_type.h"

namespace pir {
IrParser::IrParser(IrContext* ctx, std::istream& is) {
  lexer.reset(new Lexer{is});
  this->ctx = ctx;
  builder.reset(new Builder{ctx});
}

Token IrParser::ConsumeToken() { return lexer->ConsumeToken(); }

std::string IrParser::GetErrorLocationInfo() {
  return "The error occurred in line " + std::to_string(lexer->GetLine()) +
         ", column " + std::to_string(lexer->GetColumn());
}

Token IrParser::PeekToken() { return lexer->PeekToken(); }

void IrParser::ConsumeAToken(std::string expect_token_val) {
  std::string token_val = ConsumeToken().val_;
  IR_ENFORCE(token_val == expect_token_val,
             "The token value of expectation is " + expect_token_val + " ,not" +
                 token_val + "." + GetErrorLocationInfo());
}

// Type := BuiltinType | OtherDialectsDefineType
// BuiltinType := <<NULL TYPE>> | bf16 | f16 | f32 | f64
//             := | b | i8 | u8 | i16 | i32 | i64 | index | c64
//             := | c128 | VectorType
// VectorType := '[' Type(,Type)* ']'
Type IrParser::ParseType() {
  Token type_token = PeekToken();
  std::string type_val = type_token.val_;
  if (type_val == "<<NULL TYPE>>") {
    ConsumeToken();
    return Type(nullptr);
  } else if (type_val == "bf16") {
    ConsumeToken();
    return builder->bfloat16_type();
  } else if (type_val == "f16") {
    ConsumeToken();
    return builder->bfloat16_type();
  } else if (type_val == "f32") {
    ConsumeToken();
    return builder->float32_type();
  } else if (type_val == "f64") {
    ConsumeToken();
    return builder->float64_type();
  } else if (type_val == "b") {
    ConsumeToken();
    return builder->bool_type();
  } else if (type_val == "i8") {
    ConsumeToken();
    return builder->int8_type();
  } else if (type_val == "u8") {
    ConsumeToken();
    return builder->uint8_type();
  } else if (type_val == "i16") {
    ConsumeToken();
    return builder->int16_type();
  } else if (type_val == "i32") {
    ConsumeToken();
    return Int32Type::get(ctx);
  } else if (type_val == "i64") {
    ConsumeToken();
    return Int64Type::get(ctx);
  } else if (type_val == "index") {
    ConsumeToken();
    return IndexType::get(ctx);
  } else if (type_val == "c64") {
    ConsumeToken();
    return builder->complex64_type();
  } else if (type_val == "c128") {
    ConsumeToken();
    return builder->complex128_type();
  } else if (type_val == "vec") {
    ConsumeAToken("vec");
    ConsumeAToken("[");
    std::vector<Type> vec_type;
    Token vec_type_token = PeekToken();
    while (vec_type_token.val_ != "]") {
      Type cur_type = ParseType();
      vec_type.push_back(cur_type);
      vec_type_token = ConsumeToken();
    }
    return VectorType::get(ctx, vec_type);
  } else {
    IR_ENFORCE(type_val.find('.') != std::string::npos,
               "No function parsing " + type_val + " exists!" +
                   GetErrorLocationInfo());
    auto dialect_name = type_val.substr(0, type_val.find('.'));
    auto dialect = ctx->GetRegisteredDialect(dialect_name);
    return dialect->ParseType(*this);
  }
}

// Attribute := BuiltinAttribute | OtherDialectsDefineAttribute
// BuiltinAttribute := Bool | String | Float | Double | Int32 |
//                  := | Int64 | Pointer | ArrayAttribute
// ArrayAttribute   := '[' Atribute(,Attribute)* ']'
Attribute IrParser::ParseAttribute() {
  auto parenthesis_token = ConsumeToken();
  if (parenthesis_token.val_ == "true" || parenthesis_token.val_ == "false") {
    return builder->bool_attr(parenthesis_token.val_ == "true");
  } else if (parenthesis_token.token_type_ == STRING) {
    std::string val = parenthesis_token.val_;
    val = val.substr(1, val.size() - 2);
    return builder->str_attr(val);
  }
  std::string attribute_type = PeekToken().val_;
  if (attribute_type == "Float") {
    ConsumeAToken("Float");
    ConsumeAToken(")");
    std::string val = ConsumeToken().val_;
    return builder->float_attr(atof(val.c_str()));
  } else if (attribute_type == "Double") {
    ConsumeAToken("Double");
    ConsumeAToken(")");
    std::string val = ConsumeToken().val_;
    return builder->double_attr(atof(val.c_str()));
  } else if (attribute_type == "Int32") {
    ConsumeAToken("Int32");
    ConsumeAToken(")");
    std::string val = ConsumeToken().val_;
    return builder->int32_attr(atoi(val.c_str()));
  } else if (attribute_type == "Int64") {
    ConsumeAToken("Int64");
    ConsumeAToken(")");
    std::string val = ConsumeToken().val_;
    return builder->int64_attr(atoll(val.c_str()));
  } else if (attribute_type == "Pointer") {
    IR_THROW("This attribute is not currently supported by parser");
  } else if (attribute_type == "Array") {
    ConsumeAToken("Array");
    ConsumeAToken(")");
    ConsumeAToken("[");
    std::vector<Attribute> array_attribute;
    while (PeekToken().val_ != "]") {
      array_attribute.push_back(ParseAttribute());
      if (PeekToken().val_ == "]") break;
      ConsumeAToken(",");
    }
    ConsumeAToken("]");
    return builder->array_attr(array_attribute);
  } else {
    IR_ENFORCE(attribute_type.find('.') != std::string::npos,
               "No function parsing " + attribute_type + " exists!" +
                   GetErrorLocationInfo());
    auto dialect_name = attribute_type.substr(0, attribute_type.find('.'));
    auto dialect = ctx->GetRegisteredDialect(dialect_name);
    return dialect->ParseAttribute(*this);
  }
}

// Program := [ParameterList]ModuleOp
// ModuleOp := Region
std::unique_ptr<Program> IrParser::ParseProgram() {
  std::unique_ptr<Program> program(new Program{ctx});
  auto top_level_op = program->module_op();
  auto& region = top_level_op->region(0);
  ParseRegion(region);

  return program;
}

// Region := Block
void IrParser::ParseRegion(Region& region) {  // NOLINT
  ParseBlock(*region.front());
  IR_ENFORCE(PeekToken().val_ != "{",
             "Only one block in a region is supported");
}

// Block := "{" {Operation} "}"
void IrParser::ParseBlock(Block& block) {  // NOLINT
  ConsumeAToken("{");
  while (PeekToken().val_ != "}") {
    auto op = ParseOperation();
    block.push_back(op);
  }
  ConsumeAToken("}");
}

// Operation := ValueList ":=" Opname "(" OprandList ? ")" AttributeMap ":"
// FunctionType
// FunctionType := "(" TypeList ")"  "->" TypeList
Operation* IrParser::ParseOperation() {
  std::vector<std::string> value_index = ParseValueList();
  ConsumeAToken("=");

  OpInfo opinfo = ParseOpInfo();

  std::vector<Value> inputs = ParseOperandList();

  pir::AttributeMap attributeMap = ParseAttributeMap();

  ConsumeAToken(":");
  ConsumeAToken("(");
  ParseTypeList();
  ConsumeAToken(")");
  ConsumeAToken("->");

  std::vector<Type> type_vector = ParseTypeList();

  Operation* op =
      Operation::Create(inputs, attributeMap, type_vector, opinfo, 0);

  for (uint32_t i = 0; i < op->num_results(); i++) {
    std::string key_t = value_index[i];
    value_map[key_t] = op->result(i);
  }

  return op;
}

// ValueList := ValueId(,ValueId)*
std::vector<std::string> IrParser::ParseValueList() {
  std::vector<std::string> value_index{};
  ConsumeAToken("(");
  Token index_token = ConsumeToken();
  while (index_token.val_ != ")") {
    if (index_token.token_type_ == NULL_) {
      value_index.push_back("null");
    } else {
      std::string str = index_token.val_;
      value_index.push_back(str);
    }
    if (ConsumeToken().val_ == ")") break;
    index_token = ConsumeToken();
  }

  return value_index;
}

// OpName := "\"" StringIdentifer "." StringIdentifer "\""
OpInfo IrParser::ParseOpInfo() {
  Token opname_token = ConsumeToken();
  std::string opname =
      opname_token.val_.substr(1, opname_token.val_.size() - 2);
  return ctx->GetRegisteredOpInfo(opname);
}

// OprandList := ValueList
// ValueList := ValueId(,ValueId)*
std::vector<Value> IrParser::ParseOperandList() {
  ConsumeAToken("(");
  std::vector<Value> inputs{};
  Token ind_token = ConsumeToken();
  while (ind_token.val_ != ")") {
    std::string t = "";
    if (ind_token.token_type_ == NULL_) {
      inputs.emplace_back();
    } else {
      t = ind_token.val_;
      inputs.push_back(value_map[t]);
    }
    Token token = ConsumeToken();
    if (token.val_ == ")") {
      break;
    }
    ind_token = ConsumeToken();
  }
  return inputs;
}

// AttributeMap := "{" AttributeEntry,(,AttributeEntry)* "}"
// AttributeEntry := StringIdentifer:Attribute
AttributeMap IrParser::ParseAttributeMap() {
  AttributeMap attribute_map{};
  ConsumeAToken("{");
  Token key_token = ConsumeToken();
  while (key_token.val_ != "}") {
    ConsumeAToken(":");
    attribute_map[key_token.val_] = ParseAttribute();
    std::string token_val = ConsumeToken().val_;
    if (token_val == "}") {
      break;
    } else if (token_val == ",") {
      key_token = ConsumeToken();
    } else {
      IR_ENFORCE((token_val == "}") || (token_val == ","),
                 "The token value of expectation is } or , , not " + token_val +
                     "." + GetErrorLocationInfo());
    }
  }
  return attribute_map;
}

// TypeList := Type(,Type)*
std::vector<Type> IrParser::ParseTypeList() {
  std::vector<Type> type_vector{};
  while (PeekToken().val_ != "(" && PeekToken().val_ != "}" &&
         PeekToken().val_ != ")") {
    type_vector.push_back(ParseType());
    if (PeekToken().val_ == "}" || PeekToken().val_ == "(" ||
        PeekToken().val_ == ")" || PeekToken().token_type_ == EOF_)
      break;
    ConsumeAToken(",");
  }
  return type_vector;
}

Attribute Attribute::Parse(std::istream& is, IrContext* ctx) {
  IrParser parser(ctx, is);
  return parser.ParseAttribute();
}

Type Type::Parse(std::istream& is, IrContext* ctx) {
  IrParser parser(ctx, is);
  return parser.ParseType();
}

std::unique_ptr<Program> Program::Parse(std::istream& is, IrContext* ctx) {
  IrParser parser(ctx, is);
  return parser.ParseProgram();
}

}  // namespace pir
