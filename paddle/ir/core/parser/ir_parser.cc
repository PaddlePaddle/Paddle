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

#include "paddle/ir/core/ir_parser.h"

#include <cmath>

#include "paddle/ir/core/builtin_dialect.h"
#include "paddle/ir/core/builtin_type.h"

using std::vector;

namespace ir {
IrParser::IrParser(IrContext* ctx, std::istream& is) {
  lexer.reset(new Lexer{is});
  this->ctx = ctx;
  builder.reset(new Builder{ctx});
}

Token IrParser::ConsumeToken() {
  auto token = lexer->ConsumeToken();
  return token;
}

string IrParser::GetErrorLocationInfo() {
  return "The error occurred in line " + std::to_string(lexer->GetLine()) +
         ", column " + std::to_string(lexer->GetColumn());
}

Token IrParser::PeekToken() {
  auto token = lexer->ConsumeToken();
  if (token.token_type_ != EOF_) {
    lexer->Unget(token.val_.size());
  }
  return token;
}

void IrParser::ConsumeAToken(string expect_token_val) {
  string token_val = ConsumeToken().val_;
  IR_ENFORCE(token_val == expect_token_val,
             "The token value of expectation is " + expect_token_val + " ,not" +
                 token_val + "." + GetErrorLocationInfo());
}

Type IrParser::ParseType() {
  Token type_token = PeekToken();
  string type_val = type_token.val_;
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
    IR_ENFORCE(type_val.find('.') != string::npos,
               "No function parsing " + type_val + " exists!" +
                   GetErrorLocationInfo());
    auto dialect_name = type_val.substr(0, type_val.find('.'));
    auto dialect = ctx->GetRegisteredDialect(dialect_name);
    return dialect->ParseType(*this);
  }
}
Attribute IrParser::ParseAttribute() {
  auto parenthesis_token = ConsumeToken();
  if (parenthesis_token.val_ == "true" || parenthesis_token.val_ == "false") {
    return builder->bool_attr(parenthesis_token.val_ == "true");
  }
  string attribute_type = PeekToken().val_;
  if (attribute_type == "String") {
    ConsumeAToken("String");
    ConsumeAToken(")");
    string val = ConsumeToken().val_;
    return builder->str_attr(val);
  } else if (attribute_type == "Float") {
    ConsumeAToken("Float");
    ConsumeAToken(")");
    string val = ConsumeToken().val_;
    return builder->float_attr(atof(val.c_str()));
  } else if (attribute_type == "Double") {
    ConsumeAToken("Double");
    ConsumeAToken(")");
    string val = ConsumeToken().val_;
    return builder->double_attr(atof(val.c_str()));
  } else if (attribute_type == "Int32") {
    ConsumeAToken("Int32");
    ConsumeAToken(")");
    string val = ConsumeToken().val_;
    return builder->int32_attr(atoi(val.c_str()));
  } else if (attribute_type == "Int64") {
    ConsumeAToken("Int64");
    ConsumeAToken(")");
    string val = ConsumeToken().val_;
    return builder->int64_attr(atoll(val.c_str()));
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

std::unique_ptr<Program> IrParser::ParseProgram() {
  std::unique_ptr<Program> program(new Program{ctx});
  ParseBlock(*program->block());
  return program;
}

void IrParser::ParseRegion(Region& region) {  // NOLINT
  while (PeekToken().token_type_ != EOF_) {
    Block* block = new Block{};
    ParseBlock(*block);
    region.push_back(block);
  }
}

void IrParser::ParseBlock(Block& block) {  // NOLINT
  ConsumeAToken("{");
  while (PeekToken().val_ != "}") {
    auto op = ParseOperation();
    block.push_back(op);
  }
  ConsumeAToken("}");
}

Operation* IrParser::ParseOperation() {
  std::vector<string> opresultindex = ParseOpResultIndex();
  ConsumeAToken("=");

  OpInfo opinfo = ParseOpInfo();

  std::vector<OpResult> inputs = ParseOpRandList();

  ir::AttributeMap attributeMap = ParseAttributeMap();

  ConsumeAToken(":");
  ConsumeAToken("(");
  while (ConsumeToken().token_type_ != ARRAOW) {
  }

  vector<Type> type_vector = ParseFunctionTypeList();

  Operation* op =
      Operation::Create(inputs, attributeMap, type_vector, opinfo, 0);

  for (uint32_t i = 0; i < op->num_results(); i++) {
    string key_t = opresultindex[i];
    opresultmap[key_t] = op->result(i);
  }

  return op;
}

vector<string> IrParser::ParseOpResultIndex() {
  std::vector<string> opresultindex{};
  ConsumeAToken("(");
  Token index_token = ConsumeToken();
  while (index_token.val_ != ")") {
    if (index_token.token_type_ == NULL_) {
      opresultindex.push_back("null");
    } else {
      string str = index_token.val_;
      opresultindex.push_back(str);
    }
    if (ConsumeToken().val_ == ")") break;
    index_token = ConsumeToken();
  }

  return opresultindex;
}

OpInfo IrParser::ParseOpInfo() {
  Token opname_token = ConsumeToken();
  string opname = opname_token.val_.substr(1, opname_token.val_.size() - 2);
  return ctx->GetRegisteredOpInfo(opname);
}

vector<OpResult> IrParser::ParseOpRandList() {
  ConsumeAToken("(");
  std::vector<OpResult> inputs{};
  Token ind_token = ConsumeToken();
  while (ind_token.val_ != ")") {
    string t = "";
    if (ind_token.token_type_ == NULL_) {
      inputs.push_back(GetNullValue());
    } else {
      t = ind_token.val_;
      inputs.push_back(opresultmap[t]);
    }
    Token token = ConsumeToken();
    if (token.val_ == ")") {
      break;
    }
    ind_token = ConsumeToken();
  }
  return inputs;
}

AttributeMap IrParser::ParseAttributeMap() {
  AttributeMap attribute_map{};
  ConsumeAToken("{");
  Token key_token = ConsumeToken();
  while (key_token.val_ != "}") {
    ConsumeAToken(":");
    attribute_map[key_token.val_] = ParseAttribute();
    string token_val = ConsumeToken().val_;
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

vector<Type> IrParser::ParseFunctionTypeList() {
  std::vector<Type> type_vector{};
  while (PeekToken().val_ != "(" && PeekToken().val_ != "}") {
    type_vector.push_back(ParseType());
    if (PeekToken().val_ == "}" || PeekToken().val_ == "(" ||
        PeekToken().token_type_ == EOF_)
      break;
    ConsumeAToken(",");
  }
  return type_vector;
}

OpResult IrParser::GetNullValue() {
  Value* v = new Value{nullptr};
  OpResult* opresult = static_cast<OpResult*>(v);
  return *opresult;
}

Attribute Attribute::Parse(std::istream& is, IrContext* ctx) {
  IrParser parser(ctx, is);
  return parser.ParseAttribute();
}

Type Type::Parse(std::istream& is, IrContext* ctx) {
  IrParser parser(ctx, is);
  return parser.ParseType();
}

}  // namespace ir
