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
  lexer = new Lexer{is};
  this->ctx = ctx;
  builder = new Builder{ctx};
  lex_seg = LexSegment::parseOpResult;
  cur_token = lexer->GetToken(lex_seg);
}

IrParser::~IrParser() {
  delete builder;
  delete lexer;
  builder = NULL;
  lexer = NULL;
}

Token IrParser::GetToken() {
  last_token = cur_token;
  cur_token = lexer->GetToken(lex_seg);
  return last_token;
}

string IrParser::GetErrorLocationInfo() {
  return "The error occurred in line " + std::to_string(lexer->GetLine()) +
         ", column " + std::to_string(lexer->GetColumn());
}

Token IrParser::PeekToken() { return cur_token; }

void IrParser::ConsumeAToken(string expect_token_val) {
  string token_val = GetToken().val_;
  IR_ENFORCE(token_val == expect_token_val,
             "The token value of expectation is " + expect_token_val + " ,not" +
                 token_val + "." + GetErrorLocationInfo());
}

Type IrParser::ParseType() {
  Token type_token = PeekToken();
  string type_val = type_token.val_;
  if (type_val == "<<NULL TYPE>>") {
    GetToken();
    return Type(nullptr);
  } else if (type_val == "bf16") {
    GetToken();
    return builder->bfloat16_type();
  } else if (type_val == "f16") {
    GetToken();
    return builder->bfloat16_type();
  } else if (type_val == "f32") {
    GetToken();
    return builder->float32_type();
  } else if (type_val == "f64") {
    GetToken();
    return builder->float64_type();
  } else if (type_val == "b") {
    GetToken();
    return builder->bool_type();
  } else if (type_val == "i8") {
    GetToken();
    return builder->int8_type();
  } else if (type_val == "u8") {
    GetToken();
    return builder->uint8_type();
  } else if (type_val == "i16") {
    GetToken();
    return builder->int16_type();
  } else if (type_val == "i32") {
    GetToken();
    return Int32Type::get(ctx);
  } else if (type_val == "i64") {
    GetToken();
    return Int64Type::get(ctx);
  } else if (type_val == "index") {
    GetToken();
    return IndexType::get(ctx);
  } else if (type_val == "c64") {
    GetToken();
    return builder->complex64_type();
  } else if (type_val == "c128") {
    GetToken();
    return builder->complex128_type();
  } else if (type_val == "vec") {
    ConsumeAToken("vec");
    ConsumeAToken("[");
    std::vector<Type> vec_type;
    Token vec_type_token = PeekToken();
    while (vec_type_token.val_ != "]") {
      Type cur_type = ParseType();
      vec_type.push_back(cur_type);
      vec_type_token = GetToken();
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
  auto parenthesis_token = GetToken();
  if (parenthesis_token.val_ == "true" || parenthesis_token.val_ == "false") {
    return builder->bool_attr(parenthesis_token.val_ == "true");
  }
  string attribute_type = PeekToken().val_;
  if (attribute_type == "String") {
    ConsumeAToken("String");
    ConsumeAToken(")");
    string val = GetToken().val_;
    return builder->str_attr(val);
  } else if (attribute_type == "Float") {
    ConsumeAToken("Float");
    ConsumeAToken(")");
    string val = GetToken().val_;
    return builder->float_attr(atof(val.c_str()));
  } else if (attribute_type == "Double") {
    ConsumeAToken("Double");
    ConsumeAToken(")");
    string val = GetToken().val_;
    return builder->double_attr(atof(val.c_str()));
  } else if (attribute_type == "Int32") {
    ConsumeAToken("Int32");
    ConsumeAToken(")");
    string val = GetToken().val_;
    return builder->int32_attr(atoi(val.c_str()));
  } else if (attribute_type == "Int64") {
    ConsumeAToken("Int64");
    ConsumeAToken(")");
    string val = GetToken().val_;
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

std::unique_ptr<Region> IrParser::ParseRegion() {
  std::unique_ptr<Region> region(new Region{});
  while (PeekToken().token_type_ != EOF_) {
    Block* block = new Block{};
    ParseBlock(*block);
    region->push_back(block);
  }
  return region;
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
  NextLexSegment();
  OpInfo opinfo = ParseOpInfo();

  NextLexSegment();
  std::vector<OpResult> inputs = ParseOpRandList();

  NextLexSegment();
  ir::AttributeMap attributeMap = ParseAttributeMap();

  NextLexSegment();
  ConsumeAToken(":");
  ConsumeAToken("(");
  while (GetToken().token_type_ != ARRAOW) {
  }

  vector<Type> type_vector = ParseFunctionTypeList();

  NextLexSegment();
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
  Token index_token = GetToken();
  while (index_token.val_ != ")") {
    if (index_token.token_type_ == NULL_) {
      opresultindex.push_back("null");
    } else {
      string str = index_token.val_;
      opresultindex.push_back(str);
    }
    if (GetToken().val_ == ")") break;
    index_token = GetToken();
  }

  return opresultindex;
}

OpInfo IrParser::ParseOpInfo() {
  Token opname_token = GetToken();
  string opname = opname_token.val_;
  return ctx->GetRegisteredOpInfo(opname_token.val_);
}

vector<OpResult> IrParser::ParseOpRandList() {
  ConsumeAToken("(");
  std::vector<OpResult> inputs{};
  Token ind_token = GetToken();
  while (ind_token.val_ != ")") {
    string t = "";
    if (ind_token.token_type_ == NULL_) {
      inputs.push_back(GetNullValue());
    } else {
      t = ind_token.val_;
      inputs.push_back(opresultmap[t]);
    }
    Token token = GetToken();
    if (token.val_ == ")") {
      break;
    }
    ind_token = GetToken();
  }
  return inputs;
}

AttributeMap IrParser::ParseAttributeMap() {
  AttributeMap attribute_map{};
  ConsumeAToken("{");
  Token key_token = GetToken();
  while (key_token.val_ != "}") {
    ConsumeAToken(":");
    attribute_map[key_token.val_] = ParseAttribute();
    string token_val = GetToken().val_;
    if (token_val == "}") {
      break;
    } else if (token_val == ",") {
      key_token = GetToken();
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

void IrParser::NextLexSegment() {
  switch (lex_seg) {
    case (LexSegment::parseOpResult):
      lex_seg = LexSegment::parseOpInfo;
      break;
    case (LexSegment::parseOpInfo):
      lex_seg = LexSegment::parseOpRand;
      break;
    case (LexSegment::parseOpRand):
      lex_seg = LexSegment::parserAttribute;
      break;
    case (LexSegment::parserAttribute):
      lex_seg = LexSegment::parseFunctionType;
      break;
    case (LexSegment::parseFunctionType):
      lex_seg = LexSegment::parseOpResult;
      break;
  }
}
}  // namespace ir
