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
#include "paddle/ir/core/ir_context.h"
#include "paddle/ir/core/operation.h"
#include "paddle/ir/core/parser/lexer.h"
#include "paddle/ir/core/program.h"

using OpResultMap = std::map<string, ir::OpResult>;
using AttributeMap = std::unordered_map<std::string, ir::Attribute>;
using OpAttributeInfoMap = std::map<string, string>;

namespace ir {
class IrParser {
 public:
  Lexer* lexer;
  IrContext* ctx;
  OpResultMap opresultmap;
  Builder* builder;
  Token last_token;
  Token cur_token;
  LexSegment lex_seg;

 public:
  IrParser(IrContext* ctx, std::istream& is);

  ~IrParser();

  Token GetToken();

  Token PeekToken();

  std::unique_ptr<Program> ParseProgram();

  std::unique_ptr<Region> ParseRegion();

  void ParseBlock(Block& block);  // NOLINT

  Operation* ParseOperation();

  OpInfo ParseOpInfo();

  std::vector<std::string> ParseOpResultIndex();

  std::vector<OpResult> ParseOpRandList();

  AttributeMap ParseAttributeMap();

  std::vector<Type> ParseFunctionTypeList();

  OpResult GetNullValue();

  void NextLexSegment();

  Type ParseType();

  Attribute ParseAttribute();

  string GetErrorLocationInfo();

  void ConsumeAToken(string expect_token_val);
};

}  // namespace ir
