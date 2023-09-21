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
#include "paddle/pir/core/ir_context.h"
#include "paddle/pir/core/operation.h"
#include "paddle/pir/core/parser/lexer.h"
#include "paddle/pir/core/program.h"

using ValueMap = std::map<std::string, pir::Value>;
using AttributeMap = std::unordered_map<std::string, pir::Attribute>;
using OpAttributeInfoMap = std::map<std::string, std::string>;

namespace pir {
class IrParser {
 public:
  std::unique_ptr<Lexer> lexer;
  IrContext* ctx;
  ValueMap value_map;
  std::unique_ptr<Builder> builder;

 public:
  IrParser(IrContext* ctx, std::istream& is);

  ~IrParser() = default;

  Token ConsumeToken();

  Token PeekToken();

  std::unique_ptr<Program> ParseProgram();

  void ParseRegion(Region& region);  // NOLINT

  void ParseBlock(Block& block);  // NOLINT

  Operation* ParseOperation();

  OpInfo ParseOpInfo();

  std::vector<std::string> ParseValueList();

  std::vector<Value> ParseOperandList();

  AttributeMap ParseAttributeMap();

  std::vector<Type> ParseTypeList();

  Type ParseType();

  Attribute ParseAttribute();

  std::string GetErrorLocationInfo();

  void ConsumeAToken(std::string expect_token_val);
};

}  // namespace pir
