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
#include <istream>
#include "paddle/ir/core/parser/token.h"

enum LexSegment {
  parseOpResult = 0,
  parseOpInfo = 1,
  parseOpRand = 2,
  parserAttribute = 3,
  parseFunctionType = 4,
};

class Lexer {
 private:
  std::istream& is;
  size_t line = 1;
  size_t column = 1;

 public:
  explicit Lexer(std::istream& is) : is(is) {}
  ~Lexer() = default;
  Token GetToken(LexSegment seg);
  Token* LexIdentifer(LexSegment seg);
  Token* LexNumberOrArraow();
  Token* LexEndTagOrNullVal(LexSegment seg);
  Token* LexValueId();
  Token* LexEOF();
  Token* LexOpName();
  char GetChar();
  void SkipWhitespace();
  bool IsEndTag(char, LexSegment);
  bool IsSpace(char);
  size_t GetLine();
  size_t GetColumn();
};
