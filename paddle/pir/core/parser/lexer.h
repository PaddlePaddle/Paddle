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
#include <memory>

#include "paddle/pir/core/parser/token.h"

class Lexer {
 private:
  std::istream& is;
  size_t line = 1;
  size_t column = 1;

 public:
  explicit Lexer(std::istream& is) : is(is) {}
  ~Lexer() = default;
  Token ConsumeToken();
  Token PeekToken();
  std::unique_ptr<Token> LexIdentifer();
  std::unique_ptr<Token> LexNumberOrArraow();
  std::unique_ptr<Token> LexEndTagOrNullVal();
  std::unique_ptr<Token> LexValueId();
  std::unique_ptr<Token> LexEOF();
  std::unique_ptr<Token> LexString();
  char GetChar();
  void SkipWhitespace();
  bool IsEndTag(char);
  bool IsSpace(char);
  size_t GetLine();
  size_t GetColumn();
  void Unget(const int len);
};
