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

#include "paddle/ir/core/parser/lexer.h"
using std::string;

Token Lexer::GetToken(LexSegment seg) {
  SkipWhitespace();
  if (auto token = LexIdentifer(seg)) {
    return *token;
  } else if (auto token = LexNumberOrArraow()) {
    return *token;
  } else if (auto token = LexEndTagOrNullVal(seg)) {
    return *token;
  } else if (auto token = LexValueId()) {
    return *token;
  } else if (auto token = LexOpName()) {
    return *token;
  } else if (auto token = LexEOF()) {
    return *token;
  } else {
    return Token{"Error", NULL_};
  }
}

char Lexer::GetChar() {
  char c = is.get();
  if (c == '\n') {
    line++;
    column = 1;
  } else {
    column++;
  }
  return c;
}

size_t Lexer::GetColumn() { return column; }

size_t Lexer::GetLine() { return line; }

void Lexer::SkipWhitespace() {
  while (IsSpace(is.peek())) {
    GetChar();
  }
}

Token* Lexer::LexIdentifer(LexSegment seg) {
  if ((!isalpha(is.peek()) && is.peek() != '_') || IsEndTag(is.peek(), seg)) {
    return nullptr;
  }
  string token_identifier = "";
  while (isalnum(is.peek()) || is.peek() == '_' || is.peek() == '.') {
    token_identifier += GetChar();
  }
  return new Token{token_identifier, IDENTIFER};
}

Token* Lexer::LexNumberOrArraow() {
  if (!isdigit(is.peek()) && is.peek() != '-') {
    return nullptr;
  }

  string token_digit = "";
  token_digit += GetChar();

  if (token_digit[0] == '-' && is.peek() == '>') {
    GetChar();
    return new Token{"->", ARRAOW};
  }
  while (isdigit(is.peek())) {
    token_digit += GetChar();
  }
  if (is.peek() == '.') {
    token_digit += GetChar();
    while (isdigit(is.peek())) {
      token_digit += GetChar();
    }
  }
  if (is.peek() == 'e') {
    token_digit += GetChar();
    if (is.peek() == '+' || is.peek() == '-') {
      token_digit += GetChar();
    }
    while (isdigit(is.peek())) {
      token_digit += GetChar();
    }
    return new Token{token_digit, SDIGIT};
  }
  return new Token{token_digit, DIGIT};
}

Token* Lexer::LexEndTagOrNullVal(LexSegment seg) {
  if (!IsEndTag(is.peek(), seg)) {
    return nullptr;
  }
  string token_end = "";
  token_end += GetChar();
  if ((token_end[0] == '<' && (is.peek() != '<' && is.peek() != '#')) ||
      token_end[0] != '<') {
    return new Token{token_end, ENDTAG};
  }
  if (is.peek() == '<') {
    string token_null_val = "";
    GetChar();
    while (is.peek() != '>') {
      token_null_val += GetChar();
    }
    GetChar();
    GetChar();
    return new Token{"<<" + token_null_val + ">>", NULL_};
  } else {
    string token_attrnull = "";
    while (is.peek() != '>') {
      token_attrnull += GetChar();
    }
    GetChar();
    return new Token{"<" + token_attrnull + ">", NULL_};
  }
}

Token* Lexer::LexValueId() {
  if (is.peek() != '%') {
    return nullptr;
  }
  string token_valueid = "";
  token_valueid += GetChar();

  while (isdigit(is.peek())) {
    token_valueid += GetChar();
  }
  return new Token{token_valueid, VALUEID};
}

Token* Lexer::LexEOF() {
  if (is.peek() == EOF) {
    return new Token{"LEX_DOWN", EOF_};
  } else {
    return nullptr;
  }
}

Token* Lexer::LexOpName() {
  if (is.peek() != '"') {
    return nullptr;
  }
  GetChar();
  string token_opname = "";
  while (is.peek() != '"') {
    token_opname += GetChar();
  }
  GetChar();
  return new Token{token_opname, OPNAME};
}

bool Lexer::IsSpace(char c) {
  return c == ' ' || c == '\n' || c == '\t' || c == '\f';
}

bool Lexer::IsEndTag(char c, LexSegment seg) {
  if (seg == parseFunctionType) {
    return c == '{' || c == '}' || c == '(' || c == ')' || c == ':' ||
           c == '>' || c == ',' || c == ']' || c == '[' || c == '+' ||
           c == '=' || c == 'x' || c == '<';
  } else {
    return c == '{' || c == '}' || c == '(' || c == ')' || c == ':' ||
           c == '>' || c == ',' || c == ']' || c == '[' || c == '+' ||
           c == '=' || c == '<';
  }
}
