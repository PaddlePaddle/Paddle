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

#include "paddle/pir/src/core/parser/lexer.h"

Token Lexer::ConsumeToken() {
  SkipWhitespace();
  if (auto token = LexIdentifier()) {
    return *token;
  } else if (auto token = LexNumberOrArrow()) {
    return *token;
  } else if (auto token = LexEndTagOrNullVal()) {
    return *token;
  } else if (auto token = LexValueId()) {
    return *token;
  } else if (auto token = LexString()) {
    return *token;
  } else if (auto token = LexEOF()) {
    return *token;
  } else {
    return Token{"Error", NULL_};
  }
}

Token Lexer::PeekToken() {
  auto pos = is.tellg();
  size_t cache_line = line;
  size_t cache_column = column;

  auto token = ConsumeToken();

  if (is.eof()) {
    is.clear();
  }
  is.seekg(pos);
  line = cache_line;
  column = cache_column;

  return token;
}

char Lexer::GetChar() {
  char c = static_cast<char>(is.get());
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
  while (IsSpace(static_cast<char>(is.peek()))) {
    GetChar();
  }
}

std::unique_ptr<Token> Lexer::LexIdentifier() {
  if ((!isalpha(is.peek()) && is.peek() != '_') ||
      IsEndTag(static_cast<char>(is.peek()))) {
    return nullptr;
  }
  std::string token_identifier = "";
  while (isalnum(is.peek()) || is.peek() == '_' || is.peek() == '.') {
    token_identifier += GetChar();
  }
  std::unique_ptr<Token> token(new Token{token_identifier, IDENTIFIER});
  return token;
}

std::unique_ptr<Token> Lexer::LexNumberOrArrow() {
  if (!isdigit(is.peek()) && is.peek() != '-') {
    return nullptr;
  }

  std::string token_digit = "";
  token_digit += GetChar();

  if (token_digit[0] == '-' && is.peek() == '>') {
    GetChar();
    std::unique_ptr<Token> arrow_token(new Token{"->", ARROW});
    return arrow_token;
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
    std::unique_ptr<Token> sdigit_token(new Token{token_digit, SDIGIT});
    return sdigit_token;
  }
  std::unique_ptr<Token> digit_token(new Token{token_digit, DIGIT});
  return digit_token;
}

std::unique_ptr<Token> Lexer::LexEndTagOrNullVal() {
  if (!IsEndTag(static_cast<char>(is.peek()))) {
    return nullptr;
  }
  std::string token_end = "";
  token_end += GetChar();
  if ((token_end[0] == '<' && (is.peek() != '<' && is.peek() != '#')) ||
      token_end[0] != '<') {
    std::unique_ptr<Token> endtag_token(new Token{token_end, ENDTAG});
    return endtag_token;
  }
  if (is.peek() == '<') {
    std::string token_null_val = "";
    GetChar();
    while (is.peek() != '>') {
      token_null_val += GetChar();
    }
    GetChar();
    GetChar();
    std::unique_ptr<Token> null_token(
        new Token{"<<" + token_null_val + ">>", NULL_});
    return null_token;
  } else {
    std::string token_attr_null = "";
    while (is.peek() != '>') {
      token_attr_null += GetChar();
    }
    GetChar();
    std::unique_ptr<Token> null_token(
        new Token{"<" + token_attr_null + ">", NULL_});
    return null_token;
  }
}

std::unique_ptr<Token> Lexer::LexValueId() {
  if (is.peek() != '%') {
    return nullptr;
  }
  std::string token_valueid = "";
  token_valueid += GetChar();

  while (isdigit(is.peek())) {
    token_valueid += GetChar();
  }
  std::unique_ptr<Token> valueid_token(new Token{token_valueid, VALUEID});
  return valueid_token;
}

std::unique_ptr<Token> Lexer::LexEOF() {
  if (is.peek() == EOF) {
    std::unique_ptr<Token> eof_token(new Token{"LEX_DOWN", EOF_});
    return eof_token;
  } else {
    return nullptr;
  }
}

std::unique_ptr<Token> Lexer::LexString() {
  if (is.peek() != '"') {
    return nullptr;
  }
  GetChar();
  std::string token_val = "";
  while (is.peek() != '"') {
    char c = GetChar();
    if (c == '\\' && is.peek() == '\"') {
      c = GetChar();
    }
    token_val += c;
  }
  GetChar();
  std::unique_ptr<Token> string_token(
      new Token{"\"" + token_val + "\"", STRING});
  return string_token;
}

bool Lexer::IsSpace(char c) {
  return c == ' ' || c == '\n' || c == '\t' || c == '\f';
}

bool Lexer::IsEndTag(char c) {
  return c == '{' || c == '}' || c == '(' || c == ')' || c == ':' || c == '>' ||
         c == ',' || c == ']' || c == '[' || c == '+' || c == '=' || c == '<';
}

void Lexer::Unget(const int len) {
  if (is.eof()) {
    is.clear();
  }
  column -= len;
  is.seekg(-len, std::ios::cur);
}
