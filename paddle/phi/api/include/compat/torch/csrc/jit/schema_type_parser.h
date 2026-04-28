// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

#include <ATen/core/alias_info.h>
#include <ATen/core/jit_type.h>
#include <c10/macros/Macros.h>
#include <cstddef>
#include <optional>
#include <set>
#include <string>

namespace torch::jit {

struct ParsedType {
  c10::TypePtr type;
  std::optional<c10::AliasInfo> alias_info;
};

class PADDLE_API SchemaTypeParser {
 public:
  SchemaTypeParser(const std::string& schema,
                   size_t* pos,
                   size_t* next_fresh_alias_id)
      : schema_(schema),
        pos_(refFromPtr(pos, "pos")),
        next_fresh_alias_id_(
            refFromPtr(next_fresh_alias_id, "next_fresh_alias_id")) {}

  c10::TypePtr parseBaseType();
  std::optional<c10::AliasInfo> parseAliasAnnotation();
  ParsedType parseType();

 private:
  void parseAliasSetList(std::set<std::string>* sets);
  std::string parseIdentifier(const char* desc);
  std::string parseDottedIdentifier(const char* desc);
  void skipWhitespace();
  bool consumeLiteral(const char* literal);
  bool consumeChar(char c);
  char peek() const;
  unsigned char peekAsUnsigned() const;
  bool atEnd() const;
  std::string posInfo() const;

  static bool isIdentifierStart(char c);
  static bool isIdentifierChar(char c);
  static size_t& refFromPtr(size_t* ptr, const char* name);

  const std::string& schema_;
  size_t& pos_;
  size_t& next_fresh_alias_id_;
};

}  // namespace torch::jit
