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

#include "torch/csrc/jit/schema_type_parser.h"
#include "torch/csrc/jit/schema_parser_defs.h"

#include <cctype>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace torch::jit {

size_t& SchemaTypeParser::refFromPtr(size_t* ptr, const char* name) {
  TORCH_CHECK(ptr != nullptr, name, " must not be null");
  return *ptr;
}

c10::TypePtr SchemaTypeParser::parseBaseType() {
  // Map textual schema type names to compat lightweight type objects.
  std::string type_name = parseDottedIdentifier("type");
  if (type_name == TORCH_SCHEMA_TYPE_CPP_DOUBLE) {
    TORCH_CHECK(false,
                "Use `float` instead of `double` in schema declarations",
                posInfo());
  }
  if (type_name == TORCH_SCHEMA_TYPE_CPP_INT64_T) {
    TORCH_CHECK(false,
                "Use `int` instead of `int64_t` in schema declarations",
                posInfo());
  }

#define TORCH_SCHEMA_BASE_TYPE_CASE(TEXT, KIND, REPR)              \
  if (type_name == TEXT) {                                         \
    return c10::makeSchemaAtomicType(c10::TypeKind::KIND, (REPR)); \
  }
  TORCH_SCHEMA_BASE_TYPE_CONVERSION_TABLE(TORCH_SCHEMA_BASE_TYPE_CASE)
#undef TORCH_SCHEMA_BASE_TYPE_CASE

  TORCH_CHECK(false, "Unsupported type specifier `", type_name, "`", posInfo());
}

std::optional<c10::AliasInfo> SchemaTypeParser::parseAliasAnnotation() {
  // Supported alias forms:
  //   (a), (a!), (a! -> b|c), !
  // where bare '!' creates a fresh write alias set.
  skipWhitespace();
  if (consumeChar(TORCH_SCHEMA_CH_LPAREN)) {
    std::set<std::string> before_sets;
    parseAliasSetList(&before_sets);
    skipWhitespace();
    const bool is_write = consumeChar(TORCH_SCHEMA_CH_BANG);
    std::set<std::string> after_sets = before_sets;
    skipWhitespace();
    if (consumeLiteral(TORCH_SCHEMA_LIT_ARROW)) {
      after_sets.clear();
      parseAliasSetList(&after_sets);
    }
    skipWhitespace();
    TORCH_CHECK(!atEnd() && peek() == TORCH_SCHEMA_CH_RPAREN,
                "Expected `)`",
                posInfo());
    consumeChar(TORCH_SCHEMA_CH_RPAREN);
    return c10::AliasInfo(is_write, before_sets, after_sets);
  }

  if (consumeChar(TORCH_SCHEMA_CH_BANG)) {
    std::set<std::string> fresh_set{
        std::string(TORCH_SCHEMA_ALIAS_FRESH_PREFIX) +
        std::to_string(next_fresh_alias_id_++)};
    return c10::AliasInfo(true, fresh_set, fresh_set);
  }
  return std::nullopt;
}

ParsedType SchemaTypeParser::parseType() {
  // Parse a full type expression including:
  // - tuple forms: (T1, T2, ...)
  // - alias suffixes
  // - optional suffix '?'
  skipWhitespace();
  ParsedType out;
  if (consumeChar(TORCH_SCHEMA_CH_LPAREN)) {
    std::vector<c10::TypePtr> elements;
    std::vector<std::optional<c10::AliasInfo>> element_aliases;
    skipWhitespace();
    if (!consumeChar(TORCH_SCHEMA_CH_RPAREN)) {
      while (true) {
        ParsedType elem = parseType();
        elements.push_back(elem.type);
        element_aliases.push_back(std::move(elem.alias_info));
        skipWhitespace();
        if (consumeChar(TORCH_SCHEMA_CH_COMMA)) {
          continue;
        }
        TORCH_CHECK(!atEnd() && peek() == TORCH_SCHEMA_CH_RPAREN,
                    "Expected `)`",
                    posInfo());
        consumeChar(TORCH_SCHEMA_CH_RPAREN);
        break;
      }
    }
    out.type = c10::makeSchemaTupleType(std::move(elements));
    out.alias_info = parseAliasAnnotation();
    // If tuple elements carry alias info, attach them as contained aliases
    // of the tuple-level alias metadata.
    bool has_contained_alias = false;
    for (const auto& alias : element_aliases) {
      if (alias.has_value()) {
        has_contained_alias = true;
        break;
      }
    }
    if (has_contained_alias) {
      if (!out.alias_info.has_value()) {
        out.alias_info.emplace();
      }
      for (auto& alias : element_aliases) {
        if (alias.has_value()) {
          out.alias_info->addContainedType(std::move(*alias));
        }
      }
    }
  } else {
    out.type = parseBaseType();
    out.alias_info = parseAliasAnnotation();
  }

  skipWhitespace();
  if (consumeChar(TORCH_SCHEMA_CH_QMARK)) {
    out.type = c10::makeSchemaOptionalType(out.type);
  }
  return out;
}

void SchemaTypeParser::parseAliasSetList(std::set<std::string>* sets) {
  TORCH_CHECK(sets != nullptr, "Alias set output must not be null");
  // Parse alias set unions: a|b|*.
  skipWhitespace();
  while (true) {
    if (consumeChar(TORCH_SCHEMA_CH_STAR)) {
      sets->insert(TORCH_SCHEMA_ALIAS_WILDCARD);
    } else {
      sets->insert(parseIdentifier("alias set"));
    }
    skipWhitespace();
    if (!consumeChar(TORCH_SCHEMA_CH_PIPE)) {
      break;
    }
    skipWhitespace();
  }
  TORCH_CHECK(!sets->empty(), "Empty alias set annotation", posInfo());
}

std::string SchemaTypeParser::parseIdentifier(const char* desc) {
  skipWhitespace();
  TORCH_CHECK(
      !atEnd() && isIdentifierStart(peek()), "Expected ", desc, posInfo());
  const size_t start = pos_++;
  while (!atEnd() && isIdentifierChar(peek())) {
    ++pos_;
  }
  return schema_.substr(start, pos_ - start);
}

std::string SchemaTypeParser::parseDottedIdentifier(const char* desc) {
  std::string ident = parseIdentifier(desc);
  skipWhitespace();
  while (consumeChar(TORCH_SCHEMA_CH_DOT)) {
    ident.push_back(TORCH_SCHEMA_CH_DOT);
    ident += parseIdentifier(desc);
    skipWhitespace();
  }
  return ident;
}

void SchemaTypeParser::skipWhitespace() {
  while (!atEnd() && std::isspace(peekAsUnsigned())) {
    ++pos_;
  }
}

bool SchemaTypeParser::consumeLiteral(const char* literal) {
  const size_t len = std::char_traits<char>::length(literal);
  if (schema_.compare(pos_, len, literal) == 0) {
    pos_ += len;
    return true;
  }
  return false;
}

bool SchemaTypeParser::consumeChar(char c) {
  if (!atEnd() && schema_[pos_] == c) {
    ++pos_;
    return true;
  }
  return false;
}

char SchemaTypeParser::peek() const {
  TORCH_INTERNAL_ASSERT(!atEnd());
  return schema_[pos_];
}

unsigned char SchemaTypeParser::peekAsUnsigned() const {
  return static_cast<unsigned char>(peek());
}

bool SchemaTypeParser::atEnd() const { return pos_ >= schema_.size(); }

std::string SchemaTypeParser::posInfo() const {
  std::ostringstream os;
  os << " at position " << pos_ << " in schema `" << schema_ << "`";
  return os.str();
}

bool SchemaTypeParser::isIdentifierStart(char c) {
  const auto uc = static_cast<unsigned char>(c);
  return std::isalpha(uc) || c == '_';
}

bool SchemaTypeParser::isIdentifierChar(char c) {
  const auto uc = static_cast<unsigned char>(c);
  return std::isalnum(uc) || c == '_';
}

}  // namespace torch::jit
