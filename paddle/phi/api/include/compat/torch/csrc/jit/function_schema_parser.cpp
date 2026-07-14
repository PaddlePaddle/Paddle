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

// The file has been adapted from pytorch project
// Licensed under BSD-style license -
// https://github.com/pytorch/pytorch/blob/main/LICENSE

#include "torch/csrc/jit/function_schema_parser.h"
#include "glog/logging.h"
#include "torch/csrc/jit/schema_parser_defs.h"
#include "torch/csrc/jit/schema_type_parser.h"

namespace torch::jit {

namespace {

std::string parsedDeclarationToDebugString(
    const std::variant<std::string, c10::FunctionSchema>& parsed) {
  // Used only in parser debug logs so we can see whether we parsed
  // an operator name or a full function schema.
  std::ostringstream os;
  if (std::holds_alternative<std::string>(parsed)) {
    os << "name(" << std::get<std::string>(parsed) << ")";
  } else {
    os << "schema" << std::get<c10::FunctionSchema>(parsed);
  }
  return os.str();
}

std::string schemaTypeRuntimeClassName(const c10::Type& type) {
  if (dynamic_cast<const c10::detail::SchemaAtomicType*>(&type)) {
    return "c10::detail::SchemaAtomicType";
  }
  if (dynamic_cast<const c10::detail::SchemaOptionalType*>(&type)) {
    return "c10::detail::SchemaOptionalType";
  }
  if (dynamic_cast<const c10::detail::SchemaTupleType*>(&type)) {
    return "c10::detail::SchemaTupleType";
  }
  return typeid(type).name();
}

const char* schemaTypeKindName(c10::TypeKind kind) {
  switch (kind) {
#define TORCH_SCHEMA_KIND_CASE(T) \
  case c10::TypeKind::T:          \
    return #T;
    C10_FORALL_TYPES(TORCH_SCHEMA_KIND_CASE)
#undef TORCH_SCHEMA_KIND_CASE
    default:
      return "UnknownTypeKind";
  }
}

void appendTypeTree(std::ostringstream& os,
                    const c10::TypePtr& type,
                    int depth) {
  // Recursively dump the parsed type tree (e.g. Optional[Tuple[...]])
  // for verbose parser tracing.
  const std::string indent(static_cast<size_t>(depth) * 2, ' ');
  if (!type) {
    os << "\n" << indent << "- <null type>";
    return;
  }

  os << "\n"
     << indent << "- str=`" << type->str()
     << "`, kind=" << schemaTypeKindName(type->kind())
     << ", class=" << schemaTypeRuntimeClassName(*type);

  const auto children = type->containedTypes();
  for (const auto& child : children) {
    appendTypeTree(os, child, depth + 1);
  }
}

std::string buildFunctionSchemaTypeTreeDebugString(
    const c10::FunctionSchema& schema) {
  std::ostringstream os;
  os << "schema_type_tree";
  for (size_t i = 0; i < schema.arguments().size(); ++i) {
    const auto& arg = schema.arguments()[i];
    os << "\narg[" << i << "] `" << arg.name() << "`";
    appendTypeTree(os, arg.type(), 1);
  }
  for (size_t i = 0; i < schema.returns().size(); ++i) {
    const auto& ret = schema.returns()[i];
    os << "\nret[" << i << "] `" << ret.name() << "`";
    appendTypeTree(os, ret.type(), 1);
  }
  return os.str();
}

class SchemaParser final {
 public:
  explicit SchemaParser(const std::string& schema) : schema_(schema) {}

  std::variant<std::string, c10::FunctionSchema> parseExactlyOneDeclaration() {
    // Parse exactly one declaration and reject trailing characters so callers
    // can treat a successful parse as fully validated schema text.
    skipWhitespace();
    if (atEnd()) {
      return std::string();
    }
    auto result = parseDeclaration();
    skipWhitespace();
    TORCH_CHECK(atEnd(), "Unexpected trailing content", posInfo());
    return result;
  }

 private:
  std::variant<std::string, c10::FunctionSchema> parseDeclaration() {
    // Declarations are either:
    // 1) operator name only
    // 2) full schema: name(args) -> returns
    const std::string name = parseOperatorName();
    skipWhitespace();
    if (atEnd() || peek() != TORCH_SCHEMA_CH_LPAREN) {
      return name;
    }

    std::vector<c10::Argument> arguments;
    std::vector<c10::Argument> returns;
    bool kwarg_only = false;
    bool is_vararg = false;
    bool is_varret = false;
    size_t idx = 0;

    parseDelimitedList(TORCH_SCHEMA_CH_LPAREN, TORCH_SCHEMA_CH_RPAREN, [&] {
      skipWhitespace();
      if (consumeLiteral(TORCH_SCHEMA_LIT_VARARG)) {
        TORCH_CHECK(
            !is_vararg, "Duplicate vararg (...) declaration", posInfo());
        is_vararg = true;
        return;
      }
      if (consumeChar(TORCH_SCHEMA_CH_STAR)) {
        kwarg_only = true;
        return;
      }
      TORCH_CHECK(!is_vararg,
                  "... must be the last element of the argument list",
                  posInfo());
      arguments.push_back(
          parseArgument(idx++, /*is_return=*/false, /*kwarg_only=*/kwarg_only));
    });

    if (is_vararg) {
      for (const auto& arg : arguments) {
        TORCH_CHECK(!arg.default_value().has_value(),
                    "Schemas with vararg (...) cannot have default arguments");
      }
    }

    skipWhitespace();
    expectLiteral(TORCH_SCHEMA_LIT_ARROW);
    skipWhitespace();

    // In FunctionSchema, `-> (T1, T2)` means two return slots. It is not a
    // single Tuple type return unless the schema explicitly encodes that type.
    if (consumeLiteral(TORCH_SCHEMA_LIT_VARARG)) {
      is_varret = true;
    } else if (consumeChar(TORCH_SCHEMA_CH_LPAREN)) {
      skipWhitespace();
      if (!consumeChar(TORCH_SCHEMA_CH_RPAREN)) {
        size_t return_idx = 0;
        while (true) {
          skipWhitespace();
          if (consumeLiteral(TORCH_SCHEMA_LIT_VARARG)) {
            TORCH_CHECK(
                !is_varret, "Duplicate varret (...) declaration", posInfo());
            is_varret = true;
            skipWhitespace();
            TORCH_CHECK(peek() == TORCH_SCHEMA_CH_RPAREN,
                        "... must be the last element of the return list",
                        posInfo());
          } else {
            TORCH_CHECK(!is_varret,
                        "... must be the last element of the return list",
                        posInfo());
            returns.push_back(parseArgument(
                return_idx++, /*is_return=*/true, /*kwarg_only=*/false));
          }
          skipWhitespace();
          if (consumeChar(TORCH_SCHEMA_CH_COMMA)) {
            continue;
          }
          expectChar(TORCH_SCHEMA_CH_RPAREN);
          break;
        }
      }
    } else {
      returns.push_back(
          parseArgument(0, /*is_return=*/true, /*kwarg_only=*/false));
    }

    return c10::FunctionSchema(
        std::move(arguments), std::move(returns), is_vararg, is_varret);
  }

  c10::Argument parseArgument(size_t /*idx*/, bool is_return, bool kwarg_only) {
    // Type and alias syntax is parsed by SchemaTypeParser. This method handles
    // argument-level decorations such as fixed-size list suffixes, names and
    // defaults.
    SchemaTypeParser type_parser(schema_, &pos_, &next_fresh_alias_id_);
    ParsedType parsed = type_parser.parseType();
    std::optional<int32_t> N;
    std::optional<torch::IValue> default_value;
    std::string name;

    skipWhitespace();
    if (!is_return && consumeChar(TORCH_SCHEMA_CH_LBRACKET)) {
      skipWhitespace();
      const std::string n_str = parseUnsignedNumber();
      int64_t n64 = 0;
      try {
        n64 = std::stoll(n_str);
      } catch (const std::exception&) {
        TORCH_CHECK(false, "Invalid fixed-size list length", posInfo());
      }
      TORCH_CHECK(n64 >= 0 && n64 <= std::numeric_limits<int32_t>::max(),
                  "Fixed-size list length out of range",
                  posInfo());
      N = static_cast<int32_t>(n64);
      skipWhitespace();
      expectChar(TORCH_SCHEMA_CH_RBRACKET);
      skipWhitespace();
      if (consumeChar(TORCH_SCHEMA_CH_QMARK)) {
        parsed.type = c10::makeSchemaOptionalType(parsed.type);
      }
      // Container alias annotation belongs to the outer list-like container.
      // Element alias information is kept as contained type alias metadata.
      auto container_alias = type_parser.parseAliasAnnotation();
      if (container_alias.has_value() && parsed.alias_info.has_value()) {
        container_alias->addContainedType(std::move(*parsed.alias_info));
      }
      if (container_alias.has_value()) {
        parsed.alias_info = std::move(container_alias);
      }
    }

    if (is_return) {
      skipWhitespace();
      if (!atEnd() && isIdentifierStart(peek())) {
        name = parseIdentifier("return field name");
      } else {
        name = "";
      }
    } else {
      name = parseIdentifier("argument name");
      skipWhitespace();
      if (consumeChar(TORCH_SCHEMA_CH_EQUAL)) {
        default_value = parseDefaultValue(*parsed.type);
      }
    }

    return c10::Argument(std::move(name),
                         parsed.type,
                         parsed.type,
                         N,
                         std::move(default_value),
                         !is_return && kwarg_only,
                         std::move(parsed.alias_info));
  }

  torch::IValue parseDefaultValue(const c10::Type& arg_type) {
    skipWhitespace();
    TORCH_CHECK(!atEnd(), "Missing default value", posInfo());

    if (consumeKeyword(TORCH_SCHEMA_KW_NONE)) {
      return torch::IValue();
    }
    if (consumeKeyword(TORCH_SCHEMA_KW_TRUE)) {
      return torch::IValue(true);
    }
    if (consumeKeyword(TORCH_SCHEMA_KW_FALSE)) {
      return torch::IValue(false);
    }
    if (peek() == TORCH_SCHEMA_CH_DQUOTE || peek() == TORCH_SCHEMA_CH_SQUOTE) {
      return torch::IValue(parseStringLiteral());
    }
    if (peek() == TORCH_SCHEMA_CH_PLUS || peek() == TORCH_SCHEMA_CH_MINUS ||
        std::isdigit(peekAsUnsigned())) {
      return parseNumericLiteral();
    }
    if (isIdentifierStart(peek())) {
      const std::string ident = parseIdentifier("default value");
      TORCH_CHECK(arg_type.kind() == c10::TypeKind::StringType,
                  "Unsupported identifier default value `",
                  ident,
                  "`",
                  posInfo());
      return torch::IValue(ident);
    }

    TORCH_CHECK(false, "Unsupported default value", posInfo());
  }

  torch::IValue parseNumericLiteral() {
    skipWhitespace();
    const size_t start = pos_;
    bool seen_digit = false;
    bool is_float = false;

    if (peek() == TORCH_SCHEMA_CH_PLUS || peek() == TORCH_SCHEMA_CH_MINUS) {
      ++pos_;
    }
    while (!atEnd() && std::isdigit(peekAsUnsigned())) {
      ++pos_;
      seen_digit = true;
    }
    if (!atEnd() && peek() == TORCH_SCHEMA_CH_DOT) {
      is_float = true;
      ++pos_;
      while (!atEnd() && std::isdigit(peekAsUnsigned())) {
        ++pos_;
        seen_digit = true;
      }
    }
    if (!atEnd() && (peek() == TORCH_SCHEMA_CH_EXP_LOWER ||
                     peek() == TORCH_SCHEMA_CH_EXP_UPPER)) {
      is_float = true;
      ++pos_;
      if (!atEnd() &&
          (peek() == TORCH_SCHEMA_CH_PLUS || peek() == TORCH_SCHEMA_CH_MINUS)) {
        ++pos_;
      }
      bool has_exp_digit = false;
      while (!atEnd() && std::isdigit(peekAsUnsigned())) {
        ++pos_;
        has_exp_digit = true;
      }
      TORCH_CHECK(has_exp_digit, "Malformed numeric literal", posInfo());
    }

    TORCH_CHECK(seen_digit, "Malformed numeric literal", posInfo());
    const std::string literal = schema_.substr(start, pos_ - start);
    try {
      if (is_float) {
        return torch::IValue(std::stod(literal));
      }
      return torch::IValue(static_cast<int64_t>(std::stoll(literal)));
    } catch (const std::exception&) {
      TORCH_CHECK(
          false, "Failed to parse numeric literal `", literal, "`", posInfo());
    }
  }

  std::string parseStringLiteral() {
    skipWhitespace();
    TORCH_CHECK(!atEnd() && (peek() == TORCH_SCHEMA_CH_DQUOTE ||
                             peek() == TORCH_SCHEMA_CH_SQUOTE),
                "Expected string literal",
                posInfo());
    const char quote = peek();
    ++pos_;

    std::string out;
    while (!atEnd()) {
      char c = schema_[pos_++];
      if (c == quote) {
        return out;
      }
      if (c == TORCH_SCHEMA_CH_BACKSLASH) {
        TORCH_CHECK(!atEnd(), "Unterminated escape sequence", posInfo());
        const char escaped = schema_[pos_++];
        switch (escaped) {
          case TORCH_SCHEMA_CH_N:
            out.push_back('\n');
            break;
          case TORCH_SCHEMA_CH_T:
            out.push_back('\t');
            break;
          case TORCH_SCHEMA_CH_R:
            out.push_back('\r');
            break;
          case TORCH_SCHEMA_CH_BACKSLASH:
            out.push_back('\\');
            break;
          case TORCH_SCHEMA_CH_SQUOTE:
            out.push_back('\'');
            break;
          case TORCH_SCHEMA_CH_DQUOTE:
            out.push_back('"');
            break;
          default:
            out.push_back(escaped);
            break;
        }
      } else {
        out.push_back(c);
      }
    }

    TORCH_CHECK(false, "Unterminated string literal", posInfo());
  }

  template <typename Callback>
  void parseDelimitedList(char begin, char end, Callback&& callback) {
    // Shared list parser for "(...)" sections with comma-separated elements.
    skipWhitespace();
    expectChar(begin);
    skipWhitespace();
    if (consumeChar(end)) {
      return;
    }
    while (true) {
      callback();
      skipWhitespace();
      if (consumeChar(TORCH_SCHEMA_CH_COMMA)) {
        continue;
      }
      expectChar(end);
      return;
    }
  }

  std::string parseOperatorName() {
    skipWhitespace();
    const size_t start = pos_;
    while (!atEnd()) {
      const char c = peek();
      if (std::isspace(peekAsUnsigned()) || c == TORCH_SCHEMA_CH_LPAREN) {
        break;
      }
      ++pos_;
    }
    TORCH_CHECK(start != pos_, "Expected operator name", posInfo());
    return schema_.substr(start, pos_ - start);
  }

  std::string parseIdentifier(const char* desc) {
    skipWhitespace();
    TORCH_CHECK(
        !atEnd() && isIdentifierStart(peek()), "Expected ", desc, posInfo());
    const size_t start = pos_++;
    while (!atEnd() && isIdentifierChar(peek())) {
      ++pos_;
    }
    return schema_.substr(start, pos_ - start);
  }

  std::string parseUnsignedNumber() {
    skipWhitespace();
    TORCH_CHECK(!atEnd() && std::isdigit(peekAsUnsigned()),
                "Expected an unsigned number",
                posInfo());
    const size_t start = pos_++;
    while (!atEnd() && std::isdigit(peekAsUnsigned())) {
      ++pos_;
    }
    return schema_.substr(start, pos_ - start);
  }

  bool consumeKeyword(const char* kw) {
    skipWhitespace();
    const size_t len = std::char_traits<char>::length(kw);
    if (schema_.compare(pos_, len, kw) != 0) {
      return false;
    }
    const size_t next = pos_ + len;
    if (next < schema_.size() && isIdentifierChar(schema_[next])) {
      return false;
    }
    pos_ = next;
    return true;
  }

  bool consumeLiteral(const char* literal) {
    const size_t len = std::char_traits<char>::length(literal);
    if (schema_.compare(pos_, len, literal) == 0) {
      pos_ += len;
      return true;
    }
    return false;
  }

  void expectLiteral(const char* literal) {
    TORCH_CHECK(consumeLiteral(literal), "Expected `", literal, "`", posInfo());
  }

  bool consumeChar(char c) {
    if (!atEnd() && schema_[pos_] == c) {
      ++pos_;
      return true;
    }
    return false;
  }

  void expectChar(char c) {
    TORCH_CHECK(
        !atEnd() && schema_[pos_] == c, "Expected `", c, "`", posInfo());
    ++pos_;
  }

  char peek() const {
    TORCH_INTERNAL_ASSERT(!atEnd());
    return schema_[pos_];
  }

  unsigned char peekAsUnsigned() const {
    return static_cast<unsigned char>(peek());
  }

  bool atEnd() const { return pos_ >= schema_.size(); }

  void skipWhitespace() {
    while (!atEnd() && std::isspace(peekAsUnsigned())) {
      ++pos_;
    }
  }

  static bool isIdentifierStart(char c) {
    const auto uc = static_cast<unsigned char>(c);
    return std::isalpha(uc) || c == '_';
  }

  static bool isIdentifierChar(char c) {
    const auto uc = static_cast<unsigned char>(c);
    return std::isalnum(uc) || c == '_';
  }

  std::string posInfo() const {
    std::ostringstream os;
    os << " at position " << pos_ << " in schema `" << schema_ << "`";
    return os.str();
  }

  const std::string& schema_;
  size_t pos_{0};
  size_t next_fresh_alias_id_{0};
};

}  // namespace

std::variant<std::string, c10::FunctionSchema> parseSchemaOrName(
    const std::string& schemaOrName) {
  auto parsed = SchemaParser(schemaOrName).parseExactlyOneDeclaration();
  VLOG(3) << "parseSchemaOrName input=`" << schemaOrName
          << "` parsed=" << parsedDeclarationToDebugString(parsed);
  if (VLOG_IS_ON(4) && std::holds_alternative<c10::FunctionSchema>(parsed)) {
    VLOG(4) << buildFunctionSchemaTypeTreeDebugString(
        std::get<c10::FunctionSchema>(parsed));
  }
  return parsed;
}

c10::FunctionSchema parseSchema(const std::string& schema) {
  auto parsed = parseSchemaOrName(schema);
  TORCH_CHECK(
      std::holds_alternative<c10::FunctionSchema>(parsed),
      "Tried to parse a function schema but only the operator name was given");
  VLOG(3) << "parseSchema input=`" << schema
          << "` output=" << std::get<c10::FunctionSchema>(parsed);
  return std::get<c10::FunctionSchema>(std::move(parsed));
}

std::string parseName(const std::string& name) {
  auto parsed = parseSchemaOrName(name);
  TORCH_CHECK(
      std::holds_alternative<std::string>(parsed),
      "Tried to parse an operator name but a function schema was given");
  VLOG(3) << "parseName input=`" << name
          << "` output=" << std::get<std::string>(parsed);
  return std::get<std::string>(std::move(parsed));
}

std::string schemaTypeTreeToDebugString(const c10::FunctionSchema& schema) {
  return buildFunctionSchemaTypeTreeDebugString(schema);
}

}  // namespace torch::jit
