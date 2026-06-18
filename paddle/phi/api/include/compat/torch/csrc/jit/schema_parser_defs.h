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

// Common literals
#define TORCH_SCHEMA_LIT_VARARG "..."
#define TORCH_SCHEMA_LIT_ARROW "->"

// Common keywords
#define TORCH_SCHEMA_KW_NONE "None"
#define TORCH_SCHEMA_KW_TRUE "true"
#define TORCH_SCHEMA_KW_FALSE "false"

// Common characters
#define TORCH_SCHEMA_CH_LPAREN '('
#define TORCH_SCHEMA_CH_RPAREN ')'
#define TORCH_SCHEMA_CH_LBRACKET '['
#define TORCH_SCHEMA_CH_RBRACKET ']'
#define TORCH_SCHEMA_CH_COMMA ','
#define TORCH_SCHEMA_CH_STAR '*'
#define TORCH_SCHEMA_CH_BANG '!'
#define TORCH_SCHEMA_CH_PIPE '|'
#define TORCH_SCHEMA_CH_QMARK '?'
#define TORCH_SCHEMA_CH_EQUAL '='
#define TORCH_SCHEMA_CH_DOT '.'
#define TORCH_SCHEMA_CH_PLUS '+'
#define TORCH_SCHEMA_CH_MINUS '-'
#define TORCH_SCHEMA_CH_EXP_LOWER 'e'
#define TORCH_SCHEMA_CH_EXP_UPPER 'E'
#define TORCH_SCHEMA_CH_DQUOTE '"'
#define TORCH_SCHEMA_CH_SQUOTE '\''
#define TORCH_SCHEMA_CH_BACKSLASH '\\'
#define TORCH_SCHEMA_CH_N 'n'
#define TORCH_SCHEMA_CH_T 't'
#define TORCH_SCHEMA_CH_R 'r'

// Alias syntax literals
#define TORCH_SCHEMA_ALIAS_WILDCARD "*"
#define TORCH_SCHEMA_ALIAS_FRESH_PREFIX "$"

// Type spellings in schema text
#define TORCH_SCHEMA_TYPE_CPP_DOUBLE "double"
#define TORCH_SCHEMA_TYPE_CPP_INT64_T "int64_t"
#define TORCH_SCHEMA_TYPE_TENSOR "Tensor"
#define TORCH_SCHEMA_TYPE_STR "str"
#define TORCH_SCHEMA_TYPE_STRING "string"
#define TORCH_SCHEMA_TYPE_INT "int"
#define TORCH_SCHEMA_TYPE_FLOAT "float"
#define TORCH_SCHEMA_TYPE_BOOL "bool"
#define TORCH_SCHEMA_TYPE_NONE "None"
#define TORCH_SCHEMA_TYPE_NONE_TYPE "NoneType"
#define TORCH_SCHEMA_TYPE_DEVICE "Device"
#define TORCH_SCHEMA_TYPE_SCALAR "Scalar"
#define TORCH_SCHEMA_TYPE_NUMBER "number"

// Base type conversion table for schema text -> TypeKind + canonical repr.
// Entry shape:
//   _(input_text_literal, type_kind_enum_member, canonical_repr_literal)
// Notes:
// - Some entries intentionally map aliases to the same canonical repr.
// - This table only covers atomic/base types; tuple/optional are parsed
//   structurally in SchemaTypeParser::parseType().
#define TORCH_SCHEMA_BASE_TYPE_CONVERSION_TABLE(_)                      \
  _(TORCH_SCHEMA_TYPE_TENSOR, TensorType, TORCH_SCHEMA_TYPE_TENSOR)     \
  _(TORCH_SCHEMA_TYPE_STR, StringType, TORCH_SCHEMA_TYPE_STR)           \
  _(TORCH_SCHEMA_TYPE_STRING, StringType, TORCH_SCHEMA_TYPE_STR)        \
  _(TORCH_SCHEMA_TYPE_INT, IntType, TORCH_SCHEMA_TYPE_INT)              \
  _(TORCH_SCHEMA_TYPE_FLOAT, FloatType, TORCH_SCHEMA_TYPE_FLOAT)        \
  _(TORCH_SCHEMA_TYPE_BOOL, BoolType, TORCH_SCHEMA_TYPE_BOOL)           \
  _(TORCH_SCHEMA_TYPE_NONE, NoneType, TORCH_SCHEMA_TYPE_NONE_TYPE)      \
  _(TORCH_SCHEMA_TYPE_NONE_TYPE, NoneType, TORCH_SCHEMA_TYPE_NONE_TYPE) \
  _(TORCH_SCHEMA_TYPE_DEVICE, DeviceObjType, TORCH_SCHEMA_TYPE_DEVICE)  \
  _(TORCH_SCHEMA_TYPE_SCALAR, NumberType, TORCH_SCHEMA_TYPE_SCALAR)     \
  _(TORCH_SCHEMA_TYPE_NUMBER, NumberType, TORCH_SCHEMA_TYPE_SCALAR)
