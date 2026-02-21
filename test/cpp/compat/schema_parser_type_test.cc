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

#include <torch/library.h>

#include <string>
#include <vector>
#include "gtest/gtest.h"

namespace {

c10::FunctionSchema ParseAsSchema(const std::string& schema_text) {
  auto parsed = torch::jit::parseSchemaOrName(schema_text);
  EXPECT_TRUE(std::holds_alternative<c10::FunctionSchema>(parsed))
      << "schema: " << schema_text;
  return std::get<c10::FunctionSchema>(std::move(parsed));
}

}  // namespace

TEST(schema_parser_type_test, TorchCodecSchemasSmoke) {
  struct SchemaCase {
    const char* reason;
    const char* schema;
  };

  // Reason: keep a compact smoke set that exercises torchcodec-like grammar
  // shapes without duplicating all type-structure assertions in one test.
  const std::vector<SchemaCase> schemas = {
      {"optional string arg + default None",
       "create_from_file(str filename, str? seek_mode=None) -> Tensor"},
      {"multiple optional numeric args + void return",
       "encode_audio_to_file(Tensor samples, int sample_rate, str filename, "
       "int? bit_rate=None, int? num_channels=None, int? "
       "desired_sample_rate=None) -> ()"},
      {"alias annotation + kw-only section + defaults",
       "add_audio_stream(Tensor(a!) decoder, *, int? stream_index=None, int? "
       "sample_rate=None, int? num_channels=None) -> ()"},
      {"multi-value return list",
       "get_next_frame(Tensor(a!) decoder) -> (Tensor, Tensor, Tensor)"},
      {"kw-only arg in call-site-sensitive API",
       "get_frame_at_index(Tensor(a!) decoder, *, int frame_index) -> (Tensor, "
       "Tensor, Tensor)"},
      {"bool return + kw-only args",
       "_test_frame_pts_equality(Tensor(a!) decoder, *, int frame_index, float "
       "pts_seconds_to_test) -> bool"},
      {"no-arg function + string return",
       "_get_json_ffmpeg_library_versions() -> str"}};

  for (const auto& test_case : schemas) {
    EXPECT_NO_THROW({
      auto parsed = torch::jit::parseSchemaOrName(test_case.schema);
      EXPECT_TRUE(std::holds_alternative<c10::FunctionSchema>(parsed))
          << "reason: " << test_case.reason << ", schema: " << test_case.schema;
    }) << "reason: "
       << test_case.reason << ", schema: " << test_case.schema;
  }
}

TEST(schema_parser_type_test, OptionalScalarArgumentType) {
  // Reason: torchcodec uses `float? stop_seconds` in range APIs.
  const std::string schema_text =
      "get_frames_by_pts_in_range_audio(Tensor(a!) decoder, *, float "
      "start_seconds, float? stop_seconds) -> (Tensor, Tensor)";
  auto parsed = torch::jit::parseSchemaOrName(schema_text);
  ASSERT_TRUE(std::holds_alternative<c10::FunctionSchema>(parsed));

  const auto schema = std::get<c10::FunctionSchema>(parsed);
  ASSERT_EQ(schema.arguments().size(), 3UL);

  const auto& stop_seconds = schema.arguments()[2];
  ASSERT_EQ(stop_seconds.name(), "stop_seconds");
  ASSERT_NE(stop_seconds.type(), nullptr);
  EXPECT_EQ(stop_seconds.type()->kind(), c10::TypeKind::OptionalType);

  const auto optional_inner = stop_seconds.type()->containedTypes();
  ASSERT_EQ(optional_inner.size(), 1UL);
  EXPECT_EQ(optional_inner[0]->kind(), c10::TypeKind::FloatType);
}

TEST(schema_parser_type_test, OptionalTupleArgumentType) {
  // Reason: torchcodec uses optional tuple payloads for frame mappings.
  const std::string schema_text =
      "_add_video_stream(Tensor(a!) decoder, *, (Tensor, Tensor, Tensor)? "
      "custom_frame_mappings=None) -> ()";
  auto parsed = torch::jit::parseSchemaOrName(schema_text);
  ASSERT_TRUE(std::holds_alternative<c10::FunctionSchema>(parsed));

  const auto schema = std::get<c10::FunctionSchema>(parsed);
  ASSERT_EQ(schema.arguments().size(), 2UL);

  const auto& mappings = schema.arguments()[1];
  ASSERT_EQ(mappings.name(), "custom_frame_mappings");
  ASSERT_NE(mappings.type(), nullptr);
  EXPECT_EQ(mappings.type()->kind(), c10::TypeKind::OptionalType);

  const auto optional_inner = mappings.type()->containedTypes();
  ASSERT_EQ(optional_inner.size(), 1UL);
  EXPECT_EQ(optional_inner[0]->kind(), c10::TypeKind::TupleType);

  const auto tuple_elements = optional_inner[0]->containedTypes();
  ASSERT_EQ(tuple_elements.size(), 3UL);
  EXPECT_EQ(tuple_elements[0]->kind(), c10::TypeKind::TensorType);
  EXPECT_EQ(tuple_elements[1]->kind(), c10::TypeKind::TensorType);
  EXPECT_EQ(tuple_elements[2]->kind(), c10::TypeKind::TensorType);
}

TEST(schema_parser_type_test, OptionalReturnType) {
  // Reason: optional return type should be parsed as Optional[T], not raw T.
  const std::string schema_text = "maybe_decode(Tensor decoder) -> Tensor?";
  auto parsed = torch::jit::parseSchemaOrName(schema_text);
  ASSERT_TRUE(std::holds_alternative<c10::FunctionSchema>(parsed));

  const auto schema = std::get<c10::FunctionSchema>(parsed);
  ASSERT_EQ(schema.returns().size(), 1UL);
  ASSERT_NE(schema.returns()[0].type(), nullptr);
  EXPECT_EQ(schema.returns()[0].type()->kind(), c10::TypeKind::OptionalType);

  const auto optional_inner = schema.returns()[0].type()->containedTypes();
  ASSERT_EQ(optional_inner.size(), 1UL);
  EXPECT_EQ(optional_inner[0]->kind(), c10::TypeKind::TensorType);
}

TEST(schema_parser_type_test, KwOnlyDefaultAndAliasMetadata) {
  // Reason: parser should preserve kw-only/default/alias metadata for callers.
  const auto schema = ParseAsSchema(
      "alias_and_kwonly(Tensor(a! -> b) x, *, int? idx=None, str "
      "mode=\"nearest\") -> ()");
  ASSERT_EQ(schema.arguments().size(), 3UL);

  const auto& x = schema.arguments()[0];
  EXPECT_FALSE(x.kwarg_only());
  ASSERT_NE(x.alias_info(), nullptr);
  EXPECT_TRUE(x.alias_info()->isWrite());
  EXPECT_EQ(x.alias_info()->beforeSets().count("a"), 1UL);
  EXPECT_EQ(x.alias_info()->afterSets().count("b"), 1UL);

  const auto& idx = schema.arguments()[1];
  EXPECT_TRUE(idx.kwarg_only());
  ASSERT_TRUE(idx.default_value().has_value());
  EXPECT_TRUE(idx.default_value()->is_none());
  ASSERT_NE(idx.type(), nullptr);
  EXPECT_EQ(idx.type()->kind(), c10::TypeKind::OptionalType);
  ASSERT_EQ(idx.type()->containedTypes().size(), 1UL);
  EXPECT_EQ(idx.type()->containedTypes()[0]->kind(), c10::TypeKind::IntType);

  const auto& mode = schema.arguments()[2];
  EXPECT_TRUE(mode.kwarg_only());
  ASSERT_TRUE(mode.default_value().has_value());
  EXPECT_EQ(mode.default_value()->to_string(), "nearest");
}

TEST(schema_parser_type_test, MultiReturnVersusTupleReturn) {
  // Reason: "(T1, T2)" is two return slots, not one Tuple return.
  const auto multi_ret = ParseAsSchema("f(Tensor x) -> (Tensor, Tensor)");
  ASSERT_EQ(multi_ret.returns().size(), 2UL);
  EXPECT_EQ(multi_ret.returns()[0].type()->kind(), c10::TypeKind::TensorType);
  EXPECT_EQ(multi_ret.returns()[1].type()->kind(), c10::TypeKind::TensorType);

  // Reason: a single Tuple return needs an extra layer of parentheses.
  const auto tuple_ret =
      ParseAsSchema("f_tuple(Tensor x) -> ((Tensor, Tensor))");
  ASSERT_EQ(tuple_ret.returns().size(), 1UL);
  ASSERT_NE(tuple_ret.returns()[0].type(), nullptr);
  EXPECT_EQ(tuple_ret.returns()[0].type()->kind(), c10::TypeKind::TupleType);
  const auto tuple_elems = tuple_ret.returns()[0].type()->containedTypes();
  ASSERT_EQ(tuple_elems.size(), 2UL);
  EXPECT_EQ(tuple_elems[0]->kind(), c10::TypeKind::TensorType);
  EXPECT_EQ(tuple_elems[1]->kind(), c10::TypeKind::TensorType);
}

TEST(schema_parser_type_test, VariadicFlagsAndValidation) {
  // Reason: variadic arg/ret markers should map to FunctionSchema flags.
  const auto variadic = ParseAsSchema("variadic(Tensor x, ...) -> ...");
  EXPECT_TRUE(variadic.is_vararg());
  EXPECT_TRUE(variadic.is_varret());
  ASSERT_EQ(variadic.arguments().size(), 1UL);
  EXPECT_EQ(variadic.arguments()[0].name(), "x");
  EXPECT_TRUE(variadic.returns().empty());

  // Reason: parser currently forbids defaults when vararg is present.
  EXPECT_ANY_THROW(torch::jit::parseSchema("broken(int x=1, ...) -> int"));
}

TEST(schema_parser_type_test, ParseNameAndParseSchemaBoundaries) {
  // Reason: parseName and parseSchema should reject mismatched inputs.
  EXPECT_EQ(torch::jit::parseName("just_name"), "just_name");

  const auto schema = torch::jit::parseSchema("named(int x) -> int");
  ASSERT_EQ(schema.arguments().size(), 1UL);
  EXPECT_EQ(schema.arguments()[0].name(), "x");

  EXPECT_ANY_THROW(torch::jit::parseSchema("name_only"));
  EXPECT_ANY_THROW(torch::jit::parseName("has_schema(int x) -> int"));
}
