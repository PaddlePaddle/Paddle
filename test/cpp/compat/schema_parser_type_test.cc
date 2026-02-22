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
#include "test/cpp/utils/exception_test_utils.h"
#include "torch/csrc/jit/schema_type_parser.h"

c10::FunctionSchema ParseAsSchema(const std::string& schema_text) {
  auto parsed = torch::jit::parseSchemaOrName(schema_text);
  EXPECT_TRUE(std::holds_alternative<c10::FunctionSchema>(parsed))
      << "schema: " << schema_text;
  return std::get<c10::FunctionSchema>(std::move(parsed));
}

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

TEST(schema_parser_type_test, OptionalTypeShapes) {
  // Reason: cover optional scalar arg, optional tuple arg, and optional return.
  {
    const auto schema = ParseAsSchema(
        "get_frames_by_pts_in_range_audio(Tensor(a!) decoder, *, float "
        "start_seconds, float? stop_seconds) -> (Tensor, Tensor)");
    ASSERT_EQ(schema.arguments().size(), 3UL);

    const auto& stop_seconds = schema.arguments()[2];
    ASSERT_EQ(stop_seconds.name(), "stop_seconds");
    ASSERT_NE(stop_seconds.type(), nullptr);
    EXPECT_EQ(stop_seconds.type()->kind(), c10::TypeKind::OptionalType);
    const auto optional_inner = stop_seconds.type()->containedTypes();
    ASSERT_EQ(optional_inner.size(), 1UL);
    EXPECT_EQ(optional_inner[0]->kind(), c10::TypeKind::FloatType);
  }

  {
    const auto schema = ParseAsSchema(
        "_add_video_stream(Tensor(a!) decoder, *, (Tensor, Tensor, Tensor)? "
        "custom_frame_mappings=None) -> ()");
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

  {
    const auto schema =
        ParseAsSchema("maybe_decode(Tensor decoder) -> Tensor?");
    ASSERT_EQ(schema.returns().size(), 1UL);
    ASSERT_NE(schema.returns()[0].type(), nullptr);
    EXPECT_EQ(schema.returns()[0].type()->kind(), c10::TypeKind::OptionalType);
    const auto optional_inner = schema.returns()[0].type()->containedTypes();
    ASSERT_EQ(optional_inner.size(), 1UL);
    EXPECT_EQ(optional_inner[0]->kind(), c10::TypeKind::TensorType);
  }
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

TEST(schema_parser_type_test, SchemaTypeTreeDebugStringCoversNestedTypes) {
  // Reason: appendTypeTree() is only used by debug-tree rendering and should
  // recursively print nested Optional/Tuple structures.
  const auto schema = ParseAsSchema(
      "tree_dbg((Tensor, float?)? payload=None) -> (Tensor, Tensor?)");
  const auto debug_str = torch::jit::schemaTypeTreeToDebugString(schema);

  EXPECT_NE(debug_str.find("schema_type_tree"), std::string::npos);
  EXPECT_NE(debug_str.find("arg[0] `payload`"), std::string::npos);
  EXPECT_NE(debug_str.find("kind=OptionalType"), std::string::npos);
  EXPECT_NE(debug_str.find("kind=TupleType"), std::string::npos);
  EXPECT_NE(debug_str.find("kind=TensorType"), std::string::npos);
  EXPECT_NE(debug_str.find("kind=FloatType"), std::string::npos);
  EXPECT_NE(debug_str.find("ret[1] ``"), std::string::npos);
}

TEST(schema_parser_type_test, DefaultsFixedListAndNamedReturns) {
  // Reason: cover default-literal parsing branches and fixed-size list suffix.
  const auto schema = ParseAsSchema(
      R"schema(defaults(int[3]? xs=None, bool bt=true, bool bf=false, str ident=truevalue, float exp=1.25e-1, int plus=+42, str esc="a\n\t\r\\\'\"\q") -> (Tensor out, int idx))schema");

  ASSERT_EQ(schema.arguments().size(), 7UL);
  const auto& xs = schema.arguments()[0];
  ASSERT_TRUE(xs.N().has_value());
  EXPECT_EQ(*xs.N(), 3);
  ASSERT_NE(xs.type(), nullptr);
  EXPECT_EQ(xs.type()->kind(), c10::TypeKind::OptionalType);
  ASSERT_TRUE(xs.default_value().has_value());
  EXPECT_TRUE(xs.default_value()->is_none());

  const auto& bt = schema.arguments()[1];
  ASSERT_TRUE(bt.default_value().has_value());
  EXPECT_TRUE(bt.default_value()->is_bool());
  EXPECT_TRUE(bt.default_value()->to_bool());

  const auto& bf = schema.arguments()[2];
  ASSERT_TRUE(bf.default_value().has_value());
  EXPECT_TRUE(bf.default_value()->is_bool());
  EXPECT_FALSE(bf.default_value()->to_bool());

  const auto& ident = schema.arguments()[3];
  ASSERT_TRUE(ident.default_value().has_value());
  EXPECT_TRUE(ident.default_value()->is_string());
  EXPECT_EQ(ident.default_value()->to_string(), "truevalue");

  const auto& exp = schema.arguments()[4];
  ASSERT_TRUE(exp.default_value().has_value());
  EXPECT_TRUE(exp.default_value()->is_double());
  EXPECT_DOUBLE_EQ(exp.default_value()->to_double(), 0.125);

  const auto& plus = schema.arguments()[5];
  ASSERT_TRUE(plus.default_value().has_value());
  EXPECT_TRUE(plus.default_value()->is_int());
  EXPECT_EQ(plus.default_value()->to_int(), 42);

  const auto& esc = schema.arguments()[6];
  ASSERT_TRUE(esc.default_value().has_value());
  EXPECT_TRUE(esc.default_value()->is_string());
  const std::string esc_value = esc.default_value()->to_string();
  EXPECT_NE(esc_value.find('\n'), std::string::npos);
  EXPECT_NE(esc_value.find('\t'), std::string::npos);
  EXPECT_NE(esc_value.find('\r'), std::string::npos);
  EXPECT_NE(esc_value.find('\\'), std::string::npos);
  EXPECT_NE(esc_value.find('\''), std::string::npos);
  EXPECT_NE(esc_value.find('"'), std::string::npos);
  EXPECT_EQ(esc_value.back(), 'q');

  ASSERT_EQ(schema.returns().size(), 2UL);
  EXPECT_EQ(schema.returns()[0].name(), "out");
  EXPECT_EQ(schema.returns()[0].type()->kind(), c10::TypeKind::TensorType);
  EXPECT_EQ(schema.returns()[1].name(), "idx");
  EXPECT_EQ(schema.returns()[1].type()->kind(), c10::TypeKind::IntType);
}

TEST(schema_parser_type_test, ParserErrorBranchMatrix) {
  // Reason: explicitly cover parser error branches that are easy to miss.
  const auto empty = torch::jit::parseSchemaOrName("   ");
  ASSERT_TRUE(std::holds_alternative<std::string>(empty));
  EXPECT_EQ(std::get<std::string>(empty), "");

  const std::vector<std::string> bad_schemas = {
      "op(int x) -> int trailing extra",
      "dup_vararg(..., ...) -> int",
      "vararg_not_last(..., int x) -> int",
      "dup_varret() -> (..., ...)",
      "varret_not_last() -> (..., int)",
      "missing_default(int x=) -> int",
      "unsupported_default(int x=@) -> int",
      "identifier_for_non_string(int x=cpu) -> int",
      "malformed_number(float x=1e) -> int",
      "overflow_number(int x=999999999999999999999999999999999999999999)",
      "(int x) -> int",
      "missing_arg_name(int ) -> int",
      "missing_unsigned(int[] x) -> int",
      "missing_arrow(int x) int",
      "missing_rparen(int x -> int",
      "invalid_fixed_size(int[999999999999999999999999999999999999] x) -> int",
  };
  for (const auto& schema_text : bad_schemas) {
    EXPECT_ANY_THROW(torch::jit::parseSchemaOrName(schema_text))
        << "schema: " << schema_text;
  }

  EXPECT_ANY_THROW(torch::jit::parseSchemaOrName("unterminated(str s=\"abc"));
  EXPECT_ANY_THROW(torch::jit::parseSchemaOrName(
      std::string("unterminated_escape(str s=\"abc\\")));
}

TEST(schema_parser_type_test, SchemaTypeParserErrorDiagnostics) {
  // Reason: schema_type_parser.cpp has dedicated diagnostics for these inputs.
  struct ErrorCase {
    const char* schema;
    const char* expected_substr;
  };
  const std::vector<ErrorCase> cases = {
      {"bad(double x) -> int", "Use `float` instead of `double`"},
      {"bad(int64_t x) -> int", "Use `int` instead of `int64_t`"},
      {"bad(foo.bar x) -> int", "Unsupported type specifier `foo.bar`"},
      {"alias_missing_rparen(Tensor(a! x) -> ()", "Expected `)`"},
      {"alias_bad_set(Tensor(!) x) -> ()", "Expected alias set"}};
  for (const auto& test_case : cases) {
    test::utils::ExpectThrowContains<std::exception>(
        [&]() { (void)torch::jit::parseSchemaOrName(test_case.schema); },
        test_case.expected_substr,
        std::string("schema: ") + test_case.schema);
  }
}

TEST(schema_parser_type_test, SchemaTypeParserAliasAndTupleForms) {
  // Reason: cover alias shapes and tuple parsing in a single table-like test.
  {
    const auto schema =
        ParseAsSchema("fresh_alias(Tensor! a, Tensor! b) -> ()");
    ASSERT_EQ(schema.arguments().size(), 2UL);

    const auto* alias_a = schema.arguments()[0].alias_info();
    ASSERT_NE(alias_a, nullptr);
    EXPECT_TRUE(alias_a->isWrite());
    EXPECT_EQ(alias_a->beforeSets().count("$0"), 1UL);
    EXPECT_EQ(alias_a->afterSets().count("$0"), 1UL);

    const auto* alias_b = schema.arguments()[1].alias_info();
    ASSERT_NE(alias_b, nullptr);
    EXPECT_TRUE(alias_b->isWrite());
    EXPECT_EQ(alias_b->beforeSets().count("$1"), 1UL);
    EXPECT_EQ(alias_b->afterSets().count("$1"), 1UL);
  }

  {
    const auto schema = ParseAsSchema("wild_alias(Tensor(*! -> a|*) x) -> ()");
    ASSERT_EQ(schema.arguments().size(), 1UL);
    const auto* alias = schema.arguments()[0].alias_info();
    ASSERT_NE(alias, nullptr);
    EXPECT_TRUE(alias->isWrite());
    EXPECT_EQ(alias->beforeSets().count("*"), 1UL);
    EXPECT_EQ(alias->afterSets().count("a"), 1UL);
    EXPECT_EQ(alias->afterSets().count("*"), 1UL);
  }

  {
    const auto schema =
        ParseAsSchema("tuple_alias((Tensor(a), Tensor(b!)) x) -> ()");
    ASSERT_EQ(schema.arguments().size(), 1UL);
    const auto* alias = schema.arguments()[0].alias_info();
    ASSERT_NE(alias, nullptr);
    EXPECT_TRUE(alias->beforeSets().empty());
    EXPECT_TRUE(alias->afterSets().empty());
    ASSERT_EQ(alias->containedTypes().size(), 2UL);
    EXPECT_EQ(alias->containedTypes()[0].beforeSets().count("a"), 1UL);
    EXPECT_FALSE(alias->containedTypes()[0].isWrite());
    EXPECT_EQ(alias->containedTypes()[1].beforeSets().count("b"), 1UL);
    EXPECT_TRUE(alias->containedTypes()[1].isWrite());
  }

  {
    const auto schema = ParseAsSchema("empty_tuple_arg(() x) -> ()");
    ASSERT_EQ(schema.arguments().size(), 1UL);
    ASSERT_NE(schema.arguments()[0].type(), nullptr);
    EXPECT_EQ(schema.arguments()[0].type()->kind(), c10::TypeKind::TupleType);
    EXPECT_TRUE(schema.arguments()[0].type()->containedTypes().empty());
  }
}

TEST(schema_parser_type_test, SchemaTypeParserCtorRejectsNullPointers) {
  // Reason: cover refFromPtr null-check branch.
  size_t pos = 0;
  size_t fresh_id = 0;
  EXPECT_ANY_THROW(
      (void)torch::jit::SchemaTypeParser("Tensor", nullptr, &fresh_id));
  EXPECT_ANY_THROW((void)torch::jit::SchemaTypeParser("Tensor", &pos, nullptr));
}

TEST(schema_parser_type_test, JitTypeAtomicTypeOperatorsAndAnnotation) {
  // Reason: cover jit_type.h atomic helpers used by schema parser internals.
  auto tensor_type_a =
      c10::makeSchemaAtomicType(c10::TypeKind::TensorType, "Tensor");
  auto tensor_type_b =
      c10::makeSchemaAtomicType(c10::TypeKind::TensorType, "Tensor");
  auto int_type = c10::makeSchemaAtomicType(c10::TypeKind::IntType, "int");

  ASSERT_NE(tensor_type_a, nullptr);
  ASSERT_NE(tensor_type_b, nullptr);
  ASSERT_NE(int_type, nullptr);

  // Hits SchemaAtomicType::equals and global Type operator!=.
  EXPECT_TRUE(*tensor_type_a == *tensor_type_b);
  EXPECT_FALSE(*tensor_type_a != *tensor_type_b);
  EXPECT_TRUE(*tensor_type_a != *int_type);

  // Hits SchemaAtomicType::annotation_str_impl.
  EXPECT_EQ(tensor_type_a->annotation_str(), "Tensor");
}

TEST(schema_parser_type_test, JitTypeBaseWrapperAndAnnotationBranches) {
  // Reason: cover jit_type_base.h wrapper constructors/assignments and
  // Type::annotation_str/repr_str/createWithContained branches.
  c10::SingletonOrSharedTypePtr<c10::Type> from_singleton(c10::IntType::get());
  c10::SingletonOrSharedTypePtr<c10::IntType> from_singleton_exact(
      c10::IntType::get());
  c10::SingletonOrSharedTypePtr<c10::Type> from_nullptr(nullptr);
  EXPECT_EQ(from_nullptr, nullptr);
  EXPECT_FALSE(static_cast<bool>(from_nullptr));

  c10::SingletonOrSharedTypePtr<c10::Type> assigned;
  assigned = from_singleton;
  ASSERT_NE(assigned, nullptr);
  ASSERT_NE(assigned.get(), nullptr);
  ASSERT_NE(from_singleton_exact.get(), nullptr);

  auto shared_atomic =
      c10::makeSchemaAtomicType(c10::TypeKind::DynamicType, "shared");
  ASSERT_NE(shared_atomic, nullptr);
  c10::Type& shared_as_base = *shared_atomic;
  auto shared_roundtrip = shared_as_base.cast<c10::detail::SchemaAtomicType>();
  ASSERT_NE(shared_roundtrip, nullptr);
  EXPECT_EQ(shared_roundtrip->str(), "shared");

  const c10::Type& shared_as_const = *shared_atomic;
  auto shared_const_roundtrip =
      shared_as_const.cast<c10::detail::SchemaAtomicType>();
  ASSERT_NE(shared_const_roundtrip, nullptr);
  EXPECT_EQ(shared_const_roundtrip->str(), "shared");

  c10::SingletonOrSharedTypePtr<c10::detail::SchemaAtomicType> shared_exact_ptr(
      shared_roundtrip);
  c10::SingletonOrSharedTypePtr<c10::Type> shared_base_ptr(shared_roundtrip);
  ASSERT_NE(shared_exact_ptr.get(), nullptr);
  ASSERT_NE(shared_base_ptr.get(), nullptr);

  EXPECT_EQ(shared_as_base.cast<c10::detail::SchemaTupleType>(), nullptr);
  EXPECT_EQ(shared_as_const.cast<c10::detail::SchemaTupleType>(), nullptr);

  c10::Type& int_as_base = *c10::IntType::get();
  ASSERT_TRUE(static_cast<bool>(int_as_base.cast<c10::IntType>()));
  EXPECT_FALSE(static_cast<bool>(int_as_base.cast<c10::TensorType>()));

  const c10::Type& int_as_const = *c10::IntType::get();
  ASSERT_TRUE(static_cast<bool>(int_as_const.cast<c10::IntType>()));
  EXPECT_FALSE(static_cast<bool>(int_as_const.cast<c10::TensorType>()));

  EXPECT_EQ(from_singleton, c10::IntType::get());
  EXPECT_EQ(c10::IntType::get(), from_singleton);
  EXPECT_NE(shared_base_ptr, c10::IntType::get());
  EXPECT_NE(from_singleton, shared_base_ptr);

  const c10::Type& plain = *c10::TensorType::get();
  const c10::TypePrinter rename_printer = [](const c10::Type&) {
    return std::optional<std::string>("renamed");
  };
  const c10::TypePrinter passthrough_printer = [](const c10::Type&) {
    return std::optional<std::string>();
  };
  EXPECT_EQ(plain.annotation_str(), "Tensor");
  EXPECT_EQ(plain.annotation_str(rename_printer), "renamed");
  EXPECT_EQ(plain.annotation_str(passthrough_printer), "Tensor");
  EXPECT_EQ(plain.repr_str(), "Tensor");

  test::utils::ExpectThrowContains<std::exception>(
      [&]() {
        (void)plain.createWithContained(
            std::vector<c10::TypePtr>{c10::IntType::get()});
      },
      "type with contained types did not overload createWithContained");
}

TEST(schema_parser_type_test, JitTypeBaseEqualityAndSubtypeBranches) {
  // Reason: cover operator== symmetric/asymmetric branches and
  // Type::isSubtypeOfExt optional-inner matching.
  class AsymmetricRhsType final : public c10::Type {
   public:
    AsymmetricRhsType() : Type(c10::TypeKind::DynamicType) {}

    bool equals(const c10::Type& rhs) const override {
      (void)rhs;
      ++equals_calls_;
      return true;
    }

    bool symmetric() const override { return false; }

    std::string str() const override { return "asymmetric_rhs"; }

    int equals_calls() const { return equals_calls_; }

   private:
    mutable int equals_calls_{0};
  };

  const c10::Type& lhs = *c10::TensorType::get();
  AsymmetricRhsType rhs;
  EXPECT_TRUE(lhs == rhs);
  EXPECT_EQ(rhs.equals_calls(), 1);

  EXPECT_TRUE(*c10::IntType::get() == *c10::IntType::get());
  EXPECT_FALSE(*c10::IntType::get() == *c10::FloatType::get());

  auto optional_dynamic =
      c10::makeSchemaOptionalType(c10::TypePtr(c10::IntType::get()));
  EXPECT_TRUE(c10::IntType::get()->isSubtypeOfExt(*optional_dynamic, nullptr));
  EXPECT_FALSE(
      c10::TensorType::get()->isSubtypeOfExt(*optional_dynamic, nullptr));
}

TEST(schema_parser_type_test, TypePtrSingletonTypePtrCtorAndGet) {
  // Reason: cover type_ptr.h raw-pointer constructor and get().
  int value = 7;
  c10::SingletonTypePtr<int> ptr(&value);
  EXPECT_EQ(ptr.get(), &value);
  EXPECT_TRUE(static_cast<bool>(ptr));
  EXPECT_EQ(*ptr, 7);
  EXPECT_EQ(ptr.operator->(), &value);

  *ptr = 9;
  EXPECT_EQ(value, 9);

  c10::SingletonTypePtr<int> same_ptr(&value);
  EXPECT_TRUE(ptr == same_ptr);
  EXPECT_FALSE(ptr != same_ptr);

  int other = 3;
  c10::SingletonTypePtr<int> other_ptr(&other);
  EXPECT_FALSE(ptr == other_ptr);
  EXPECT_TRUE(ptr != other_ptr);
}
