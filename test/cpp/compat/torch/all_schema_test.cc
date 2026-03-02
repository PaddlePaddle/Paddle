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

#include <torch/all.h>
#include <torch/library.h>
#include <string>

#include "gtest/gtest.h"
#include "test/cpp/utils/exception_test_utils.h"
#include "torch/csrc/jit/schema_type_parser.h"

// -----  test/cpp/jit/test_misc.cpp start -----

namespace torch {
namespace jit {

// 暂不支持 ListType
// test a few features that are not directly used in schemas yet
// TEST(SchemaParserTest, NestedArrays) {
//   // nested arrays
//   auto s = parseSchema("at::what(int[][4] foo) -> ()");
//   ASSERT_TRUE(s.arguments().at(0).N() == 4);
//   ASSERT_TRUE(IntType::get()->isSubtypeOf(*s.arguments()
//                                                .at(0)
//                                                .type()
//                                                ->expectRef<ListType>()
//                                                .getElementType()
//                                                ->expectRef<ListType>()
//                                                .getElementType()));
//   auto s2 = parseSchema("at::what(int[][] foo) -> ()");
//   ASSERT_TRUE(IntType::get()->isSubtypeOf(*s2.arguments()
//                                                .at(0)
//                                                .type()
//                                                ->expectRef<ListType>()
//                                                .getElementType()
//                                                ->expectRef<ListType>()
//                                                .getElementType()));
// }

TEST(SchemaParserTest, OutVariant) {
  auto schema_with_out = parseSchema(
      "at::foo(Tensor self, *, Tensor(a!) f, Tensor(b!) l) -> (Tensor(a!) f, "
      "Tensor(b!) l)");
  ASSERT_TRUE(schema_with_out.arguments().at(1).is_out());
  ASSERT_TRUE(schema_with_out.arguments().at(2).is_out());

  auto schema_without_out =
      parseSchema("at::foo(Tensor self, *, int scalar) -> (int)");

  for (const auto& arg : schema_without_out.arguments()) {
    ASSERT_TRUE(!arg.is_out());
  }

  auto schema_with_is_write = parseSchema(
      "aten::ne_.Scalar(Tensor(a!) self, Scalar other) -> (Tensor(a!))");

  for (const auto& arg : schema_with_is_write.arguments()) {
    ASSERT_TRUE(!arg.is_out());
  }
}

TEST(SchemaParserTest, NamedReturns) {
  // named returns
  parseSchema("at::what(Tensor! i_will_be_written_to) -> ()");
  auto s3 =
      parseSchema("at::what() -> (Tensor the_return, Tensor the_return2)");
  ASSERT_EQ(s3.returns().at(0).name(), "the_return");
  ASSERT_EQ(s3.returns().at(1).name(), "the_return2");
}

// 不支持 FutureType
// TEST(SchemaParserTest, Futures) {
//   // futures
//   auto s4 = parseSchema("at::what(Future(int) foo) -> ()");
//   ASSERT_TRUE(IntType::get()->isSubtypeOf(
//       *s4.arguments().at(0).type()->expectRef<FutureType>().getElementType()));
// }

TEST(SchemaParserTest, AnnotatedAliasSets) {
  // test tensor with annotated alias sets
  parseSchema("at::what(Tensor(a) foo) -> (Tensor(a))");
}

// 不支持 ListType，这里的测试是一个不定长数组，目前的 parser
// 实现中只支持定长数组 TEST(SchemaParserTest, TensorListAnnotatedAliasSets) {
//   const auto s = parseSchema(
//       "at::foo(Tensor(a!) self, Tensor(b!)[] out)"
//       " -> ()");
//   const AliasInfo* selfAliasInfo = s.arguments().at(0).alias_info();
//   const AliasInfo* outAliasInfo = s.arguments().at(1).alias_info();
//   ASSERT_TRUE(
//       selfAliasInfo->beforeSets() ==
//       std::unordered_set<std::string>{std::string("alias::a")});
//   ASSERT_TRUE(selfAliasInfo->isWrite());

//   ASSERT_TRUE(outAliasInfo->isWrite());
//   ASSERT_TRUE(outAliasInfo->beforeSets().empty());
//   ASSERT_EQ(outAliasInfo->containedTypes().size(), 1);

//   auto containedType = outAliasInfo->containedTypes()[0];

//   ASSERT_TRUE(containedType.isWrite());
//   ASSERT_TRUE(
//       containedType.beforeSets() ==
//       std::unordered_set<std::string>{std::string("alias::b")});
// }

TEST(SchemaParserTest, AnnotatedAliasWithoutBeforeSet) {
  const std::string schema = "at::foo(Tensor(!) self) -> Tensor";
  test::utils::ExpectThrowContains<std::exception>(
      [&]() { (void)parseSchema(schema); },
      "Expected alias set",
      std::string("schema: ") + schema);
}

// 不支持 ListType
// TEST(SchemaParserTest, BeforeAfterSets) {
//   const auto s = parseSchema(
//       "at::what(Tensor(b|c)[](a!) list, Tensor(c) element)"
//       " -> (Tensor(b|c)[](a!))");

//   // The list itself is annotated with `a`
//   const AliasInfo* aliasInfo = s.arguments().at(0).alias_info();
//   ASSERT_NE(aliasInfo, nullptr);
//   ASSERT_TRUE(
//       aliasInfo->beforeSets() ==
//       std::unordered_set<std::string>{std::string("alias::a")});
//   ASSERT_TRUE(aliasInfo->isWrite());

//   // Check the contained types
//   ASSERT_TRUE(!aliasInfo->containedTypes().empty());
//   const auto& containedAliasInfo = aliasInfo->containedTypes()[0];
//   const auto expected = std::unordered_set<std::string>{
//       std::string("alias::b"),
//       std::string("alias::c"),
//   };
//   ASSERT_TRUE(containedAliasInfo.beforeSets() == expected);
//   ASSERT_TRUE(containedAliasInfo.afterSets() == expected);
//   ASSERT_FALSE(containedAliasInfo.isWrite());
// }

// 不支持 ListType
// TEST(SchemaParserTest, BeforeAfterSets2) {
//   const auto s = parseSchema(
//       "at::what(Tensor(b -> b|c)[](a!) list, Tensor(c) element)"
//       " -> (Tensor(b|c)[](a!))");

//   // The list itself is annotated with `a`
//   const AliasInfo* aliasInfo = s.arguments().at(0).alias_info();
//   ASSERT_NE(aliasInfo, nullptr);
//   ASSERT_EQ(
//       aliasInfo->beforeSets(),
//       std::unordered_set<std::string>{std::string("alias::a")});
//   ASSERT_EQ(
//       aliasInfo->afterSets(),
//       std::unordered_set<std::string>{std::string("alias::a")});
//   ASSERT_TRUE(aliasInfo->isWrite());
//   ASSERT_EQ(aliasInfo->containedTypes().size(), 1);

//   // Check the contained types
//   ASSERT_TRUE(!aliasInfo->containedTypes().empty());
//   const auto& containedAliasInfo = aliasInfo->containedTypes()[0];
//   const auto expectedBefore = std::unordered_set<std::string>{
//       std::string("alias::b"),
//   };
//   const auto expectedAfter = std::unordered_set<std::string>{
//       std::string("alias::b"), std::string("alias::c")};
//   ASSERT_TRUE(containedAliasInfo.beforeSets() == expectedBefore);
//   ASSERT_TRUE(containedAliasInfo.afterSets() == expectedAfter);
//   ASSERT_FALSE(containedAliasInfo.isWrite());
// }

}  // namespace jit
}  // namespace torch

// -----  test/cpp/jit/test_misc.cpp end -----

// -----  test/cpp/jit/test_schema_info.cpp start -----

TEST(FunctionSchemaIsAliasingTest, Basic) {
  c10::FunctionSchema schema = torch::jit::parseSchema(
      "aten::test.Tensor(Tensor(a) self, Tensor(b!) other, Tensor more_other) "
      "-> (Tensor(a), Tensor(b!))");
  ASSERT_TRUE(schema.is_aliasing({c10::SchemaArgType::output, 0}));
  ASSERT_TRUE(schema.is_aliasing({c10::SchemaArgType::output, 1}));
  ASSERT_TRUE(schema.is_aliasing({c10::SchemaArgType::input, 0}));
  ASSERT_TRUE(schema.is_aliasing({c10::SchemaArgType::input, 1}));
  ASSERT_FALSE(schema.is_aliasing({c10::SchemaArgType::input, 2}));
}

TEST(FunctionSchemaIsAliasingTest, InvalidArgument) {
  c10::FunctionSchema schema = torch::jit::parseSchema(
      "aten::sub_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> "
      "(Tensor(a!))");
  test::utils::ExpectThrowContains<std::exception>(
      [&]() {
        (void)schema.is_aliasing({c10::SchemaArgType::input, 4});
      },
      "Schema input index 4 is out of bounds",
      "input index out of bounds");
  test::utils::ExpectThrowContains<std::exception>(
      [&]() {
        (void)schema.is_aliasing({c10::SchemaArgType::output, 4});
      },
      "Schema output index 4 is out of bounds",
      "output index out of bounds");
}

TEST(FunctionSchemaIsMutableTest, Basic) {
  c10::FunctionSchema schema = torch::jit::parseSchema(
      "aten::sub_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> "
      "(Tensor(a!))");
  ASSERT_TRUE(schema.is_mutable({c10::SchemaArgType::output, 0}));
  ASSERT_TRUE(schema.is_mutable({c10::SchemaArgType::input, 0}));
  ASSERT_TRUE(schema.is_mutable("self"));
  ASSERT_FALSE(schema.is_mutable({c10::SchemaArgType::input, 1}));
  ASSERT_FALSE(schema.is_mutable("other"));
  ASSERT_FALSE(schema.is_mutable({c10::SchemaArgType::input, 2}));
  ASSERT_FALSE(schema.is_mutable("alpha"));
}

TEST(FunctionSchemaIsMutableTest, InvalidArgument) {
  c10::FunctionSchema schema = torch::jit::parseSchema(
      "aten::sub_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> "
      "(Tensor(a!))");
  test::utils::ExpectThrowContains<std::exception>(
      [&]() {
        (void)schema.is_mutable({c10::SchemaArgType::input, 4});
      },
      "Schema input index 4 is out of bounds",
      "mutable input index out of bounds");
  test::utils::ExpectThrowContains<std::exception>(
      [&]() {
        (void)schema.is_mutable({c10::SchemaArgType::output, 4});
      },
      "Schema output index 4 is out of bounds",
      "mutable output index out of bounds");
  test::utils::ExpectThrowContains<std::exception>(
      [&]() { (void)schema.is_mutable("named_argument"); },
      "Tried to test mutability of nonexistent name named_argument",
      "mutable name not found");
}

// 不支持 SchemaInfo, see:
// https://github.com/pytorch/pytorch/blob/main/torch/csrc/utils/schema_info.h
// TEST(SchemaInfoIsMutableTest, Basic) {
//   SchemaInfo schema(
//       "aten::sub_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) ->
//       (Tensor(a!))");
//   ASSERT_TRUE(schema.is_mutable({SchemaArgType::input, 0}));
//   ASSERT_TRUE(schema.is_mutable("self"));
//   ASSERT_FALSE(schema.is_mutable({SchemaArgType::input, 1}));
//   ASSERT_FALSE(schema.is_mutable("other"));
//   ASSERT_FALSE(schema.is_mutable({SchemaArgType::input, 2}));
//   ASSERT_FALSE(schema.is_mutable("alpha"));
// }

// 不支持 SchemaInfo
// TEST(SchemaInfoIsMutableTest, InvalidArgument) {
//   SchemaInfo schema(
//       "aten::sub_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) ->
//       (Tensor(a!))");
//   ASSERT_THROW(schema.is_mutable({SchemaArgType::input, 4}), c10::Error);
//   ASSERT_THROW(schema.is_mutable("named_argument"), c10::Error);
// }

// 不支持 SchemaInfo
// TEST(SchemaInfoIsMutableTest, AliasingInputs) {
//   SchemaInfo schema(
//       "aten::test.Tensor(Tensor(a!) self, Tensor(b) other, *, Scalar alpha=1)
//       -> (Tensor(a!), Tensor(b))");
//   ASSERT_TRUE(schema.is_mutable({SchemaArgType::input, 0}));
//   ASSERT_TRUE(schema.is_mutable({SchemaArgType::output, 0}));
//   ASSERT_TRUE(schema.is_mutable("self"));
//   ASSERT_FALSE(schema.is_mutable({SchemaArgType::input, 1}));
//   ASSERT_FALSE(schema.is_mutable({SchemaArgType::output, 1}));
//   ASSERT_FALSE(schema.is_mutable("other"));
//   at::Tensor input = at::randn({3, 3});
//   schema.addArgumentValue("self", input);
//   schema.addArgumentValue("other", input);
//   ASSERT_TRUE(schema.is_mutable({SchemaArgType::input, 1}));
//   ASSERT_TRUE(schema.is_mutable({SchemaArgType::output, 1}));
//   ASSERT_TRUE(schema.is_mutable("other"));
// }

// 不支持 SchemaInfo
// TEST(SchemaInfoIsMutableTest, InstanceNorm) {
//   SchemaInfo schema_info(
//       "aten::instance_norm(Tensor input, Tensor? weight, Tensor? bias,
//       Tensor? running_mean, Tensor? running_var, bool use_input_stats, float
//       momentum, float eps, bool cudnn_enabled) -> Tensor");
//   ASSERT_TRUE(schema_info.is_mutable("running_mean"));
//   ASSERT_TRUE(schema_info.is_mutable("running_var"));
//   schema_info.addArgumentValue("use_input_stats", false);
//   ASSERT_FALSE(schema_info.is_mutable("running_mean"));
//   ASSERT_FALSE(schema_info.is_mutable("running_var"));
// }

// 不支持 SchemaInfo
// TEST(SchemaInfoIsMutableTest, BatchNorm) {
//   SchemaInfo schema_info(
//       "aten::batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor?
//       running_mean, Tensor? running_var, bool training, float momentum, float
//       eps, bool cudnn_enabled) -> Tensor");
//   ASSERT_TRUE(schema_info.is_mutable("running_mean"));
//   ASSERT_TRUE(schema_info.is_mutable("running_var"));
//   schema_info.addArgumentValue("training", false);
//   ASSERT_FALSE(schema_info.is_mutable("running_mean"));
//   ASSERT_FALSE(schema_info.is_mutable("running_var"));
// }

// 不支持 SchemaInfo
// TEST(SchemaInfoIsNonDeterministicTest, Basic) {
//   SchemaInfo deterministic_schema_info(
//       "aten::sub_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) ->
//       (Tensor(a!))");
//   SchemaInfo nondeterministic_schema_info(
//       "aten::bernoulli(Tensor self, *, Generator? generator) -> Tensor");
//   ASSERT_FALSE(deterministic_schema_info.is_nondeterministic());
//   ASSERT_TRUE(nondeterministic_schema_info.is_nondeterministic());
// }

// 不支持 SchemaInfo
// TEST(SchemaInfoIsNonDeterministicTest, Dropout) {
//   SchemaInfo droupout_schema_info(
//       "aten::dropout(Tensor input, float p, bool train) -> Tensor");
//   ASSERT_TRUE(droupout_schema_info.is_nondeterministic());
//   droupout_schema_info.addArgumentValue("train", false);
//   ASSERT_FALSE(droupout_schema_info.is_nondeterministic());
// }

TEST(FunctionSchemaMayAliasTest, Basic) {
  c10::FunctionSchema schema = torch::jit::parseSchema(
      "aten::sub_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> "
      "(Tensor(a!))");
  ASSERT_TRUE(schema.may_alias({c10::SchemaArgType::input, 0},
                               {c10::SchemaArgType::output, 0}));
  ASSERT_FALSE(schema.may_alias({c10::SchemaArgType::input, 1},
                                {c10::SchemaArgType::output, 0}));
  ASSERT_FALSE(schema.may_alias({c10::SchemaArgType::input, 1},
                                {c10::SchemaArgType::input, 0}));
}

TEST(FunctionSchemaMayAliasTest, InvalidArgument) {
  c10::FunctionSchema schema = torch::jit::parseSchema(
      "aten::sub_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> "
      "(Tensor(a!))");
  test::utils::ExpectThrowContains<std::exception>(
      [&]() {
        (void)schema.may_alias({c10::SchemaArgType::input, 15},
                               {c10::SchemaArgType::output, 0});
      },
      "Schema input index 15 is out of bounds",
      "may_alias input index out of bounds");
  test::utils::ExpectThrowContains<std::exception>(
      [&]() {
        (void)schema.may_alias({c10::SchemaArgType::input, 0},
                               {c10::SchemaArgType::output, 15});
      },
      "Schema output index 15 is out of bounds",
      "may_alias output index out of bounds");
}

TEST(FunctionSchemaMayAliasTest, Wildcard) {
  c10::FunctionSchema schema = torch::jit::parseSchema(
      "aten::test.Tensor(Tensor(*) self) -> (Tensor(*), Tensor)");
  ASSERT_TRUE(schema.may_alias({c10::SchemaArgType::output, 0},
                               {c10::SchemaArgType::input, 0}));
  ASSERT_FALSE(schema.may_alias({c10::SchemaArgType::output, 1},
                                {c10::SchemaArgType::input, 0}));
}

// 不支持 SchemaInfo
// TEST(SchemaInfoMayAliasTest, AliasingInputs) {
//   SchemaInfo schema(
//       "aten::sub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) ->
//       Tensor");
//   ASSERT_FALSE(
//       schema.may_alias({SchemaArgType::input, 0}, {SchemaArgType::input,
//       1}));
//   at::Tensor input = at::randn({3, 3});
//   schema.addArgumentValue("self", input);
//   schema.addArgumentValue("other", input);
//   ASSERT_TRUE(
//       schema.may_alias({SchemaArgType::input, 0}, {SchemaArgType::input,
//       1}));
// }

// 不支持 SchemaInfo
// TEST(SchemaInfoMayAliasTest, AliasingOutputs) {
//   SchemaInfo schema(
//       "aten::aminmax.out(Tensor self, *, int? dim=None, bool keepdim=False,
//       Tensor(a!) min, Tensor(b!) max) -> (Tensor(a!) min, Tensor(b!) max)");
//   ASSERT_FALSE(
//       schema.may_alias({SchemaArgType::output, 0}, {SchemaArgType::output,
//       1}));
//   at::Tensor input = at::randn({3, 3});
//   schema.addArgumentValue("min", input);
//   schema.addArgumentValue("max", input);
//   ASSERT_TRUE(
//       schema.may_alias({SchemaArgType::output, 0}, {SchemaArgType::output,
//       1}));
// }

// 不支持 SchemaInfo
// TEST(SchemaInfoMayAliasTest, AliasingInputOutput) {
//   SchemaInfo schema(
//       "aten::sub_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) ->
//       (Tensor(a!))");
//   ASSERT_TRUE(
//       schema.may_alias({SchemaArgType::input, 0}, {SchemaArgType::output,
//       0}));
//   ASSERT_FALSE(
//       schema.may_alias({SchemaArgType::input, 1}, {SchemaArgType::output,
//       0}));
//   at::Tensor input = at::randn({3, 3});
//   schema.addArgumentValue("self", input);
//   schema.addArgumentValue("other", input);
//   ASSERT_TRUE(
//       schema.may_alias({SchemaArgType::input, 0}, {SchemaArgType::output,
//       0}));
//   ASSERT_TRUE(
//       schema.may_alias({SchemaArgType::input, 1}, {SchemaArgType::output,
//       0}));
// }

// 不支持 SchemaInfo
// TEST(SchemaInfoMayAliasTest, MultipleWildcardInputs) {
//   SchemaInfo schema(
//       "aten::test.Tensor(Tensor(a) a, Tensor(*) b, Tensor(*) c) ->
//       (Tensor(a), Tensor(*))");
//   ASSERT_TRUE(
//       schema.may_alias({SchemaArgType::input, 0}, {SchemaArgType::output,
//       0}));
//   ASSERT_TRUE(
//       schema.may_alias({SchemaArgType::input, 1}, {SchemaArgType::output,
//       1}));
//   ASSERT_TRUE(
//       schema.may_alias({SchemaArgType::input, 2}, {SchemaArgType::output,
//       1}));
//   ASSERT_FALSE(
//       schema.may_alias({SchemaArgType::input, 0}, {SchemaArgType::input,
//       1}));
//   ASSERT_FALSE(
//       schema.may_alias({SchemaArgType::input, 0}, {SchemaArgType::input,
//       2}));
//   ASSERT_FALSE(
//       schema.may_alias({SchemaArgType::input, 0}, {SchemaArgType::output,
//       1}));
//   ASSERT_FALSE(
//       schema.may_alias({SchemaArgType::input, 1}, {SchemaArgType::output,
//       0}));
//   at::Tensor input = at::randn({3, 3});
//   schema.addArgumentValue("a", input);
//   schema.addArgumentValue("b", input);
//   ASSERT_TRUE(
//       schema.may_alias({SchemaArgType::input, 0}, {SchemaArgType::output,
//       0}));
//   ASSERT_TRUE(
//       schema.may_alias({SchemaArgType::input, 1}, {SchemaArgType::output,
//       1}));
//   ASSERT_TRUE(
//       schema.may_alias({SchemaArgType::input, 2}, {SchemaArgType::output,
//       1}));
//   ASSERT_TRUE(
//       schema.may_alias({SchemaArgType::input, 0}, {SchemaArgType::input,
//       1}));
//   ASSERT_TRUE(
//       schema.may_alias({SchemaArgType::input, 0}, {SchemaArgType::input,
//       2}));
//   ASSERT_TRUE(
//       schema.may_alias({SchemaArgType::input, 0}, {SchemaArgType::output,
//       1}));
//   ASSERT_TRUE(
//       schema.may_alias({SchemaArgType::input, 1}, {SchemaArgType::output,
//       0}));
// }

// 不支持 SchemaInfo
// TEST(SchemaInfoMayAliasTest, MultipleNonWildcardInputs) {
//   SchemaInfo schema(
//       "aten::test.Tensor(Tensor(a) a, Tensor(a) b, Tensor(*) c, Tensor(b) d)
//       -> (Tensor(a), Tensor(*))");
//   ASSERT_TRUE(
//       schema.may_alias({SchemaArgType::input, 0}, {SchemaArgType::input,
//       1}));
//   ASSERT_TRUE(
//       schema.may_alias({SchemaArgType::input, 0}, {SchemaArgType::input,
//       2}));
//   ASSERT_TRUE(
//       schema.may_alias({SchemaArgType::input, 2}, {SchemaArgType::input,
//       1}));
//   ASSERT_TRUE(
//       schema.may_alias({SchemaArgType::input, 2}, {SchemaArgType::output,
//       0}));
// }

// 不支持 SchemaInfo
// TEST(SchemaInfoMayAliasTest, MultipleNonWildcardOutputs) {
//   SchemaInfo schema(
//       "aten::test.Tensor(Tensor(a) a, Tensor(*) b) -> (Tensor(a),
//       Tensor(a))");
//   ASSERT_TRUE(
//       schema.may_alias({SchemaArgType::input, 0}, {SchemaArgType::input,
//       1}));
//   ASSERT_TRUE(
//       schema.may_alias({SchemaArgType::output, 0}, {SchemaArgType::output,
//       1}));
//   ASSERT_TRUE(
//       schema.may_alias({SchemaArgType::output, 0}, {SchemaArgType::input,
//       1}));
// }

// 不支持 SchemaInfo
// TEST(SchemaInfoMayAliasTest, MismatchingTypes) {
//   SchemaInfo schema("aten::test.Tensor(Tensor(a) a) -> int(a)");
//   ASSERT_FALSE(
//       schema.may_alias({SchemaArgType::input, 0}, {SchemaArgType::output,
//       0}));
// }

TEST(FunctionSchemaMayContainAliasTest, Basic) {
  c10::FunctionSchema schema = torch::jit::parseSchema(
      "aten::sub_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> "
      "(Tensor(a!))");
  ASSERT_TRUE(schema.may_contain_alias({c10::SchemaArgType::input, 0},
                                       {c10::SchemaArgType::output, 0}));
  ASSERT_FALSE(schema.may_contain_alias({c10::SchemaArgType::input, 1},
                                        {c10::SchemaArgType::output, 0}));
  ASSERT_FALSE(schema.may_contain_alias({c10::SchemaArgType::input, 1},
                                        {c10::SchemaArgType::input, 0}));
}

TEST(FunctionSchemaMayContainAliasTest, Wildcard) {
  c10::FunctionSchema schema = torch::jit::parseSchema(
      "aten::test.Tensor(Tensor(*) self) -> (Tensor[], Tensor)");
  ASSERT_FALSE(schema.may_alias({c10::SchemaArgType::output, 0},
                                {c10::SchemaArgType::input, 0}));
  ASSERT_TRUE(schema.may_contain_alias({c10::SchemaArgType::output, 0},
                                       {c10::SchemaArgType::input, 0}));
  ASSERT_TRUE(schema.may_contain_alias(
      {c10::SchemaArgType::output, 0}, {c10::SchemaArgType::input, 0}, false));
  ASSERT_FALSE(schema.may_contain_alias(
      {c10::SchemaArgType::input, 0}, {c10::SchemaArgType::output, 0}, false));
  ASSERT_FALSE(schema.may_alias({c10::SchemaArgType::output, 1},
                                {c10::SchemaArgType::input, 0}));
}

TEST(FunctionSchemaMayContainAliasTest, InputAndOutputContainers) {
  c10::FunctionSchema schema =
      torch::jit::parseSchema("aten::test.Tensor(Tensor[] self) -> Tensor[]");
  ASSERT_FALSE(schema.may_alias({c10::SchemaArgType::output, 0},
                                {c10::SchemaArgType::input, 0}));
  ASSERT_TRUE(schema.may_contain_alias({c10::SchemaArgType::output, 0},
                                       {c10::SchemaArgType::input, 0}));
  ASSERT_TRUE(schema.may_contain_alias(
      {c10::SchemaArgType::output, 0}, {c10::SchemaArgType::input, 0}, false));
  ASSERT_TRUE(schema.may_contain_alias(
      {c10::SchemaArgType::input, 0}, {c10::SchemaArgType::output, 0}, false));
}

// 不支持 SchemaInfo
// TEST(SchemaInfoMayContainAliasTest, ContainAliasInputsEqual) {
//   SchemaInfo schema(
//       "aten::sub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) ->
//       Tensor");
//   ASSERT_FALSE(schema.may_contain_alias(
//       {SchemaArgType::input, 0}, {SchemaArgType::input, 1}));
//   at::Tensor input = at::randn({3, 3});
//   schema.addArgumentValue("self", input);
//   schema.addArgumentValue("other", input);
//   ASSERT_TRUE(schema.may_contain_alias(
//       {SchemaArgType::input, 0}, {SchemaArgType::input, 1}));
//   ASSERT_TRUE(schema.may_contain_alias(
//       {SchemaArgType::input, 0}, {SchemaArgType::input, 1}, false));
//   ASSERT_TRUE(schema.may_contain_alias(
//       {SchemaArgType::input, 1}, {SchemaArgType::input, 0}, false));
// }

// 不支持 SchemaInfo
// TEST(SchemaInfoMayContainAliasTest, ContainAliasInputsContained) {
//   SchemaInfo schema(
//       "aten::test.Tensor(Tensor[] self, Tensor other, *, Scalar alpha=1) ->
//       Tensor");
//   ASSERT_FALSE(schema.may_contain_alias(
//       {SchemaArgType::input, 0}, {SchemaArgType::input, 1}));
//   at::Tensor input = at::randn({3, 3});
//   schema.addArgumentValue("self", c10::List<at::Tensor>({input}));
//   schema.addArgumentValue("other", input);
//   ASSERT_TRUE(schema.may_contain_alias(
//       {SchemaArgType::input, 0}, {SchemaArgType::input, 1}));
//   ASSERT_TRUE(schema.may_contain_alias(
//       {SchemaArgType::input, 0}, {SchemaArgType::input, 1}, false));
//   ASSERT_FALSE(schema.may_contain_alias(
//       {SchemaArgType::input, 1}, {SchemaArgType::input, 0}, false));
// }

// 不支持 SchemaInfo
// TEST(SchemaInfoMayContainAliasTest, ContainAliasOutputs) {
//   SchemaInfo schema(
//       "aten::aminmax.out(Tensor self, *, int? dim=None, bool keepdim=False,
//       Tensor(a!) min, Tensor(b!) max) -> (Tensor(a!) min, Tensor(b!) max)");
//   ASSERT_FALSE(schema.may_contain_alias(
//       {SchemaArgType::output, 0}, {SchemaArgType::output, 1}));
//   at::Tensor input = at::randn({3, 3});
//   schema.addArgumentValue("min", input);
//   schema.addArgumentValue("max", input);
//   ASSERT_TRUE(schema.may_contain_alias(
//       {SchemaArgType::output, 0}, {SchemaArgType::output, 1}));
// }

// 不支持 SchemaInfo
// TEST(SchemaInfoMayContainAliasTest, ContainAliasInputOutput) {
//   SchemaInfo schema(
//       "aten::test.tensor(Tensor(a) self, Tensor[] other) -> Tensor(a)");
//   ASSERT_FALSE(schema.may_contain_alias(
//       {SchemaArgType::output, 0}, {SchemaArgType::input, 1}));
//   at::Tensor input = at::randn({3, 3});
//   schema.addArgumentValue("other", c10::List<at::Tensor>({input}));
//   schema.addArgumentValue("self", input);
//   ASSERT_TRUE(schema.may_contain_alias(
//       {SchemaArgType::output, 0}, {SchemaArgType::input, 1}));
//   ASSERT_FALSE(schema.may_contain_alias(
//       {SchemaArgType::output, 0}, {SchemaArgType::input, 1}, false));
//   ASSERT_TRUE(schema.may_contain_alias(
//       {SchemaArgType::input, 1}, {SchemaArgType::output, 0}, false));
// }

// 不支持 SchemaInfo
// TEST(SchemaInfoMayContainAliasTest, InputAndOutputContainers) {
//   SchemaInfo schema(
//       "aten::test.tensor(Tensor self, Tensor[] other) -> Tensor[]");
//   ASSERT_TRUE(schema.may_contain_alias(
//       {SchemaArgType::output, 0}, {SchemaArgType::input, 1}));
//   ASSERT_FALSE(schema.may_contain_alias(
//       {SchemaArgType::output, 0}, {SchemaArgType::input, 0}));
//   ASSERT_FALSE(schema.may_contain_alias(
//       {SchemaArgType::input, 0}, {SchemaArgType::input, 1}));
//   at::Tensor input = at::randn({3, 3});
//   schema.addArgumentValue("other", c10::List<at::Tensor>({input}));
//   schema.addArgumentValue("self", input);
//   ASSERT_TRUE(schema.may_contain_alias(
//       {SchemaArgType::output, 0}, {SchemaArgType::input, 1}));
//   ASSERT_TRUE(schema.may_contain_alias(
//       {SchemaArgType::output, 0}, {SchemaArgType::input, 0}));
//   ASSERT_TRUE(schema.may_contain_alias(
//       {SchemaArgType::input, 0}, {SchemaArgType::input, 1}));
// }

// 不支持 SchemaInfo
// TEST(SchemaInfoMayContainAliasTest, Wildcard) {
//   SchemaInfo schema(
//       "aten::test.tensor(Tensor a, Tensor[] b, Tensor(*) c) -> Tensor[]");
//   ASSERT_FALSE(schema.may_contain_alias(
//       {SchemaArgType::input, 0}, {SchemaArgType::input, 2}));
//   ASSERT_FALSE(schema.may_contain_alias(
//       {SchemaArgType::input, 0}, {SchemaArgType::input, 1}));
//   ASSERT_TRUE(schema.may_contain_alias(
//       {SchemaArgType::input, 2}, {SchemaArgType::input, 1}));
//   at::Tensor input = at::randn({3, 3});
//   schema.addArgumentValue("b", c10::List<at::Tensor>({input}));
//   schema.addArgumentValue("a", input);
//   ASSERT_TRUE(schema.may_contain_alias(
//       {SchemaArgType::input, 0}, {SchemaArgType::input, 2}));
//   ASSERT_TRUE(schema.may_contain_alias(
//       {SchemaArgType::input, 0}, {SchemaArgType::input, 1}));
//   ASSERT_TRUE(schema.may_contain_alias(
//       {SchemaArgType::input, 2}, {SchemaArgType::input, 1}));
// }

// -----  test/cpp/jit/test_schema_info.cpp end -----

// -----  test/cpp/jit/test_alias_analysis.cpp start -----

// 暂不支持 Dispatcher, 并且暂时也没有 nondeterministic_seeded 相关标记
// TEST(NonDeterminismBackwardsCompatibility, BackwardsCompatibility) {
//   static const std::vector<std::string> nondeterministic_ops = {
//       "aten::dropout(Tensor input, float p, bool train) -> Tensor",
//       "aten::_fused_dropout(Tensor self, float p, Generator? generator) ->
//       (Tensor, Tensor)", "aten::_standard_gamma(Tensor self, Generator?
//       generator) -> Tensor", "aten::bernoulli(Tensor self, *, Generator?
//       generator) -> Tensor", "aten::bernoulli(Tensor self, float p, *,
//       Generator? generator) -> Tensor", "aten::multinomial(Tensor self, int
//       num_samples, bool replacement, *, Generator? generator) -> Tensor",
//       "aten::native_dropout(Tensor input, float p, bool? train) -> (Tensor,
//       Tensor)", "aten::normal.Tensor_Tensor(Tensor mean, Tensor std, *,
//       Generator? generator) -> Tensor", "aten::normal.float_Tensor(float
//       mean, Tensor std, *, Generator? generator) -> Tensor",
//       "aten::normal.Tensor_float(Tensor mean, float std, *, Generator?
//       generator) -> Tensor", "aten::poisson(Tensor self, Generator?
//       generator) -> Tensor", "aten::binomial(Tensor count, Tensor prob,
//       Generator? generator=None) -> Tensor", "aten::rrelu(Tensor self, Scalar
//       lower, Scalar upper, bool training, Generator? generator) -> Tensor",
//       "aten::rrelu_with_noise(Tensor self, Tensor noise, Scalar lower, Scalar
//       upper, bool training, Generator? generator) -> Tensor",
//       "aten::rand(int[] size, *, int? dtype, int? layout, Device? device,
//       bool? pin_memory) -> Tensor", "aten::rand_like(Tensor self, *, int?
//       dtype=None, int? layout=None, Device? device=None, bool?
//       pin_memory=None, MemoryFormat? memory_format=None) -> Tensor",
//       "aten::rand_like.generator(Tensor self, *, Generator? generator, int?
//       dtype=None, int? layout=None, Device? device=None, bool?
//       pin_memory=None, MemoryFormat? memory_format=None) -> Tensor",
//       "aten::randint(int high, int[] size, *, int? dtype, int? layout,
//       Device? device, bool? pin_memory) -> Tensor", "aten::randint(int low,
//       int high, int[] size, *, int? dtype, int? layout, Device? device, bool?
//       pin_memory) -> Tensor", "aten::randint_like(Tensor self, int high, *,
//       int? dtype=None, int? layout=None, Device? device=None, bool?
//       pin_memory=None, MemoryFormat? memory_format=None) -> Tensor",
//       "aten::randint_like.generator(Tensor self, int high, *, Generator?
//       generator, int? dtype=None, int? layout=None, Device? device=None,
//       bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor",
//       "aten::randint_like.Tensor(Tensor self, Tensor high, *, int?
//       dtype=None, int? layout=None, Device? device=None, bool?
//       pin_memory=None, MemoryFormat? memory_format=None) -> Tensor",
//       "aten::randint_like.Tensor_generator(Tensor self, Tensor high, *,
//       Generator? generator, int? dtype=None, int? layout=None, Device?
//       device=None, bool? pin_memory=None, MemoryFormat? memory_format=None)
//       -> Tensor", "aten::randint_like.low_dtype(Tensor self, int low, int
//       high, *, int? dtype=None, int? layout=None, Device? device=None, bool?
//       pin_memory=None, MemoryFormat? memory_format=None) -> Tensor",
//       "aten::randint_like.low_generator_dtype(Tensor self, int low, int high,
//       *, Generator? generator, int? dtype=None, int? layout=None, Device?
//       device=None, bool? pin_memory=None, MemoryFormat? memory_format=None)
//       -> Tensor", "aten::randn(int[] size, *, int? dtype, int? layout,
//       Device? device, bool? pin_memory) -> Tensor", "aten::randn_like(Tensor
//       self, *, int? dtype=None, int? layout=None, Device? device=None, bool?
//       pin_memory=None, MemoryFormat? memory_format=None) -> Tensor",
//       "aten::randn_like.generator(Tensor self, *, Generator? generator, int?
//       dtype=None, int? layout=None, Device? device=None, bool?
//       pin_memory=None, MemoryFormat? memory_format=None) -> Tensor",
//       "aten::randperm(int n, *, int? dtype, int? layout, Device? device,
//       bool? pin_memory) -> Tensor"};
//   for (const std::string& op : nondeterministic_ops) {
//     const c10::FunctionSchema& schema = torch::jit::parseSchema(op);
//     const auto& op_handle = c10::Dispatcher::singleton().findOp(
//         c10::OperatorName(schema.name(), schema.overload_name()));
//     ASSERT_TRUE(op_handle->hasTag(at::Tag::nondeterministic_seeded));
//   }
// }

// -----  test/cpp/jit/test_alias_analysis.cpp end -----

// -----  aten/src/ATen/core/boxing/impl/kernel_function_test.cpp start -----

// 不支持从 cpp func 反向推导 schema
// TEST(OperatorRegistrationTestFunctionBasedKernel,
// givenKernel_whenRegisteredWithoutSpecifyingSchema_thenInfersSchema) {
//   auto registrar = RegisterOperators()
//       .op("_test::no_schema_specified",
//       RegisterOperators::options().catchAllKernel<decltype(kernelForSchemaInference),
//       &kernelForSchemaInference>());

//   auto op =
//   c10::Dispatcher::singleton().findSchema({"_test::no_schema_specified",
//   ""}); ASSERT_TRUE(op.has_value());

//   std::optional<std::string> differences =
//   c10::findSchemaDifferences(torch::jit::parseSchema("_test::no_schema_specified(Tensor
//   arg1, int arg2, Tensor[] arg3) -> (int, Tensor)"), op->schema());
//   EXPECT_FALSE(differences.has_value());
// }

// -----  aten/src/ATen/core/boxing/impl/kernel_function_test.cpp end -----

// -----  aten/src/ATen/core/boxing/impl/kernel_lambda_test.cpp start -----
// 不支持从 cpp func 反向推导 schema
// TEST(OperatorRegistrationTestLambdaBasedKernel,
// givenKernel_whenRegisteredWithoutSpecifyingSchema_thenInfersSchema) {
//   auto registrar = RegisterOperators()
//       .op("_test::no_schema_specified",
//       RegisterOperators::options().catchAllKernel([] (Tensor arg1, int64_t
//       arg2, const c10::List<Tensor>& arg3) -> std::tuple<int64_t, Tensor>
//       {return {};}));

//   auto op =
//   c10::Dispatcher::singleton().findSchema({"_test::no_schema_specified",
//   ""}); ASSERT_TRUE(op.has_value());

//   std::optional<std::string> differences =
//   c10::findSchemaDifferences(torch::jit::parseSchema("_test::no_schema_specified(Tensor
//   arg1, int arg2, Tensor[] arg3) -> (int, Tensor)"), op->schema());
//   EXPECT_FALSE(differences.has_value());
// }
// -----  aten/src/ATen/core/boxing/impl/kernel_lambda_test.cpp end -----

// ----- aten/src/ATen/core/boxing/impl/kernel_function_legacy_test.cpp start
// ----- 不支持从 cpp func 反向推导 schema
// TEST(OperatorRegistrationTestLegacyFunctionBasedKernel,
// givenKernel_whenRegisteredWithoutSpecifyingSchema_thenInfersSchema) {
//   auto registrar = RegisterOperators()
//       .op("_test::no_schema_specified", &kernelForSchemaInference);

//   auto op =
//   c10::Dispatcher::singleton().findSchema({"_test::no_schema_specified",
//   ""}); ASSERT_TRUE(op.has_value());

//   std::optional<std::string> differences =
//   c10::findSchemaDifferences(torch::jit::parseSchema("_test::no_schema_specified(Tensor
//   arg1, int arg2, Tensor[] arg3) -> (int, Tensor)"), op->schema());
//   EXPECT_FALSE(differences.has_value());
// }
// ----- aten/src/ATen/core/boxing/impl/kernel_function_legacy_test.cpp end
// -----

// ----- aten/src/ATen/core/boxing/impl/kernel_lambda_legacy_test.cpp start
// ----- 不支持从 cpp func 反向推导 schema
// TEST(OperatorRegistrationTestLegacyLambdaBasedKernel,
// givenKernel_whenRegisteredWithoutSpecifyingSchema_thenInfersSchema) {
//   auto registrar = RegisterOperators()
//       .op("_test::no_schema_specified", [] (Tensor arg1, int64_t arg2, const
//       std::vector<Tensor>& arg3) -> std::tuple<int64_t, Tensor> {return
//       {};});

//   auto op =
//   c10::Dispatcher::singleton().findSchema({"_test::no_schema_specified",
//   ""}); ASSERT_TRUE(op.has_value());

//   std::optional<std::string> differences =
//   c10::findSchemaDifferences(torch::jit::parseSchema("_test::no_schema_specified(Tensor
//   arg1, int arg2, Tensor[] arg3) -> (int, Tensor)"), op->schema());
//   EXPECT_FALSE(differences.has_value());
// }
// ----- aten/src/ATen/core/boxing/impl/kernel_lambda_legacy_test.cpp end -----

// ----- aten/src/ATen/core/boxing/impl/make_boxed_from_unboxed_functor_test.cpp
// start ----- 不支持从 cpp func 反向推导 schema
// TEST(OperatorRegistrationTestFunctorBasedKernel,
// givenKernel_whenRegisteredWithoutSpecifyingSchema_thenInfersSchema) {
//   auto registrar = RegisterOperators()
//       .op("_test::no_schema_specified",
//       RegisterOperators::options().kernel<KernelForSchemaInference>(DispatchKey::CPU));

//   auto op =
//   c10::Dispatcher::singleton().findSchema({"_test::no_schema_specified",
//   ""}); ASSERT_TRUE(op.has_value());

//   std::optional<std::string> differences =
//   c10::findSchemaDifferences(torch::jit::parseSchema("_test::no_schema_specified(Tensor
//   arg1, int arg2, Tensor[] arg3) -> (int, Tensor)"), op->schema());
//   EXPECT_FALSE(differences.has_value());
// }

// 不支持从 cpp func 反向推导 schema
// TEST(OperatorRegistrationTestFunctorBasedKernel,
// givenKernel_whenRegisteredCatchAllWithoutSpecifyingSchema_thenInfersSchema) {
//   auto registrar = RegisterOperators()
//       .op("_test::no_schema_specified",
//       RegisterOperators::options().catchAllKernel<KernelForSchemaInference>());

//   auto op =
//   c10::Dispatcher::singleton().findSchema({"_test::no_schema_specified",
//   ""}); ASSERT_TRUE(op.has_value());

//   std::optional<std::string> differences =
//   c10::findSchemaDifferences(torch::jit::parseSchema("_test::no_schema_specified(Tensor
//   arg1, int arg2, Tensor[] arg3) -> (int, Tensor)"), op->schema());
//   EXPECT_FALSE(differences.has_value());
// }
// ----- aten/src/ATen/core/boxing/impl/make_boxed_from_unboxed_functor_test.cpp
// end -----

// ----- aten/src/ATen/core/op_registration/op_registration_test.cpp start -----

// 不支持 AliasAnalysisKind 选择
// TEST(NewOperatorRegistrationTest, schema) {
//   auto m = MAKE_TORCH_LIBRARY(test);
//   m.def("def1(Tensor self) -> Tensor");
//   m.def(torch::schema("def2(Tensor self) -> Tensor"));
//   m.def(torch::schema("def3(Tensor self) -> Tensor",
//   c10::AliasAnalysisKind::PURE_FUNCTION));
//   m.def(torch::jit::parseSchema("def4(Tensor self) -> Tensor"));

//   ASSERT_TRUE(Dispatcher::singleton().findSchema({"test::def1",
//   ""}).has_value());
//   ASSERT_TRUE(Dispatcher::singleton().findSchema({"test::def2",
//   ""}).has_value());
//   ASSERT_TRUE(Dispatcher::singleton().findSchema({"test::def3",
//   ""}).has_value());
//   ASSERT_TRUE(Dispatcher::singleton().findSchema({"test::def4",
//   ""}).has_value());

//   EXPECT_EQ(Dispatcher::singleton().findSchema({"test::def1",
//   ""})->schema().aliasAnalysis(), c10::AliasAnalysisKind::FROM_SCHEMA);
//   EXPECT_EQ(Dispatcher::singleton().findSchema({"test::def2",
//   ""})->schema().aliasAnalysis(), c10::AliasAnalysisKind::FROM_SCHEMA);
//   EXPECT_EQ(Dispatcher::singleton().findSchema({"test::def3",
//   ""})->schema().aliasAnalysis(), c10::AliasAnalysisKind::PURE_FUNCTION);
//   ASSERT_TRUE(Dispatcher::singleton().findSchema({"test::def4",
//   ""})->schema().isDefaultAliasAnalysisKind());
// }

// ----- aten/src/ATen/core/op_registration/op_registration_test.cpp end -----
