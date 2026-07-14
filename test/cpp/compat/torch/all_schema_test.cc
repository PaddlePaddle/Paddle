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

#include <torch/all.h>
#include <torch/library.h>
#include <string>

#include "gtest/gtest.h"
#include "test/cpp/utils/exception_test_utils.h"

// -----  test/cpp/jit/test_misc.cpp start -----

namespace torch {
namespace jit {

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

TEST(SchemaParserTest, AnnotatedAliasSets) {
  // test tensor with annotated alias sets
  parseSchema("at::what(Tensor(a) foo) -> (Tensor(a))");
}

TEST(SchemaParserTest, AnnotatedAliasWithoutBeforeSet) {
  const std::string schema = "at::foo(Tensor(!) self) -> Tensor";
  test::utils::ExpectThrowContains<std::exception>(
      [&]() { (void)parseSchema(schema); },
      "Expected alias set",
      std::string("schema: ") + schema);
}

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

// -----  test/cpp/jit/test_schema_info.cpp end -----
