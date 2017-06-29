#include "attribute_reader.h"
#include <gtest/gtest.h>

TEST(AttributeReader, ReadPlain) {
  using paddle::framework::AttributeMap;
  AttributeMap test;
  test["floatValue"].set_f(0.23);
  test["strValue"].set_s("unittest string");
  test["intValue"].set_i(-1);

  using paddle::framework::AttributeReader;

  AttributeReader reader(test);

  ASSERT_TRUE(reader.ContainPlain<int>("intValue"));
  ASSERT_TRUE(reader.ContainPlain<std::string>("strValue"));
  ASSERT_TRUE(reader.ContainPlain<float>("floatValue"));

  ASSERT_EQ(-1, reader.Get<int>("intValue"));
  ASSERT_EQ("unittest string", reader.Get<std::string>("strValue"));
  ASSERT_NEAR(0.23f, reader.Get<float>("floatValue"), 1e-5);

  ASSERT_FALSE(reader.ContainPlain<float>("intValue"));
  ASSERT_FALSE(reader.ContainPlain<int>("otherValue"));
}

TEST(AttributeReader, ReadArray) {
  using paddle::framework::AttributeMap;
  AttributeMap test;
  auto ilist = test["listInt"].mutable_list()->mutable_ints();
  auto strlist = test["listStr"].mutable_list()->mutable_strings();
  constexpr int length = 4;
  ilist->Reserve(length);
  strlist->Reserve(length);
  std::vector<int> expected;
  std::vector<std::string> expectedStr;
  expected.reserve(length);
  for (int i = 0; i < length; ++i) {
    *ilist->Add() = i;
    *strlist->Add() = std::to_string(i);
    expected.push_back(i);
    expectedStr.push_back(std::to_string(i));
  }
  std::vector<int> actual;
  std::vector<std::string> actualStr;
  using paddle::framework::AttributeReader;
  AttributeReader reader(test);
  reader.GetArray("listInt", &actual);
  reader.GetArray("listStr", &actualStr);
  ASSERT_EQ(expected, actual);
  ASSERT_EQ(expectedStr, actualStr);
}