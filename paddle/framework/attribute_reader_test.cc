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