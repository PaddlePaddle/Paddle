#include <gtest/gtest.h>
#include <paddle/framework/attr_helper.h>
#include <random>
#include "attr_test.pb.h"

TEST(AttrHelper, plainTypes) {
  paddle::framework::AttributeTestMessage msg;
  std::random_device dev;
  unsigned int seed = dev();
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> distInt(-1000, 1000);
  std::uniform_real_distribution<float> distFloat(-1000.0f, 1000.0f);
  paddle::framework::AttributeWriter writer(msg.mutable_attrs());
  paddle::framework::AttributeReader reader(msg.attrs());
  for (size_t i = 0; i < 1000; ++i) {
    std::string key = "str_" + std::to_string(i);
    switch (i % 3) {
      case 0:
        ASSERT_TRUE(writer.Set<int>(key, distInt(rng)).isOK());
        break;
      case 1:
        ASSERT_TRUE(writer.Set<float>(key, distFloat(rng)).isOK());
        break;
      case 2:
        ASSERT_TRUE(writer
                        .Set<std::string>(
                            key, "random_str_" + std::to_string(distInt(rng)))
                        .isOK());
        break;
      default:
        ASSERT_TRUE(false);
    }
  }

  std::mt19937 rng2(seed);

  for (size_t i = 0; i < 1000; ++i) {
    std::string key = "str_" + std::to_string(i);
    int intVal;
    float floatVal;
    std::string strVal;

    switch (i % 3) {
      case 0:
        ASSERT_TRUE(reader.Get<int>(key, &intVal).isOK());
        ASSERT_EQ(distInt(rng2), intVal);
        break;
      case 1:
        ASSERT_TRUE(reader.Get<float>(key, &floatVal).isOK());
        ASSERT_EQ(distFloat(rng2), floatVal);
        break;
      case 2:
        ASSERT_TRUE(reader.Get<std::string>(key, &strVal).isOK());
        ASSERT_EQ("random_str_" + std::to_string(distInt(rng2)), strVal);
        break;
      default:
        ASSERT_TRUE(false);
    }
  }
}

template <typename Container>
inline void TestArrayImpl(const Container& container) {
  paddle::framework::AttributeTestMessage msg;
  paddle::framework::AttributeWriter writer(msg.mutable_attrs());
  paddle::framework::AttributeReader reader(msg.attrs());

  auto err =
      writer.SetArray<typename Container::value_type>("test_array", container);
  ASSERT_TRUE(err.isOK());

  Container tmp;
  err = reader.GetArray("test_array", &tmp);
  ASSERT_TRUE(err.isOK());
  ASSERT_EQ(container, tmp);
}

TEST(AttrHelper, array) {
  TestArrayImpl(std::vector<float>{0.7, 0.8, 0.9});
  TestArrayImpl(std::vector<int>{-1, -2, 1, 2, 0});
  TestArrayImpl(std::vector<std::string>{"a", "", "c", "e", "01304fksd"});
}