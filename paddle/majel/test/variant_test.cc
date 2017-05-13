#include "mapbox/variant.hpp"
#include "gtest/gtest.h"

TEST(Variant, Get) {
  typedef mapbox::util::variant<int, std::string> variant_type;
  variant_type str = "Hello world";
  // mapbox::util::get equal to boost::get
  ASSERT_EQ(mapbox::util::get<std::string>(str), "Hello world");
}

TEST(Variant, BadGet) {
  typedef mapbox::util::variant<int, std::string> variant_type;
  variant_type str;
  try {
    mapbox::util::get<std::string>(str);
  } catch (mapbox::util::bad_variant_access) {
    // mapbox::util::bad_variant_access equal to boost::bad_get
    str = "bad access";
  }
  ASSERT_EQ(mapbox::util::get<std::string>(str), "bad access");
}

struct Check : public mapbox::util::static_visitor<> {
  void operator()(int const& val) const { ASSERT_EQ(val, 0); }
  void operator()(std::string const& val) const {
    ASSERT_EQ(val, "hello world");
  }
};

TEST(Variant, Visitor) {
  typedef mapbox::util::variant<std::string, int, double> variant_type;
  // mapbox::util::apply_visitor equal to boost::apply_visitor
  variant_type v(0);
  mapbox::util::apply_visitor(Check(), v);
  v = "hello world";
  mapbox::util::apply_visitor(Check(), v);
}
