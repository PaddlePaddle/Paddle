#include "third_party/protobuf_test/example_lib.h"

#include "gtest/gtest.h"

namespace third_party {
namespace protobuf_test {

TEST(ProtobufTest, GetGreet) {
  Greeting g;
  g.set_name("Paddle");
  EXPECT_EQ("Hello Paddle", get_greet(g));
}

}  // namespace protobuf_test
}  // namespace third_party
