#include "libadder.h"
#include <iostream>
#include "gtest/gtest.h"

TEST(Cgo, Invoke) {
  EXPECT_EQ(GoAdder(30, 12), 42);
}
