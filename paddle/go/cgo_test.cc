#include <iostream>
#include "gtest/gtest.h"
#include "libadder.h"

TEST(Cgo, Invoke) { EXPECT_EQ(GoAdder(30, 12), 42); }
