#include <gtest/gtest.h>
#include <paddle/platform/must_check.h>

int __must_check SomeFunctionMustCheck() { return 0; }

TEST(MustCheck, all) {
  //  This line should not be compiled, because the
  //  return value of SomeFunctionMustCheck marked as __must_check
  //  SomeFunctionMustCheck();
}