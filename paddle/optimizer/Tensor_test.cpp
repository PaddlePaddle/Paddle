#include <iostream>
#include "gtest/gtest.h"
#include "tensor.h"

using namespace paddle;
using namespace paddle::optimizer;

TEST(Tensor, indexer) {
  real* ptr = new real[3];
  Tensor t(ptr, 3);
  for (auto i = 0; i < t.size(); ++i) {
    t[i] = i;
  }
  ASSERT_EQ(t[2], 2);
  ASSERT_EQ(t[1], 1);
  delete ptr;
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
