#include "Tensor.h"
#include "gtest/gtest.h"

TEST(Tensor, testDot) {
  int[] a = {1, 2, 3};
  int[] b = {1, 2, 3};
  int[] c = {2, 4, 6};
  size_t size = 3;
  paddle::Tensor<int> T1(&a, size);
  paddle::Tensor<int> T2(&b, size);
  auto T1 += T2;
  ASSERT_EQ(T1, T2);
}
