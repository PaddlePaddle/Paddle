#include "optimizer.h"
#include "gtest/gtest.h"

template <class T>
class Opitmizer_C_Test : public testing::Test {
private:
  Tensor<T> parameter;
  Tensor<T> gradient;
};

void applyGradientDescent_TEST() {}
