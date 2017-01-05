#include "PaddleCAPI.h"
#include "gtest/gtest.h"

TEST(CAPIVector, create) {
  PD_IVector vec;
  ASSERT_EQ(kPD_NO_ERROR, PDIVecCreateNone(&vec));
  ASSERT_EQ(kPD_NO_ERROR, PDIVecDestroy(vec));
}
