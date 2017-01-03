#include "PaddleCAPI.h"
#include "gtest/gtest.h"

TEST(CAPIVector, create) {
  PD_Vector tmp;
  ASSERT_EQ(PD_NO_ERROR, PDVecCreate(&tmp, 128, false));
  bool isSparse;
  ASSERT_EQ(PD_NO_ERROR, PDVecIsSparse(tmp, &isSparse));
  ASSERT_FALSE(isSparse);
  ASSERT_EQ(PD_NO_ERROR, PDVecDestroy(tmp));
}
