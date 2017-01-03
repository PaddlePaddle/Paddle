#include "PaddleCAPI.h"
#include "gtest/gtest.h"

TEST(CAPIMatrix, create) {
  PD_Matrix mat;
  ASSERT_EQ(PD_NO_ERROR, PDMatCreate(&mat, 128, 32, false));
  std::vector<pd_real> sampleRow;
  sampleRow.resize(32);
  for (size_t i = 0; i < sampleRow.size(); ++i) {
    sampleRow[i] = 1.0 / (i + 1.0);
  }
  ASSERT_EQ(PD_NO_ERROR, PDMatCopyToRow(mat, 0, sampleRow.data()));
  ASSERT_EQ(PD_OUT_OF_RANGE, PDMatCopyToRow(mat, 128, sampleRow.data()));

  pd_real* arrayPtr;

  ASSERT_EQ(PD_NO_ERROR, PDMatGetRow(mat, 0, &arrayPtr));
  for (size_t i = 0; i < sampleRow.size(); ++i) {
    ASSERT_NEAR(sampleRow[i], arrayPtr[i], 1e-5);
  }

  uint64_t height, width;
  ASSERT_EQ(PD_NO_ERROR, PDMatGetShape(mat, &height, &width));
  ASSERT_EQ(128, height);
  ASSERT_EQ(32, width);
  ASSERT_EQ(PD_NO_ERROR, PDMatDestroy(mat));
}

TEST(CAPIMatrix, createNone) {
  PD_Matrix mat;
  ASSERT_EQ(PD_NO_ERROR, PDMatCreateNone(&mat));
  ASSERT_EQ(PD_NO_ERROR, PDMatDestroy(mat));
}
