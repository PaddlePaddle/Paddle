#include "PaddleCAPI.h"
#include "gtest/gtest.h"
#include "paddle/utils/ThreadLocal.h"

static std::vector<pd_real> randomBuffer(size_t bufSize) {
  auto& eng = paddle::ThreadLocalRandomEngine::get();
  std::uniform_real_distribution<pd_real> dist(-1.0, 1.0);
  std::vector<pd_real> retv;
  retv.reserve(bufSize);
  for (size_t i = 0; i < bufSize; ++i) {
    retv.push_back(dist(eng));
  }
  return retv;
}

TEST(CAPIArguments, create) {
  PD_Arguments args;
  ASSERT_EQ(PD_NO_ERROR, PDArgsCreateNone(&args));
  uint64_t size;
  ASSERT_EQ(PD_NO_ERROR, PDArgsGetSize(args, &size));
  ASSERT_EQ(0UL, size);
  ASSERT_EQ(PD_NO_ERROR, PDArgsDestroy(args));
}

TEST(CAPIArguments, value) {
  PD_Arguments args;
  ASSERT_EQ(PD_NO_ERROR, PDArgsCreateNone(&args));
  ASSERT_EQ(PD_NO_ERROR, PDArgsResize(args, 1));

  PD_Matrix mat;
  ASSERT_EQ(PD_NO_ERROR, PDMatCreate(&mat, 128, 64, false));
  for (size_t i = 0; i < 128; ++i) {
    std::vector<pd_real> sampleBuf = randomBuffer(64);
    PDMatCopyToRow(mat, i, sampleBuf.data());
  }
  ASSERT_EQ(PD_NO_ERROR, PDArgsSetValue(args, 0, mat));

  PD_Matrix val;
  ASSERT_EQ(PD_NO_ERROR, PDMatCreateNone(&val));

  ASSERT_EQ(PD_NO_ERROR, PDArgsGetValue(args, 0, val));

  for (size_t i = 0; i < 128; ++i) {
    pd_real* row1;
    pd_real* row2;

    ASSERT_EQ(PD_NO_ERROR, PDMatGetRow(mat, i, &row1));
    ASSERT_EQ(PD_NO_ERROR, PDMatGetRow(val, i, &row2));
    ASSERT_EQ(row1, row2);
  }
  ASSERT_EQ(PD_NO_ERROR, PDMatDestroy(val));
  ASSERT_EQ(PD_NO_ERROR, PDMatDestroy(mat));
  ASSERT_EQ(PD_NO_ERROR, PDArgsDestroy(args));
}
