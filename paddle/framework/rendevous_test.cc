#include "paddle/framework/rendevous.h"

#include <gtest/gtest.h>

namespace paddle {
namespace framework {

TEST(PairKey, Hash) {
  PairKey ka{"var0:gpu0", "var1:gpu1", 0, 1};
  auto s = CreateKey(ka);
  auto kb = ParseKey(s);
  // PairKey kb{"var1:gpu1", "var0:gpu0", 1, 0};
  EXPECT_EQ(Hash64(ka), Hash64(kb));
}

}  // namespace framework
}  // namespace paddle
