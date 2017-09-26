#include "paddle/framework/rendevous.h"

#include <gtest/gtest.h>

namespace paddle {
namespace framework {

TEST(PairKey, Hash) {
  PairKey ka{"var0:gpu0", "var1:gpu1", 0, 1};
  PairKey kb{"var1:gpu1", "var0:gpu0", 1, 0};
  LOG(INFO) << Hash64(ka);
}

}  // namespace framework
}  // namespace paddle
