#include <gtest/gtest.h>

namespace paddle {
namespace framework {
namespace tmp {
extern int InitializeB();
}
}
}

TEST(GENERIC_CMAKE, ALL) {
  ASSERT_EQ(0, paddle::framework::tmp::InitializeB());
}