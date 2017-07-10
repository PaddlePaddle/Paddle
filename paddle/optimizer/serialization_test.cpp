#include "serialization.h"
#include "gtest/gtest.h"

using namespace paddle;
using namespace paddle::optimizer;

TEST(TensorToProto, Case1) {
  Tensor t(3), t1(3);
  for (size_t i = 0; i < t.size(); ++i) {
    t[i] = i;
    t1[i] = 0;
  }

  TensorProto proto;
  TensorToProto(t, &proto);
  ProtoToTensor(proto, &t1);
  for (size_t i = 0; i < t1.size(); ++i) {
    EXPECT_EQ(t1[i], t[i]);
  }
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
