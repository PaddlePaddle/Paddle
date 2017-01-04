#include <gtest/gtest.h>
#include <stdlib.h>
#include <string.h>
#include "PaddleCAPI.h"

TEST(GradientMachine, load) {
  void* buf;
  int size;
  ASSERT_EQ(
      PD_NO_ERROR,
      PDParseTrainerConfigFromFile(strdup("./vgg_16_cifar.py"), &buf, &size));
  free(buf);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  std::vector<char*> argvs;
  argvs.push_back(strdup("--use_gpu=false"));
  PDInit((int)argvs.size(), argvs.data());
  for (auto each : argvs) {
    free(each);
  }
  return RUN_ALL_TESTS();
}
