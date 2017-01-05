#include <gtest/gtest.h>
#include <paddle/trainer/TrainerConfigHelper.h>
#include <stdlib.h>
#include <string.h>
#include "PaddleCAPI.h"

TEST(GradientMachine, load) {
  paddle::TrainerConfigHelper config("./vgg_16_cifar.py");
  std::string buffer;
  ASSERT_TRUE(config.getModelConfig().SerializeToString(&buffer));
  PD_GradiemtMachine machine;

  ASSERT_EQ(PD_NO_ERROR,
            PDGradientMachineCreateForPredict(
                &machine, &buffer[0], (int)buffer.size()));
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
