/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <gtest/gtest.h>
#include <paddle/gserver/gradientmachines/GradientMachine.h>
#include <paddle/trainer/TrainerConfigHelper.h>
#include <stdlib.h>
#include <string.h>
#include <type_traits>
#include "capi.h"
#include "paddle/utils/ThreadLocal.h"

static std::vector<paddle_real> randomBuffer(size_t bufSize) {
  auto& eng = paddle::ThreadLocalRandomEngine::get();
  std::uniform_real_distribution<paddle_real> dist(-1.0, 1.0);
  std::vector<paddle_real> retv;
  retv.reserve(bufSize);
  for (size_t i = 0; i < bufSize; ++i) {
    retv.push_back(dist(eng));
  }
  return retv;
}

TEST(GradientMachine, testPredict) {
  //! TODO(yuyang18): Test GPU Code.
  paddle::TrainerConfigHelper config("./test_predict_network.py");
  std::string buffer;
  ASSERT_TRUE(config.getModelConfig().SerializeToString(&buffer));
  paddle_gradient_machine machine;

  ASSERT_EQ(kPD_NO_ERROR,
            paddle_gradient_machine_create_for_inference(
                &machine, &buffer[0], (int)buffer.size()));
  std::unique_ptr<paddle::GradientMachine> gm(
      paddle::GradientMachine::create(config.getModelConfig()));
  ASSERT_NE(nullptr, gm);
  gm->randParameters();
  gm->saveParameters("./");

  ASSERT_EQ(kPD_NO_ERROR,
            paddle_gradient_machine_load_parameter_from_disk(machine, "./"));

  paddle_gradient_machine machineSlave;
  ASSERT_EQ(kPD_NO_ERROR,
            paddle_gradient_machine_create_shared_param(
                machine, &buffer[0], (int)buffer.size(), &machineSlave));
  std::swap(machineSlave, machine);
  paddle_arguments outArgs = paddle_arguments_create_none();

  paddle_arguments inArgs = paddle_arguments_create_none();
  ASSERT_EQ(kPD_NO_ERROR, paddle_arguments_resize(inArgs, 1));
  paddle_matrix mat = paddle_matrix_create(1, 100, false);
  static_assert(std::is_same<paddle_real, paddle::real>::value, "");

  auto data = randomBuffer(100);
  paddle_real* rowPtr;
  ASSERT_EQ(kPD_NO_ERROR, paddle_matrix_get_row(mat, 0, &rowPtr));
  memcpy(rowPtr, data.data(), data.size() * sizeof(paddle_real));

  ASSERT_EQ(kPD_NO_ERROR, paddle_arguments_set_value(inArgs, 0, mat));
  ASSERT_EQ(kPD_NO_ERROR,
            paddle_gradient_machine_forward(machine, inArgs, outArgs, false));

  uint64_t sz;
  ASSERT_EQ(kPD_NO_ERROR, paddle_arguments_get_size(outArgs, &sz));
  ASSERT_EQ(1UL, sz);

  ASSERT_EQ(kPD_NO_ERROR, paddle_arguments_get_value(outArgs, 0, mat));
  std::vector<paddle::Argument> paddleInArgs;
  std::vector<paddle::Argument> paddleOutArgs;
  paddleInArgs.resize(1);
  paddleInArgs[0].value =
      paddle::Matrix::create(data.data(), 1, 100, false, false);

  gm->forward(paddleInArgs, &paddleOutArgs, paddle::PASS_TEST);

  auto matPaddle = paddleOutArgs[0].value;

  uint64_t height, width;
  ASSERT_EQ(kPD_NO_ERROR, paddle_matrix_get_shape(mat, &height, &width));
  ASSERT_EQ(matPaddle->getHeight(), height);
  ASSERT_EQ(matPaddle->getWidth(), width);

  ASSERT_EQ(kPD_NO_ERROR, paddle_matrix_get_row(mat, 0, &rowPtr));
  for (size_t i = 0; i < width; ++i) {
    ASSERT_NEAR(matPaddle->getData()[i], rowPtr[i], 1e-5);
  }

  ASSERT_EQ(kPD_NO_ERROR, paddle_matrix_destroy(mat));
  ASSERT_EQ(kPD_NO_ERROR, paddle_arguments_destroy(inArgs));
  ASSERT_EQ(kPD_NO_ERROR, paddle_arguments_destroy(outArgs));
  std::swap(machineSlave, machine);
  ASSERT_EQ(kPD_NO_ERROR, paddle_gradient_machine_destroy(machineSlave));
  ASSERT_EQ(kPD_NO_ERROR, paddle_gradient_machine_destroy(machine));
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  std::vector<char*> argvs;
  argvs.push_back(strdup("--use_gpu=false"));
  paddle_init((int)argvs.size(), argvs.data());
  for (auto each : argvs) {
    free(each);
  }
  return RUN_ALL_TESTS();
}
