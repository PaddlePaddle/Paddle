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

#undef PADDLE_DISABLE_TIMER
#include <gtest/gtest.h>
#include <paddle/utils/PythonUtil.h>
#include <algorithm>
#include <cstdlib>

#include "paddle/testing/TestUtil.h"
#include "paddle/trainer/Trainer.h"
#include "paddle/utils/Stat.h"

using namespace paddle;  // NOLINT
using namespace std;     // NOLINT

DECLARE_int32(gpu_id);
DECLARE_double(checkgrad_eps);
DEFINE_bool(use_label, true, "input label or sequence label");
DEFINE_bool(static_para, false, "static parameter");

struct DataIn {
  std::vector<Argument> inArgs;
  std::vector<MatrixPtr> outGrads;
  std::vector<VectorPtr> paraValues;
};

struct DataOut {
  std::vector<MatrixPtr> outValues;
  std::vector<VectorPtr> paraGrads;
};

void initArgument(DataIn& data,
                  const std::string& configPath,
                  bool useGpu = FLAGS_use_gpu) {
  TrainerConfigHelper config(configPath);
  size_t batchSize = config.getOptConfig().batch_size();

  for (const auto& layer_name : config.getModelConfig().input_layer_names()) {
    auto layer_config = std::find_if(config.getModelConfig().layers().begin(),
                                     config.getModelConfig().layers().end(),
                                     [=](const LayerConfig& layer_config) {
                                       return layer_config.name() == layer_name;
                                     });
    CHECK(layer_config != config.getModelConfig().layers().end());

    size_t layerSize = layer_config->size();
    Argument arg;
    arg.value = Matrix::create(batchSize, layerSize, false, useGpu);
    arg.grad = Matrix::create(batchSize, layerSize, false, useGpu);
    arg.value->randomizeUniform();
    arg.value->add(-0.5);
    arg.value->sigmoid(*arg.value);
    arg.grad->zeroMem();
    if (FLAGS_use_label) {
      arg.ids = VectorT<int>::create(batchSize, useGpu);
      arg.ids->rand(layerSize);
    }
    generateSequenceStartPositions(batchSize, arg.sequenceStartPositions);
    data.inArgs.push_back(arg);
  }

  for (const auto& layer_name : config.getModelConfig().output_layer_names()) {
    auto layer_config = std::find_if(config.getModelConfig().layers().begin(),
                                     config.getModelConfig().layers().end(),
                                     [=](const LayerConfig& layer_config) {
                                       return layer_config.name() == layer_name;
                                     });
    CHECK(layer_config != config.getModelConfig().layers().end());

    size_t layerSize = layer_config->size();
    MatrixPtr grad = Matrix::create(batchSize, layerSize, false, useGpu);
    grad->randomizeUniform();
    data.outGrads.push_back(grad);
  }

  for (const auto& para_config : config.getModelConfig().parameters()) {
    VectorPtr value = Vector::create(para_config.size(), useGpu);
    value->randnorm(0, 2);
    data.paraValues.push_back(value);
  }
}

void calcGradient(DataIn& in, DataOut& out, const std::string& configPath) {
  *ThreadLocalRand::getSeed() = 0;
  srand(0);

  Trainer trainer;
  auto config = std::make_shared<TrainerConfigHelper>(configPath);
  trainer.init(config, false);

  std::vector<ParameterPtr> parameters;
  vector<Argument> outArgs;

  auto gradientMachine = trainer.getGradientMachine();
  parameters = gradientMachine->getParameters();
  if (FLAGS_static_para) {
    for (size_t i = 0; i < parameters.size(); i++) {
      parameters[i]->getBuf(PARAMETER_VALUE)->one();
    }
  } else {
    for (size_t i = 0; i < in.paraValues.size(); i++) {
      parameters[i]->getBuf(PARAMETER_VALUE)->copyFrom(*in.paraValues[i]);
    }
  }
  gradientMachine->start();
  gradientMachine->forward(in.inArgs, &outArgs, PASS_TRAIN);
  for (size_t i = 0; i < in.outGrads.size(); i++) {
    // If the all the layers in the config have no parameters, also
    // not set NeedGradient(), the outArgs[i] will be nullptr.
    outArgs[i].grad->copyFrom(*in.outGrads[i]);
  }
  gradientMachine->backward();
  for (size_t i = 0; i < in.outGrads.size(); i++) {
    MatrixPtr value = Matrix::create(outArgs[i].value->getHeight(),
                                     outArgs[i].value->getWidth(),
                                     false,
                                     false);
    value->copyFrom(*outArgs[i].value);
    out.outValues.push_back(value);
  }
  for (size_t i = 0; i < in.paraValues.size(); i++) {
    VectorPtr grad = Vector::create(
        parameters[i]->getBuf(PARAMETER_GRADIENT)->getSize(), false);
    grad->copyFrom(*parameters[i]->getBuf(PARAMETER_GRADIENT));
    out.paraGrads.push_back(grad);
  }

  for (int i = 0; i < 20; i++) {
    REGISTER_TIMER("forward");
    gradientMachine->forward(in.inArgs, &outArgs, PASS_TRAIN);
  }
  for (int i = 0; i < 20; i++) {
    REGISTER_TIMER("backward");
    gradientMachine->backward();
  }

  gradientMachine->finish();
}

void checkBuffer(real* A,
                 const char* desA,
                 real* B,
                 const char* desB,
                 size_t len,
                 size_t width = 1) {
  int nNum = 0;
  for (size_t i = 0; i < len; ++i) {
    real diff = fabs(A[i] - B[i]);
    if (diff > 0.0f &&
        diff / std::max(fabs(A[i]), fabs(B[i])) > FLAGS_checkgrad_eps) {
      nNum++;
      LOG(INFO) << "Row: " << i / width << ", " << desA << " : " << A[i]
                << "    " << desB << " : " << B[i];
    }
  }
  EXPECT_EQ(0, nNum);
}

void compareGradient(DataOut& outA, DataOut& outB) {
  LOG(INFO) << "------------------------------"
            << " Check Network Output "
            << "------------------------------";
  for (size_t i = 0; i < outA.outValues.size(); ++i) {
    LOG(INFO) << "OUTPUT VALUE: " << i;
    checkBuffer(outA.outValues[i]->getData(),
                "network A output",
                outB.outValues[i]->getData(),
                "network B output",
                outA.outValues[i]->getElementCnt(),
                outA.outValues[i]->getWidth());
  }

  if (!FLAGS_static_para) {
    LOG(INFO) << "------------------------------"
              << " Check Parameters "
              << "------------------------------";
    for (size_t i = 0; i < outA.paraGrads.size(); ++i) {
      LOG(INFO) << "PARAMETER GRADIENT: " << i;
      checkBuffer(outA.paraGrads[i]->getData(),
                  "Network A",
                  outB.paraGrads[i]->getData(),
                  "Network B",
                  outA.paraGrads[i]->getSize());
    }
  }
}

void compareNetwork(const std::string& config_file_a,
                    const std::string& config_file_b) {
  DataIn in;
  initArgument(in, config_file_a);

  DataOut dataA;
  calcGradient(in, dataA, config_file_a);
  LOG(INFO) << "forwardBackward of Network A is finished";
  globalStat.printSegTimerStatus();
  globalStat.reset();
  LOG(INFO) << "\n\n";

  DataOut dataB;
  calcGradient(in, dataB, config_file_b);
  LOG(INFO) << "forwardBackward of the Network B is finished";
  globalStat.printSegTimerStatus();
  globalStat.reset();
  LOG(INFO) << "\n\n";

  compareGradient(dataA, dataB);
}

TEST(Compare, concat_dotmul) {
  std::string config_file_a = "./gserver/tests/concat_dotmul_a.conf";
  std::string config_file_b = "./gserver/tests/concat_dotmul_b.conf";
  compareNetwork(config_file_a, config_file_b);
}

TEST(Compare, concat_fullmatrix) {
  std::string config_file_a = "./gserver/tests/concat_fullmatrix_a.conf";
  std::string config_file_b = "./gserver/tests/concat_fullmatrix_b.conf";
  compareNetwork(config_file_a, config_file_b);
}

TEST(Compare, concat_table) {
  std::string config_file_a = "./gserver/tests/concat_table_a.conf";
  std::string config_file_b = "./gserver/tests/concat_table_b.conf";
  compareNetwork(config_file_a, config_file_b);
}

TEST(Compare, concat_slice) {
  std::string config_file_a = "./gserver/tests/concat_slice_a.conf";
  std::string config_file_b = "./gserver/tests/concat_slice_b.conf";
  compareNetwork(config_file_a, config_file_b);
}

#ifdef PADDLE_WITH_CUDA
TEST(Compare, img_pool) {
  std::string config_file_a = "./gserver/tests/img_pool_a.conf";
  std::string config_file_b = "./gserver/tests/img_pool_b.conf";
  bool useGpu = FLAGS_use_gpu;
  FLAGS_use_gpu = true;
  compareNetwork(config_file_a, config_file_b);
  FLAGS_use_gpu = useGpu;
}

TEST(Compare, img_conv) {
  std::string config_file_a = "./gserver/tests/img_conv_a.conf";
  std::string config_file_b = "./gserver/tests/img_conv_b.conf";
  bool useGpu = FLAGS_use_gpu;
  FLAGS_use_gpu = true;
  compareNetwork(config_file_a, config_file_b);
  FLAGS_use_gpu = useGpu;
}

// Test cudnn_conv and exconv give the same result
TEST(Compare, img_conv2) {
  std::string config_file_a = "./gserver/tests/img_conv_cudnn.py";
  std::string config_file_b = "./gserver/tests/img_conv_exconv.py";
  bool useGpu = FLAGS_use_gpu;
  double eps = FLAGS_checkgrad_eps;
  FLAGS_use_gpu = true;
  // Sometimes, this unit test will fail with 1e-2
  FLAGS_checkgrad_eps = 4e-2;
  compareNetwork(config_file_a, config_file_b);
  FLAGS_use_gpu = useGpu;
  FLAGS_checkgrad_eps = eps;
}
#endif

DEFINE_string(config_file_a, "", "config of one network to compare");
DEFINE_string(config_file_b, "", "config of another network to compare");
TEST(Compare, network) {
  if (FLAGS_config_file_a != "" && FLAGS_config_file_b != "") {
    compareNetwork(FLAGS_config_file_a, FLAGS_config_file_b);
  }
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  paddle::initMain(argc, argv);
  initPython(argc, argv);
  int ret = RUN_ALL_TESTS();
  return ret;
}
