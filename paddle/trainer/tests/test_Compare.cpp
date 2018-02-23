/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <paddle/utils/PythonUtil.h>

#include "paddle/trainer/Trainer.h"

#include <gtest/gtest.h>
#include <cstdlib>

using namespace paddle;  // NOLINT
using namespace std;     // NOLINT

static const string& configFile = "trainer/tests/sample_trainer_config.conf";

DECLARE_int32(gpu_id);
DECLARE_bool(use_gpu);
DECLARE_string(config);
DECLARE_string(config_args);

struct comData {
  vector<Argument> outArgs;
  vector<ParameterPtr> parameters;
};

void calcGradient(bool useGpu, comData& Data) {
  FLAGS_use_gpu = useGpu;
  FLAGS_config = configFile;

  *ThreadLocalRand::getSeed() = 0;
  srand(0);
  Trainer trainer;
  trainer.init(TrainerConfigHelper::createFromFlagConfig());

  Data.parameters = trainer.getGradientMachine()->getParameters();
  DataBatch dataBatch;
  int32_t batchSize = trainer.getConfig().opt_config().batch_size();
  trainer.getDataProvider()->setSkipShuffle();
  trainer.getDataProvider()->getNextBatch(batchSize, &dataBatch);
  CHECK(dataBatch.getSize()) << "No data from data provider";
  vector<Argument>& inArgs = dataBatch.getStreams();
  trainer.getGradientMachine()->start();
  for (int i = 0; i < 2; ++i) {
    trainer.getGradientMachine()->forwardBackward(
        inArgs, &Data.outArgs, PASS_TRAIN);
  }
  trainer.getGradientMachine()->finish();
}

void compareGradient(comData& comDataCpu, comData& comDataGpu);

TEST(Trainer, create) {
  int devCount = 0;
  devCount = hl_get_device_count();
  FLAGS_config_args = "drop_rate=0";

  comData comDataCpu;
  calcGradient(false, comDataCpu);
  LOG(INFO) << "Cpu is completed";

  {
    LOG(INFO) << "Test GPU";
    comData comData;
    calcGradient(true, comData);
    compareGradient(comDataCpu, comData);
    LOG(INFO) << "Gpu is completed";
  }

  {
    LOG(INFO) << "Test test multi gpu";
    comData comData;
    FLAGS_trainer_count = devCount;
    calcGradient(true, comData);
    compareGradient(comDataCpu, comData);
    LOG(INFO) << "Gpu4 is completed";
  }

  {
    LOG(INFO) << "Test use_sparse_update=true";
    comData comData;
    calcGradient(false, comData);
    compareGradient(comDataCpu, comData);
    LOG(INFO) << "Cpu4 is completed";
  }
}

double checkBuffer(real* A, real* B, size_t len) {
#ifdef PADDLE_TYPE_DOUBLE
  double precision = 1e-7;
#else
  double precision = 2e-3;
#endif
  int nNum = 0;
  double maxE = 0;
  for (size_t i = 0; i < len; ++i) {
    double e = fabs(A[i] - B[i]);
    maxE = std::max(e, maxE);
    nNum += e > precision * fabs(A[i]);
  }
  EXPECT_EQ(0, nNum);
  return maxE;
}

void compareGradient(comData& comDataCpu, comData& comDataGpu) {
  /*compare outArgs*/
  vector<Argument> outArgs1 = comDataCpu.outArgs;
  vector<Argument> outArgs2 = comDataGpu.outArgs;
  CpuMatrix out1(outArgs1[0].value->getHeight(), outArgs1[0].value->getWidth());
  CpuMatrix out2(outArgs2[0].value->getHeight(), outArgs2[0].value->getWidth());
  out1.copyFrom(*outArgs1[0].value);
  out2.copyFrom(*outArgs2[0].value);
  checkBuffer(out1.getData(), out2.getData(), out1.getElementCnt());

  /*compare parameters*/
  vector<ParameterPtr>& parameters1 = comDataCpu.parameters;
  vector<ParameterPtr>& parameters2 = comDataGpu.parameters;
  for (size_t i = 0; i < parameters1.size(); ++i) {
    ParameterPtr parameter1, parameter2;
    parameter1 = parameters1[i];
    parameter2 = parameters2[i];
    /*compare parameters value*/
    CpuVector para1(parameter1->getSize());
    CpuVector para2(parameter2->getSize());
    para1.copyFrom(*parameter1->getBuf(PARAMETER_VALUE));
    para2.copyFrom(*parameter2->getBuf(PARAMETER_VALUE));
    checkBuffer(para1.getData(), para2.getData(), para1.getSize());

    /*compare parameters grad*/
    CpuVector cpuGrad1(*parameter1->getBuf(PARAMETER_GRADIENT));
    CpuVector cpuGrad2(*parameter2->getBuf(PARAMETER_GRADIENT));
    double e =
        checkBuffer(cpuGrad1.getData(), cpuGrad2.getData(), cpuGrad1.getSize());
    LOG(INFO) << parameter1->getName() << " max error=" << e;
  }
}

int main(int argc, char** argv) {
#ifndef PADDLE_WITH_CUDA
  exit(0);
#endif
  paddle::initMain(argc, argv);
  testing::InitGoogleTest(&argc, argv);
  initPython(argc, argv);
  int ret = RUN_ALL_TESTS();
  exit(ret);
}
