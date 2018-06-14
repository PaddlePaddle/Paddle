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
#include <math.h>
#include <paddle/utils/PythonUtil.h>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include "ModelConfig.pb.h"
#include "paddle/gserver/layers/DataLayer.h"
#include "paddle/gserver/layers/FullyConnectedLayer.h"
#include "paddle/gserver/layers/Layer.h"
#include "paddle/gserver/layers/SelectiveFullyConnectedLayer.h"
#include "paddle/math/CpuSparseMatrix.h"

using namespace paddle;  // NOLINT
using namespace std;     // NOLINT

DECLARE_bool(use_gpu);
DECLARE_int32(num_passes);
DECLARE_string(config);
DECLARE_string(init_model_path);
DECLARE_string(config_args);

size_t fcLayerWidth = 1024;

struct ComData {
  vector<Argument> outArgs;
  vector<ParameterPtr> parameters;
};

int randint(int* data, size_t int_max, size_t size) {
  srand((size_t)(time(NULL)));
  if (int_max < size) {
    return -1;
  }
  size_t count = 0;
  std::map<int, int> tmp;
  int this_int = 0;

  while (count < size) {
    this_int = std::rand() % int_max;  // NOLINT
    if (tmp.find(this_int) == tmp.end()) {
      tmp[this_int] = 0;
      count += 1;
    }
  }

  if (tmp.size() != size) {
    return -1;
  }
  count = 0;
  for (auto itr = tmp.begin(); itr != tmp.end(); ++itr) {
    data[count] = itr->first;
    count += 1;
  }
  return 0;
}

void calcOutput(ComData& comData,
                const string configFile,
                const string configArgs,
                bool useGpu) {
  FLAGS_config = configFile;
  FLAGS_config_args = configArgs;
  FLAGS_use_gpu = useGpu;
  FLAGS_init_model_path = "gserver/tests/SelectiveFcTest/model";
  *ThreadLocalRand::getSeed() = 0;
  srand(0);

  Trainer trainer;
  trainer.init(TrainerConfigHelper::createFromFlags(), false);

  comData.parameters = trainer.getGradientMachine()->getParameters();

  auto dataProvider = trainer.getDataProvider();
  int32_t batchSize = trainer.getConfig().opt_config().batch_size();
  DataBatch dataBatch;
  dataProvider->setSkipShuffle();
  dataProvider->reset();
  dataProvider->getNextBatch(batchSize, &dataBatch);
  CHECK(dataBatch.getSize()) << "No data from data provider";

  vector<Argument>& inArgs = dataBatch.getStreams();
  trainer.getGradientMachine()->start(trainer.getConfig(), nullptr);
  trainer.getGradientMachine()->forwardBackward(
      inArgs, &comData.outArgs, PASS_TRAIN);
  trainer.getGradientMachine()->finish();
}

void checkMatrix(real* A, real* B, size_t matSize) {
#ifndef PADDLE_TYPE_DOUBLE
  real err = 1e-3;
#else
  real err = 1e-10;
#endif
  int diffNum = 0;
  for (size_t i = 0; i < matSize; ++i) {
    if (std::isinf(A[i]) || std::isnan(A[i]) || std::isinf(B[i]) ||
        std::isnan(B[i])) {
    } else if (fabs(A[i] - B[i]) > err) {
      diffNum++;
    }
  }
  EXPECT_EQ(0, diffNum);
}

void checkTranspose(real* matrix,
                    real* transpose,
                    size_t width,
                    size_t matSize) {
#ifndef PADDLE_TYPE_DOUBLE
  real err = 1e-3;
#else
  real err = 1e-10;
#endif
  size_t height = matSize / width;
  int diffNum = 0;
  size_t rowId = 0;
  size_t colId = 0;
  for (size_t i = 0; i < matSize; ++i) {
    if (i % width == 0 && i) {
      rowId++;
    }
    colId = i % width;
    if (fabs(matrix[i] - transpose[colId * height + rowId]) > err) {
      diffNum++;
      LOG(INFO) << i << " diff : " << matrix[i] << "\t"
                << transpose[colId * height + rowId];
    }
  }
  EXPECT_EQ(0, diffNum);
}

void compareOutput(ComData& fcData, ComData& selFcData) {
  vector<Argument> outArgsFc = fcData.outArgs;
  vector<Argument> outArgsSelfc = selFcData.outArgs;

  // check cost
  LOG(INFO) << "Check cost";
  CpuMatrix fcCost(outArgsFc[0].value->getHeight(),
                   outArgsFc[0].value->getWidth());
  CpuMatrix selfcCost(outArgsSelfc[0].value->getHeight(),
                      outArgsSelfc[0].value->getWidth());
  fcCost.copyFrom(*outArgsFc[0].value);
  selfcCost.copyFrom(*outArgsSelfc[0].value);
  checkMatrix(fcCost.getData(), selfcCost.getData(), fcCost.getElementCnt());

  // check selective fc output and fc output
  LOG(INFO) << "Compare output of SelectiveFullyConectedLayer "
            << "with FullyConectedLayer";
  CpuMatrix fcOut(outArgsFc[1].value->getHeight(),
                  outArgsFc[1].value->getWidth());
  CpuMatrix selfcOut(outArgsSelfc[1].value->getHeight(),
                     outArgsSelfc[1].value->getWidth());

  fcOut.copyFrom(*outArgsFc[1].value);
  selfcOut.copyFrom(*outArgsSelfc[1].value);
  checkMatrix(fcOut.getData(), selfcOut.getData(), fcOut.getElementCnt());

  // check gradient math
  vector<ParameterPtr>& fcParam = fcData.parameters;
  vector<ParameterPtr>& selfcParam = selFcData.parameters;
  for (size_t i = 0; i < fcParam.size(); ++i) {
    ParameterPtr p1, p2;
    p1 = fcParam[i];
    p2 = selfcParam[i];

    string paramName = p1->getName();
    LOG(INFO) << "check parameter : " << paramName;

    // check parameter value
    CpuVector paraValue1(p1->getSize());
    CpuVector paraValue2(p2->getSize());
    paraValue1.copyFrom(*p1->getBuf(PARAMETER_VALUE));
    paraValue2.copyFrom(*p2->getBuf(PARAMETER_VALUE));

    // check gradient
    CpuVector paraGrad1(*p1->getBuf(PARAMETER_GRADIENT));
    CpuVector paraGrad2(*p2->getBuf(PARAMETER_GRADIENT));
    if (paramName == "rand_fc_param.bias") {
      checkMatrix(
          paraValue1.getData(), paraValue2.getData(), paraValue1.getSize());
      checkMatrix(
          paraGrad1.getData(), paraGrad2.getData(), paraGrad1.getSize());
    } else {
      checkTranspose(paraValue1.getData(),
                     paraValue2.getData(),
                     fcLayerWidth,
                     paraValue1.getSize());
      checkTranspose(paraGrad1.getData(),
                     paraGrad2.getData(),
                     fcLayerWidth,
                     paraGrad1.getSize());
    }
  }
}

void compareSparseMulOutput(
    real* fcOutput,
    real* selOutput,
    size_t nnz,
    const std::shared_ptr<std::vector<std::pair<int*, size_t>>>& selCols) {
#ifndef PADDLE_TYPE_DOUBLE
  real err = 1e-3;
#else
  real err = 1e-10;
#endif
  size_t nnzCount =
      std::accumulate(selCols->begin(),
                      selCols->end(),
                      0UL,
                      [](size_t a, const std::pair<int*, size_t>& arr) {
                        return a + arr.second;
                      });
  EXPECT_EQ(nnz, nnzCount);

  size_t sampleNum = selCols->size();
  int diffNum = 0;
  size_t count = 0;
  for (size_t i = 0; i < sampleNum; ++i) {
    for (size_t j = 0; j < (*selCols)[i].second; ++j) {
      size_t selIdx = (*selCols)[i].first[j];
      if (fabs(fcOutput[i * fcLayerWidth + selIdx] - selOutput[count]) > err) {
        diffNum++;
        LOG(INFO) << count << " diff : " << fcOutput[i * fcLayerWidth + selIdx]
                  << "\t" << selOutput[count];
      }
      count++;
    }
  }
  EXPECT_EQ(0, diffNum);
}

LayerPtr creatDataLayer(string name,
                        size_t batchSize,
                        size_t layerSize,
                        std::vector<real>& values,
                        bool useGpu) {
  LayerConfig dataConfig;
  dataConfig.set_name(name);
  dataConfig.set_type("data");
  dataConfig.set_size(layerSize);
  LayerPtr layer = LayerPtr(new DataLayer(dataConfig));

  Argument data;
  data.value = Matrix::create(batchSize, layerSize, false, useGpu);
  data.value->copyFrom(values.data(), batchSize * layerSize);

  DataLayerPtr dataLayer = std::dynamic_pointer_cast<DataLayer>(layer);
  dataLayer->setData(data);
  dataLayer->forward(PASS_TEST);
  return layer;
}

ParameterPtr creatParameter(
    string name, int pid, size_t paraSize, string paramFile, bool useGpu) {
  ParameterConfig paraConfig;
  paraConfig.set_name(name);
  paraConfig.set_size(paraSize);

  ParameterPtr parameter =
      std::make_shared<Parameter>(paraConfig, useGpu, /*initialize */ false);
  parameter->enableType(PARAMETER_VALUE);
  parameter->randomize();
  parameter->setID(pid);
  parameter->load(paramFile);
  return parameter;
}

LayerPtr initFcLayer(LayerPtr dataLayer,
                     LayerConfig layerConfig,
                     int dataLayerSize,
                     int fcLayerSize,
                     string paraName,
                     string paraFile,
                     bool useGpu) {
  LayerMap layerMap;
  ParameterMap parameterMap;

  layerMap[dataLayer->getName()] = dataLayer;
  ParameterPtr para = creatParameter(
      paraName, 0, dataLayerSize * fcLayerSize, paraFile, useGpu);
  parameterMap[para->getName()] = para;

  layerConfig.add_inputs();
  LayerInputConfig& input = *(layerConfig.mutable_inputs(0));
  input.set_input_layer_name(dataLayer->getName());
  input.set_input_parameter_name(paraName);

  LayerPtr testLayer = Layer::create(layerConfig);
  layerMap[testLayer->getName()] = testLayer;

  testLayer->setNeedGradient(false);
  testLayer->init(layerMap, parameterMap);
  return testLayer;
}

#ifndef PADDLE_TYPE_DOUBLE
// The parameter file used in fc.conf and selective_fc.conf is float
TEST(Layer, SelectiveFcLayer_train_dense_mul) {
  const string& fcConfig = "gserver/tests/SelectiveFcTest/conf/fc.conf";
  const string& fcConfigArgs =
      "filelist=gserver/tests/SelectiveFcTest/dense_mul_list";
  const string& selFcConfig =
      "gserver/tests/SelectiveFcTest/conf/selective_fc.conf";
  const string& selConfigArgs =
      "filelist=gserver/tests/SelectiveFcTest/dense_mul_list";

  for (auto useGpu : {false, true}) {
#ifndef PADDLE_WITH_CUDA
    if (useGpu) {
      break;
    }
#endif
    LOG(INFO) << "FullyConnectedLayer forwardBackward()";
    ComData fcData;
    calcOutput(fcData, fcConfig, fcConfigArgs, useGpu);

    LOG(INFO) << "SelectiveFullyConnectedLayer forwardBackward()";
    ComData selFcData;
    calcOutput(selFcData, selFcConfig, selConfigArgs, useGpu);
    compareOutput(fcData, selFcData);
  }
}
#endif  // PADDLE_TYPE_DOUBLE

void testSelectiveFcLayerTrainSparseMul(const LayerConfig& config,
                                        bool useGpu) {
  FLAGS_use_gpu = useGpu;
  size_t batchSize = 100;
  size_t dataLayerSize = 512;
  std::vector<real> values(batchSize * dataLayerSize);
  for (size_t j = 0; j < batchSize * dataLayerSize; ++j) {
    values[j] = std::rand() / real(RAND_MAX);
  }
  LayerPtr dataLayer =
      creatDataLayer("data", batchSize, dataLayerSize, values, useGpu);

  const string& selfcParaFile =
      "gserver/tests/SelectiveFcTest/model/rand_fc_param.w.transpose";
  const string& selfcParaName = "rand_fc_param.w.transpose";

  std::shared_ptr<SelectiveFullyConnectedLayer> selfcLayer =
      std::dynamic_pointer_cast<SelectiveFullyConnectedLayer>(
          initFcLayer(dataLayer,
                      config,
                      dataLayerSize,
                      fcLayerWidth,
                      selfcParaName,
                      selfcParaFile,
                      useGpu));

  // create selected columns
  std::shared_ptr<std::vector<std::pair<int*, size_t>>> selCols(
      new std::vector<std::pair<int*, size_t>>(batchSize));
  size_t maxNNZ = 30;
  srand((size_t)(time(NULL)));
  int total = 0;
  while (total == 0) {
    for (size_t i = 0; i < batchSize; ++i) {
      size_t num = std::rand() % maxNNZ;
      int* data = new int[num];
      randint(data, fcLayerWidth, num);
      (*selCols)[i] = std::make_pair(data, num);
      total += num;
    }
  }
  selfcLayer->fillSelectiveData(selCols);
  selfcLayer->forward(PASS_TEST);

  MatrixPtr outMatSelfc = selfcLayer->getOutputValue();
  CpuSparseMatrixPtr cpuOutMatSelfc(
      new CpuSparseMatrix(outMatSelfc->getHeight(),
                          outMatSelfc->getWidth(),
                          outMatSelfc->getElementCnt()));
  cpuOutMatSelfc->copyFrom(*outMatSelfc, HPPL_STREAM_DEFAULT);
#ifdef PADDLE_WITH_CUDA
  if (useGpu) {
    hl_stream_synchronize(HPPL_STREAM_DEFAULT);
  }
#endif
  real* outValueSelfc = cpuOutMatSelfc->getValue();
  size_t nnz = cpuOutMatSelfc->getElementCnt();

  const string& fcParaFile =
      "gserver/tests/SelectiveFcTest/model/rand_fc_param.w";
  const string& fcParaName = "rand_fc_param.w";
  LayerConfig fcLayerConfig;
  fcLayerConfig.set_name("fc_layer");
  fcLayerConfig.set_type("fc");
  fcLayerConfig.set_active_type("linear");
  fcLayerConfig.set_size(fcLayerWidth);

  LayerPtr fcLayer = initFcLayer(dataLayer,
                                 fcLayerConfig,
                                 dataLayerSize,
                                 fcLayerWidth,
                                 fcParaName,
                                 fcParaFile,
                                 useGpu);
  fcLayer->forward(PASS_TEST);

  MatrixPtr outMatFc = fcLayer->getOutputValue();
  MatrixPtr cpuOutMatFc(
      new CpuMatrix(outMatFc->getHeight(), outMatFc->getWidth()));
  cpuOutMatFc->copyFrom(*outMatFc, HPPL_STREAM_DEFAULT);
#ifdef PADDLE_WITH_CUDA
  if (useGpu) {
    hl_stream_synchronize(HPPL_STREAM_DEFAULT);
  }
#endif
  real* outValueFc = cpuOutMatFc->getData();

  compareSparseMulOutput(outValueFc, outValueSelfc, nnz, selCols);
  for (size_t i = 0; i < batchSize; ++i) {
    delete[](*selCols)[i].first;
  }
}

#ifndef PADDLE_TYPE_DOUBLE
// The parameter file used in testSelectiveFcLayerTrainSparseMul is float
TEST(Layer, SelectiveFcLayer_train_sparse_mul) {
  LayerConfig selLayerConfig;
  selLayerConfig.set_name("sel_fc");
  selLayerConfig.set_type("selective_fc");
  selLayerConfig.set_active_type("linear");
  selLayerConfig.set_has_selected_colums(false);
  selLayerConfig.set_selective_fc_pass_generation(true);
  selLayerConfig.set_size(fcLayerWidth);

  testSelectiveFcLayerTrainSparseMul(selLayerConfig, false);
#ifdef PADDLE_WITH_CUDA
  testSelectiveFcLayerTrainSparseMul(selLayerConfig, true);
#endif
}
#endif  // PADDLE_TYPE_DOUBLE

// TODO(dangqingqing) test multi threads after support in matrix
// TEST(Layer, SelectiveFcLayer_train_sparse_mul_parallel) {
//   LayerConfig selLayerConfig;
//   selLayerConfig.set_name("sel_fc");
//   selLayerConfig.set_type("selective_fc");
//   selLayerConfig.set_active_type("linear");
//   selLayerConfig.set_has_selected_colums(false);
//   selLayerConfig.set_selective_fc_pass_generation(true);
//   selLayerConfig.set_selective_fc_parallel_plain_mul_thread_num(10);
//   selLayerConfig.set_selective_fc_full_mul_ratio(1000);
//   selLayerConfig.set_size(fcLayerWidth);
//   SelectiveFcLayer_test(selLayerConfig, false);
// }

int main(int argc, char** argv) {
  paddle::initMain(argc, argv);
  testing::InitGoogleTest(&argc, argv);
  initPython(argc, argv);
  int ret = RUN_ALL_TESTS();
  return ret;
}
