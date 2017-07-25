/* Copyright (c) 2017 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "LayerGradUtil.h"
#include "MkldnnTestFunc.h"
#include "paddle/gserver/layers/MkldnnBase.h"

namespace paddle {
bool isMkldnnLayer(const LayerConfig& config) {
  // if layer type started with "mkldnn_"
  const std::string dnn("mkldnn_");
  const std::string& type = config.type();
  return type.compare(0, dnn.length(), dnn) == 0;
}

bool isMkldnnAct(const LayerConfig& config) {
  // if layer activation started with "mkldnn_"
  const std::string dnn("mkldnn_");
  const std::string& act = config.active_type();
  return act.compare(0, dnn.length(), dnn) == 0;
}

// Get delta percent
// if many wrong point return the max(diff/ref)
// else return sum(abs(a-b)) / sum(abs(b))
double getDelta(const real* d1, const real* d2, size_t len,
  const float failRate = 1e-3, const float thres = 0.1) {
  double delta = 0, sum = 0;
  int failCnt = 0;
  const double eps = 1e-5;
  double maxOut = 0;
  for (size_t i = 0; i < len; ++i) {
    double ref = fabs(d2[i]);
    double diff = fabs(d1[i] - d2[i]);
    delta += diff;
    sum += ref;
    if (ref > eps && fabs(d1[i]) > eps && diff / ref > thres) {
      maxOut = std::max(maxOut, diff / ref);
      failCnt++;
    }
  }
  EXPECT_TRUE(std::isnormal(sum));
  EXPECT_FALSE(std::isinf(sum));
  EXPECT_FALSE(std::isnan(delta));
  VLOG(DNN_TESTS_MORE) << "reference avg data: " << sum / len
    << ", delta: " << delta / sum << ", failCnt:" << failCnt;
  return (failCnt / (float)len) > failRate ? maxOut : delta / sum;
}

double compareMatrix(const MatrixPtr& m1, const MatrixPtr& m2) {
  CHECK_EQ(m1->getElementCnt(), m2->getElementCnt());
  return getDelta(m1->getData(), m2->getData(), m1->getElementCnt());
}

double compareVector(const VectorPtr& v1, const VectorPtr& v2) {
  CHECK_EQ(v1->getSize(), v2->getSize());
  return getDelta(v1->getData(), v2->getData(), v1->getSize());
}

void printMatrix(const MatrixPtr& m, int level = DNN_ALL) {
  const real* pd = m->getData();
  const int width = m->getWidth();
  const int height = m->getHeight();
  for(int h = 0; h < height; ++h) {
    std::stringstream row;
    for (int w = 0; w < width; ++w) {
      row << pd[width * h + w] << ", ";
    }
    VLOG(level) << row.str();
  }
}

void printVector(const VectorPtr& v, int level = DNN_ALL) {
  const real* pd = v->getData();
  const int sz = v->getSize();
  std::stringstream row;
  for(int i = 0; i < sz; ++i) {
    row << pd[i] << ", ";
  }
  VLOG(level) << row.str();
}

void testLayerFunc(std::vector<TestConfig>& configs, size_t batchSize,
  size_t inputImgH, size_t inputImgW, float epsilon) {
  CHECK(isMkldnnLayer(configs[0].layerConfig)
    || isMkldnnAct(configs[0].layerConfig)) << "test type go first";

  const bool isBN = configs[0].layerConfig.type() == "mkldnn_batch_norm";
  const bool trans = false;
  const bool useGpu = false;
  const int lvl = DNN_TESTS_MORE;  // vlog level
  vector<string> layerNames = {"tgt", "ref"};
  CHECK_EQ(configs.size(), 2);
  vector<vector<DataLayerPtr>> dataLayers(2);
  vector<LayerMap> layerMap(2);
  vector<vector<Argument>> datas(2);
  vector<vector<ParameterPtr>> parameters(2);
  vector<LayerPtr> testLayer(2);
  for (size_t i = 0; i < 2; ++i) {
    configs[i].layerConfig.set_name(layerNames[i]);
    // data layer initialize
    initDataLayer(configs[i], &(dataLayers[i]), &(datas[i]),
      &(layerMap[i]), layerNames[i], batchSize, trans, useGpu);
    initTestLayer(configs[i], &(layerMap[i]), &(parameters[i]), &(testLayer[i]));

  }
  CHECK_EQ(dataLayers[0].size(), dataLayers[1].size());
  VLOG(DNN_TESTS) << "Test MKLDNN functionality: "
    << configs[0].layerConfig.type() << " vs " << configs[1].layerConfig.type();

  const size_t iter = 3;
  vector<double> deltaFwd;
  vector<double> deltaBwd;
  vector<double> deltaParam;
  // init parameters: [0] target, [1] reference
  EXPECT_EQ(parameters[0].size(), parameters[1].size());
  for (size_t i = 0; i < parameters[0].size(); ++i) {
    parameters[0][i]->randomize();
    VectorPtr srcValue = parameters[0][i]->getBuf(PARAMETER_VALUE);
    VectorPtr dstValue = parameters[1][i]->getBuf(PARAMETER_VALUE);
    const VectorPtr& tstGrad = parameters[0][i]->getBuf(PARAMETER_GRADIENT);
    const VectorPtr& refGrad = parameters[1][i]->getBuf(PARAMETER_GRADIENT);
    dstValue->copyFrom(*srcValue);
    VLOG(lvl) << "initial "<< parameters[0][i]->getName() << ":";
    printVector(srcValue, lvl);
    if (tstGrad) {
      tstGrad->zeroMem();
    }
    if (refGrad) {
      refGrad->zeroMem();
    }
  }
  // clear botdiffs
  for (size_t idx = 0; idx < dataLayers[0].size(); ++idx) {
    dataLayers[0][idx]->getOutputGrad()->zeroMem();
    dataLayers[1][idx]->getOutputGrad()->zeroMem();
  }

  // set image size
  // TODO(TJ): fix me when concat and elewise
  for (size_t idx = 0; idx < dataLayers[0].size(); ++idx) {
    dataLayers[0][idx]->getOutput().setFrameHeight(inputImgH);
    dataLayers[0][idx]->getOutput().setFrameWidth(inputImgW);
    dataLayers[1][idx]->getOutput().setFrameHeight(inputImgH);
    dataLayers[1][idx]->getOutput().setFrameWidth(inputImgW);
  }
  // repeat some times, make sure all of them pass
  for (size_t i = 0; i < iter; ++i) {
    VLOG(lvl) << "Check Iteration " << i;
    // random botdata, copy same to ref
    for (size_t idx = 0; idx < dataLayers[0].size(); ++idx) {
      dataLayers[0][idx]->getOutputValue()->randomizeUniform();
      dataLayers[1][idx]->getOutputValue()->copyFrom(
        *(dataLayers[0][idx]->getOutputValue()));
      VLOG(lvl) << "Forward Input " << idx << " BotData: ";
      printMatrix(dataLayers[0][idx]->getOutputValue(), lvl);
    }
    // forward
    testLayer[0]->forward(PASS_TRAIN);
    testLayer[1]->forward(PASS_TRAIN);

    // print foward output
    for (int n = 0; n < 2; ++n) {
      VLOG(lvl) << configs[n].layerConfig.type() << " forward output TopData: ";
      printMatrix(testLayer[n]->getOutputValue(), lvl);
    }

    // random test layer topdiff, copy same to ref
    testLayer[0]->getOutputGrad()->randomizeUniform();
    testLayer[1]->getOutputGrad()->copyFrom(*(testLayer[0]->getOutputGrad()));
    
    VLOG(lvl) << "Backward Input TopDiff: ";
    printMatrix(testLayer[0]->getOutputGrad());

    // backward
    testLayer[0]->backward(nullptr);
    testLayer[1]->backward(nullptr);

    // Get compared delta of forward outputs and backward outputs
    // FWD: topdata, BWD: botdiff, weight and bias
    deltaFwd.push_back(compareMatrix(
      testLayer[0]->getOutputValue(),
      testLayer[1]->getOutputValue()));
    for (size_t idx = 0; idx < dataLayers[0].size(); ++idx) {
      const MatrixPtr& diffA = dataLayers[0][idx]->getOutputGrad();
      const MatrixPtr& diffB = dataLayers[1][idx]->getOutputGrad();
      deltaBwd.push_back(compareMatrix(diffA, diffB));
      VLOG(lvl) << "Mkldnn Backward Output " << idx << " BotDiff: ";
      printMatrix(dataLayers[0][idx]->getOutputGrad(), lvl);
      VLOG(lvl) << "Reference Backward Output " << idx << " BotDiff: ";
      printMatrix(dataLayers[1][idx]->getOutputGrad(), lvl);
      if (isBN) {
        // the other two inputs in batch norm are for moving mean and var
        break;
      }
    }
    testLayer[0]->cvtWgtToPaddle();
    for (size_t idx = 0; idx < parameters[0].size(); ++idx) {
      const VectorPtr& tgt = parameters[0][idx]->getBuf(PARAMETER_VALUE);
      const VectorPtr& ref = parameters[1][idx]->getBuf(PARAMETER_VALUE);
      deltaParam.push_back(compareVector(tgt, ref));

      VLOG(lvl) << "Mkldnn Output " << parameters[0][idx]->getName() << ":";
      printVector(tgt, lvl);
      VLOG(lvl) << "Reference Output " << parameters[1][idx]->getName() << ":";
      printVector(ref, lvl);
    }

    // clear topdata
    testLayer[0]->getOutputValue()->zeroMem();
    testLayer[1]->getOutputValue()->zeroMem();
    // clear botdiff
    for (size_t idx = 0; idx < dataLayers[0].size(); ++idx) {
      dataLayers[0][idx]->getOutputGrad()->zeroMem();
      dataLayers[1][idx]->getOutputGrad()->zeroMem();
    }
    // clear param diff
    for (size_t idx = 0; idx < parameters[0].size(); ++idx) {
      const VectorPtr& gradA = parameters[0][idx]->getBuf(PARAMETER_GRADIENT);
      const VectorPtr& gradB = parameters[1][idx]->getBuf(PARAMETER_GRADIENT);
      if (gradA) {
        gradA->zeroMem();
      }
      if (gradB) {
        gradB->zeroMem();
      }
    }
  }

  // check Fwd delta
  for (size_t i = 0; i < deltaFwd.size(); ++i) {
    VLOG(DNN_TESTS_DETAILS) << "Check iter " << i << ", Top data";
    EXPECT_LE(fabs(deltaFwd[i]), epsilon);
  }
  // check Bwd delta
  for (size_t i = 0; i < deltaBwd.size(); ++i) {
    VLOG(DNN_TESTS_DETAILS) << "Check iter " << i << ", Bot"
      << (!isBN ? i % dataLayers[0].size() : 0) << " diff";
    EXPECT_LE(fabs(deltaBwd[i]), epsilon);
  }
  // check Param delta
  for (size_t i = 0; i < deltaParam.size(); ++i) {
    VLOG(DNN_TESTS_DETAILS) << "Check iter " << i << ", "
      << parameters[0][i % parameters[0].size()]->getName();
    EXPECT_LE(fabs(deltaParam[i]), epsilon);
  }
}


}  //  namespace paddle
