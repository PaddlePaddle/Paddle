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

#include "MkldnnTester.h"
#include "paddle/gserver/layers/MkldnnBase.h"

namespace paddle {

// init data layer and test layer of both dnn and reference
void MkldnnTester::reset(
  const TestConfig& dnn, const TestConfig& ref, size_t batchSize) {
  const bool trans = false;
  const bool useGpu = false;

  // clear
  configs_.clear();
  layerNames_.clear();
  dataLayers_.clear();
  datas_.clear();
  layerMaps_.clear();
  parameters_.clear();
  testLayers_.clear();

  // resize
  configs_.resize(NUM);
  layerNames_.resize(NUM);
  dataLayers_.resize(NUM);
  datas_.resize(NUM);
  layerMaps_.resize(NUM);
  parameters_.resize(NUM);
  testLayers_.resize(NUM);

  // reset configs and layer names
  configs_[DNN] = dnn;
  configs_[REF] = ref;
  layerNames_[DNN] = "mkldnn";  // the first is mkldnn layer
  layerNames_[REF] = "reference";  // second is reference layer

  // reset others
  for (size_t i = 0; i < NUM; ++i) {
    configs_[i].layerConfig.set_name(layerNames_[i]);
    initDataLayer(configs_[i], &(dataLayers_[i]), &(datas_[i]), &(layerMaps_[i]),
      layerNames_[i], batchSize, trans, useGpu);
    initTestLayer(configs_[i], &(layerMaps_[i]), &(parameters_[i]),
      &(testLayers_[i]));
  }
  dnnLayer_ = testLayers_[DNN];
  refLayer_ = testLayers_[REF];
  EXPECT_EQ(dataLayers_[DNN].size(), dataLayers_[REF].size());
  EXPECT_EQ(parameters_[DNN].size(), parameters_[REF].size());
  
  setInputImgSize();
}

void MkldnnTester::setInputImgSize() {
  for (size_t n = 0; n < dataLayers_.size(); ++n) {
     for (size_t i = 0; i < dataLayers_[n].size(); ++i) {
      // TODO(TJ): fix me when concat and elewise ready
      dataLayers_[n][i]->getOutput().setFrameHeight(ih_);
      dataLayers_[n][i]->getOutput().setFrameWidth(iw_);
    }
  }
}

// init randome parameters of ref, and copy to mkldnn
void MkldnnTester::randomWgtDatas() {
  EXPECT_EQ(parameters_[DNN].size(), parameters_[REF].size());
  for (size_t i = 0; i < parameters_[REF].size(); ++i) {
    const VectorPtr& dnnValue = parameters_[DNN][i]->getBuf(PARAMETER_VALUE);
    const VectorPtr& refValue = parameters_[REF][i]->getBuf(PARAMETER_VALUE);
    parameters_[REF][i]->randomize();
    dnnValue->copyFrom(*refValue);

    VLOG(lvl_) << "Random weight data " << parameters_[DNN][i]->getName();
    printVector(dnnValue);
  }
}

// random botdata of ref layer and copy same to mkldnn
void MkldnnTester::randomBotDatas() {
  CHECK_EQ(dataLayers_.size(), NUM);
  for (size_t i = 0; i < dataLayers_[DNN].size(); ++i) {
    dataLayers_[REF][i]->getOutputValue()->randomizeUniform();
    dataLayers_[DNN][i]->getOutputValue()->copyFrom(
      *(dataLayers_[REF][i]->getOutputValue()));
    VLOG(lvl_) << "Input " << i << " data:";
    printMatrix(dataLayers_[REF][i]->getOutputValue());
  }
}

void MkldnnTester::randomTopDiffs() {
  refLayer_->getOutputGrad()->randomizeUniform();
  dnnLayer_->getOutputGrad()->copyFrom(*(refLayer_->getOutputGrad()));
  VLOG(lvl_) << "Random dom Backward Input, TopDiff: ";
  printMatrix(refLayer_->getOutputGrad());
}

void MkldnnTester::checkForward() {
  printTopDatas();
  double delta = compareMatrix(
        testLayers_[DNN]->getOutputValue(),
        testLayers_[REF]->getOutputValue());
  VLOG(DNN_TESTS_DETAILS) << "Check Forward";
  EXPECT_LE(fabs(delta), eps_);
}

void MkldnnTester::checkBackwardData() {
  const bool isBN = dnnLayer_->getType() == "mkldnn_batch_norm";
  for (size_t i = 0; i < dataLayers_[DNN].size(); ++i) {
    const MatrixPtr& dnnDiff = dataLayers_[DNN][i]->getOutputGrad();
    const MatrixPtr& refDiff = dataLayers_[REF][i]->getOutputGrad();
    VLOG(lvl_) << "Mkldnn Backward Output BotDiff " << i;
    printMatrix(dnnDiff);
    VLOG(lvl_) << "Reference Backward Output BotDiff " << i;
    printMatrix(refDiff);
    
    double delta = compareMatrix(dnnDiff, refDiff);
    EXPECT_LE(fabs(delta), eps_);
    if (isBN) {
      // the other two inputs in batch norm are for moving mean and var
      break;
    }
  }
}

void MkldnnTester::checkBackwardWgts() {
  CHECK_EQ(parameters_[DNN].size(), parameters_[REF].size());
  vector<VectorPtr> dnnWgts;  // used to temply save mkldnn weights
  saveWgt(parameters_[DNN], dnnWgts);

  dnnLayer_->cvtWgtToPaddle();
  for (size_t i = 0; i < parameters_[DNN].size(); ++i) {
    const VectorPtr& dnn = parameters_[DNN][i]->getBuf(PARAMETER_VALUE);
    const VectorPtr& ref = parameters_[REF][i]->getBuf(PARAMETER_VALUE);
    VLOG(lvl_) << "Mkldnn Output weight " << parameters_[DNN][i]->getName();
    printVector(dnn);
    VLOG(lvl_) << "Reference Output weight " << parameters_[REF][i]->getName();
    printVector(ref);

    double delta = compareVector(dnn, ref);
    EXPECT_LE(fabs(delta), eps_);
  }

  VLOG(DNN_TESTS_DETAILS) << "Restore dnn weights before comapre";
  restoreWgt(dnnWgts, parameters_[DNN]);
}

void MkldnnTester::saveWgt(
  const vector<ParameterPtr>& from, vector<VectorPtr>& to) {
  const bool useGpu = false;
  to.resize(from.size());
  for (size_t i = 0; i < to.size(); ++i) {
    const VectorPtr& wgt = from[i]->getBuf(PARAMETER_VALUE);
    to[i] = Vector::create(wgt->getSize(), useGpu);
    to[i]->copyFrom(*wgt);
  }
}

void MkldnnTester::restoreWgt(
  const vector<VectorPtr>& from, vector<ParameterPtr>& to) {
  CHECK_EQ(from.size(), to.size());
  for (size_t i = 0; i < from.size(); ++i) {
    const VectorPtr& wgt = to[i]->getBuf(PARAMETER_VALUE);
    wgt->copyFrom(*from[i]);
  }
}


// clear parameters grad
void MkldnnTester::clearWgtDiffs() {
  for (size_t n = 0; n < parameters_.size(); ++n) {
    for (size_t i = 0; i < parameters_[n].size(); ++i) {
      const VectorPtr& grad = parameters_[n][i]->getBuf(PARAMETER_GRADIENT);
      if (grad) {
        grad->zeroMem();
      }
    }
  }
}

void MkldnnTester::clearBotDiffs() {
  // dnn and ref
  for (size_t n = 0; n < dataLayers_.size(); ++n) {
    // all inputs layers
    for (size_t i = 0; i < dataLayers_[n].size(); ++i) {
      dataLayers_[n][i]->getOutputGrad()->zeroMem();
    }
  }
}

void MkldnnTester::clearBotDiffs(int n) {
  CHECK_LT(n, NUM);
  // all inputs layers
  for (size_t i = 0; i < dataLayers_[n].size(); ++i) {
    dataLayers_[n][i]->getOutputGrad()->zeroMem();
  }
}

void MkldnnTester::clearTopDatas() {
  for (size_t i = 0; i < testLayers_.size(); ++i) {
    testLayers_[i]->getOutputValue()->zeroMem();
  }
}

void MkldnnTester::printTopDatas() {
  for (int n = 0; n < NUM; ++n) {
    VLOG(lvl_) << testLayers_[n]->getType() << " forward output TopData: ";
    printMatrix(testLayers_[n]->getOutputValue());
  }
}

void MkldnnTester::printMatrix(const MatrixPtr& m) {
  const real* pd = m->getData();
  const int width = m->getWidth();
  const int height = m->getHeight();
  for(int h = 0; h < height; ++h) {
    std::stringstream row;
    for (int w = 0; w < width; ++w) {
      row << pd[width * h + w] << ", ";
    }
    VLOG(lvl_) << row.str();
  }
}

void MkldnnTester::printVector(const VectorPtr& v) {
  const real* pd = v->getData();
  const int sz = v->getSize();
  std::stringstream row;
  for(int i = 0; i < sz; ++i) {
    row << pd[i] << ", ";
  }
  VLOG(lvl_) << row.str();
}

double MkldnnTester::getDelta(const real* d1, const real* d2, size_t len,
  const float failRate, const float thres) {
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

double MkldnnTester::compareMatrix(const MatrixPtr& m1, const MatrixPtr& m2) {
  CHECK_EQ(m1->getElementCnt(), m2->getElementCnt());
  return getDelta(m1->getData(), m2->getData(), m1->getElementCnt());
}

double MkldnnTester::compareVector(const VectorPtr& v1, const VectorPtr& v2) {
  CHECK_EQ(v1->getSize(), v2->getSize());
  return getDelta(v1->getData(), v2->getData(), v1->getSize());
}

void MkldnnTester::runOnce() {
  // test forward
  randomBotDatas();
  dnnLayer_->forward(PASS_TRAIN);
  refLayer_->forward(PASS_TRAIN);
  checkForward();
  
  // test backward
  randomTopDiffs();
  dnnLayer_->backward(nullptr);
  refLayer_->backward(nullptr);
  checkBackwardData();
  checkBackwardWgts();

  // clear buffers
  // ref code will addto the diff, dnn code will writeto it
  clearBotDiffs(REF);
  // below two should be coverd by test layers
  // clearTopDatas();
  // clearWgtDiffs();
}

void MkldnnTester::run(
  const TestConfig& dnn, const TestConfig& ref, size_t batchSize,
  size_t inputImgH, size_t inputImgW, size_t iter, float epsilon, int level) {
  VLOG(DNN_TESTS) << "Test MKLDNN functionality: "
      << dnn.layerConfig.type() << " vs " << ref.layerConfig.type();
  ih_ = inputImgH;
  iw_ = inputImgW;
  iter_ = iter;
  eps_ = epsilon;
  lvl_ = level;

  // Firstly always set flag false to initial from paddle weight
  TestConfig first = dnn;
  first.layerConfig.set_init_wgt_from_mkldnn(false);

  // reset and run once
  reset(first, ref, batchSize);
  randomWgtDatas();
  clearWgtDiffs();
  clearBotDiffs();

  VLOG(DNN_TESTS) << "Check Iteration 0";
  runOnce();

  // firstly get the flag
  bool initWgtFromMkldnn = dnn.layerConfig.has_init_wgt_from_mkldnn()
    && dnn.layerConfig.init_wgt_from_mkldnn();
  if (initWgtFromMkldnn) {
    // after run once the mkldnn weight has been stored in dnnlayer
    // then save the weigths and restart again
    vector<VectorPtr> dnnWgts, refWgts;
    CHECK_EQ(parameters_[DNN].size(), parameters_[REF].size());
    saveWgt(parameters_[DNN], dnnWgts);
    saveWgt(parameters_[REF], refWgts);

    // restart again with flag true
    reset(dnn, ref, batchSize);

    // restore wgt
    restoreWgt(dnnWgts, parameters_[DNN]);
    restoreWgt(refWgts, parameters_[REF]);
    clearWgtDiffs();
    clearBotDiffs();

    // at least run once
    runOnce();
  }

  for (size_t i = 1; i < iter_; ++i) {
    VLOG(DNN_TESTS) << "Check Iteration " << i;
    runOnce();
  }
}

}  //  namespace paddle
