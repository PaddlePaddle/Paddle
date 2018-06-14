/* Copyright (c) 2017 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <string>
#include <vector>
#include "LayerGradUtil.h"
#include "paddle/gserver/layers/MKLDNNBase.h"
#include "paddle/gserver/layers/MKLDNNLayer.h"

namespace paddle {

/**
 * @brief test the functionality of MKLDNNlayers and MKLDNNActivations
 * refer to paddle original function
 */
class MKLDNNTester {
  enum {
    DNN = 0,  // MKLDNN layer
    REF = 1,  // Reference layer
    NUM = 2,  // Number of total
  };

  struct DataIn {
    std::vector<std::vector<Argument>> inArgs;
    std::vector<std::vector<MatrixPtr>> outGrads;
    std::vector<VectorPtr> paraValues;
  };

  struct DataOut {
    std::vector<MatrixPtr> outValues;
    std::vector<VectorPtr> paraValues;
  };

 protected:
  std::vector<TestConfig> configs_;
  vector<string> layerNames_;
  vector<vector<DataLayerPtr>> dataLayers_;
  vector<vector<Argument>> datas_;
  vector<LayerMap> layerMaps_;
  vector<vector<ParameterPtr>> parameters_;
  vector<LayerPtr> testLayers_;
  LayerPtr refLayer_, dnnLayer_;

  /// run some iterations, all the result should pass
  size_t iter_;
  /// whether to print out the details
  bool log_;
  /// epsilon
  float eps_;
  /// input image size, default 1
  size_t ih_, iw_;
  /// passType, PASS_TRAIN, PASS_TEST or PASS_GC (Gradient Check pass)
  PassType passType_;

 public:
  explicit MKLDNNTester(size_t iter = 3, float epsilon = 1e-4) {
    iter_ = iter;
    eps_ = epsilon;
    log_ = false;
    passType_ = PASS_TRAIN;
  }

  ~MKLDNNTester() {}

 public:
  void run(const TestConfig& dnn,
           const TestConfig& ref,
           size_t batchSize,
           size_t inputImgH = 1,
           size_t inputImgW = 1,
           PassType passType = PASS_TRAIN,
           bool printDetails = false,
           size_t iter = 3,
           float epsilon = 1e-4);
  static void runNetTest(const std::string& configPath,
                         size_t iter = 2,
                         float eps = 1e-4);
  static void initArgument(DataIn& data,
                           const std::string& configPath,
                           size_t iter = 2);
  static void getOutResult(const std::string& configPath,
                           DataIn& in,
                           DataOut& out,
                           bool use_mkldnn,
                           size_t iter = 2);

 private:
  void reset(const TestConfig& dnn, const TestConfig& ref, size_t batchSize);
  void setInputImgSize();
  void runOnce();

  void randomWgtDatas();
  void randomBotDatas();
  void randomTopDiffs();

  void checkForward();
  void checkBackwardData();
  void checkBackwardWgts();

  // clear specific layer, clear all when id equals NUM
  void clearWgtDiffs(size_t id = NUM);
  void clearBotDiffs(size_t id = NUM);
  void clearTopDatas(size_t id = NUM);

  void printTopDatas();
  void printMatrix(const MatrixPtr& m);
  void printVector(const VectorPtr& v);

  void saveWgt(const vector<ParameterPtr>& from, vector<VectorPtr>& to);
  void restoreWgt(const vector<VectorPtr>& from, vector<ParameterPtr>& to);

  static double compareMatrix(const MatrixPtr& m1, const MatrixPtr& m2);
  static double compareVector(const VectorPtr& v1, const VectorPtr& v2);
  static void compareResult(DataOut& ref, DataOut& dnn, float eps = 1e-4);

  /**
   * Get delta percent
   * if many(>failRate) wrong(abs(val-ref)/abs(ref) > thres) points
   * return the max(diff/ref)
   * else return sum(abs(diff)) / sum(abs(ref))
   * The return value should be smaller than eps when passing.
   */
  static double getDelta(const real* refer,
                         const real* value,
                         size_t len,
                         const float failRate = 1e-3,
                         const float thres = 0.1);
};

}  //  namespace paddle
