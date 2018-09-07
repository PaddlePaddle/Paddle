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

#pragma once
#include "ModelConfig.pb.h"
#include "paddle/gserver/layers/DataLayer.h"

#include "paddle/testing/TestUtil.h"
using namespace std;  // NOLINT

namespace paddle {
enum InputType {
  INPUT_DATA,         // dense vector
  INPUT_LABEL,        // id
  INPUT_DATA_TARGET,  // dense vector, but no gradient
  INPUT_SEQUENCE_DATA,
  INPUT_HASSUB_SEQUENCE_DATA,  // sequence has sub-sequence
  INPUT_SEQUENCE_MDIM_DATA,
  INPUT_SEQUENCE_LABEL,
  INPUT_SPARSE_NON_VALUE_DATA,
  INPUT_SPARSE_FLOAT_VALUE_DATA,
  INPUT_DENSE_DIM_DATA,    // using sequence length to init dense data
  INPUT_SELF_DEFINE_DATA,  // support customizing for input value
};

struct ParaSparse {
  bool sparse;
  string format;
  // if equalNnzPerSample is set true,
  // every row of the sparse matrix in a format of CSR has a same
  // number of nnz values. Currently, this flag is only used for
  // selective_fc layer
  bool equalNnzPerSample;
  ParaSparse(const string& formatIn = "") {  // NOLINT
    if (formatIn == "") {
      sparse = false;
    } else {
      sparse = true;
    }
    equalNnzPerSample = false;
  }
  ParaSparse(const string& formatIn, bool equalNnz) {
    format = formatIn;
    sparse = true;
    equalNnzPerSample = equalNnz;
  }
};

struct InputDef {
  InputType inputType;
  string name;
  size_t dim;
  size_t paraSize;
  ParaSparse sparse;
  bool isStatic;
  std::vector<int> labelInitValue;
  std::vector<int> labelSeqStartPositions;
  std::vector<int> labelSubSeqStartPositions;
  std::vector<int> ids;
  MatrixPtr selfDefinedData;

  InputDef(InputType type, string nameIn, size_t dimIn, size_t sizeIn) {
    inputType = type;
    name = nameIn;
    dim = dimIn;
    paraSize = sizeIn;
    sparse = {""};
    isStatic = false;
  }

  InputDef(InputType type,
           string nameIn,
           MatrixPtr selfDefinedData,
           std::vector<int> selfDefinedSeqStartPos = {},
           std::vector<int> selfDefinedSubSeqStartPos = {})
      : labelSeqStartPositions(selfDefinedSeqStartPos),
        labelSubSeqStartPositions(selfDefinedSubSeqStartPos),
        selfDefinedData(selfDefinedData) {
    inputType = type;
    name = nameIn;
    dim = 0;
    sparse = {""};
    paraSize = 0;
    isStatic = false;
  }

  InputDef(InputType type,
           string nameIn,
           const std::vector<int>& ids,
           const std::vector<int>& selfDefinedSeqStartPos = {},
           const std::vector<int>& selfDefinedSubSeqStartPos = {})
      : labelSeqStartPositions(selfDefinedSeqStartPos),
        labelSubSeqStartPositions(selfDefinedSubSeqStartPos),
        ids(ids) {
    selfDefinedData = nullptr;
    inputType = type;
    name = nameIn;
    dim = 0;
    sparse = {""};
    paraSize = 0;
    isStatic = false;
  }

  InputDef(InputType type,
           string nameIn,
           size_t dimIn,
           size_t sizeIn,
           const std::vector<int>& labelInitValue,
           const std::vector<int>& labelSeqStartPositions)
      : labelInitValue(labelInitValue),
        labelSeqStartPositions(labelSeqStartPositions) {
    inputType = type;
    name = nameIn;
    dim = dimIn;
    paraSize = sizeIn;
    sparse = {""};
    isStatic = false;
  }

  InputDef(InputType type,
           string nameIn,
           size_t dimIn,
           size_t sizeIn,
           ParaSparse sparseIn) {
    inputType = type;
    name = nameIn;
    dim = dimIn;
    paraSize = sizeIn;
    sparse = sparseIn;
  }
};

struct TestConfig {
  LayerConfig layerConfig;
  std::vector<InputDef> inputDefs;
  size_t biasSize;
  real paramInitialMean;
  real paramInitialStd;
  bool testAccumulate;
  bool testState;
  bool staticBias;
  bool testBatchState;
  TestConfig()
      : biasSize(0),
        paramInitialMean(0.0),
        paramInitialStd(1.0),
        testAccumulate(true),
        testState(false),
        staticBias(false),
        testBatchState(false) {}
};

real getCostSum(ParameterPtr& parameter,
                CpuVector& cpuPara,
                LayerPtr& testLayer,
                MatrixPtr weights = nullptr);

real getDiffAndPrint(real newCost1,
                     real newCost2,
                     real callbackCount,
                     char fill,
                     string testLayerName,
                     string name,
                     real step,
                     real delta);

/**
 * @brief verify that sequentially running forward() one timestamp at one time
 *        has same result as running forward() with one whole sequence
 *
 * @param testLayer[in/out]    testLayer
 * @param dataLayers[in/out]   dataLayers
 * @param datas[in/out]        data of dataLayers
 */
void testState(LayerPtr testLayer,
               vector<DataLayerPtr>& dataLayers,
               vector<Argument>& datas);

/**
 * @brief verify that sequentially running forward() with short sequences one
 *        time has same result as running forward() with long sequences.
 *
 * @param testLayer[in/out]    testLayer
 * @param dataLayers[in/out]   dataLayers
 * @param datas[in/out]        data of dataLayers
 */
void testBatchState(LayerPtr testLayer,
                    vector<DataLayerPtr>& dataLayers,
                    vector<Argument>& datas);

/**
 * @brief Generate a perturbation so that it is roughly aligned with the
 *        gradient direction. This is to make sure that change along this
 *        direction will make cost increase (or decrease) in a meaningful
 *        way so that the finite difference can be used to approximate the
 *        directional dirivative well.
 *
 * @param oldGrad[in]  input gradient
 *        newGrad[out] output gradient
 *        dim          dimension of oldGrad/newGrad
 *
 * @return sum_i(oldGrad[i] * newGrad[i])
 */
double genPerturbation(const real* oldGrad, real* newGrad, size_t dim);

void initWeight(MatrixPtr& weights);

void initBatchState(LayerPtr dataLayer,
                    LayerPtr testLayer,
                    LayerStatePtr state,
                    bool useGpu);

/**
 * @brief initialize the dataLayer by its inputType
 *
 * @param testConf[in]        test config
 *        dataLayers[out]     dataLayers
 *        datas[out]          initialized data of dataLayers
 *        layerMap[out]       layerMap
 */
void initDataLayer(TestConfig testConf,
                   std::vector<DataLayerPtr>* dataLayers,
                   vector<Argument>* datas,
                   LayerMap* layerMap,
                   string testLayerName,
                   size_t batchSize,
                   bool trans,
                   bool useGpu);

/**
 * @brief initialize the parameter of testLayer
 *
 * @param testConf[in/out]    test config
 *        layerMap[out]       layerMap
 *        parameters[out]     parameters of testLayer
 *        testLayer[out]      testLayer
 */
void initTestLayer(TestConfig testConf,
                   LayerMap* layerMap,
                   std::vector<ParameterPtr>* parameters,
                   LayerPtr* testLayer);

/**
 * @brief Test whether the layer's forward calculation is stable by adding
 *        perturbation to its parameters
 *
 * @param testConf[in]         test config
 *        weights[in]          weights of testLayer
 *        state[in]            state of testLayer
 *        cost[in]             input cost
 *        callbackCount[in]    number of done callback
 *        maxDiff[in/out]      max of all previous diff
 *        testLayer[in/out]    testLayer
 *        parameters[in/out]   parameters of testLayer
 */
void testPerturbParameter(TestConfig testConf,
                          const MatrixPtr weights,
                          const LayerStatePtr state,
                          real cost,
                          real callbackCount,
                          real* maxDiff,
                          LayerPtr testLayer,
                          std::vector<ParameterPtr>* parameters);

/**
 * @brief Test whether the layer's forward calculation is stable by adding
 *        perturbation to its input layers
 *
 * @param testConf[in]         test config
 *        weights[in]          weights of testLayer
 *        state[in]            state of testLayer
 *        cost[in]             input cost
 *        callbackCount[in]    number of done callback
 *        maxDiff[in/out]      max of all previous diff
 *        testLayer[in/out]    testLayer
 *        dataLayers[in/out]   dataLayers
 */
void testPerturbInput(TestConfig testConf,
                      const MatrixPtr weights,
                      const LayerStatePtr state,
                      real cost,
                      real callbackCount,
                      real* maxDiff,
                      LayerPtr testLayer,
                      std::vector<DataLayerPtr> dataLayers);

void testLayerGradKernel(TestConfig testConf,
                         string testLayerName,
                         size_t batchSize,
                         bool trans,
                         bool useGpu,
                         bool useWeight = false,
                         float epsilon = 0.02);

void testLayerGrad(TestConfig testConf,
                   string testLayerName,
                   size_t batchSize,
                   bool trans,
                   bool useGpu,
                   bool useWeight = false,
                   float epsilon = 0.02);

void testProjectionGrad(ProjectionConfig conf,
                        InputType inputType,
                        size_t parameterSize,
                        size_t batchSize,
                        bool useGpu,
                        bool testState = false,
                        int biasSize = 0,
                        bool sharedBias = false);

void testOperatorGrad(TestConfig& config,
                      OperatorConfig& operatorConf,
                      size_t batchSize,
                      bool useGpu,
                      bool testState = false);

}  //  namespace paddle
