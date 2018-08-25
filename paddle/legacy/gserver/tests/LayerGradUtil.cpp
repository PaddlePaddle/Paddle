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

#include "LayerGradUtil.h"

DECLARE_bool(thread_local_rand_use_global_seed);

namespace paddle {
real getCostSum(LayerPtr& testLayer, MatrixPtr weights) {
  testLayer->forward(PASS_GC);
  std::vector<Argument> outArgs;
  outArgs.push_back(testLayer->getOutput());
  if (weights) {
    outArgs[0].value->dotMul(*outArgs[0].value, *weights);
  }
  return Argument::sum(outArgs);
}

real getDiffAndPrint(real newCost1,
                     real newCost2,
                     real callbackCount,
                     char fill,
                     string testLayerName,
                     string name,
                     real step,
                     real delta) {
  EXPECT_FALSE(std::isnan(newCost1));
  EXPECT_FALSE(std::isnan(newCost2));

  real trueDelta = (newCost1 - newCost2) * (callbackCount / 2.);
  real diff = (1e-20 + trueDelta) / (1e-20 + delta) - 1;
  LOG(INFO) << setiosflags(ios::left) << setfill(fill) << setw(20)
            << testLayerName << " " << setw(20) << name << "step=" << setw(15)
            << step << "cost1=" << setw(10) << newCost1 << "cost2=" << setw(10)
            << newCost2 << "true_delta=" << setw(15) << trueDelta
            << "analytic_delta=" << setw(15) << delta << "diff=" << diff
            << (abs(diff) > 0.01 ? " ***" : "");
  if (fabs(diff - 1) < 0.02) {
    LOG(INFO) << "The previous diff might be caused by not accumulating"
              << " parameter gradients in backward()";
  }
  return diff;
}

void testState(LayerPtr testLayer,
               vector<DataLayerPtr>& dataLayers,
               vector<Argument>& datas) {
  auto batchSize = datas[0].getBatchSize();
  Argument data;
  ICpuGpuVectorPtr sequenceStartPositions =
      ICpuGpuVector::create(2, /* useGpu= */ false);
  sequenceStartPositions->getMutableData(false)[0] = 0;
  sequenceStartPositions->getMutableData(false)[1] = batchSize;
  data.sequenceStartPositions = sequenceStartPositions;
  testLayer->resetState();
  for (size_t j = 0; j < datas.size(); ++j) {
    if (datas[j].value) {
      data.value = datas[j].value;
    }
    if (datas[j].ids) {
      data.ids = datas[j].ids;
    }
    dataLayers[j]->setData(data);
    dataLayers[j]->forward(PASS_TEST);
  }
  testLayer->forward(PASS_TEST);
  Argument batchOut;
  batchOut.resizeAndCopyFrom(testLayer->getOutput(), /* useGpu= */ false);

  sequenceStartPositions->getMutableData(false)[1] = 1;
  testLayer->resetState();

  auto testLayerState = [&](int batchId) {
    for (size_t j = 0; j < datas.size(); ++j) {
      if (datas[j].value) {
        data.value = datas[j].value->subMatrix(batchId, 1);
      }
      if (datas[j].ids) {
        data.ids = IVector::create(
            datas[j].ids->getData() + batchId, 1, FLAGS_use_gpu);
      }
      dataLayers[j]->setData(data);
      dataLayers[j]->forward(PASS_TEST);
    }

    testLayer->forward(PASS_TEST);
    Argument out;
    out.resizeAndCopyFrom(testLayer->getOutput(), /* useGpu= */ false);
    if (batchOut.value) {
      size_t dim = batchOut.value->getWidth();
      ASSERT_TRUE((bool)out.value);
      EXPECT_EQ(dim, out.value->getWidth());
      EXPECT_EQ(1UL, out.value->getHeight());
      auto ret = std::mismatch(batchOut.value->getData() + batchId * dim,
                               batchOut.value->getData() + (batchId + 1) * dim,
                               out.value->getData());
      if (ret.second != out.value->getData() + dim) {
        // If reaches here, the test will fail
        EXPECT_EQ(*ret.first, *ret.second);
      }
    } else if (batchOut.ids) {
      ASSERT_TRUE((bool)out.ids);
      EXPECT_EQ(1UL, out.ids->getSize());
      EXPECT_EQ(batchOut.ids->getElement(batchId), out.ids->getElement(0));
    }
  };

  CHECK_GT(batchSize, 0);
  std::vector<LayerStatePtr> statePtrs;
  statePtrs.reserve(batchSize);

  // Test layer setState() and getState()
  for (int i = 0; i < batchSize; ++i) {
    statePtrs.push_back(testLayer->getState());
    testLayerState(i);
  }
  for (int k = 0; k < batchSize - 1; ++k) {
    testLayer->setState(statePtrs[k]);
    for (int i = k; i < batchSize; ++i) {
      testLayerState(i);
    }
  }
}

void testBatchState(LayerPtr testLayer,
                    vector<DataLayerPtr>& dataLayers,
                    vector<Argument>& datas) {
  auto batchSize = datas[0].getBatchSize();
  Argument data;
  /*two sequences*/
  size_t numSequences = 2;
  ICpuGpuVectorPtr sequenceStartPositions =
      ICpuGpuVector::create(numSequences + 1, /* useGpu= */ false);
  int* cpuStarts = sequenceStartPositions->getMutableData(false);
  int len = ::rand() % (batchSize - 1);
  cpuStarts[0] = 0;
  cpuStarts[1] = len > 0 ? len : 1;
  cpuStarts[2] = batchSize;

  data.sequenceStartPositions = sequenceStartPositions;
  for (size_t j = 0; j < datas.size(); ++j) {
    if (datas[j].value) {
      data.value = datas[j].value;
    }
    if (datas[j].ids) {
      data.ids = datas[j].ids;
    }
    dataLayers[j]->setData(data);
    dataLayers[j]->forward(PASS_TEST);
  }
  testLayer->resetState();
  testLayer->forward(PASS_TEST);
  Argument batchOut;
  batchOut.resizeAndCopyFrom(testLayer->getOutput(), /* useGpu= */ false);

  /*split one miniBatch into two miniBatchs*/
  std::vector<int> seqSplitPos;
  for (size_t seqId = 0; seqId < numSequences; ++seqId) {
    int len = ::rand() % (cpuStarts[seqId + 1] - cpuStarts[seqId]);
    len = len > 0 ? len : 1;
    seqSplitPos.push_back(cpuStarts[seqId] + len);
  }

  std::vector<int> start; /*seq start pos in source data*/
  for (size_t seqId = 0; seqId < numSequences; ++seqId) {
    start.push_back(cpuStarts[seqId]);
  }
  testLayer->resetState();
  Argument splitData;
  for (size_t batchId = 0; batchId < 2; ++batchId) {
    size_t splitBatchSize = 0;
    std::vector<int> seqLens;
    for (size_t seqId = 0; seqId < numSequences; ++seqId) {
      int seqLen = (batchId == 0) ? seqSplitPos[seqId] - cpuStarts[seqId]
                                  : cpuStarts[seqId + 1] - seqSplitPos[seqId];
      seqLens.push_back(seqLen);
      splitBatchSize += seqLen;
    }
    ICpuGpuVectorPtr cpuSeqStartPos =
        ICpuGpuVector::create(3, /* useGpu= */ false);
    int* seqStartPosData = cpuSeqStartPos->getMutableData(false);
    seqStartPosData[0] = 0;
    seqStartPosData[1] = seqLens[0];
    seqStartPosData[2] = splitBatchSize;

    CHECK_GT(splitBatchSize, size_t(0));
    splitData.sequenceStartPositions = cpuSeqStartPos;
    for (size_t j = 0; j < datas.size(); ++j) {
      if (datas[j].value) {
        Matrix::resizeOrCreate(splitData.value,
                               splitBatchSize,
                               datas[j].value->getWidth(),
                               false,
                               FLAGS_use_gpu);
        for (size_t seqId = 0; seqId < numSequences; ++seqId) {
          if (seqLens[seqId]) {
            splitData.value->subMatrix(seqStartPosData[seqId], seqLens[seqId])
                ->copyFrom(
                    *datas[j].value->subMatrix(start[seqId], seqLens[seqId]));
          }
        }
      }
      if (datas[j].ids) {
        IVector::resizeOrCreate(splitData.ids, splitBatchSize, FLAGS_use_gpu);
        for (size_t seqId = 0; seqId < numSequences; ++seqId) {
          if (seqLens[seqId]) {
            splitData.ids->subVec(seqStartPosData[seqId], seqLens[seqId])
                ->copyFrom(*datas[j].ids->subVec(start[seqId], seqLens[seqId]));
          }
        }
      }
      dataLayers[j]->setData(splitData);
      dataLayers[j]->forward(PASS_TEST);
    }

    testLayer->forward(PASS_TEST);
    Argument out;
    out.resizeAndCopyFrom(testLayer->getOutput(), /* useGpu= */ false);
    if (batchOut.value) {
      size_t dim = batchOut.value->getWidth();
      ASSERT_TRUE((bool)out.value);
      EXPECT_EQ(dim, out.value->getWidth());
      for (size_t seqId = 0; seqId < numSequences; ++seqId) {
        if (seqLens[seqId]) {
          out.value->subMatrix(seqStartPosData[seqId], seqLens[seqId])
              ->sub(*batchOut.value->subMatrix(start[seqId], seqLens[seqId]));
        }
      }
    }

    std::vector<Argument> args;
    args.push_back(out);
    ASSERT_NEAR(0, Argument::sum(args), 1e-5) << "testBatchState failed";
    for (size_t seqId = 0; seqId < numSequences; ++seqId) {
      start[seqId] += seqLens[seqId];
    }
  }
}

double genPerturbation(const real* oldGrad, real* newGrad, size_t dim) {
  double gradNorm = 0, dNorm = 0;
  for (size_t i = 0; i < dim; ++i) {
    newGrad[i] = 2. * rand() / RAND_MAX - 1;  // NOLINT
    dNorm += newGrad[i] * newGrad[i];
    gradNorm += oldGrad[i] * oldGrad[i];
  }
  if (gradNorm > 0) {
    real s = 0.5 * sqrt(gradNorm / dNorm);
    for (size_t i = 0; i < dim; ++i) {
      newGrad[i] = s * newGrad[i] + oldGrad[i];
    }
  }
  double delta = 0;
  for (size_t i = 0; i < dim; ++i) {
    delta += oldGrad[i] * newGrad[i];
  }
  return delta;
}

void initWeight(MatrixPtr& weights) {
  MatrixPtr tmpMat = weights->clone();
  for (int i = 0; i < int(tmpMat->getElementCnt()); i++) {
    tmpMat->getData()[i] = (11 - 2 * (i % 11));
  }
  weights->copyFrom(*tmpMat);
}

void initBatchState(LayerPtr dataLayer,
                    LayerPtr testLayer,
                    LayerStatePtr state,
                    bool useGpu) {
  int sequenceNum = dataLayer->getOutput().getNumSequences();
  MatrixPtr prevBatchOutput =
      Matrix::create(sequenceNum, testLayer->getSize(), false, useGpu);
  MatrixPtr prevBatchState =
      Matrix::create(sequenceNum, testLayer->getSize(), false, useGpu);
  prevBatchOutput->randomizeUniform();
  prevBatchState->randomizeUniform();
  state->value.clear();
  state->value.push_back(prevBatchOutput);
  state->value.push_back(prevBatchState);
}

void initDataLayer(TestConfig testConf,
                   std::vector<DataLayerPtr>* dataLayers,
                   vector<Argument>* datas,
                   LayerMap* layerMap,
                   string testLayerName,
                   size_t batchSize,
                   bool trans,
                   bool useGpu) {
  ICpuGpuVectorPtr sequenceStartPositions;
  ICpuGpuVectorPtr subSequenceStartPositions;
  IVectorPtr cpuSequenceDims;
  for (size_t i = 0; i < testConf.inputDefs.size(); ++i) {
    if (testConf.inputDefs[i].inputType != INPUT_SEQUENCE_LABEL) continue;

    const std::vector<int>& labelSeqStartPositions =
        testConf.inputDefs[i].labelSeqStartPositions;
    if (labelSeqStartPositions.size() != 0) {
      CHECK(!sequenceStartPositions);
      CHECK_GE(static_cast<int>(labelSeqStartPositions.size()), 2);

      sequenceStartPositions =
          ICpuGpuVector::create(labelSeqStartPositions.size(), useGpu);
      sequenceStartPositions->copyFrom(
          labelSeqStartPositions.data(), labelSeqStartPositions.size(), useGpu);
    }
  }

  for (size_t i = 0; i < testConf.inputDefs.size(); ++i) {
    LayerConfig config;
    config.set_name(testConf.inputDefs[i].name);
    config.set_type("data");
    config.set_size(testConf.inputDefs[i].dim);
    LayerPtr layer = LayerPtr(new DataLayer(config));
    size_t numSequence = sequenceStartPositions
                             ? sequenceStartPositions->getSize() - 1
                             : batchSize / 10 + 1;

    Argument data;
    auto fillData = [&](bool trans, int height, int width) {
      int newHeight = trans ? height : width;
      int newWidth = trans ? width : height;
      data.value = Matrix::create(newHeight, newWidth, false, useGpu);
      data.grad = Matrix::create(newHeight, newWidth, false, useGpu);
    };
    switch (testConf.inputDefs[i].inputType) {
      case INPUT_DATA:
      case INPUT_SEQUENCE_DATA:
      case INPUT_HASSUB_SEQUENCE_DATA:
      case INPUT_DATA_TARGET:
      case INPUT_SEQUENCE_MDIM_DATA:
        fillData(trans, layer->getSize(), batchSize);
        data.value->randomizeUniform();
        // make sure that multi-class-cross-entry won't encounter negatives
        // make sure that multi_binary_label satisfies 0~1
        data.value->add(-0.5);
        if (testLayerName != "prelu") {
          data.value->sigmoid(*data.value);
        }
        data.grad->zeroMem();
        break;
      case INPUT_LABEL:
      case INPUT_SEQUENCE_LABEL:
        if (testConf.inputDefs[i].labelInitValue.size() != 0) {
          const std::vector<int>& labelInitValue =
              testConf.inputDefs[i].labelInitValue;
          CHECK_EQ(labelInitValue.size(), batchSize);
          data.ids = VectorT<int>::create(batchSize, useGpu);
          data.ids->copyFrom(labelInitValue.data(), batchSize);
        } else {
          data.ids = VectorT<int>::create(batchSize, useGpu);
          // now rand number can be 0 to inputDefs[i].dim
          data.ids->rand(testConf.inputDefs[i].dim);
        }
        break;
      case INPUT_SPARSE_NON_VALUE_DATA:
        data.value = makeRandomSparseMatrix(
            batchSize,
            layer->getSize(),
            /* withValue= */ false,
            useGpu,
            testConf.inputDefs[i].sparse.equalNnzPerSample);
        break;
      case INPUT_SPARSE_FLOAT_VALUE_DATA:
        data.value = makeRandomSparseMatrix(batchSize,
                                            layer->getSize(),
                                            /* withValue= */ true,
                                            useGpu);
        break;
      case INPUT_DENSE_DIM_DATA:
        fillData(trans, layer->getSize(), numSequence);
        data.value->randomizeUniform();
        data.value->add(-0.5);
        data.value->sigmoid(*data.value);
        data.grad->zeroMem();
        break;
      case INPUT_SELF_DEFINE_DATA: {
        if (testConf.inputDefs[i].ids.size()) {
          data.ids = IVector::create(testConf.inputDefs[i].ids.size(), useGpu);
          data.ids->copyFrom(testConf.inputDefs[i].ids.data(),
                             testConf.inputDefs[i].ids.size());
        } else if (testConf.inputDefs[i].selfDefinedData) {
          size_t height = testConf.inputDefs[i].selfDefinedData->getHeight();
          size_t width = testConf.inputDefs[i].selfDefinedData->getWidth();
          CHECK_GT(static_cast<int>(height), 0);
          CHECK_GT(static_cast<int>(width), 0);
          data.value = Matrix::create(height, width, false, useGpu);
          data.grad = Matrix::create(height, width, false, useGpu);
          data.value->copyFrom(*testConf.inputDefs[i].selfDefinedData);
          data.grad->zeroMem();
        } else {
          LOG(FATAL) << "No self-defined data are given.";
          return;
        }

        const std::vector<int>& labelSeqStartPositions =
            testConf.inputDefs[i].labelSeqStartPositions;
        if (labelSeqStartPositions.size() != 0) {
          CHECK_GE(static_cast<int>(labelSeqStartPositions.size()), 2);

          sequenceStartPositions =
              ICpuGpuVector::create(labelSeqStartPositions.size(), useGpu);
          sequenceStartPositions->copyFrom(labelSeqStartPositions.data(),
                                           labelSeqStartPositions.size(),
                                           useGpu);
          data.sequenceStartPositions = sequenceStartPositions;
        }

        const std::vector<int>& labelSubSeqStartPositions =
            testConf.inputDefs[i].labelSubSeqStartPositions;
        if (labelSubSeqStartPositions.size() != 0) {
          CHECK_GE(static_cast<int>(labelSubSeqStartPositions.size()), 2);

          subSequenceStartPositions =
              ICpuGpuVector::create(labelSubSeqStartPositions.size(), useGpu);
          subSequenceStartPositions->copyFrom(labelSubSeqStartPositions.data(),
                                              labelSubSeqStartPositions.size(),
                                              useGpu);
          data.subSequenceStartPositions = subSequenceStartPositions;
        }
        break;
      }
      default:
        LOG(FATAL) << " unknown inputType ";
        return;
    }
    if (testConf.inputDefs[i].inputType == INPUT_SEQUENCE_DATA ||
        testConf.inputDefs[i].inputType == INPUT_HASSUB_SEQUENCE_DATA ||
        testConf.inputDefs[i].inputType == INPUT_SEQUENCE_LABEL ||
        testConf.inputDefs[i].inputType == INPUT_SEQUENCE_MDIM_DATA) {
      if (!sequenceStartPositions) {
        generateSequenceStartPositions(batchSize, sequenceStartPositions);
      }
      data.sequenceStartPositions = sequenceStartPositions;
    }
    if (testConf.inputDefs[i].inputType == INPUT_HASSUB_SEQUENCE_DATA) {
      if (!subSequenceStartPositions) {
        generateSubSequenceStartPositions(sequenceStartPositions,
                                          subSequenceStartPositions);
      }
      data.subSequenceStartPositions = subSequenceStartPositions;
    }
    if (testConf.inputDefs[i].inputType == INPUT_SEQUENCE_MDIM_DATA) {
      if (!cpuSequenceDims) {
        generateMDimSequenceData(sequenceStartPositions, cpuSequenceDims);
      }
      data.cpuSequenceDims = cpuSequenceDims;
    }

    DataLayerPtr dataLayer = std::dynamic_pointer_cast<DataLayer>(layer);
    dataLayer->setData(data);
    dataLayer->forward(PASS_GC);
    dataLayers->push_back(dataLayer);
    (*layerMap)[config.name()] = layer;
    datas->push_back(data);
  }
}

void initTestLayer(TestConfig testConf,
                   LayerMap* layerMap,
                   std::vector<ParameterPtr>* parameters,
                   LayerPtr* testLayer) {
  ParameterMap parameterMap;
  size_t index = 0;
  LayerConfig testConfig = testConf.layerConfig;
  CHECK_EQ(testConf.inputDefs.size(),
           size_t(testConf.layerConfig.inputs_size()));

  auto initParameter = [&](string paraName,
                           size_t paraSize,
                           bool isStatic,
                           bool initialize,
                           ParameterConfig paraConfig) {
    paraConfig.set_name(paraName);
    paraConfig.set_size(paraSize);
    paraConfig.set_is_static(isStatic);
    auto para =
        std::make_shared<Parameter>(paraConfig, FLAGS_use_gpu, initialize);
    para->enableType(PARAMETER_VALUE);
    if (!para->isStatic()) {
      para->enableType(PARAMETER_GRADIENT);
      para->enableType(PARAMETER_MOMENTUM);
    }
    para->randomize();
    para->setID(index++);
    parameters->push_back(para);
    parameterMap[paraConfig.name()] = para;
  };

  for (size_t i = 0; i < testConf.inputDefs.size(); i++) {
    InputDef inputDef = testConf.inputDefs[i];
    size_t paraSize = inputDef.paraSize;
    bool sparse = inputDef.sparse.sparse;
    LayerInputConfig& input = *(testConfig.mutable_inputs(i));
    input.set_input_layer_name(inputDef.name);

    if (paraSize) {
      constexpr int kParaNameLen = 20;
      char paraName[kParaNameLen];
      snprintf(paraName, kParaNameLen, "para_%d", (int)i);
      input.set_input_parameter_name(paraName);
      ParameterConfig paraConfig;
      paraConfig.set_is_sparse(sparse);
      paraConfig.set_format(inputDef.sparse.format);
      if (sparse) {
        paraConfig.add_dims((*layerMap)[input.input_layer_name()]->getSize());
        paraConfig.add_dims(testConf.layerConfig.size());
      }
      CHECK_GE(testConf.paramInitialStd, 0);
      paraConfig.set_initial_mean(testConf.paramInitialMean);
      paraConfig.set_initial_std(testConf.paramInitialStd);
      initParameter(paraName, paraSize, inputDef.isStatic, false, paraConfig);
    }
  }
  if (testConf.biasSize) {
    testConfig.set_bias_parameter_name("bias");
    ParameterConfig paraConfig;
    initParameter(testConfig.bias_parameter_name(),
                  testConf.biasSize,
                  testConf.staticBias,
                  true,
                  paraConfig);
  }

  *testLayer = Layer::create(testConfig);
  (*layerMap)[testConfig.name()] = *testLayer;
  (*testLayer)->init((*layerMap), parameterMap);
  (*testLayer)->setNeedGradient(true);
}

void testPerturbParameter(TestConfig testConf,
                          const MatrixPtr weights,
                          const LayerStatePtr state,
                          real cost,
                          real callbackCount,
                          real* maxDiff,
                          LayerPtr testLayer,
                          std::vector<ParameterPtr>* parameters) {
  char fill = ' ';
  for (auto& parameter : *parameters) {
    if (parameter->isStatic()) {
      continue;
    }

    size_t dim = parameter->getSize();
    CpuVector oldPara(dim);
    CpuVector newPara(dim);
    VectorPtr v = parameter->getBuf(PARAMETER_VALUE);
    oldPara.copyFrom(*parameter->getBuf(PARAMETER_VALUE));
    real* newp = newPara.getData();
    real* oldp = oldPara.getData();
    CpuVector cpuGrad(*parameter->getBuf(PARAMETER_GRADIENT));
    vector<real> d(dim);

    double delta = genPerturbation(cpuGrad.getData(), &d[0], dim);
    // use a step such that delta / cost is FLAGS_checkgrad_eps
    real step =
        (delta != 0) ? cost / delta * FLAGS_checkgrad_eps : FLAGS_checkgrad_eps;
    if (fabs(step) < 1e-6) step = 1e-6;
    delta *= step;

    // compute newCost
    real newCost[2];
    for (int k = 0; k < 2; k++) {
      for (size_t i = 0; i < dim; ++i) {
        newp[i] = (k == 0) ? oldp[i] + step * d[i] : oldp[i] - step * d[i];
      }
      if (testConf.testBatchState) {
        testLayer->setState(state);
      }
      parameter->getBuf(PARAMETER_VALUE)->copyFrom(newPara);
      parameter->setValueUpdated();
      newCost[k] = getCostSum(testLayer, weights);
    }
    real diff = getDiffAndPrint(newCost[0],
                                newCost[1],
                                callbackCount,
                                fill,
                                testLayer->getName(),
                                parameter->getName(),
                                step,
                                delta);
    *maxDiff = std::max(*maxDiff, abs(diff));
    // restore parameter
    parameter->getBuf(PARAMETER_VALUE)->copyFrom(oldPara);
    parameter->setValueUpdated();
    fill = (fill == ' ') ? '.' : ' ';
  }
}

void testPerturbInput(TestConfig testConf,
                      const MatrixPtr weights,
                      const LayerStatePtr state,
                      real cost,
                      real callbackCount,
                      real* maxDiff,
                      LayerPtr testLayer,
                      std::vector<DataLayerPtr> dataLayers) {
  char fill = ' ';
  for (size_t index = 0; index < testConf.inputDefs.size(); index++) {
    InputType inputType = testConf.inputDefs[index].inputType;
    if (inputType != INPUT_DATA && inputType != INPUT_SEQUENCE_DATA &&
        inputType != INPUT_HASSUB_SEQUENCE_DATA) {
      continue;
    }

    MatrixPtr outV = dataLayers[index]->getOutputValue();
    int height = outV->getHeight();
    int width = outV->getWidth();
    size_t dim = height * width;

    CpuMatrix oldPara(height, width);
    CpuMatrix newPara(height, width);
    oldPara.copyFrom(*outV);
    real* newp = newPara.getData();
    real* oldp = oldPara.getData();
    CpuMatrix cpuGrad(height, width);
    cpuGrad.copyFrom(*(dataLayers[index]->getOutputGrad()));
    CpuMatrix d(height, width);
    real* data = d.getData();

    double delta = genPerturbation(cpuGrad.getData(), data, dim);
    // use a step such that delta / cost is FLAGS_checkgrad_eps
    real step =
        (delta != 0) ? cost / delta * FLAGS_checkgrad_eps : FLAGS_checkgrad_eps;
    if (fabs(step) < 1e-6) step = 1e-6;
    delta *= step;

    real newCost[2];
    for (int k = 0; k < 2; k++) {
      for (size_t i = 0; i < dim; ++i) {
        newp[i] =
            (k == 0) ? oldp[i] + step * data[i] : oldp[i] - step * data[i];
      }
      if (testConf.testBatchState) {
        testLayer->setState(state);
      }
      outV->copyFrom(newPara);
      newCost[k] = getCostSum(testLayer, weights);
    }

    real diff = getDiffAndPrint(newCost[0],
                                newCost[1],
                                callbackCount,
                                fill,
                                testLayer->getName(),
                                dataLayers[index]->getName(),
                                step,
                                delta);
    *maxDiff = std::max(*maxDiff, abs(diff));
    // restore parameter
    outV->copyFrom(oldPara);
    fill = (fill == ' ') ? '.' : ' ';
  }
}

void testLayerGradKernel(TestConfig testConf,
                         string testLayerName,
                         size_t batchSize,
                         bool trans,
                         bool useGpu,
                         bool useWeight,
                         float epsilon) {
#ifndef PADDLE_WITH_CUDA
  if (useGpu) return;
#endif
  FLAGS_use_gpu = useGpu;
  FLAGS_prev_batch_state = testConf.testBatchState;
  MatrixPtr weights = nullptr;
  testConf.layerConfig.set_name(testLayerName);
  LOG(INFO) << " layer_type=" << testConf.layerConfig.type()
            << " useGpu=" << useGpu;

  // data layer initialize
  std::vector<DataLayerPtr> dataLayers;
  LayerMap layerMap;
  vector<Argument> datas;
  initDataLayer(testConf,
                &dataLayers,
                &datas,
                &layerMap,
                testLayerName,
                batchSize,
                trans,
                useGpu);
  // test layer initialize
  std::vector<ParameterPtr> parameters;
  LayerPtr testLayer;
  initTestLayer(testConf, &layerMap, &parameters, &testLayer);

  LayerStatePtr state = std::make_shared<LayerState>();
  if (testConf.testBatchState) {
    initBatchState(dataLayers[0], testLayer, state, useGpu);
    testLayer->resetState();
    testLayer->setState(state);
  }

  testLayer->forward(PASS_GC);
  if (useWeight && weights == nullptr) {
    weights = testLayer->getOutput().value->clone(0, 0, useGpu);
    initWeight(weights);
  }
  std::vector<Argument> outArgs;
  outArgs.push_back(testLayer->getOutput());
  if (useWeight) {
    outArgs[0].value = outArgs[0].value->clone(0, 0, useGpu);
    outArgs[0].value->dotMul(*testLayer->getOutput().value, *weights);
  }

  real cost = Argument::sum(outArgs);
  LOG(INFO) << " cost " << cost;
  EXPECT_FALSE(std::isnan(cost));

  // Test whether the callback is called for a parameter
  if (testLayer->getOutputGrad()) {
    useWeight ? testLayer->getOutput().grad->copyFrom(*weights)
              : testLayer->getOutputGrad()->resetOne();
  }
  vector<int> callbackFlags(parameters.size(), 0);
  auto callback = [&](Parameter* para) { ++callbackFlags[para->getID()]; };
  testLayer->backward(callback);

  // do forward and backward for another time to test that gradient is doubled
  int callbackCount = 1;
  if (testConf.testAccumulate) {
    if (testConf.testBatchState) {
      testLayer->setState(state);
    }
    testLayer->forward(PASS_GC);
    if (testLayer->getOutputGrad()) {
      useWeight ? testLayer->getOutput().grad->copyFrom(*weights)
                : testLayer->getOutputGrad()->resetOne();
    }
    testLayer->backward(callback);
    ++callbackCount;
  }
  for (size_t i = 0; i < parameters.size(); ++i) {
    EXPECT_EQ(parameters[i]->isStatic() ? 0 : callbackCount, callbackFlags[i]);
  }

  // Test whether the layer's forward calculation is stable
  // by adding perturbation to its parameters or its input layers
  real maxDiff = 0;
  testPerturbParameter(testConf,
                       weights,
                       state,
                       cost,
                       callbackCount,
                       &maxDiff,
                       testLayer,
                       &parameters);
  testPerturbInput(testConf,
                   weights,
                   state,
                   cost,
                   callbackCount,
                   &maxDiff,
                   testLayer,
                   dataLayers);
  EXPECT_LE(fabs(maxDiff), epsilon);

  if (testConf.testState) {
    testState(testLayer, dataLayers, datas);
  }
  if (testConf.testBatchState) {
    testBatchState(testLayer, dataLayers, datas);
  }
}

void testLayerGrad(TestConfig testConf,
                   string testLayerName,
                   size_t batchSize,
                   bool trans,
                   bool useGpu,
                   bool useWeight,
                   float epsilon) {
  testLayerGradKernel(
      testConf, testLayerName, batchSize, trans, useGpu, useWeight, epsilon);
  bool isStaticTest = false;
  LayerConfig testConfig = testConf.layerConfig;
  for (size_t i = 0; i < testConf.inputDefs.size(); i++) {
    InputDef inputDef = testConf.inputDefs[i];
    // Some layer must set isStatic true, like DataNormLayer
    // so use !isStatic in if
    if (inputDef.paraSize && (!inputDef.isStatic)) {
      testConf.inputDefs[i].isStatic = true;
      isStaticTest = true;
    }
  }

  if (testConf.biasSize) {
    testConf.staticBias = true;
    isStaticTest = true;
  }
  if (isStaticTest) {
    testLayerGradKernel(
        testConf, testLayerName, batchSize, trans, useGpu, useWeight, epsilon);
  }
}

void testProjectionGrad(ProjectionConfig conf,
                        InputType inputType,
                        size_t parameterSize,
                        size_t batchSize,
                        bool useGpu,
                        bool testState,
                        int biasSize,
                        bool sharedBias) {
  TestConfig config;
  conf.set_name(conf.type());
  config.layerConfig.set_type("mixed");
  config.layerConfig.set_size(conf.output_size());
  config.biasSize = biasSize == 0 ? config.layerConfig.size() : biasSize;
  config.layerConfig.set_bias_size(config.biasSize);
  config.layerConfig.set_shared_biases(sharedBias);
  config.inputDefs.push_back({inputType,
                              "layer_0",
                              static_cast<size_t>(conf.input_size()),
                              parameterSize});
  *config.layerConfig.add_inputs()->mutable_proj_conf() = conf;
  config.testState = testState;
  testLayerGrad(config, "mixed", batchSize, false, useGpu);
}

void testOperatorGrad(TestConfig& config,
                      OperatorConfig& operatorConf,
                      size_t batchSize,
                      bool useGpu,
                      bool testState) {
  config.layerConfig.set_type("mixed");

  operatorConf.set_output_size(config.layerConfig.size());
  for (size_t i = 0; i < config.inputDefs.size(); ++i) {
    operatorConf.add_input_indices(i);
    operatorConf.add_input_sizes(config.inputDefs[i].dim);
  }

  config.testState = testState;
  testLayerGrad(config, "mixed", batchSize, false, useGpu);
}
}  //  namespace paddle
