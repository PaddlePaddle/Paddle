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
#include <paddle/utils/Version.h>
#include <vector>
#include "ModelConfig.pb.h"
#include "paddle/gserver/layers/DataLayer.h"
#include "paddle/gserver/layers/Layer.h"

#include "paddle/testing/TestUtil.h"

using namespace paddle;  // NOLINT
using namespace std;     // NOLINT
DECLARE_bool(use_gpu);
DECLARE_bool(rnn_use_batch);
DECLARE_int32(fixed_seq_length);

void checkError(const Matrix& matrix1, const Matrix& matrix2) {
  CHECK(matrix1.getHeight() == matrix2.getHeight());
  CHECK(matrix1.getWidth() == matrix2.getWidth());
#ifndef PADDLE_TYPE_DOUBLE
  real err = 1e-3;
#else
  real err = 1e-10;
#endif

  int height = matrix1.getHeight();
  int width = matrix1.getWidth();
  const real* data1 = matrix1.getData();
  const real* data2 = matrix2.getData();
  int count = 0;
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      if (fabs(data1[i * width + j] - data2[i * width + j]) > err) {
        count++;
      }
    }
  }
  EXPECT_EQ(count, 0) << "There are " << count << " different element.";
}

void checkError(const CpuVector& vector1, const CpuVector& vector2) {
  CHECK(vector1.getSize() == vector2.getSize());
#ifndef PADDLE_TYPE_DOUBLE
  real err = 1e-3;
#else
  real err = 1e-10;
#endif

  int size = vector1.getSize();
  const real* data1 = vector1.getData();
  const real* data2 = vector2.getData();
  int count = 0;
  for (int i = 0; i < size; i++) {
    if (fabs(data1[i] - data2[i]) > err) {
      count++;
    }
  }
  EXPECT_EQ(count, 0) << "There are " << count << " different element.";
}

LayerPtr creatDataLayer(string name,
                        size_t batchSize,
                        int layerSize,
                        bool useGpu) {
  LayerConfig dataConfig;
  dataConfig.set_name(name);
  dataConfig.set_type("data");
  dataConfig.set_size(layerSize);
  LayerPtr layer = LayerPtr(new DataLayer(dataConfig));

  Argument data;
  data.value = Matrix::create(batchSize, layer->getSize(), false, useGpu);
  data.grad = Matrix::create(batchSize, layer->getSize(), false, useGpu);
  data.value->randomizeUniform();
  data.value->add(-0.5);
  data.value->sigmoid(*data.value);
  data.grad->zeroMem();

  generateSequenceStartPositions(batchSize, data.sequenceStartPositions);

  DataLayerPtr dataLayer = std::dynamic_pointer_cast<DataLayer>(layer);
  dataLayer->setData(data);
  dataLayer->forward(PASS_GC);

  return layer;
}

ParameterPtr creatParameter(string name,
                            int pid,
                            size_t paraSize,
                            bool useGpu) {
  ParameterConfig paraConfig;
  paraConfig.set_name(name);
  paraConfig.set_size(paraSize);

  ParameterPtr parameter =
      std::make_shared<Parameter>(paraConfig, useGpu, /*initialize */ false);
  parameter->enableType(PARAMETER_VALUE);
  parameter->enableType(PARAMETER_GRADIENT);
  parameter->randomize();
  parameter->setID(pid);

  return parameter;
}

ParameterPtr creatParameterBias(string name,
                                int pid,
                                size_t paraSize,
                                bool useGpu) {
  ParameterConfig paraConfig;
  paraConfig.set_name(name);
  paraConfig.set_size(paraSize);
  paraConfig.set_initial_std(1);

  ParameterPtr parameter =
      std::make_shared<Parameter>(paraConfig, useGpu, /*initialize */ true);
  parameter->randomize();
  parameter->setID(pid);

  return parameter;
}

LayerPtr initRecurrentLayer(LayerConfig layerConfig,
                            size_t batchSize,
                            int layerSize,
                            bool useGpu) {
  FLAGS_use_gpu = useGpu;
  LayerMap layerMap;
  ParameterMap parameterMap;
  LayerPtr dataLayer = creatDataLayer("layer_0", batchSize, layerSize, useGpu);
  layerMap[dataLayer->getName()] = dataLayer;

  ParameterPtr para =
      creatParameter("para_0", 0, layerSize * layerSize, useGpu);
  parameterMap[para->getName()] = para;

  layerConfig.add_inputs();
  LayerInputConfig& input = *(layerConfig.mutable_inputs(0));
  input.set_input_layer_name("layer_0");
  input.set_input_parameter_name("para_0");
  LayerPtr testLayer = Layer::create(layerConfig);
  layerMap[testLayer->getName()] = testLayer;

  testLayer->init(layerMap, parameterMap);
  testLayer->setNeedGradient(true);

  return testLayer;
}

void checkRecurrentLayer(LayerPtr testLayer) {
  const VectorPtr& weightGrad =
      (testLayer->getParameters()[0])->getBuf(PARAMETER_GRADIENT);
  const MatrixPtr& inputGrad = testLayer->getPrev(0)->getOutputGrad();
  CpuVector seqPara(weightGrad->getSize());
  CpuVector batPara(weightGrad->getSize());
  CpuMatrix seqInputGrad(inputGrad->getHeight(), inputGrad->getWidth());
  CpuMatrix batInputGrad(inputGrad->getHeight(), inputGrad->getWidth());

  CpuMatrix outputGrad(inputGrad->getHeight(), inputGrad->getWidth());
  outputGrad.randomizeUniform();

  /* use sequence calculate */
  FLAGS_rnn_use_batch = false;
  weightGrad->zero();
  inputGrad->zero();
  testLayer->forward(PASS_GC);
  testLayer->getOutputGrad()->copyFrom(outputGrad);
  testLayer->backward();
  seqPara.copyFrom(*weightGrad);
  seqInputGrad.copyFrom(*inputGrad);

  /* use batch calculate */
  FLAGS_rnn_use_batch = true;
  weightGrad->zero();
  inputGrad->zero();
  testLayer->forward(PASS_GC);
  testLayer->getOutputGrad()->copyFrom(outputGrad);
  testLayer->backward();
  batPara.copyFrom(*weightGrad);
  batInputGrad.copyFrom(*inputGrad);

  /* check */
  checkError(seqInputGrad, batInputGrad);
  checkError(seqPara, batPara);
}

TEST(Layer, RecurrentLayer) {
  LayerConfig layerConfig;
  layerConfig.set_name("rnn");
  layerConfig.set_type("recurrent");
  layerConfig.set_active_type("tanh");
  for (auto layerSize : {1, 10, 64, 128, 256, 512}) {
    for (auto batchSize : {1, 5, 20, 100, 128}) {
      for (auto useGpu : {false, true}) {
        for (auto reversed : {false, true}) {
          LOG(INFO) << " layerSize=" << layerSize << " batchSize=" << batchSize
                    << " useGpu=" << useGpu << " reversed=" << reversed;
          layerConfig.set_size(layerSize);
          layerConfig.set_reversed(reversed);
          LayerPtr testLayer =
              initRecurrentLayer(layerConfig, batchSize, layerSize, useGpu);
          checkRecurrentLayer(testLayer);
        }
      }
    }
  }
}

#define protected public
#include "paddle/gserver/layers/GatedRecurrentLayer.h"
#include "paddle/gserver/layers/LstmLayer.h"
#include "paddle/gserver/layers/RecurrentLayer.h"
template <class T>
class TestRecurrentLayer {
 public:
  LayerConfig config_;
  bool useGpu_;
  bool useBatch_;
  LayerPtr testLayer_;
  LayerPtr dataLayer_;
  ParameterPtr para_;
  ParameterPtr bias_;
  LayerMap layerMap_;
  ParameterMap parameterMap_;
  TestRecurrentLayer(const LayerConfig& config,
                     bool useGpu,
                     bool useBatch = false)
      : config_(config), useGpu_(useGpu), useBatch_(useBatch) {}
  void init(size_t batchSize) {
    FLAGS_use_gpu = useGpu_;
    testLayer_ = Layer::create(config_);
    if (typeid(T) == typeid(GatedRecurrentLayer)) {
      dataLayer_ = creatDataLayer(config_.mutable_inputs(0)->input_layer_name(),
                                  batchSize,
                                  config_.size() * 3,
                                  useGpu_);
      para_ = creatParameter(config_.mutable_inputs(0)->input_parameter_name(),
                             0,
                             config_.size() * config_.size() * 3,
                             useGpu_);
      bias_ = creatParameterBias(
          config_.bias_parameter_name(), 1, config_.size() * 3, useGpu_);
    } else if (typeid(T) == typeid(LstmLayer)) {
      dataLayer_ = creatDataLayer(config_.mutable_inputs(0)->input_layer_name(),
                                  batchSize,
                                  config_.size() * 4,
                                  useGpu_);
      para_ = creatParameter(config_.mutable_inputs(0)->input_parameter_name(),
                             0,
                             config_.size() * config_.size() * 4,
                             useGpu_);
      bias_ = creatParameterBias(
          config_.bias_parameter_name(), 1, config_.size() * 7, useGpu_);
    }
    layerMap_[dataLayer_->getName()] = dataLayer_;
    parameterMap_[para_->getName()] = para_;
    parameterMap_[bias_->getName()] = bias_;

    layerMap_[testLayer_->getName()] = testLayer_;
    testLayer_->init(layerMap_, parameterMap_);
    testLayer_->setNeedGradient(true);
    (dynamic_cast<T*>(testLayer_.get()))->useBatch_ = useBatch_;
  }
  void forward() {
    FLAGS_use_gpu = useGpu_;
    testLayer_->forward(PASS_GC);
  }
  void backward() {
    FLAGS_use_gpu = useGpu_;
    testLayer_->backward(nullptr);
  }
};

template <class T>
void checkRecurrentLayer(LayerConfig layerConfig,
                         size_t batchSize,
                         bool cpuBatch,
                         bool gpuBatch) {
  TestRecurrentLayer<T> testCpu(layerConfig, false, cpuBatch);
  TestRecurrentLayer<T> testGpu(layerConfig, true, gpuBatch);
  testCpu.init(batchSize);
  testGpu.init(batchSize);
  auto checkError = [](
      MatrixPtr cpu, MatrixPtr gpu, int numSequences, const char* str) {
    CpuMatrix check(gpu->getHeight(), gpu->getWidth());
    check.copyFrom(*gpu);
    int height = cpu->getHeight();
    int width = cpu->getWidth();
    const real* data1 = cpu->getData();
    const real* data2 = check.getData();
    int count = 0;
    for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
        if (fabs(data1[i * width + j] - data2[i * width + j]) / numSequences >
            1e-4) {
          count++;
        }
      }
    }
    EXPECT_EQ(count, 0) << "[" << str << "]"
                        << "There are " << count << " different element.";
  };
  T* cpuLayer = dynamic_cast<T*>(testCpu.testLayer_.get());
  T* gpuLayer = dynamic_cast<T*>(testGpu.testLayer_.get());

  Argument& cpuInput = testCpu.dataLayer_->getOutput();
  Argument& gpuInput = testGpu.dataLayer_->getOutput();
  gpuInput.resizeAndCopyFrom(cpuInput, true);

  const VectorPtr& cpuVec = testCpu.para_->getBuf(PARAMETER_VALUE);
  const VectorPtr& gpuVec = testGpu.para_->getBuf(PARAMETER_VALUE);
  gpuVec->copyFrom(*cpuVec);

  const VectorPtr& cpuBiasVec = testCpu.bias_->getBuf(PARAMETER_VALUE);
  const VectorPtr& gpuBiasVec = testGpu.bias_->getBuf(PARAMETER_VALUE);
  gpuBiasVec->copyFrom(*cpuBiasVec);

  /* check forward */
  testCpu.forward();
  testGpu.forward();

  checkError(
      cpuLayer->getOutputValue(), gpuLayer->getOutputValue(), 1, "outputValue");

  /* check backward */
  cpuLayer->getOutputGrad()->randomizeUniform();
  gpuLayer->getOutputGrad()->copyFrom(*cpuLayer->getOutputGrad());
  hl_stream_synchronize(HPPL_STREAM_DEFAULT);

  testCpu.backward();
  testGpu.backward();

  // check input grad
  checkError(cpuInput.grad, gpuInput.grad, 1, "inputGrad");
  // check weight grad
  int numSequences = cpuInput.getNumSequences();
  checkError(cpuLayer->weight_->getWGrad(),
             gpuLayer->weight_->getWGrad(),
             numSequences,
             "weightGrad");
  // check bias grad
  checkError(cpuLayer->bias_->getWGrad(),
             gpuLayer->bias_->getWGrad(),
             numSequences,
             "biasGrad");
}

TEST(Layer, GatedRecurrentLayer) {
  LayerConfig layerConfig;
  layerConfig.set_type("gated_recurrent");
  layerConfig.set_active_type("sigmoid");
  layerConfig.set_active_gate_type("sigmoid");

  layerConfig.add_inputs();
  LayerInputConfig& input = *(layerConfig.mutable_inputs(0));
  input.set_input_layer_name("layer_0");
  input.set_input_parameter_name("para_0");
  layerConfig.set_bias_parameter_name("bias");

  for (auto frameSize : {32, 64, 128, 256, 512}) {
    for (auto batchSize : {1, 5, 100, 500}) {
      for (auto reversed : {false, true}) {
        for (auto cpuBatch : {false, true}) {
          for (auto gpuBatch : {false, true}) {
            LOG(INFO) << " batchSize=" << batchSize
                      << " frameSize=" << frameSize << " reversed=" << reversed
                      << " cpuBatch=" << cpuBatch << " gpuBatch=" << gpuBatch;
            layerConfig.set_size(frameSize);
            layerConfig.set_reversed(reversed);
            checkRecurrentLayer<GatedRecurrentLayer>(
                layerConfig, batchSize, cpuBatch, gpuBatch);
          }
        }
      }
    }
  }
}

TEST(Layer, LstmLayer) {
  LayerConfig layerConfig;
  layerConfig.set_type("lstmemory");
  layerConfig.set_active_type("relu");
  layerConfig.set_active_state_type("tanh");
  layerConfig.set_active_gate_type("sigmoid");

  layerConfig.add_inputs();
  LayerInputConfig& input = *(layerConfig.mutable_inputs(0));
  input.set_input_layer_name("layer_0");
  input.set_input_parameter_name("para_0");
  layerConfig.set_bias_parameter_name("bias");

  for (auto frameSize : {32, 64, 128, 256, 512}) {
    for (auto batchSize : {1, 5, 100, 500}) {
      for (auto reversed : {false, true}) {
        for (auto cpuBatch : {false, true}) {
          for (auto gpuBatch : {false, true}) {
            LOG(INFO) << " batchSize=" << batchSize
                      << " frameSize=" << frameSize << " reversed=" << reversed
                      << " cpuBatch=" << cpuBatch << " gpuBatch=" << gpuBatch;
            layerConfig.set_size(frameSize);
            layerConfig.set_reversed(reversed);
            checkRecurrentLayer<LstmLayer>(
                layerConfig, batchSize, cpuBatch, gpuBatch);
          }
        }
      }
    }
  }
}

#ifdef PADDLE_WITH_MKLML

#include "paddle/gserver/layers/MKLPackedRecurrentLayer.h"

LayerPtr initMKLPackedLayer(LayerConfig layerConfig,
                            bool reversed,
                            int layerSize,
                            LayerPtr dataLayer,
                            ParameterPtr para,
                            ParameterPtr bias = nullptr) {
  LayerMap layerMap;
  ParameterMap parameterMap;
  layerMap[dataLayer->getName()] = dataLayer;
  parameterMap[para->getName()] = para;
  if (bias) {
    parameterMap[bias->getName()] = bias;
    layerConfig.set_bias_parameter_name("bias_0");
  }

  layerConfig.set_size(layerSize);
  layerConfig.set_reversed(reversed);
  layerConfig.add_inputs();
  LayerInputConfig& input = *(layerConfig.mutable_inputs(0));
  input.set_input_layer_name("layer_0");
  input.set_input_parameter_name("para_0");

  LayerPtr testLayer = Layer::create(layerConfig);
  layerMap[testLayer->getName()] = testLayer;

  testLayer->init(layerMap, parameterMap);
  testLayer->setNeedGradient(true);

  return testLayer;
}

void checkMKLPackedLayer(LayerConfig layerConfig1,
                         LayerConfig layerConfig2,
                         bool reversed,
                         int layerSize,
                         int batchSize,
                         bool useBatch1,
                         bool useBatch2) {
  LayerPtr dataLayer;
  ParameterPtr para, bias;

  if (layerConfig1.type() == "recurrent") {
    dataLayer = creatDataLayer("layer_0", batchSize, layerSize, false);
    para = creatParameter("para_0", 0, layerSize * layerSize, false);
    bias = nullptr;
  } else if (layerConfig1.type() == "gated_recurrent") {
    dataLayer = creatDataLayer("layer_0", batchSize, layerSize * 3, false);
    para = creatParameter("para_0", 0, layerSize * layerSize * 3, false);
    bias = creatParameterBias("bias_0", 1, layerSize * 3, false);
  }

  LayerPtr testLayer1 = initMKLPackedLayer(
      layerConfig1, reversed, layerSize, dataLayer, para, bias);
  LayerPtr testLayer2 = initMKLPackedLayer(
      layerConfig2, reversed, layerSize, dataLayer, para, bias);

  const VectorPtr& weightGrad =
      (testLayer1->getParameters()[0])->getBuf(PARAMETER_GRADIENT);
  const MatrixPtr& inputGrad = testLayer1->getPrev(0)->getOutputGrad();
  CpuVector wgt_grad1(weightGrad->getSize());
  CpuVector wgt_grad2(weightGrad->getSize());
  CpuMatrix input_grad1(inputGrad->getHeight(), inputGrad->getWidth());
  CpuMatrix input_grad2(inputGrad->getHeight(), inputGrad->getWidth());

  for (int i = 0; i < 2; i++) {
    FLAGS_rnn_use_batch = useBatch1;

    testLayer1->forward(PASS_GC);

    FLAGS_rnn_use_batch = useBatch2;
    testLayer2->forward(PASS_GC);

    testLayer1->getOutputGrad()->randomizeUniform();
    testLayer2->getOutputGrad()->copyFrom(*testLayer1->getOutputGrad());

    weightGrad->zero();
    inputGrad->zero();
    FLAGS_rnn_use_batch = useBatch1;
    testLayer1->backward(nullptr);

    wgt_grad1.copyFrom(*weightGrad);
    input_grad1.copyFrom(*inputGrad);

    weightGrad->zero();
    inputGrad->zero();
    FLAGS_rnn_use_batch = useBatch2;
    testLayer2->backward(nullptr);

    wgt_grad2.copyFrom(*weightGrad);
    input_grad2.copyFrom(*inputGrad);

    checkError(*testLayer1->getOutputValue(), *testLayer2->getOutputValue());
    checkError(wgt_grad1, wgt_grad2);
    checkError(input_grad1, input_grad2);
  }
}

TEST(MKLPackedLayer, RecurrentLayer) {
  LayerConfig layerConfig1;
  LayerConfig layerConfig2;

  layerConfig1.set_name("paddle-rnn");
  layerConfig1.set_type("recurrent");
  layerConfig1.set_active_type("relu");

  layerConfig2.set_name("mkl-packed-rnn");
  layerConfig2.set_type("mkl_packed_recurrent");
  layerConfig2.set_active_type("relu");

  FLAGS_use_gpu = false;

  for (auto layerSize : {32, 64, 128, 256, 512}) {
    for (auto batchSize : {1, 5, 100, 500}) {
      for (auto reversed : {true, false}) {
        for (auto paddle_use_batch : {true, false}) {
          for (auto MKLPacked_use_batch : {true, false}) {
            LOG(INFO) << " layerSize=" << layerSize
                      << " batchSize=" << batchSize << " reversed=" << reversed
                      << " paddle_use_batch=" << paddle_use_batch
                      << " MKLPacked_use_batch=" << MKLPacked_use_batch;

            checkMKLPackedLayer(layerConfig1,
                                layerConfig2,
                                reversed,
                                layerSize,
                                batchSize,
                                paddle_use_batch,
                                MKLPacked_use_batch);
          }
        }
      }
    }
  }
}
#endif

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  initMain(argc, argv);
  if (!version::isWithGpu()) {
    testing::GTEST_FLAG(filter) = "-Layer.*";
  }
  return RUN_ALL_TESTS();
}
