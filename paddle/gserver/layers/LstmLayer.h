/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve.

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

#include "Layer.h"
#include "paddle/math/Matrix.h"
#include "paddle/math/BaseMatrix.h"
#include "SequenceToBatch.h"
#include "LstmCompute.h"
namespace paddle {

/*
LstmLayer takes 1 input layer with size * 4.
Input layer is diveded into 4 equal parts:
  (input_s, input_ig, input_fg, input_og)

For each sequence [start, end] it performs the following computation:

out_i   = actState(state_i) * actGate(outputGate_i)
state_i = actInput(input_s_i + bias_s + output_{i-1} * recurrIW)
          * actGate(inputGate_i) + actGate(forgetGate_i) * state_{i-1}
inputGate = input_ig_i + bias_ig + output_{i-1} * recurrIGW
            + state_{i-1} * inputCheck
ouputGate = input_og_i + bias_og + output_{i-1} * recurrOGW
            + state_{i} * outputCheck
forgetGate = input_fg_i + bias_fg + output_{i-1} * recurrFGW
             + state_{i-1} * forgetCheck

parameter[0] consists of (recurrIW, recurrIGW, recurrFGW, recurrOGW)
baisParameter consists of
  (bias_s, bias_ig, bias_og, bias_fg, inputCheck, forgetCheck, outputCheck)

actInput is defined by config active_type
actState is defined by config active_state_type
actGate is defined by config actvie_gate_type
*/

class LstmLayer : public Layer, public LstmCompute {
public:
  explicit LstmLayer(const LayerConfig &config) : Layer(config) {}

  bool init(const LayerMap &layerMap, const ParameterMap &parameterMap);

  void forward(PassType passType);

  void backward(const UpdateCallback &callback);

  void resetState();

  void setState(LayerStatePtr state);

  LayerStatePtr getState();

protected:
  void forwardSequence(int batchSize, size_t numSequences,
                       const int *starts, MatrixPtr inputValue);
  void backwardSequence(int batchSize, size_t numSequences,
                        const int *starts, MatrixPtr inputGrad);

  void forwardBatch(int batchSize, size_t numSequences,
                    const int *starts, MatrixPtr inputValue);
  void backwardBatch(int batchSize, size_t numSequences,
                     const int *starts, MatrixPtr inputGrad);

  void forwardSeqParallel(int batchSize, size_t numSequences,
                          const int *starts, MatrixPtr inputValue);
  void backwardSeqParallel(int batchSize, size_t numSequences,
                           const int *starts, MatrixPtr inputGrad);
  void getPrevBatchOutput(size_t numSequences);
  void getPrevBatchState(size_t numSequences);

protected:
  std::unique_ptr<Weight> weight_;
  std::unique_ptr<Weight> bias_;
  /* real bias and peephole for different gates */
  MatrixPtr localBias_, checkIg_, checkFg_, checkOg_;
  /* the gradient of, real bias and peephole for different gates */
  MatrixPtr localBiasGrad_, checkIgGrad_, checkFgGrad_, checkOgGrad_;

  Argument state_;
  Argument preOutput_;
  Argument gate_;
  bool reversed_;
  bool useBatch_;
  bool useSeqParallel_;
  std::unique_ptr<SequenceToBatch> batchValue_;
  std::unique_ptr<SequenceToBatch> batchGrad_;

  MatrixPtr prevState_;
  MatrixPtr prevOutput_;
  MatrixPtr prevBatchOutput2_;
  MatrixPtr totalState_;
};

}  // namespace paddle
