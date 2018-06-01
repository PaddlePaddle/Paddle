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

#include "Layer.h"
#include "LstmCompute.h"
#include "SequenceToBatch.h"
#include "paddle/math/BaseMatrix.h"
#include "paddle/math/Matrix.h"
namespace paddle {

/**
 * @brief LstmLayer takes 1 input layer with size * 4.
 * Input layer is diveded into 4 equal parts:
 *   (input_s, input_ig, input_fg, input_og)
 *
 * For each sequence [start, end] it performs the following computation:
 * @code
 * output_{i} = actState(state_{i}) * actGate(outputGate_{i})
 * state_{i} = actInput(input_s_{i} + bias_s +
 *             output_{i-1} * recurrIW) * actGate(inputGate_{i}) +
 *             actGate(forgetGate_{i}) * state_{i-1}
 * inputGate = input_ig_{i} + bias_ig + output_{i-1} * recurrIGW +
 *             state_{i-1} * inputCheck
 * ouputGate = input_og_{i} + bias_og + output_{i-1} * recurrOGW +
 *             state_{i} * outputCheck
 * forgetGate = input_fg_{i} + bias_fg + output_{i-1} * recurrFGW +
 *              state_{i-1} * forgetCheck
 * @endcode
 *
 * - parameter[0] consists of (recurrIW, recurrIGW, recurrFGW, recurrOGW)
 * - baisParameter consists of
 *   (bias_s, bias_ig, bias_og, bias_fg, inputCheck, forgetCheck, outputCheck)
 *
 * - actInput is defined by config active_type.
 * - actState is defined by config active_state_type.
 * - actGate is defined by config actvie_gate_type.
 *
 * There are two ways to compute, namely one sequence by one sequence or
 * one batch by one batch. By default and no setting pre_batch_state true,
 * it will compute batch by batch.
 *
 * The formula in the paper is as follows:
 * \f[
 * i_t = \sigma(W_{xi}x_{t} + W_{hi}h_{t-1} + W_{ci}c_{t-1} + b_i) \\
 * f_t = \sigma(W_{xf}x_{t} + W_{hf}h_{t-1} + W_{cf}c_{t-1} + b_f) \\
 * \tilde{c_t} = tanh (W_{xc}x_t+W_{hc}h_{t-1} + b_c) \\
 * o_t = \sigma(W_{xo}x_{t} + W_{ho}h_{t-1} + W_{co}c_t + b_o) \\
 * c_t = f_t * c_{t-1} + i_t * \tilde{c_t} \\
 * h_t = o_t tanh(c_t)
 * \f]
 *
 * @note These \f$W_{xi}x_{t}, W_{xf}x_{t}, W_{xc}x_{t}, W_{xo}x_{t}\f$
 * operations on the input sequence were NOT included in LstmLayer. So
 * users should use fc_layer or mixed_layer before lstm_later.
 *
 * The weight ([size, 4*size]) contains \f$W_{hi}, W_{hf}, W_{hc}, W_{ho}\f$.
 * The bias contains \f$b_i, b_f, b_c, b_o\f$ and \f$W_{ci}, W_{cf}, W_{co}\f$.
 */

class LstmLayer : public Layer, public LstmCompute {
 public:
  explicit LstmLayer(const LayerConfig &config) : Layer(config) {}

  bool init(const LayerMap &layerMap,
            const ParameterMap &parameterMap) override;

  void forward(PassType passType) override;

  void backward(const UpdateCallback &callback) override;

  void resetState() override;

  void setState(LayerStatePtr state) override;

  LayerStatePtr getState() override;

 protected:
  /**
   * @brief Compute lstm forward one sequence by one sequence.
   * @param batchSize The batchSize is not equal to the batch_size in
   * the config file. It is the total words number of all samples
   * in this forward batch.
   * @param numSequences The sample number. It is equal to the batch_size
   * in the config file.
   * @param starts Each start position of each samples.
   * @param inputValue The input values.
   */
  void forwardSequence(int batchSize,
                       size_t numSequences,
                       const int *starts,
                       MatrixPtr inputValue);
  /**
   * Compute lstm backward one sequence by one sequence.
   */
  void backwardSequence(int batchSize,
                        size_t numSequences,
                        const int *starts,
                        MatrixPtr inputGrad);

  /**
   * Compute lstm forward one batch by one batch. The batch value is
   * reorganized by SequenceToBatch class. The batch output value will
   * be convert into sequence value after finishing forward. Here, one
   * batch contains one word of each sample. If the length of each sample
   * is not equality, the batch will not pads zero and contains less words.
   * The total batch numbers are the max length of the sequence. The details
   * can refer to SequenceToBatch class. On GPU mode, it will launch GPU
   * kernel for loop.
   *
   * @code
   * for (int i = 0; i < numBatch(max_sequence_length); ++i) {
   *   compute one batch.
   * }
   * @endcode
   */
  void forwardBatch(int batchSize,
                    size_t numSequences,
                    const int *starts,
                    MatrixPtr inputValue);
  /**
   * Compute lstm backward one batch by one batch.
   */
  void backwardBatch(int batchSize,
                     size_t numSequences,
                     const int *starts,
                     MatrixPtr inputGrad);

  /**
   * This function only supports GPU. It not need to reorganize input into
   * batch value. It will launch one kernel to parallelly compute forward
   * propagation in sequence level.
   */
  void forwardSeqParallel(int batchSize,
                          size_t numSequences,
                          const int *starts,
                          MatrixPtr inputValue);
  /**
   * Backward propagation corresponding to forwardSeqParallel.
   */
  void backwardSeqParallel(int batchSize,
                           size_t numSequences,
                           const int *starts,
                           MatrixPtr inputGrad);
  /**
   * This function is used for sequence generation and get output after
   * forwardBatch.
   */
  void getPrevBatchOutput(size_t numSequences);
  /**
   * This function is used for sequence generation and get state after
   * forwardBatch.
   */
  void getPrevBatchState(size_t numSequences);

 protected:
  /// Learned parameters, shape: (size, 4*size).
  /// The weight ([size, 4*size]) contains \f$W_{hi}, W_{hf}, W_{hc}, W_{ho}\f$.
  std::unique_ptr<Weight> weight_;
  /// Learned bias parameter, shape: (1, 7 * size).
  /// The bias contains \f$b_i, b_f, b_c, b_o\f$ and \f$W_{ci}, W_{cf},
  /// W_{co}\f$.
  std::unique_ptr<Weight> bias_;
  /// The reeal bias, point to \f$b_i, b_f, b_c, b_o\f$.
  MatrixPtr localBias_;
  /// The peephole connection for input gate.
  MatrixPtr checkIg_;
  /// The peephole connection for forget gate.
  MatrixPtr checkFg_;
  /// The peephole connection for output gate.
  MatrixPtr checkOg_;
  /// The gradient of real bias
  MatrixPtr localBiasGrad_;
  /// The gradient of peephole connection for input gates.
  MatrixPtr checkIgGrad_;
  /// The gradient of peephole connection for forget gates.
  MatrixPtr checkFgGrad_;
  /// The gradient of peephole connection for output gates.
  MatrixPtr checkOgGrad_;

  /// Stores the cell state of previous time step, namely \f$c_{t-1}\f$.
  Argument state_;
  /// Stores the hidden of previous time step, namely \f$h_{t-1}\f$.
  Argument preOutput_;
  /// Stores the value and gradient of four gates, namely
  /// \f$i_t, f_t, o_t, c_t\f$.
  Argument gate_;
  /// Whether it is reversed lstm.
  bool reversed_;
  /// Whether to use batch method to compute.
  bool useBatch_;
  /// Whether to use sequence parallell method to compute.
  bool useSeqParallel_;
  /// batchValue_ is used in method of batch calculation. It stores the
  /// batch value after reorganized input.
  std::unique_ptr<SequenceToBatch> batchValue_;
  /// The gradient of batchValue_.
  std::unique_ptr<SequenceToBatch> batchGrad_;

  /// Used in generation and stores the state of previous time step.
  MatrixPtr prevState_;
  /// Used in generation and stores the output of previous time step.
  MatrixPtr prevOutput_;
  MatrixPtr prevBatchOutput2_;
  /// The total state.
  MatrixPtr totalState_;
};

}  // namespace paddle
