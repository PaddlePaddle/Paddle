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
#include <gflags/gflags.h>
#include "Layer.h"
#include "SequenceToBatch.h"
#include "paddle/utils/Stat.h"

namespace paddle {

/**
 * @brief RecurrentLayer takes 1 input layer. The output size is the same with
 * input layer.
 * For each sequence [start, end] it performs the following computation:
 * \f[
 *    out_{i} = act(in_{i})     \      \      \text{for} \ i = start \\
 *    out_{i} = act(in_{i} + out_{i-1} * W) \ \ \text{for} \ start < i <= end
 *
 * \f]
 * If reversed is true, the order is reversed:
 * \f[
 *   out_{i} = act(in_{i})           \    \   \text{for} \ i = end  \\
 *   out_{i} = act(in_{i} + out_{i+1} * W) \ \ \text{for} \ start <= i < end
 * \f]
 * There are two methods to calculate rnn. One way is to compute rnn one
 * sequence by one sequence. The other way is to reorganize the input
 * into batches, then compute rnn one batch by one batch. Users can select
 * them by rnn_use_batch flag.
 */

class RecurrentLayer : public Layer {
 public:
  explicit RecurrentLayer(const LayerConfig& config) : Layer(config) {}

  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override;

  void forward(PassType passType) override;

  void backward(const UpdateCallback& callback) override;

  void resetState() override;

  void setState(LayerStatePtr state) override;

  LayerStatePtr getState() override;

 protected:
  /**
   * @brief If user do not set --rnn_use_batch=true, it will
   * compute rnn forward one sequence by one sequence in default.
   * @param batchSize Total words number of all samples in this batch.
   * @param numSequences The sample number.
   * @param starts Each start position of each samples.
   */
  void forwardSequence(int batchSize, size_t numSequences, const int* starts);
  /**
   * @brief Compute rnn forward by one sequence.
   * @param start The start position of this sequence (or sample).
   * @param length The length of this sequence (or sample), namely the words
   * number of this sequence.
   */
  void forwardOneSequence(int start, int length);
  /**
   * @brief Compute rnn backward one sequence by onesequence.
   * @param batchSize Total words number of all samples in this batch.
   * @param numSequences The sample number.
   * @param starts Each start position of each samples.
   */
  void backwardSequence(int batchSize, size_t numSequences, const int* starts);
  /**
   * @brief Compute rnn backward by one sequence.
   * @param start The start position of this sequence (or sample).
   * @param length The length of this sequence (or sample), namely the words
   * number of this sequence.
   */
  void backwardOneSequence(int start, int length);

  /**
   * @brief Reorganize input into batches and compute rnn forward batch
   * by batch. It will convert batch shape to sequence after finishing forward.
   * The batch info can refer to SequenceToBatch class.
   * @param batchSize Total words number of all samples in this batch.
   * @param numSequences The sample number.
   * @param starts Each start position of each samples.
   */
  virtual void forwardBatch(int batchSize,
                            size_t numSequences,
                            const int* starts);

  /**
   * @brief Reorganize input into batches and compute rnn forward batch
   * by batch.
   * @param batchSize Total words number of all samples in this batch.
   * @param numSequences The sample number.
   * @param starts Each start position of each samples.
   */
  virtual void backwardBatch(int batchSize,
                             size_t numSequences,
                             const int* starts);

 protected:
  std::unique_ptr<Weight> weight_;
  std::unique_ptr<Weight> bias_;

  /// frameOutput_[i] is used to hold the i-th sample of output_
  std::vector<Argument> frameOutput_;
  MatrixPtr prevOutput_;
  /// Whether compute rnn by reverse.
  bool reversed_;
  /// If compute batch by batch, batchValue_ will be used to save the
  /// reorganized input value.
  std::unique_ptr<SequenceToBatch> batchValue_;
  /// If compute batch by batch, batchGrad_ will be used to save the
  /// gradient with respect to reorganized input value.
  std::unique_ptr<SequenceToBatch> batchGrad_;
};

}  // namespace paddle
