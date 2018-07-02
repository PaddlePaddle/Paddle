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

namespace paddle {

/**
 * @brief A layer integrating the open-source warp-ctc library
 *        <https://github.com/baidu-research/warp-ctc> to compute connectionist
 *        temporal classification cost.
 *
 * The config file api is warp_ctc_layer.
 */
class WarpCTCLayer : public Layer {
 public:
  explicit WarpCTCLayer(const LayerConfig& config) : Layer(config) {}
  ~WarpCTCLayer() {}

  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override;
  void forward(PassType passType) override;
  void backward(const UpdateCallback& callback) override;

 protected:
  /**
   * sequence matrix and batch matrix copy:
   * sequence (s0, s0, s0, s0; s1, s1; s2, s2, s2; s3)
   * batch    (s0, s1, s2, s3; s0, s1, s2, 0; s0, 0, s2, 0; s0, 0, 0, 0)
   */
  void seq2batchPadding(const MatrixPtr& seqValue,
                        MatrixPtr& batchValue,
                        const ICpuGpuVectorPtr& seqStartPositions);
  void batch2seqPadding(const MatrixPtr& seqValue,
                        MatrixPtr& batchValue,
                        const ICpuGpuVectorPtr& seqStartPositions,
                        bool normByTimes);

 protected:
  size_t numClasses_;
  size_t blank_;
  size_t maxSequenceLength_;
  bool normByTimes_;

  MatrixPtr batchValue_;
  MatrixPtr batchGrad_;
  VectorPtr workspace_;

  IVectorPtr cpuLabels_;
  MatrixPtr cpuCosts_;
};

}  // namespace paddle
