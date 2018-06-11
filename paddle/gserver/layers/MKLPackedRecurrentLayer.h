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

#include "MKLPackedWeight.h"
#include "RecurrentLayer.h"

DECLARE_bool(rnn_use_batch);

namespace paddle {

/**
 * @brief MKLPackedRecurrentLayer is almost the same with RecurrentLayer
 * but is optimized with MKL cblas packed gemm.
 * More details:
 * https://github.com/PaddlePaddle/Paddle/blob/develop/doc/design/mkl/mkl_packed.md
 */

class MKLPackedRecurrentLayer : public RecurrentLayer {
 public:
  explicit MKLPackedRecurrentLayer(const LayerConfig& config)
      : RecurrentLayer(config) {}

  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override;

  void backward(const UpdateCallback& callback) override;

 protected:
  void forwardBatch(int batchSize,
                    size_t numSequences,
                    const int* starts) override;

  void backwardBatch(int batchSize,
                     size_t numSequences,
                     const int* starts) override;

 protected:
  /// packed_weight_ contains same data with
  /// RecurrentLayer::weight_ but is packed
  std::unique_ptr<MKLPackedWeight> packed_weight_;
  /// packed_weightT_ is the transposition matrix of packed_weight_
  std::unique_ptr<MKLPackedWeight> packed_weightT_;
};

}  // namespace paddle
