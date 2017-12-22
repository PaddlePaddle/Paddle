/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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
#include "MKLPackedWeight.h"
#include "RecurrentLayer.h"
#include "SequenceToBatch.h"
#include "paddle/utils/Stat.h"

DECLARE_bool(rnn_use_batch);

namespace paddle {

/**
 * @brief MKLPackedRecurrentLayer takes 1 input layer. The output size is the
 * same with
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
  std::unique_ptr<MKLPackedWeight> packed_weight_;
  std::unique_ptr<MKLPackedWeight> packed_weightT_;
};

}  // namespace paddle
