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

#include "Layer.h"
#include "LstmCompute.h"
#include "LstmLayer.h"
#include "MKLPackedWeight.h"
#include "SequenceToBatch.h"
#include "paddle/math/BaseMatrix.h"
#include "paddle/math/MathFunctions.h"
#include "paddle/math/Matrix.h"
#include "paddle/utils/Stat.h"

namespace paddle {

/**
 * @brief MKLPackedLstmLayer is the same with LstmLayer but is optimized
 * with MKL cblas packed gemm.
 * more details:
 * https://github.com/PaddlePaddle/Paddle/blob/develop/doc/design/mkl/mkl_packed.md
 */

class MKLPackedLstmLayer : public LstmLayer {
public:
  explicit MKLPackedLstmLayer(const LayerConfig &config) : LstmLayer(config) {}

  bool init(const LayerMap &layerMap, const ParameterMap &parameterMap);

  void forward(PassType passType);

  void backward(const UpdateCallback &callback);

protected:
  void forwardBatch(int batchSize,
                    size_t numSequences,
                    const int *starts,
                    MatrixPtr inputValue);

protected:
  /// packed_weight_ contains the same data with
  /// RecurrentLayer::weight_ but is packed
  std::unique_ptr<MKLPackedWeight> packedWeight_;
};

}  // namespace paddle
