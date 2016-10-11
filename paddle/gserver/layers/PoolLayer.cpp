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

#include "paddle/utils/Logging.h"
#include "PoolLayer.h"
#include "PoolProjectionLayer.h"
#ifndef PADDLE_ONLY_CPU
#include "CudnnPoolLayer.h"
#endif
namespace paddle {

REGISTER_LAYER_CREATE_FUNC(pool, &PoolLayer::create);

bool PoolLayer::init(const LayerMap& layerMap,
                     const ParameterMap& parameterMap) {
  /* Initialize the basic parent class */
  Layer::init(layerMap, parameterMap);

  /* the size of inputs for pool-layer is 1 */
  CHECK_EQ(config_.inputs_size(), 1);

  const PoolConfig& conf = config_.inputs(0).pool_conf();
  poolType_ = conf.pool_type();
  channels_ = conf.channels();
  sizeX_ = conf.size_x();
  stride_ = conf.stride();
  outputX_ = conf.output_x();
  imgSize_ = conf.img_size();
  confPadding_ = conf.padding();

  sizeY_ = conf.has_size_y() ? conf.size_y() : conf.size_x();
  imgSizeY_ = conf.has_img_size_y() ? conf.img_size_y() : conf.img_size();
  strideY_ = conf.has_stride_y() ? conf.stride_y() : conf.stride();
  confPaddingY_ = conf.has_padding_y() ? conf.padding_y() : conf.padding();
  outputY_ = conf.has_output_y() ? conf.output_y() : conf.output_x();

  return true;
}

Layer* PoolLayer::create(const LayerConfig& config) {
  CHECK_EQ(config.inputs_size(), 1);
  const std::string& pool = config.inputs(0).pool_conf().pool_type();
  if (pool == "max-projection") {
    return new MaxPoolProjectionLayer(config);
  } else if (pool == "avg-projection") {
    return new AvgPoolProjectionLayer(config);
#ifndef PADDLE_ONLY_CPU
  } else if (CudnnPoolLayer::typeCheck(pool)) {
    return new CudnnPoolLayer(config);
#endif
  } else {
    LOG(FATAL) << "Unknown pool type: " << pool;
    return nullptr;
  }
}

REGISTER_LAYER(seqpool, SequencePoolLayer);

bool SequencePoolLayer::init(const LayerMap& layerMap,
                             const ParameterMap& parameterMap) {
  /* Initialize the basic parent class */
  Layer::init(layerMap, parameterMap);

  // seqlastins/max/average layer should have exactly 1 input
  CHECK_EQ(1U, inputLayers_.size());

  /* initialize biases_ */
  if (biasParameter_.get() != NULL) {
    biases_ = std::unique_ptr<Weight>(new Weight(1, getSize(), biasParameter_));
  }
  // transform to which sequence type
  if (config_.trans_type() == "non-seq") {
    type_ = kNonSeq;
  } else if (config_.trans_type() == "seq") {
    type_ = kSeq;
  } else {
    LOG(FATAL) << "Unknown trans_type: " << config_.trans_type();
  }
  setNeedSequenceInfo(false);
  return true;
}

void SequencePoolLayer::forward(PassType passType) {
  Layer::forward(passType);

  const Argument& input = getInput(0);
  newBatchSize_ = type_ ? input.getNumSubSequences() : input.getNumSequences();
  size_t dim = getSize();
  // check
  CHECK_EQ(dim, input.value->getWidth());
  startPositions_ =
      type_ ? input.subSequenceStartPositions : input.sequenceStartPositions;
  auto starts = startPositions_->getVector(false);
  CHECK_EQ(starts->getData()[newBatchSize_], input.getBatchSize());
  CHECK_EQ(newBatchSize_, starts->getSize() - 1);

  resetOutput(newBatchSize_, dim);
  if (type_) {
    // when trans_type = seq, input must hasSubseq
    CHECK_EQ(input.hasSubseq(), 1UL);
  }
  /* If type_ = kNonSeq, both seq has or not has sub-seq degrade to a non-seq,
   * thus, in this case, output_ has no sequenceStartPositions.
   * If type_ = kSeq, seq has sub-seq degrades to a seq, thus, only in this
   * case, we should compute the new sequenceStartPositions.
  */
  if (type_) {
    output_.degradeSequence(input, useGpu_);
  }
}

void SequencePoolLayer::backward(const UpdateCallback& callback) {
  /* Do derivation */ { backwardActivation(); }

  if (biases_ && biases_->getWGrad()) {
    biases_->getWGrad()->collectBias(*getOutputGrad(), 1);

    // Increasing the number of gradient
    biases_->getParameterPtr()->incUpdate(callback);
  }
}

}  // namespace paddle
