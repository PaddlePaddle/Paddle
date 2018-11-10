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

#include "Layer.h"

namespace paddle {

class KmaxSeqScoreLayer : public Layer {
 private:
  MatrixPtr scores_;
  size_t beamSize_;
  void kmaxScorePerSeq(const real* score,
                       real* sortedRes,
                       const ICpuGpuVectorPtr seqStartPos);

 public:
  explicit KmaxSeqScoreLayer(const LayerConfig& config) : Layer(config) {}

  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override;

  void forward(PassType passType) override;
  void backward(const UpdateCallback& callback = nullptr) override;
};

REGISTER_LAYER(kmax_seq_score, KmaxSeqScoreLayer);

bool KmaxSeqScoreLayer::init(const LayerMap& layerMap,
                             const ParameterMap& parameterMap) {
  bool ret = Layer::init(layerMap, parameterMap);
  CHECK_EQ(1U, inputLayers_.size());

  beamSize_ = config_.beam_size();
  CHECK_GE(beamSize_, 1U);

  setNeedSequenceInfo(false);
  setNeedGradient(false);
  return ret;
}

void KmaxSeqScoreLayer::kmaxScorePerSeq(const real* scores,
                                        real* sortedIds,
                                        const ICpuGpuVectorPtr seqStartPos) {
  int* starts = seqStartPos->getMutableData(false);
  std::vector<real> indices;
  for (size_t i = 0; i < seqStartPos->getSize() - 1; ++i) {
    int seqLen = starts[i + 1] - starts[i];
    int k = std::min(static_cast<int>(beamSize_), seqLen);

    indices.resize(seqLen, 0);
    std::iota(begin(indices), end(indices), 0.);
    std::vector<real> tmpScore(scores + starts[i], scores + starts[i + 1]);
    std::partial_sort(
        begin(indices),
        begin(indices) + k,
        end(indices),
        [&](size_t a, size_t b) { return tmpScore[a] > tmpScore[b]; });
    memcpy(sortedIds + (i * beamSize_), indices.data(), k * sizeof(real));
  }
}

void KmaxSeqScoreLayer::forward(PassType passType) {
  Layer::forward(passType);

  const Argument& input = getInput(0);
  const MatrixPtr inputScore = getInputValue(0);

  CHECK(input.hasSeq() || input.hasSubseq())
      << "input of " << getName()
      << " must be a sequence or a nested sequence.";
  CHECK_EQ(input.value->getWidth(), 1UL)
      << "input of " << getName() << " are scores over a sequence or "
      << "a nested sequence, so its width must be 1.";

  if (useGpu_) {
    /*
     * currently, this Layer only runs in CPU, if the other part of the model is
     * runing on GPU, then copy the input to this layer from GPU to CPU.
     */
    Matrix::resizeOrCreate(scores_,
                           inputScore->getHeight(),
                           1,
                           false /* trans */,
                           false /* useGpu */);
    scores_->copyFrom(*inputScore);
  } else {
    scores_ = inputScore;
  }

  /*
   * TODO(caoying)
   * In PaddePaddle, currently all matrices are real number types,
   * but output of this layer which is some selected indices of the give
   * sequence are actually filled with int types so that storing int types
   * information in a real number matrix is dangerous, since real numbers will
   * be convered to int types.
   */
  Matrix::resizeOrCreate(
      output_.value,
      input.hasSubseq() ? input.getNumSubSequences() : input.getNumSequences(),
      beamSize_,
      false,
      false);
  output_.value->one();
  output_.value->mulScalar(-1.);

  kmaxScorePerSeq(scores_->getData(),
                  output_.value->getData(),
                  input.hasSubseq() ? input.subSequenceStartPositions
                                    : input.sequenceStartPositions);
}

void KmaxSeqScoreLayer::backward(const UpdateCallback& callback) {}

}  // namespace paddle
