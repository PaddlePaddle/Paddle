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

#include <set>
#include <vector>

#include "paddle/math/Vector.h"
#include "paddle/utils/StringUtil.h"

#include "Evaluator.h"

namespace paddle {

/**
 * Chunk evaluator is used to evaluate segment labelling accuracy for a
 * sequence. It calculates the chunk detection F1 score.
 *
 * A chunk is correctly detected if its beginning, end and type are correct.
 * Other chunk type is ignored.
 * For each label in the label sequence, we have
 *
 * @code
 * tagType = label % numTagType
 * chunkType = label / numTagType
 * otherChunkType = numChunkTypes
 * @endcode
 *
 * The total number of different labels is numTagType*numChunkTypes+1
 * We support 4 labelling scheme
 * The tag type for each of the scheme is shown as follows:
 *
 * @code
 *  Scheme Begin Inside End   Single
 *   plain  0     -      -     -
 *   IOB    0     1      -     -
 *   IOE    -     0      1     -
 *   IOBES  0     1      2     3
 * @endcode
 *
 * 'plain' means the whole chunk must contain exactly the same chunk label.
 */
class ChunkEvaluator : public Evaluator {
  int otherChunkType_;
  int numChunkTypes_;  // number of chunk types besides other chunk type
  int numTagTypes_;
  int tagBegin_;
  int tagInside_;
  int tagEnd_;
  int tagSingle_;

  int64_t numLabelSegments_;
  int64_t numOutputSegments_;
  int64_t numCorrect_;

  struct Segment {
    int begin;
    int end;
    int type;
    bool operator==(const Segment& y) const {
      return begin == y.begin && end == y.end && type == y.type;
    }
  };

  std::vector<Segment> labelSegments_;
  std::vector<Segment> outputSegments_;
  std::set<int> excludedChunkTypes_;
  mutable std::unordered_map<std::string, real> values_;

 public:
  virtual void init(const EvaluatorConfig& config) {
    Evaluator::init(config);
    if (config.chunk_scheme() == "IOB") {
      numTagTypes_ = 2;
      tagBegin_ = 0;
      tagInside_ = 1;
      tagEnd_ = -1;
      tagSingle_ = -1;
    } else if (config.chunk_scheme() == "IOE") {
      numTagTypes_ = 2;
      tagBegin_ = -1;
      tagInside_ = 0;
      tagEnd_ = 1;
      tagSingle_ = -1;
    } else if (config.chunk_scheme() == "IOBES") {
      numTagTypes_ = 4;
      tagBegin_ = 0;
      tagInside_ = 1;
      tagEnd_ = 2;
      tagSingle_ = 3;
    } else if (config.chunk_scheme() == "plain") {
      numTagTypes_ = 1;
      tagBegin_ = -1;
      tagInside_ = -1;
      tagEnd_ = -1;
      tagSingle_ = -1;
    } else {
      LOG(FATAL) << "Unknown chunk scheme: " << config.chunk_scheme();
    }
    CHECK(config.has_num_chunk_types()) << "Missing num_chunk_types in config";
    otherChunkType_ = numChunkTypes_ = config.num_chunk_types();

    // the chunks of types in excludedChunkTypes_ will not be counted
    auto& tmp = config.excluded_chunk_types();
    excludedChunkTypes_.insert(tmp.begin(), tmp.end());
  }

  virtual void start() {
    Evaluator::start();
    numLabelSegments_ = 0;
    numOutputSegments_ = 0;
    numCorrect_ = 0;
  }

  virtual void printStats(std::ostream& os) const {
    storeLocalValues();
    os << config_.name() << "=" << values_["F1-score"]
       << " true_chunks=" << numLabelSegments_
       << " result_chunks=" << numOutputSegments_
       << " correct_chunks=" << numCorrect_;
  }

  virtual void distributeEval(ParameterClient2* client) {
    int64_t buf[3] = {numLabelSegments_, numOutputSegments_, numCorrect_};
    client->reduce(buf, buf, 3, FLAGS_trainer_id, 0);
    numLabelSegments_ = buf[0];
    numOutputSegments_ = buf[1];
    numCorrect_ = buf[2];
  }

  virtual real evalImp(std::vector<Argument>& arguments) {
    CHECK_EQ(arguments.size(), (size_t)2);
    IVectorPtr& output = arguments[0].ids;
    IVectorPtr& label = arguments[1].ids;
    CHECK(!output->useGpu() && !label->useGpu()) << "Not supported";
    auto sequenceStartPositions =
        arguments[1].sequenceStartPositions->getVector(false);
    CHECK_EQ(output->getSize(), label->getSize());
    CHECK(sequenceStartPositions);
    size_t numSequences = sequenceStartPositions->getSize() - 1;
    const int* starts = sequenceStartPositions->getData();
    for (size_t i = 0; i < numSequences; ++i) {
      eval1(output->getData() + starts[i],
            label->getData() + starts[i],
            starts[i + 1] - starts[i]);
    }
    return 0;
  }

  void eval1(int* output, int* label, int length) {
    getSegments(output, length, outputSegments_);
    getSegments(label, length, labelSegments_);
    size_t i = 0, j = 0;
    while (i < outputSegments_.size() && j < labelSegments_.size()) {
      if (outputSegments_[i] == labelSegments_[j] &&
          excludedChunkTypes_.count(outputSegments_[i].type) != 1) {
        ++numCorrect_;
      }
      if (outputSegments_[i].end < labelSegments_[j].end) {
        ++i;
      } else if (outputSegments_[i].end > labelSegments_[j].end) {
        ++j;
      } else {
        ++i;
        ++j;
      }
    }
    for (auto& segment : labelSegments_) {
      if (excludedChunkTypes_.count(segment.type) != 1) ++numLabelSegments_;
    }
    for (auto& segment : outputSegments_) {
      if (excludedChunkTypes_.count(segment.type) != 1) ++numOutputSegments_;
    }
  }

  void getSegments(int* label, int length, std::vector<Segment>& segments) {
    segments.clear();
    segments.reserve(length);
    int chunkStart = 0;
    bool inChunk = false;
    int tag = -1;
    int type = otherChunkType_;
    for (int i = 0; i < length; ++i) {
      int prevTag = tag;
      int prevType = type;
      CHECK_LE(label[i], numChunkTypes_ * numTagTypes_);
      tag = label[i] % numTagTypes_;
      type = label[i] / numTagTypes_;
      if (inChunk && isChunkEnd(prevTag, prevType, tag, type)) {
        Segment segment{
            chunkStart,  // begin
            i - 1,       // end
            prevType,
        };
        segments.push_back(segment);
        inChunk = false;
      }
      if (isChunkBegin(prevTag, prevType, tag, type)) {
        chunkStart = i;
        inChunk = true;
      }
    }
    if (inChunk) {
      Segment segment{
          chunkStart,  // begin
          length - 1,  // end
          type,
      };
      segments.push_back(segment);
    }
  }

  // whether (prevTag, prevType) is the end of a chunk
  bool isChunkEnd(int prevTag, int prevType, int tag, int type) {
    if (prevType == otherChunkType_) return false;
    if (type == otherChunkType_) return true;
    if (type != prevType) return true;
    if (prevTag == tagBegin_) return tag == tagBegin_ || tag == tagSingle_;
    if (prevTag == tagInside_) return tag == tagBegin_ || tag == tagSingle_;
    if (prevTag == tagEnd_) return true;
    if (prevTag == tagSingle_) return true;
    return false;
  }

  // whether (tag, type) is the beginning of a chunk
  bool isChunkBegin(int prevTag, int prevType, int tag, int type) {
    if (prevType == otherChunkType_) return type != otherChunkType_;
    if (type == otherChunkType_) return false;
    if (type != prevType) return true;
    if (tag == tagBegin_) return true;
    if (tag == tagInside_) return prevTag == tagEnd_ || prevTag == tagSingle_;
    if (tag == tagEnd_) return prevTag == tagEnd_ || prevTag == tagSingle_;
    if (tag == tagSingle_) return true;
    return false;
  }

  // three metrics: precision, recall and F1-score
  void getNames(std::vector<std::string>* names) {
    storeLocalValues();
    names->reserve(names->size() + values_.size());
    for (auto it = values_.begin(); it != values_.end(); ++it) {
      names->push_back(config_.name() + "." + it->first);
    }
  }

  // get value by field name
  real getValue(const std::string& name, Error* err) const {
    storeLocalValues();
    std::vector<std::string> buffers;
    paddle::str::split(name, '.', &buffers);
    auto it = values_.find(buffers.back());
    if (it == values_.end()) {  // not found
      *err = Error("No such key %s", name.c_str());
      return 0.0f;
    }

    return it->second;
  }

  // get type of evaluator
  std::string getType(const std::string& name, Error* err) const {
    this->getValue(name, err);
    if (!err->isOK()) {
      return "";
    }
    return "chunk";
  }

 private:
  void storeLocalValues() const {
    CHECK_GE(numOutputSegments_, 0);
    CHECK_GE(numLabelSegments_, 0);
    double precision =
        !numOutputSegments_ ? 0 : (double)numCorrect_ / numOutputSegments_;
    double recall =
        !numLabelSegments_ ? 0 : (double)numCorrect_ / numLabelSegments_;
    values_["precision"] = precision;
    values_["recall"] = recall;
    values_["F1-score"] =
        !numCorrect_ ? 0 : 2 * precision * recall / (precision + recall);
  }
};

REGISTER_EVALUATOR(chunk, ChunkEvaluator);

}  // namespace paddle
