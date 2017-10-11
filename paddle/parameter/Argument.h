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

#include "hl_gpu.h"

#include "paddle/math/Matrix.h"
#include "paddle/math/Vector.h"
#include "paddle/parameter/Parameter.h"
#include "paddle/utils/Locks.h"
#include "paddle/utils/Util.h"

namespace paddle {

typedef std::shared_ptr<std::vector<std::string>> SVectorPtr;

struct Argument {
  Argument()
      : in(nullptr),
        value(nullptr),
        ids(nullptr),
        grad(nullptr),
        strs(nullptr),
        frameHeight(0),
        frameWidth(0),
        sequenceStartPositions(nullptr),
        subSequenceStartPositions(nullptr),
        cpuSequenceDims(nullptr),
        deviceId(-1),
        allCount(0),
        valueCount(0),
        gradCount(0),
        dataId(0) {}
  Argument(const Argument& argument) {
    *this = argument;
    valueCount = 0;
    gradCount = 0;
    dataId = argument.dataId;
  }
  ~Argument() {}

  void operator=(const Argument& argument) {
    in = argument.in;
    value = argument.value;
    ids = argument.ids;
    grad = argument.grad;
    strs = argument.strs;
    sequenceStartPositions = argument.sequenceStartPositions;
    subSequenceStartPositions = argument.subSequenceStartPositions;
    cpuSequenceDims = argument.cpuSequenceDims;
    deviceId = argument.deviceId;
    allCount = argument.allCount;
    frameHeight = argument.frameHeight;
    frameWidth = argument.frameWidth;
    dataId = argument.dataId;
  }

  MatrixPtr in;  // used if needed
  MatrixPtr value;
  IVectorPtr ids;  // a sequence of ids. Can be use for class id for costLayer
  MatrixPtr grad;  // If empty, gradient is not needed.
  SVectorPtr strs;

  // A dataBatch includes batchSize frames, one frame maybe not only vector
  size_t frameHeight;
  size_t frameWidth;

  // If NULL, each position is treated independently.
  // Otherwise, its size should be #NumberOfSequences + 1.
  // The first position is always 0 and
  // the last position should be equal to batchSize.
  ICpuGpuVectorPtr sequenceStartPositions;

  // If NULL, each sequence has no subsequence.
  // Otherwise, its size should be #NumberOfSubSequences + 1.
  // The first position is always 0 and
  // the last position should be equal to batchSize.
  ICpuGpuVectorPtr subSequenceStartPositions;

  // dimension of sequence, stored only in CPU
  IVectorPtr cpuSequenceDims;

  int deviceId;            // the GPU device id which the argument in
  int allCount;            // the number of output layers using this argument
  mutable int valueCount;  // waiting this member when layer do forward
  mutable int gradCount;   // waiting this member when layer do backward
  mutable LockedCondition valueReadyCond;
  mutable LockedCondition gradReadyCond;

  int dataId;  // dataProvider id

  /* Increase the reference count of the argument. */
  void countIncrement() { allCount++; }

  int getAllCount() const { return allCount; }

  void waitValueReady() const {
    valueReadyCond.wait([this] { return (valueCount != 0); });

    std::lock_guard<std::mutex> guard(*valueReadyCond.mutex());
    valueCount--;
  }

  void notifyValueReady() const {
    valueReadyCond.notify_all([this] { valueCount = allCount; });
  }

  void waitGradReady() const {
    gradReadyCond.wait([this] { return (gradCount == allCount); });
    gradCount = 0;
  }

  void notifyGradReady() const {
    gradReadyCond.notify_all([this] { gradCount++; });
  }

  int64_t getBatchSize() const {
    if (value) return value->getHeight();
    if (ids) return ids->getSize();
    if (grad) return grad->getHeight();
    if (in) return in->getHeight();
    if (strs) return strs->size();
    return 0;
  }
  size_t getFrameHeight() const { return frameHeight; }
  size_t getFrameWidth() const { return frameWidth; }
  void setFrameHeight(size_t h) { frameHeight = h; }
  void setFrameWidth(size_t w) { frameWidth = w; }

  int64_t getNumSequences() const {
    return sequenceStartPositions ? sequenceStartPositions->getSize() - 1
                                  : getBatchSize();
  }

  int64_t getNumSubSequences() const {
    return subSequenceStartPositions ? subSequenceStartPositions->getSize() - 1
                                     : getBatchSize();
  }

  bool hasSubseq() const { return subSequenceStartPositions != nullptr; }

  const int* getCpuStartPositions() const {
    return hasSubseq() ? subSequenceStartPositions->getData(false)
                       : sequenceStartPositions->getData(false);
  }

  static inline real sum(const std::vector<Argument>& arguments) {
    real cost = 0;
    for (auto& arg : arguments) {
      if (arg.value) {
        SetDevice device(arg.deviceId);
        cost += arg.value->getSum();
      }
    }
    return cost;
  }

  /**
   * @brief (value, ids, grad, sequenceStartPositions) of output are subset of
   *        input. Note that, output share the same memory of input.
   *
   * @param input[in]       input
   * @param offset[in]      offset in terms of rows
   * @param height[in]      height of output.value
   * @param width[in]       width of output.value
   * @param useGpu[in]
   * @param trans[in]       whether input.value is transform
   * @param seqFlag[in]     whether input has sequenceStartPositions
   * @param seqStart[in]    offset of input.sequenceStartPositions
   * @param seqSize[in]     lenght of output.sequenceStartPositions
   */
  void subArgFrom(const Argument& input,
                  size_t offset,
                  size_t height,
                  size_t width,
                  bool useGpu,
                  bool trans = false,
                  bool seqFlag = false,
                  size_t seqStart = 0,
                  size_t seqSize = 0);
  /*
   * for sequence input:
   *   startSeq: the sequence id of start
   *   copySize: how many sequences need to copy
   *   return value: how many samples are copied
   * for non-sequence input:
   *   startSeq: the sample id of start
   *   copySize: how many samples need to copy
   *   return value: how many samples are copied
   * Note that when specifying the stream explicitly in this case,
   * synchronize should also be called somewhere after this function
   */
  int32_t resizeAndCopyFrom(const Argument& src,
                            int32_t startSeq,
                            int32_t copySize,
                            bool useGpu,
                            hl_stream_t stream);

  /*
   * same with the above function, except that the stream is
   * HPPL_STREAM_DEFAULT and synchronize is automatically called
   * inside it
   */
  int32_t resizeAndCopyFrom(const Argument& src,
                            int32_t startSeq,
                            int32_t copySize,
                            bool useGpu = FLAGS_use_gpu);

  void resizeAndCopyFrom(const Argument& src, bool useGpu, hl_stream_t stream);

  /*
   * same with the above function, except that the stream is
   * HPPL_STREAM_DEFAULT and synchronize is automatically called
   * inside it
   */
  void resizeAndCopyFrom(const Argument& src, bool useGpu = FLAGS_use_gpu);

  /*
    @brief Concatenate several arguments into one and put the result into it.
    @param args : a vector of argument, each element of which is a frame in a
    batch of sequences.
    @param selectRows : select several row of args to concatenate
    @param seqStartPos : sequence start positions in the final Argument
    @param hl_stream_t : cuda stream
    @param passTyoe : type of task, training or testing
   */
  void concat(const std::vector<Argument>& args,
              const std::vector<int>& selectRows,
              const std::vector<int>& seqStartPos,
              bool useGpu,
              hl_stream_t stream,
              PassType passType);

  /*
    Concatenate several args into one and put the result into this.
   */
  void concat(const std::vector<Argument>& src,
              bool useGpu = FLAGS_use_gpu,
              hl_stream_t stream = HPPL_STREAM_DEFAULT,
              PassType passType = PASS_TEST);

  /*
   * split vector<Argument> to several vectors according to dataId
   */
  static void splitByDataId(const std::vector<Argument>& argus,
                            std::vector<std::vector<Argument>>* arguGroups);

  struct SeqInfo {
    // Equal to sequence length for sequence data
    // Equal to number of subsequences for subsequence data
    int topLevelLength;

    int seqStart;
    int seqId;

    // Equal to topLevelLength for sequence data
    // Equal to sum of the length of subsequences for subsequence data
    int subLevelLength;

    // Only used for subsequence data, start position of this sequence
    // is subSequenceStartPositions, i.e.
    // subSequenceStartPositions[subSeqStart] == seqStart
    int subSeqStart;
  };
  /*
    Get SeqInfo for each sequence of this argument
    Elements in *seqInfo are sorted by topLevelLength in descending order
  */
  void getSeqInfo(std::vector<SeqInfo>* segInfo) const;

  /*
   Check Whether sequenceStartPositions is subset of
   subSequenceStartPositions.
   */
  void checkSubset() const;

  /*
   sequence has sub-sequence degrades to a sequence.
   */
  void degradeSequence(const Argument& input);

  /*
   After pooling with stride n (n is smaller than sequence length),
   a long sequence will be shorten.
   This function is invalid for sequence having sub-sequence.
   */
  void poolSequenceWithStride(const Argument& input,
                              size_t stride,
                              IVectorPtr* stridePositions,
                              bool reversed = false);
  /**
   * @brief getValueString will return the argument's output in string. There
   * are several kinds of output. The keys of output dictionary are 'value',
   * 'id', 'sequence pos', 'sub-sequence pos'.
   * @param out [out]: the return values.
   */
  void getValueString(std::unordered_map<std::string, std::string>* out) const;

  /**
   * @brief printValueString will print the argument's output in order of
   * 'value', 'id', 'sequence pos', 'sub-sequence pos'.
   * @param stream: Output stream
   * @param prefix: line prefix for printing.
   */
  void printValueString(std::ostream& stream,
                        const std::string& prefix = "") const;
};

}  // namespace paddle
