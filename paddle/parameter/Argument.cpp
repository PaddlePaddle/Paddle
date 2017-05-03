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

#include "Argument.h"
#include "paddle/math/SparseMatrix.h"

#include <algorithm>

namespace paddle {
static void resizeAndCopy(MatrixPtr& dest,
                          const MatrixPtr& src,
                          bool useGpu,
                          hl_stream_t stream) {
  if (src) {
    if (!dest) {
      dest = src->clone(0, 0, useGpu);
    } else {
      CHECK_EQ(dest->useGpu(), useGpu);
      dest->resize(src->getHeight(), src->getWidth());
    }
    dest->copyFrom(*src, stream);
  } else {
    dest.reset();
  }
}

static void resizeAndCopy(IVectorPtr& dest,
                          const IVectorPtr& src,
                          bool useGpu,
                          hl_stream_t stream) {
  if (src) {
    IVector::resizeOrCreate(dest, src->getSize(), useGpu);
    dest->copyFrom(*src, stream);
  } else {
    dest.reset();
  }
}

static void resizeAndCopy(ICpuGpuVectorPtr& dest,
                          const ICpuGpuVectorPtr& src,
                          bool useGpu,
                          hl_stream_t stream) {
  if (src) {
    ICpuGpuVector::resizeOrCreate(dest, src->getSize(), useGpu);
    dest->copyFrom(*src, stream);
  } else {
    dest.reset();
  }
}

static void resizeAndCopy(MatrixPtr& dest,
                          const MatrixPtr& src,
                          int32_t startRow,
                          int32_t copySize,
                          bool useGpu,
                          hl_stream_t stream = HPPL_STREAM_DEFAULT) {
  if (src) {
    CHECK_LE((size_t)startRow + copySize, src->getHeight());
    int height = copySize;
    int width = src->getWidth();
    if (!dest) {
      dest = src->clone(height, width, useGpu);
    } else {
      CHECK_EQ(dest->useGpu(), useGpu);
      dest->resize(height, width);
    }
    MatrixPtr submat = src->subMatrix(startRow, copySize);
    if (dynamic_cast<GpuSparseMatrix*>(dest.get())) {
      // copy a subMatrix of CpuSparseMatrix to GpuSparseMatrix.
      // First copy it to CPU, and then copy it to the GPU.
      MatrixPtr tmp = src->clone(height, width, false);
      tmp->copyFrom(*submat, stream);
      dest->copyFrom(*tmp, stream);
    } else {
      dest->copyFrom(*submat, stream);
    }
  } else {
    dest.reset();
  }
}

static void resizeAndCopy(IVectorPtr& dest,
                          const IVectorPtr& src,
                          int32_t startPos,
                          int32_t copySize,
                          bool useGpu,
                          hl_stream_t stream = HPPL_STREAM_DEFAULT) {
  if (src) {
    CHECK_LE((size_t)startPos + copySize, src->getSize());

    int height = copySize;
    IVector::resizeOrCreate(dest, height, useGpu);
    dest->copyFrom(src->getData() + startPos, height, stream);
  } else {
    dest.reset();
  }
}

static void resizeAndCopy(ICpuGpuVectorPtr& dest,
                          const ICpuGpuVectorPtr& src,
                          int32_t startPos,
                          int32_t copySize,
                          bool useGpu,
                          hl_stream_t stream = HPPL_STREAM_DEFAULT) {
  if (src) {
    CHECK_LE((size_t)startPos + copySize, src->getSize());

    ICpuGpuVector::resizeOrCreate(dest, copySize, useGpu);
    dest->copyFrom(*src, startPos, copySize, useGpu, stream);
  } else {
    dest.reset();
  }
}

static void resizeAndCopy(SVectorPtr& dest,
                          const SVectorPtr& src,
                          bool useGpu,
                          hl_stream_t stream) {
  if (src) {
    size_t height = src->size();
    if (!dest) {
      dest = std::make_shared<std::vector<std::string>>(height);
    } else {
      dest->resize(height);
    }
    std::copy_n(src->begin(), height, dest->begin());
  } else {
    dest.reset();
  }
}

static void resizeAndCopy(SVectorPtr& dest,
                          const SVectorPtr& src,
                          int32_t startPos,
                          int32_t copySize,
                          bool useGpu,
                          hl_stream_t stream = HPPL_STREAM_DEFAULT) {
  if (src) {
    CHECK_LE((size_t)startPos + copySize, src->size());
    size_t height = copySize;
    if (!dest) {
      dest = std::make_shared<std::vector<std::string>>(height);
    } else {
      dest->resize(height);
    }
    std::copy_n(src->begin() + startPos, height, dest->begin());
  } else {
    dest.reset();
  }
}

void Argument::resizeAndCopyFrom(const Argument& src, bool useGpu) {
  resizeAndCopyFrom(src, useGpu, HPPL_STREAM_DEFAULT);
  hl_stream_synchronize(HPPL_STREAM_DEFAULT);
}

void Argument::resizeAndCopyFrom(const Argument& src,
                                 bool useGpu,
                                 hl_stream_t stream) {
  dataId = src.dataId;
  resizeAndCopy(value, src.value, useGpu, stream);
  resizeAndCopy(grad, src.grad, useGpu, stream);
  resizeAndCopy(in, src.in, useGpu, stream);
  resizeAndCopy(ids, src.ids, useGpu, stream);
  resizeAndCopy(sequenceStartPositions,
                src.sequenceStartPositions,
                false /* useGpu */,
                stream);
  if (src.hasSubseq()) {
    resizeAndCopy(subSequenceStartPositions,
                  src.subSequenceStartPositions,
                  false /* useGpu */,
                  stream);
  }
  resizeAndCopy(strs, src.strs, useGpu, stream);
  frameWidth = src.frameWidth;
  frameHeight = src.frameHeight;
}

int32_t Argument::resizeAndCopyFrom(const Argument& src,
                                    int32_t startSeq,
                                    int32_t copySize,
                                    bool useGpu) {
  int32_t size =
      resizeAndCopyFrom(src, startSeq, copySize, useGpu, HPPL_STREAM_DEFAULT);
  hl_stream_synchronize(HPPL_STREAM_DEFAULT);
  return size;
}

int32_t Argument::resizeAndCopyFrom(const Argument& src,
                                    int32_t startSeq,
                                    int32_t copySize,
                                    bool useGpu,
                                    hl_stream_t stream) {
  dataId = src.dataId;
  frameWidth = src.frameWidth;
  frameHeight = src.frameHeight;

  if (!src.sequenceStartPositions) {
    // non-sequence input, copy samples directly
    int32_t startRow = startSeq;
    resizeAndCopy(in, src.in, startRow, copySize, useGpu, stream);
    resizeAndCopy(value, src.value, startRow, copySize, useGpu, stream);
    resizeAndCopy(grad, src.grad, startRow, copySize, useGpu, stream);
    resizeAndCopy(ids, src.ids, startRow, copySize, useGpu, stream);
    resizeAndCopy(strs, src.strs, startRow, copySize, useGpu, stream);
    return copySize;
  } else {
    // sequence input
    const int* sequence = src.sequenceStartPositions->getData(false);
    int32_t startRow = sequence[startSeq];           // sample start from here
    int32_t endRow = sequence[startSeq + copySize];  // sample end
    int32_t copyFeatureSize = endRow - startRow;     // num of samples
    resizeAndCopy(in, src.in, startRow, copyFeatureSize, useGpu, stream);
    resizeAndCopy(value, src.value, startRow, copyFeatureSize, useGpu, stream);
    resizeAndCopy(grad, src.grad, startRow, copyFeatureSize, useGpu, stream);
    resizeAndCopy(ids, src.ids, startRow, copyFeatureSize, useGpu, stream);
    resizeAndCopy(sequenceStartPositions,
                  src.sequenceStartPositions,
                  startSeq,
                  copySize + 1,
                  false,
                  stream);
    // modify new sequenceStartPositions
    int* destSequences = sequenceStartPositions->getMutableData(false);
    for (int i = 0; i < copySize + 1; i++) {
      destSequences[i] -= startRow;
    }
    CHECK_EQ(destSequences[0], 0);
    CHECK_EQ(destSequences[copySize], copyFeatureSize);
    if (src.hasSubseq()) {
      // sequence has sub-sequence
      int* subSequence = src.subSequenceStartPositions->getMutableData(false);
      int32_t subStartSeq = 0;
      int32_t subEndSeq = 0;
      int numSubSequences = src.getNumSubSequences();
      for (int i = 0; i < numSubSequences + 1; i++) {
        if (subSequence[i] == startRow) {
          subStartSeq = i;
        } else if (subSequence[i] == endRow) {
          subEndSeq = i;
          break;
        }
      }
      int32_t copySubSize = subEndSeq - subStartSeq;
      resizeAndCopy(subSequenceStartPositions,
                    src.subSequenceStartPositions,
                    subStartSeq,
                    copySubSize + 1,
                    false,
                    stream);
      // modify new subSequenceStartPositions
      int* destSubSequences = subSequenceStartPositions->getMutableData(false);
      for (int i = 0; i < copySubSize + 1; i++) {
        destSubSequences[i] -= startRow;
      }
      CHECK_EQ(destSubSequences[0], 0);
      CHECK_EQ(destSubSequences[copySubSize], copyFeatureSize);
    }
    resizeAndCopy(strs, src.strs, startRow, copySize, useGpu, stream);
    return copyFeatureSize;
  }
}

void Argument::concat(const std::vector<Argument>& args,
                      const std::vector<int>& selectRows,
                      const std::vector<int>& seqStartPos,
                      bool useGpu,
                      hl_stream_t stream,
                      PassType passType) {
  CHECK(!subSequenceStartPositions)
      << "undefined behavior for subsequence positions";

  size_t batchSize = selectRows.size();
  auto copyArg = [batchSize, stream](MatrixPtr& dst,
                                     MatrixPtr src,
                                     int startRow,
                                     int pos,
                                     int size,
                                     bool useGpu) {
    if (!src) {
      dst.reset();
      return;
    }
    size_t width = src->getWidth();
    if (!dst) {
      dst = src->clone(batchSize, width, useGpu);
    } else {
      dst->resize(batchSize, width);
    }

    MatrixPtr tmpMatrix = dst->subMatrix(startRow, size);
    tmpMatrix->copyFrom(*src->subMatrix(pos, size), stream);
  };

  auto copyIds = [batchSize, stream](IVectorPtr& dst,
                                     const IVectorPtr& src,
                                     int startRow,
                                     int pos,
                                     int size,
                                     bool useGpu) {
    if (!src) {
      dst.reset();
      return;
    }
    IVector::resizeOrCreate(dst, batchSize, useGpu);
    dst->subVec(startRow, size)->copyFrom(*src->subVec(pos, size), stream);
  };

  auto copyStrs = [batchSize, stream](SVectorPtr& dst,
                                      const SVectorPtr& src,
                                      int startRow,
                                      int pos,
                                      int size,
                                      bool useGpu) {
    if (!src) {
      dst.reset();
      return;
    }
    if (!dst) {
      dst = std::make_shared<std::vector<std::string>>(batchSize);
    } else {
      dst->resize(batchSize);
    }
    std::copy(
        src->begin() + pos, src->begin() + pos + size, dst->begin() + startRow);
  };

  dataId = args[0].dataId;
  CHECK_NE(seqStartPos.size(), 0UL);
  size_t sampleNum = seqStartPos.size() - 1;
  for (size_t i = 0; i < sampleNum; ++i) {
    int startPos = seqStartPos[i];
    int endPos = seqStartPos[i + 1];
    CHECK_GE(args.size(), static_cast<size_t>(endPos - startPos));
    for (int j = startPos; j < endPos; ++j) {
      const Argument& arg = args[j - startPos];
      CHECK_EQ(arg.dataId, dataId) << "Arguments in concat should have"
                                   << " same dataId";
      const int copySize = 1;
      const int rowIdx = selectRows[j];
      copyArg(in, arg.in, j, rowIdx, copySize, useGpu);
      copyArg(value, arg.value, j, rowIdx, copySize, useGpu);
      if (passType != PASS_TEST) {
        copyArg(grad, arg.grad, j, rowIdx, copySize, useGpu);
      }
      copyIds(ids, arg.ids, j, rowIdx, copySize, useGpu);
      copyStrs(strs, arg.strs, j, rowIdx, copySize, useGpu);
    }
  }
  ICpuGpuVector::resizeOrCreate(
      sequenceStartPositions, seqStartPos.size(), useGpu);
  sequenceStartPositions->copyFrom(
      seqStartPos.data(), seqStartPos.size(), useGpu);
}

void Argument::concat(const std::vector<Argument>& args,
                      bool useGpu,
                      hl_stream_t stream,
                      PassType passType) {
  int32_t batchSize = 0;
  int64_t numSequences = 0;
  int64_t numSubSequences = 0;
  for (auto& arg : args) {
    batchSize += arg.getBatchSize();
    numSequences += arg.getNumSequences();
    numSubSequences += arg.getNumSubSequences();
  }

  auto copyArg = [batchSize, stream](
      MatrixPtr& dst, MatrixPtr src, int startRow, bool useGpu) {
    if (!src) {
      dst.reset();
      return;
    }
    size_t width = src->getWidth();
    if (!dst) {
      dst = src->clone(batchSize, width, useGpu);
    } else {
      dst->resize(batchSize, width);
    }

    MatrixPtr tmpMatrix = dst->subMatrix(startRow, src->getHeight());
    tmpMatrix->copyFrom(*src, stream);
  };

  auto copyIds = [batchSize, stream](
      IVectorPtr& dst, const IVectorPtr& src, int startRow, bool useGpu) {
    if (!src) {
      dst.reset();
      return;
    }
    IVector::resizeOrCreate(dst, batchSize, useGpu);
    dst->subVec(startRow, src->getSize())->copyFrom(*src, stream);
  };

  auto copyStrs = [batchSize, stream](
      SVectorPtr& dst, const SVectorPtr& src, int startRow, bool useGpu) {
    if (!src) {
      dst.reset();
      return;
    }
    if (!dst) {
      dst = std::make_shared<std::vector<std::string>>(batchSize);
    } else {
      dst->resize(batchSize);
    }
    std::copy(src->begin(), src->end(), dst->begin() + startRow);
  };

  auto copySequencePos = [](ICpuGpuVectorPtr& dstSeq,
                            const ICpuGpuVectorPtr& srcSeq,
                            int dstNumSequences,
                            int srcNumSequences,
                            int& startSequences,
                            int startRow) {
    if (srcSeq) {
      ICpuGpuVector::resizeOrCreate(dstSeq, dstNumSequences + 1, false);
      const int* src = srcSeq->getData(false);
      int* dest = dstSeq->getMutableData(false);
      for (int i = 0; i < srcNumSequences + 1; ++i) {
        dest[i + startSequences] = src[i] + startRow;
      }
      startSequences += srcNumSequences;
    } else {
      dstSeq.reset();
    }
  };

  int startRow = 0;
  int startSequences = 0;
  int startSubSequences = 0;
  dataId = args[0].dataId;
  for (auto& arg : args) {
    CHECK_EQ(arg.dataId, dataId) << "Arguments in concat should have"
                                 << " same dataId";
    copyArg(in, arg.in, startRow, useGpu);
    copyArg(value, arg.value, startRow, useGpu);
    if (passType != PASS_TEST) copyArg(grad, arg.grad, startRow, useGpu);
    copyIds(ids, arg.ids, startRow, useGpu);
    copySequencePos(sequenceStartPositions,
                    arg.sequenceStartPositions,
                    numSequences,
                    arg.getNumSequences(),
                    startSequences,
                    startRow);
    copySequencePos(subSequenceStartPositions,
                    arg.subSequenceStartPositions,
                    numSubSequences,
                    arg.getNumSubSequences(),
                    startSubSequences,
                    startRow);
    copyStrs(strs, arg.strs, startRow, useGpu);
    startRow += arg.getBatchSize();
  }
}

void Argument::splitByDataId(const std::vector<Argument>& argus,
                             std::vector<std::vector<Argument>>* arguGroups) {
  arguGroups->clear();
  int lastDataId = -1;
  for (const auto& argu : argus) {
    if (argu.dataId == -1) {
      // is -1, then create a new group
      arguGroups->emplace_back();
      lastDataId = -1;
    } else if (argu.dataId != lastDataId) {
      // not -1, also not equal to last Argument, then create a new group
      arguGroups->emplace_back();
      lastDataId = argu.dataId;
    } else {
      // not -1, and equal to last Argument, do nothing
    }
    arguGroups->back().push_back(argu);
  }
}

void Argument::getSeqInfo(std::vector<SeqInfo>* seqInfo) const {
  const int* starts = sequenceStartPositions->getData(false);
  const int* subStarts =
      hasSubseq() ? subSequenceStartPositions->getData(false) : nullptr;
  size_t numSequences = getNumSequences();
  seqInfo->reserve(numSequences);
  int subSeqEnd = 0;
  for (size_t i = 0; i < numSequences; ++i) {
    SeqInfo info;
    info.seqStart = starts[i];
    info.subLevelLength = starts[i + 1] - starts[i];
    info.seqId = i;
    if (hasSubseq()) {
      info.subSeqStart = subSeqEnd;
      while (subStarts[subSeqEnd] < starts[i + 1]) {
        ++subSeqEnd;
      }
      info.topLevelLength = subSeqEnd - info.subSeqStart;
    } else {
      info.topLevelLength = info.subLevelLength;
      info.subSeqStart = 0;  // not used
    }
    seqInfo->push_back(info);
  }
  std::sort(
      seqInfo->begin(), seqInfo->end(), [](const SeqInfo& a, const SeqInfo& b) {
        return a.topLevelLength > b.topLevelLength;
      });
}

void Argument::checkSubset() const {
  if (getNumSequences() > getNumSubSequences()) {
    LOG(FATAL) << "numSubSequences is less than numSequences ("
               << getNumSubSequences() << " vs. " << getNumSequences() << ")";
  }
  const int* start = sequenceStartPositions->getData(false);
  const int* subStart = subSequenceStartPositions->getData(false);
  int seqId = 0;
  int subSeqId = 0;
  while (seqId < getNumSequences() && subSeqId < getNumSubSequences()) {
    if (start[seqId] > subStart[subSeqId]) {
      ++subSeqId;
    } else if (start[seqId] == subStart[subSeqId]) {
      ++subSeqId;
      ++seqId;
    } else {
      LOG(FATAL) << "seqStartPositions is not subset of subSeqStartPositions";
    }
  }
  if (seqId < getNumSequences()) {
    LOG(FATAL) << "seqStartPositions is not subset of subSeqStartPositions";
  }
}

void Argument::degradeSequence(const Argument& input) {
  CHECK_EQ(input.hasSubseq(), 1UL);
  size_t numSequences = input.getNumSequences();
  size_t numSubSequences = input.getNumSubSequences();
  ICpuGpuVector::resizeOrCreate(
      sequenceStartPositions, numSequences + 1, false);
  int* tgtBuf = sequenceStartPositions->getMutableData(false);
  const int* starts = input.sequenceStartPositions->getData(false);
  const int* subStarts = input.subSequenceStartPositions->getData(false);
  int seqId = 0;
  for (size_t subSeqId = 0; subSeqId < numSubSequences; ++subSeqId) {
    if (subStarts[subSeqId] == starts[seqId]) {
      tgtBuf[seqId] = subSeqId;
      seqId++;
    }
  }
  tgtBuf[numSequences] = numSubSequences;
}

void Argument::poolSequenceWithStride(const Argument& input,
                                      size_t stride,
                                      IVectorPtr* stridePostions,
                                      bool reversed) {
  // If input.sequenceStartPositions = [0, 9, 14, 17, 30] and stride = 5,
  // then sequenceStartPositions = [0, 2, 3, 4, 7].
  // If reversed = false, stridePostions = [0, 5, 9, 14, 17, 22, 27, 30];
  // else reversed = true, stridePostions = [0, 4, 9, 14, 17, 20, 25, 30]

  CHECK(input.sequenceStartPositions);
  CHECK_EQ(input.hasSubseq(), 0UL);
  CHECK_GT(stride, 0UL) << "stride must larger than 0";
  size_t numSequences = input.getNumSequences();
  ICpuGpuVector::resizeOrCreate(
      sequenceStartPositions, numSequences + 1, false);
  const int* starts = input.sequenceStartPositions->getData(false);
  int* tgtBuf = sequenceStartPositions->getMutableData(false);
  // first index of target sequence and stride positions are both 0
  tgtBuf[0] = 0;
  std::vector<int> stridePos;
  for (size_t seqId = 0; seqId < numSequences; ++seqId) {
    size_t seqLength = starts[seqId + 1] - starts[seqId];
    stridePos.emplace_back(starts[seqId]);
    if (seqLength == 0) {
      // empty sequence
      tgtBuf[seqId + 1] = tgtBuf[seqId];
    } else {
      int size = ceil((float)seqLength / stride);
      tgtBuf[seqId + 1] = tgtBuf[seqId] + size;
      for (int i = 0; i < size - 1; ++i) {
        int cur = reversed ? starts[seqId + 1] - (size - 1 - i) * stride
                           : stridePos.back() + stride;
        stridePos.emplace_back(cur);
      }
    }
  }
  stridePos.emplace_back(starts[numSequences]);
  int size = stridePos.size();
  CHECK_EQ(size - 1, tgtBuf[numSequences]);
  IVector::resizeOrCreate(*stridePostions, size, false);
  (*stridePostions)->copyFrom(stridePos.data(), size);
}

void Argument::getValueString(
    std::unordered_map<std::string, std::string>* out) const {
  if (value) {
    std::ostringstream os;
    value->print(os);
    out->insert({"value", os.str()});
  }
  if (ids) {
    std::ostringstream os;
    ids->print(os, ids->getSize());
    out->insert({"ids", os.str()});
  }
  if (sequenceStartPositions) {
    std::ostringstream os;
    sequenceStartPositions->getVector(false)->print(
        os, sequenceStartPositions->getSize());
    out->insert({"sequence pos", os.str()});
  }
  if (subSequenceStartPositions) {
    std::ostringstream os;
    subSequenceStartPositions->getVector(false)->print(
        os, subSequenceStartPositions->getSize());
    out->insert({"sub-sequence pos", os.str()});
  }
}

void Argument::printValueString(std::ostream& stream,
                                const std::string& prefix) const {
  std::unordered_map<std::string, std::string> out;
  getValueString(&out);
  for (auto field : {"value", "id", "sequence pos", "sub-sequence pos"}) {
    auto it = out.find(field);
    if (it != out.end()) {
      stream << prefix << field << ":\n" << it->second;
    }
  }
}

void Argument::subArgFrom(const Argument& input,
                          size_t offset,
                          size_t height,
                          size_t width,
                          bool useGpu,
                          bool trans,
                          bool seqFlag,
                          size_t seqStart,
                          size_t seqSize) {
  if (input.value) {
    value = Matrix::create(
        input.value->getData() + offset * width, height, width, trans, useGpu);
  }
  if (input.ids) {
    ids = IVector::create(input.ids->getData() + offset, height, useGpu);
  }
  if (input.grad) {
    grad = Matrix::create(
        input.grad->getData() + offset * width, height, width, trans, useGpu);
  }
  if (seqFlag) {
    sequenceStartPositions = std::make_shared<ICpuGpuVector>(
        *(input.sequenceStartPositions), seqStart, seqSize);
  }
}

}  // namespace paddle
