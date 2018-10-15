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

#include "SequenceToBatch.h"
#include <string.h>
#include <algorithm>
#include <iostream>
#include <vector>

namespace paddle {

void SequenceToBatch::resizeOrCreateBatch(int batchSize,
                                          size_t numSequences,
                                          const int *seqStarts,
                                          bool reversed,
                                          bool prevBatchState) {
  CHECK_EQ(seqStarts[numSequences], batchSize);
  IVector::resizeOrCreate(seq2BatchIdx_, batchSize, useGpu_);
  if (!useGpu_) {
    cpuSeq2BatchIdx_ = seq2BatchIdx_;
  } else {
    IVector::resizeOrCreate(cpuSeq2BatchIdx_, batchSize, false);
  }

  /*
   * calculate the length of each sequence & sort sequence index by the length
   * Exampel:  Sequences = {s0, s1, s2}
   *           s0: 0 0 0 0, s1: 1 1 1 1 1, s2: 2 2 2
   *           seqStartAndLength[3] = {(4, 5, 1), (0, 4, 0), (9, 3, 2)}
   */
  struct SeqStartAndLength {
    int start_;
    int length_;
    int seqIdx_;
    SeqStartAndLength(int start, int length, int seqIdx)
        : start_(start), length_(length), seqIdx_(seqIdx) {}
  };
  std::vector<SeqStartAndLength> seqStartAndLength;
  for (size_t seqId = 0; seqId < numSequences; ++seqId) {
    int length = seqStarts[seqId + 1] - seqStarts[seqId];
    seqStartAndLength.emplace_back(seqStarts[seqId], length, seqId);
  }
  std::sort(seqStartAndLength.begin(),
            seqStartAndLength.end(),
            [](SeqStartAndLength a, SeqStartAndLength b) {
              return a.length_ > b.length_;
            });

  /*
   * calculate the start position of each batch
   * (numBatch equal the maxLength of sequences)
   * Exampel:  Sequences = {s0, s1, s2}
   *           s0: 0 0 0 0, s1: 1 1 1 1 1, s2: 2 2 2
   *           numBatch = 5,
   *           batchIndex = {b0, b1, b2, b3, b4}
   *           b0: 1 0 2, b1: 1 0 2, b2: 1 0 2, b3: 1 0, b4: 1
   *           batchStartPositions[6] = {0, 3, 6, 9, 11, 12}
   */
  numBatch_ = (size_t)seqStartAndLength[0].length_;

  IVector::resizeOrCreate(batchStartPositions_, numBatch_ + 1, false);
  int *batchStartPositions = batchStartPositions_->getData();
  batchStartPositions[0] = 0;
  for (size_t n = 0; n < numBatch_; n++) {
    int batchId = batchStartPositions[n];
    for (size_t i = 0; i < seqStartAndLength.size(); ++i) {
      size_t seqLength = seqStartAndLength[i].length_;
      int start = seqStartAndLength[i].start_;
      if (n < seqLength) {
        if (!reversed) {
          cpuSeq2BatchIdx_->getData()[batchId] = start + n;
        } else {
          cpuSeq2BatchIdx_->getData()[batchId] = start + seqLength - 1 - n;
        }
        batchId++;
      } else {
        break;
      }
    }
    batchStartPositions[n + 1] = batchId;
  }
  if (useGpu_) {
    seq2BatchIdx_->copyFrom(*cpuSeq2BatchIdx_);
  }
  if (prevBatchState) {
    IVector::resizeOrCreate(seqIdx_, numSequences, useGpu_);
    IVector::resizeOrCreate(seqEndIdxInBatch_, numSequences, useGpu_);
    if (!useGpu_) {
      cpuSeqIdx_ = seqIdx_;
      cpuSeqEndIdxInBatch_ = seqEndIdxInBatch_;
    } else {
      IVector::resizeOrCreate(cpuSeqIdx_, numSequences, false);
      IVector::resizeOrCreate(cpuSeqEndIdxInBatch_, numSequences, false);
    }
    int *seqIdx = cpuSeqIdx_->getData();
    int *seqEndIdxInBatch = cpuSeqEndIdxInBatch_->getData();
    for (size_t i = 0; i < seqStartAndLength.size(); ++i) {
      seqIdx[i] = seqStartAndLength[i].seqIdx_;
    }
    for (size_t i = 0; i < seqStartAndLength.size(); ++i) {
      if (seqStartAndLength[i].length_ > 0) {
        seqEndIdxInBatch[seqStartAndLength[i].seqIdx_] =
            batchStartPositions[seqStartAndLength[i].length_ - 1] + i;
      } else {
        seqEndIdxInBatch[seqStartAndLength[i].seqIdx_] = 0;
      }
    }
    if (useGpu_) {
      seqIdx_->copyFrom(*cpuSeqIdx_);
      seqEndIdxInBatch_->copyFrom(*cpuSeqEndIdxInBatch_);
    }
  }
}

void SequenceToBatch::resizeOrCreate(Matrix &seqValue) {
  Matrix::resizeOrCreate(batchValue_,
                         seqValue.getHeight(),
                         seqValue.getWidth(),
                         /* trans= */ false,
                         useGpu_);
}

MatrixPtr SequenceToBatch::getBatchValue(int batchId, int numRows) {
  return getBatchValue(*batchValue_, batchId, numRows);
}

MatrixPtr SequenceToBatch::getBatchValue(Matrix &batchValue,
                                         int batchId,
                                         int numRows) {
  int *batchStartPositions = batchStartPositions_->getData();
  int start = batchStartPositions[batchId];
  int maxRows = batchStartPositions[batchId + 1] - batchStartPositions[batchId];
  if (numRows == 0) {
    numRows = maxRows;
  } else {
    CHECK_LE(numRows, maxRows);
  }
  return batchValue.subMatrix(start, numRows);
}

void SequenceToBatch::prevOutput2Batch(Matrix &src, Matrix &dst) {
  sequence2BatchCopy(dst, src, *seqIdx_, true);
}

void SequenceToBatch::getSeqOutputFromBatch(Matrix &sequence, Matrix &batch) {
  sequence2BatchCopy(sequence, batch, *seqEndIdxInBatch_, true);
}

void SequenceToBatch::sequence2BatchCopy(Matrix &batch,
                                         Matrix &sequence,
                                         IVector &seq2BatchIdx,
                                         bool seq2batch) {
  int seqWidth = sequence.getWidth();
  int batchCount = batch.getHeight();
  real *batchData = batch.getData();
  real *seqData = sequence.getData();
  int *idxData = seq2BatchIdx.getData();

  if (useGpu_) {
    hl_sequence2batch_copy(
        batchData, seqData, idxData, seqWidth, batchCount, seq2batch);
  } else {
    if (seq2batch) {
#ifdef PADDLE_USE_MKLML
      const int blockMemSize = 8 * 1024;
      const int blockSize = blockMemSize / sizeof(real);
#pragma omp parallel for collapse(2)
      for (int i = 0; i < batchCount; ++i) {
        for (int j = 0; j < seqWidth; j += blockSize) {
          memcpy(batch.rowBuf(i) + j,
                 sequence.rowBuf(idxData[i]) + j,
                 (j + blockSize > seqWidth) ? (seqWidth - j) * sizeof(real)
                                            : blockMemSize);
        }
      }
#else
      for (int i = 0; i < batchCount; ++i) {
        memcpy(batch.rowBuf(i),
               sequence.rowBuf(idxData[i]),
               seqWidth * sizeof(real));
      }
#endif
    } else {
#ifdef PADDLE_USE_MKLML
#pragma omp parallel for
#endif
      for (int i = 0; i < batchCount; ++i) {
        memcpy(sequence.rowBuf(idxData[i]),
               batch.rowBuf(i),
               seqWidth * sizeof(real));
      }
    }
  }
}

void SequenceToBatch::sequence2BatchAdd(Matrix &batch,
                                        Matrix &sequence,
                                        IVector &seq2BatchIdx,
                                        bool seq2batch) {
  int seqWidth = sequence.getWidth();
  int batchCount = batch.getHeight();
  real *batchData = batch.getData();
  real *seqData = sequence.getData();
  int *idxData = seq2BatchIdx.getData();

  if (useGpu_) {
    hl_sequence2batch_add(
        batchData, seqData, idxData, seqWidth, batchCount, seq2batch);
  } else {
    for (int i = 0; i < batchCount; ++i) {
      if (seq2batch) {
        batch.subMatrix(i, 1)->add(*sequence.subMatrix(idxData[i], 1));
      } else {
        sequence.subMatrix(idxData[i], 1)->add(*batch.subMatrix(i, 1));
      }
    }
  }
}

void SequenceToBatch::copyFromSeq(Matrix &seqValue) {
  Matrix::resizeOrCreate(batchValue_,
                         seqValue.getHeight(),
                         seqValue.getWidth(),
                         /* trans= */ false,
                         useGpu_);
  sequence2BatchCopy(*batchValue_, seqValue, *seq2BatchIdx_, true);
}

void SequenceToBatch::copyBackSeq(Matrix &seqValue) {
  sequence2BatchCopy(*batchValue_, seqValue, *seq2BatchIdx_, false);
}

void SequenceToBatch::copy(Matrix &seqValue,
                           Matrix &batchValue,
                           bool seq2batch) {
  sequence2BatchCopy(batchValue, seqValue, *seq2BatchIdx_, seq2batch);
}

void SequenceToBatch::add(Matrix &seqValue,
                          Matrix &batchValue,
                          bool seq2batch) {
  sequence2BatchAdd(batchValue, seqValue, *seq2BatchIdx_, seq2batch);
}

}  // namespace paddle
