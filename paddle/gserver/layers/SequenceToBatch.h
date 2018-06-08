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
#include "paddle/math/Matrix.h"
#include "paddle/math/Vector.h"

namespace paddle {

/*
 * This class can used to modify the matrix structure of sequence matrix into
 * batch structure.
 * sequence matrix: [C1_s ... Cn_s | ...... | C1_t ... Cn_t]
 * batch matrix:    [C1_s ... C1_t | ...... | Cn_s ... Cn_t]
 * Cn_s is the state for sequence s at time n.
 *
 * Exampel:  sequence matrix = {{0, 0, 0, 0}, {1, 1, 1, 1, 1}, {2, 2, 2}}
 *           s0: 0 0 0 0, s1: 1 1 1 1 1, s2: 2 2 2
 *           batch matrix = {{1, 0, 2}, {1, 0, 2}, {1, 0, 2}, {1, 0}, {1}}
 *           b0: 1 0 2, b1: 1 0 2, b2: 1 0 2, b3: 1 0, b4: 1
 *
 * Use:
 * Input: seqMatrix, seqStarts(Sequence Start Positions)
 * Output: batchMatrix
 * 1. SequenceToBatch seq2batch;
 * 2. seq2batch.resizeOrCreateBatch(seqStarts);     // calculate seq2BatchIdx
 * 3. seq2batch.copy(seqMatrix, batchMatrix, true); // copy seq to batch matrix
 *
 */
class SequenceToBatch {
 public:
  explicit SequenceToBatch(bool useGpu) : useGpu_(useGpu) {}

  /* resize and calculate the batchIndex_ */
  void resizeOrCreateBatch(int batchSize,
                           size_t numSequences,
                           const int *seqStarts,
                           bool reversed,
                           bool prevBatchState = false);

  /* sequence matrix and batch matrix copy:
   * seq2batch: copy(seqValue, batchValue, true);
   * batch2seq: copy(seqValue, batchValue, false);
   */
  void copy(Matrix &seqValue, Matrix &batchValue, bool seq2batch);
  /* sequence/batch matrix add to batch/sequence matrix */
  void add(Matrix &seqValue, Matrix &batchValue, bool seq2batch);
  MatrixPtr getBatchValue(Matrix &batchValue, int batchId, int numRows = 0);

  size_t getNumBatch() const { return numBatch_; }

  /* resize or create a batch matrix(batchValue_) */
  void resizeOrCreate(Matrix &seqValue);
  /* copy seqValue to batchValue_ */
  void copyFromSeq(Matrix &seqValue);
  /* copy batchValue_ to seqValue */
  void copyBackSeq(Matrix &seqValue);
  MatrixPtr getBatchValue(int batchId, int numRows = 0);
  MatrixPtr getBatchValue() { return batchValue_; }
  /*tranfer preBatchOutput to batch struct*/
  void prevOutput2Batch(Matrix &src, Matrix &dst);
  /*get sequence output from batch struct*/
  void getSeqOutputFromBatch(Matrix &sequence, Matrix &batch);

  /* Copy the index from another seq2batch. */
  void shareIndexWith(const SequenceToBatch &seq2batch) {
    CHECK(useGpu_ == seq2batch.useGpu_);
    batchStartPositions_ = seq2batch.batchStartPositions_;
    seq2BatchIdx_ = seq2batch.seq2BatchIdx_;
    cpuSeq2BatchIdx_ = seq2batch.cpuSeq2BatchIdx_;
    numBatch_ = seq2batch.numBatch_;
  }

 protected:
  void sequence2BatchCopy(Matrix &batch,
                          Matrix &sequence,
                          IVector &seq2BatchIdx,
                          bool seq2batch);
  void sequence2BatchAdd(Matrix &batch,
                         Matrix &sequence,
                         IVector &seq2BatchIdx,
                         bool seq2batch);

  IVectorPtr batchStartPositions_;
  IVectorPtr seq2BatchIdx_;
  IVectorPtr cpuSeq2BatchIdx_;
  IVectorPtr cpuSeqIdx_;
  IVectorPtr cpuSeqEndIdxInBatch_;
  IVectorPtr seqIdx_;
  IVectorPtr seqEndIdxInBatch_;
  size_t numBatch_;
  bool useGpu_;
  MatrixPtr batchValue_;
};

}  // namespace paddle
