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
#include <string.h>
#include <algorithm>
#include "Matrix.h"
#include "RowBuffer.h"
#include "paddle/utils/Util.h"

namespace paddle {

/**
 * Sparse Row
 */
class SparseRowCpuMatrix : public CpuMatrix {
public:
  struct IndexDict {
    // In the following, global id means the row id in the original matrix.
    // Local id means the row id in the local storage which only contains
    // the sparse rows.
    std::vector<unsigned int> localIndices;   // local id -> global id
    std::vector<unsigned int> globalIndices;  // global id -> local id
  };
  typedef std::shared_ptr<IndexDict> IndexDictPtr;

  /// heightStore is max number of rows of the sparse matrix.
  SparseRowCpuMatrix(CpuMemHandlePtr dataHandle,
                     size_t height,
                     size_t width,
                     IndexDictPtr indexDictHandle = nullptr,
                     bool trans = false)
      : CpuMatrix(nullptr, height, width, trans),
        indexDictHandle_(indexDictHandle) {
    init(height, width);
    buf_.reset(new RowBuffer(dataHandle, width));
  }

  virtual ~SparseRowCpuMatrix() {}

public:
  /**
   *  Get the row buf
   *
   *  @param row row id in the original matrix
   */
  real* getRow(size_t row) {
    CHECK_NE(globalIndices_[row], kUnusedId_);
    return getLocalRow(globalIndices_[row]);
  }

  /**
   *  Get the row buf
   *
   *  @param row row id in local storage
   */
  real* getLocalRow(size_t row) { return buf_->getWithAutoGrowth(row); }

  /**
   *  reserve the storage for rows according to current size of
   * indexDictHandle.
   *
   *  This is only used when SparseRowCpuMatrix is constructed with
   *  indexDictHandle.
   */
  void reserveStore() { buf_->resize(localIndices_->size()); }

  // row is the row id in the original matrix
  virtual real* getRowBuf(size_t row) { return getRow(row); }

  virtual void mul(CpuSparseMatrix* a, CpuMatrix* b, real scaleAB, real scaleT);

  /**
   * Fill data according to row indexs added, setup indices inside.
   *
   * *src* and *size* are data and size of normal dense CpuMatrix.
   */
  virtual void copyFrom(const real* src, size_t size);
  virtual void zeroMem();

  /**
   * apply L1 to all sparse rows, should be apply after indices ready.
   */
  virtual void applyL1(real learningRate, real decayRate);

  void clearIndices() { clearRows(); }
  void zeroMemThread(size_t tid, size_t numThreads);

  /**
   *  value -= grad * learningRate,  this is gradient.
   *
   * If L1 decay set use L1, else if L2 set use L2, otherwise no decay atall.
   *
   * t0 is a int vector used by L1/L2 decay, size = height of parameter
   * matrix,
   * store the time that each weight row last updated.
   *
   * Time is batchId, currentTime is current batchId.
   *
   * While pass finished, caller should call this func one more time
   *  with (fini=true) to let weight decay catch up current time.
   */
  void sgdUpdate(BaseMatrix& value,
                 IVector& t0,
                 real learningRate,
                 int currentTime,
                 real decayRate,
                 bool useL1,
                 bool fini = false);

  /**
   *  merge rows in *this* to *dest* for designated thread
   *
   *  values add to *dest* matrix
   *
   *  ids occured in *this* append to *ids*
   *  filtered by  (id % numThreads == tid)
   */
  void addTo(BaseMatrix& dest,
             std::vector<uint32_t>& ids,
             size_t tid,
             size_t numThreads);

  /**
   *  the second version addTo(), *dest* is a SparseRowCpuMatrix.
   *
   *  The dest's indices should be setup already, addTo() will
   *  check src ids is exist in dest's indices.
   */
  void addTo(SparseRowCpuMatrix& dest, size_t tid, size_t numThreads);

  const IndexDictPtr& getIndexDictHandle() const { return indexDictHandle_; }

  /**
   *  check all local and global indices consistency
   */
  void checkIndices();
  /**
   *  check whether row *i* exist in indices
   */
  void checkIndex(size_t i) {
    size_t localId = globalIndices_[i];
    CHECK_LT(localId, localIndices_->size());
    CHECK_EQ((*localIndices_)[localId], i);
  }

  std::vector<unsigned int>& getLocalIndices() const {
    return indexDictHandle_->localIndices;
  }

protected:
  template <typename Func>
  void apply(Func f) {
    f(buf_->data(), localIndices_->size() * width_);
  }

  void init(size_t height, size_t width);

  /// clear row indices.
  void clearRows() {
    for (auto id : *localIndices_) {
      globalIndices_[id] = kUnusedId_;
    }
    localIndices_->clear();
    buf_->clear();
  }

  inline void checkStoreSize() {
    if (buf_->isAutoGrowth()) {
      if (buf_->getRowCount() > 0.5 * height_) {
        LOG(WARNING) << "There are more than 0.5*height ("
                     << localIndices_->size() << ") rows are used for sparse "
                     << "update, which is not efficient. Considering not use "
                     << "sparse_update.";
      }
    } else {
      CHECK_LE(localIndices_->size(), buf_->getRowCount());
    }
  }

  std::unique_ptr<RowBuffer> buf_;
  IndexDictPtr indexDictHandle_;
  std::vector<unsigned int>* localIndices_;  // =&indexDictHandle_->localIndices
  unsigned int* globalIndices_;  // =indexDictHandle_->globalIndices.data();
  static const unsigned int kUnusedId_;
};

class SyncThreadPool;

/// For prefetching parameters from remote Parameter server
class SparsePrefetchRowCpuMatrix : public SparseRowCpuMatrix {
public:
  SparsePrefetchRowCpuMatrix(CpuMemHandlePtr dataHandle,
                             size_t height,
                             size_t width,
                             IndexDictPtr indexDictHandle = nullptr,
                             SyncThreadPool* pool = nullptr,
                             bool trans = false)
      : SparseRowCpuMatrix(dataHandle, height, width, indexDictHandle, trans),
        pool_(pool) {}

  /**
   * Extract feature ids from *input*, to fill row indexs.
   *
   * *input* must be sparse matrix.
   *
   * Can call many times before setup.
   */
  void addRows(MatrixPtr input);
  void addRows(IVectorPtr ids);

  /**
   * setup global indices of SparseRowMatrix after finish add rows.
   */
  void setupIndices();

protected:
  void addRows(const unsigned int* ids, size_t len);
  SyncThreadPool* pool_;
};

class SparseAutoGrowRowCpuMatrix : public SparseRowCpuMatrix {
public:
  SparseAutoGrowRowCpuMatrix(size_t height,
                             size_t width,
                             IndexDictPtr indexDictHandle = nullptr,
                             bool trans = false)
      : SparseRowCpuMatrix(nullptr, height, width, indexDictHandle, trans) {}

  real* getRow(size_t row) {
    auto id = globalIndices_[row];
    if (id == kUnusedId_) {
      id = globalIndices_[row] = localIndices_->size();
      localIndices_->push_back(row);
      checkStoreSize();
    }
    return getLocalRow(id);
  }

  virtual real* getRowBuf(size_t row) { return getRow(row); }

  virtual void mul(CpuSparseMatrix* a, CpuMatrix* b, real scaleAB, real scaleT);
};

class CacheRowCpuMatrix : public SparseAutoGrowRowCpuMatrix {
public:
  CacheRowCpuMatrix(size_t height,
                    size_t width,
                    IndexDictPtr indexDictHandle = nullptr,
                    bool trans = false)
      : SparseAutoGrowRowCpuMatrix(height, width, indexDictHandle, trans),
        sourceData_(nullptr) {}

  void setSourceData(CpuVectorPtr sourceVec) {
    sourceDataVec_ = sourceVec;
    sourceData_ = sourceVec->getData();
  }

  real* getRow(size_t row) {
    auto id = globalIndices_[row];
    if (id == kUnusedId_) {
      id = globalIndices_[row] = localIndices_->size();
      localIndices_->push_back(row);
      checkStoreSize();
      memcpy(
          getLocalRow(id), sourceData_ + width_ * row, sizeof(float) * width_);
    }
    return getLocalRow(id);
  }

  virtual real* getRowBuf(size_t row) { return getRow(row); }

  virtual void mul(CpuSparseMatrix* a, CpuMatrix* b, real scaleAB, real scaleT);

public:
  CpuVectorPtr sourceDataVec_;
  real* sourceData_;
};

/**
 * Sparse Row Ids Matrix.
 *
 * mostly same as CpuMatrix, but maintain sparse row ids occured,
 * ids are hashed by worker thread id.
 */
class SparseRowIdsCpuMatrix : public CpuMatrix {
public:
  SparseRowIdsCpuMatrix(CpuMemHandlePtr dataHandle,
                        size_t height,
                        size_t width,
                        bool trans = false)
      : CpuMatrix(dataHandle, height, width, trans) {}

  void setNumOfThreads(size_t numOfThreads) { idsArray_.resize(numOfThreads); }

  std::vector<uint32_t>& getIds(size_t threadId) { return idsArray_[threadId]; }

private:
  std::vector<std::vector<uint32_t>> idsArray_;
};

}  // namespace paddle
