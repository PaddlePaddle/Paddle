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
#include <vector>
#include "MemoryHandle.h"
#include "paddle/utils/Util.h"

namespace paddle {

/**
 * @brief The RowBuffer class
 * Represent the SparseRow Matrix Data.
 *
 * If not set memory handler, then the data could be auto growth.
 */
class RowBuffer {
public:
  /**
   * @brief RowBuffer create a auto-growth row buffer. The row length is width.
   * @param width the length of each row, a.k.a matrix width.
   */
  explicit RowBuffer(size_t width) : width_(width) {}

  /**
   * @brief RowBuffer create a row buffer, which cannot be auto-growth.
   * @param mem the pre-allocated memory.
   * @param width the length of each row, a.k.a matrix width.
   */
  RowBuffer(const CpuMemHandlePtr& mem, size_t width)
      : preallocatedBuf_(mem), width_(width) {}

  /**
   * @brief resize resize the buffer with rowCount
   * @param rowCnt number of row. matrix height.
   */
  inline void resize(int rowCnt) {
    if (preallocatedBuf_) {
      CHECK(preallocatedBuf_->getSize() >= rowCnt * width_ * sizeof(real));
    } else {
      rowStore_.resize(rowCnt * width_);
    }
  }

  /**
   * @brief get a row buffer with row index.
   * @param row the index of row.
   * @return row buffer.
   */
  inline real* get(int row) const {
    if (preallocatedBuf_) {
      CHECK_LE((row + 1) * width_ * sizeof(real), preallocatedBuf_->getSize());
      return reinterpret_cast<real*>(preallocatedBuf_->getBuf()) + row * width_;
    } else {
      CHECK_LE((row + 1) * width_, rowStore_.size());
      return const_cast<real*>(rowStore_.data() + row * width_);
    }
  }

  /**
   * @brief get a row buffer with row index. If row index is larger than local
   *        buffer, the size of local buffer will grow.
   * @param row the index of row.
   * @return row buffer.
   */
  inline real* getWithAutoGrowth(int row) {
    if (preallocatedBuf_) {
      return get(row);
    } else {
      if ((rowStore_.size() <= row * width_)) {
        rowStore_.resize((row + 1) * width_);
      }
      return rowStore_.data() + row * width_;
    }
  }

  /**
   * @return raw data buffer.
   */
  inline real* data() {
    if (preallocatedBuf_) {
      return reinterpret_cast<real*>(preallocatedBuf_->getBuf());
    } else {
      return rowStore_.data();
    }
  }

  /**
   * @brief clear local buffer. It only affect auto-growth buffer.
   */
  inline void clear() { rowStore_.clear(); }

  /**
   * @brief get current number of rows.
   * @return number of rows.
   */
  inline size_t getRowCount() const {
    if (preallocatedBuf_) {
      return preallocatedBuf_->getSize() / sizeof(real) / width_;
    } else {
      return rowStore_.size() / width_;
    }
  }

  /**
   * @brief get is this buffer can automatically grow or not.
   * @return ture if can automacitally grow.
   */
  inline bool isAutoGrowth() const { return !preallocatedBuf_; }

  /**
   * @brief return the width of matrix. a.k.a length of row.
   * @return width of matrix
   */
  inline size_t getWidth() const { return width_; }

private:
  //! TODO(yuyang18): Add resize method to CpuMemHandlePtr, then we can get rid
  //! of std::vector here.
  CpuMemHandlePtr preallocatedBuf_;
  std::vector<real, AlignedAllocator<real, 32>> rowStore_;
  size_t width_;
};
}  // namespace paddle
