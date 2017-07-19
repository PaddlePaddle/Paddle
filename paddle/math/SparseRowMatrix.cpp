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

#include "SparseRowMatrix.h"
#include "CpuSparseMatrix.h"

#include <algorithm>

#include "paddle/utils/Logging.h"

#include "SIMDFunctions.h"

#include "paddle/utils/Thread.h"
#include "paddle/utils/Util.h"

namespace paddle {

const unsigned int SparseRowCpuMatrix::kUnusedId_ = -1U;

void SparseRowCpuMatrix::init(size_t height, size_t width) {
  height_ = height;
  if (!indexDictHandle_) {
    indexDictHandle_.reset(new IndexDict);
    indexDictHandle_->globalIndices.assign(height, kUnusedId_);
  }
  localIndices_ = &indexDictHandle_->localIndices;
  globalIndices_ = indexDictHandle_->globalIndices.data();
}

void SparseRowCpuMatrix::mul(CpuSparseMatrix* a,
                             CpuMatrix* b,
                             real scaleAB,
                             real scaleT) {
  CpuMatrix::mul<CpuMatrix, SparseRowCpuMatrix>(a, b, this, scaleAB, scaleT);
}

void SparseRowCpuMatrix::copyFrom(const real* src, size_t size) {
  LOG(FATAL) << "This should not be called";
}

void SparseRowCpuMatrix::zeroMem() {
  apply([](real* buf, size_t len) { memset(buf, 0, sizeof(real) * len); });
  clearRows();
}

void SparseRowCpuMatrix::applyL1(real learningRate, real decayRate) {
  apply([=](real* buf, size_t len) {
    CpuVector value(0, nullptr);
    value.subVecFrom(buf, 0, len);
    value.applyL1(learningRate, decayRate);
  });
}

void SparseRowCpuMatrix::sgdUpdate(BaseMatrix& value,
                                   IVector& t0,
                                   real learningRate,
                                   int currentTime,
                                   real decayRate,
                                   bool useL1,
                                   bool fini) {
  std::vector<unsigned int>& localIndices = indexDictHandle_->localIndices;

  // t0 and value are vectors
  CHECK_EQ(t0.getSize(), this->height_);
  CHECK_EQ(value.width_, this->height_ * this->width_);

  if (decayRate == 0.0f) {
    if (fini) {
      return;
    }

    for (size_t i = 0; i < localIndices.size(); ++i) {
      real* g = getLocalRow(i);
      real* v = value.rowBuf(localIndices[i]);
      for (size_t j = 0; j < this->width_; ++j) {
        v[j] -= learningRate * g[j];
      }
    }
    return;
  }  // else

  if (useL1) {  // L1 decay
    if (fini) {
      for (size_t i = 0; i < this->height_; ++i) {
        real* v = value.rowBuf(i);
        int* t = t0.getData() + i;
        if (t[0] < currentTime) {
          // W(t0) -> W(t+1)
          int tDiff = currentTime - t[0];
          real delta = tDiff * learningRate * decayRate;
          simd::decayL1(v, v, delta, this->width_);
        }
      }
      return;
    }  // else

    for (size_t i = 0; i < localIndices.size(); ++i) {
      real* g = getLocalRow(i);
      real* v = value.rowBuf(localIndices[i]);
      int* t = t0.getData() + localIndices[i];
      if (t[0] < currentTime) {
        // W(t0) -> W(t)
        int tDiff = currentTime - t[0];
        real delta = tDiff * learningRate * decayRate;
        simd::decayL1(v, v, delta, this->width_);
      }

      // W(t) -> W(t+1)
      for (size_t j = 0; j < this->width_; ++j) {
        v[j] -= learningRate * g[j];
      }
      simd::decayL1(v, v, learningRate * decayRate, this->width_);

      // state update to t+1
      t[0] = currentTime + 1;
    }

  } else {  // L2 decay
    if (fini) {
      for (size_t i = 0; i < this->height_; ++i) {
        real* v = value.rowBuf(i);
        int* t = t0.getData() + i;
        if (t[0] < currentTime) {
          // W(t0) -> W(t+1)
          int tDiff = currentTime - t[0];
          real recip = 1.0f / (1.0f + tDiff * learningRate * decayRate);
          for (size_t j = 0; j < this->width_; ++j) {
            v[j] *= recip;
          }
        }
      }
      return;
    }  // else

    real recipDecay = 1.0f / (1.0f + learningRate * decayRate);

    for (size_t i = 0; i < localIndices.size(); ++i) {
      real* g = getLocalRow(i);
      real* v = value.rowBuf(localIndices[i]);
      int* t = t0.getData() + localIndices[i];
      if (t[0] < currentTime) {
        // W(t0) -> W(t)
        int tDiff = currentTime - t[0];
        real recip = 1.0f / (1.0f + tDiff * learningRate * decayRate);
        for (size_t j = 0; j < this->width_; ++j) {
          v[j] *= recip;
        }
      }

      // W(t) -> W(t+1)
      for (size_t j = 0; j < this->width_; ++j) {
        v[j] = recipDecay * (v[j] - learningRate * g[j]);
      }

      // state update to t+1
      t[0] = currentTime + 1;
    }
  }
}

void SparseRowCpuMatrix::addTo(BaseMatrix& dest,
                               std::vector<uint32_t>& ids,
                               size_t tid,
                               size_t numThreads) {
  CHECK(!dest.useGpu_);
  CHECK_EQ(dest.height_ * dest.width_, this->height_ * this->width_);

  std::vector<unsigned int>& localIndices = indexDictHandle_->localIndices;
  for (size_t i = 0; i < localIndices.size(); ++i) {
    uint32_t id = localIndices[i];
    if (id % numThreads == tid) {
      simd::addTo(dest.rowBuf(id), getLocalRow(i), this->width_);
      ids.push_back(id);
    }
  }
}

void SparseRowCpuMatrix::addTo(SparseRowCpuMatrix& dest,
                               size_t tid,
                               size_t numThreads) {
  CHECK(!dest.useGpu_);
  CHECK_EQ(dest.height_ * dest.width_, this->height_ * this->width_);

  std::vector<unsigned int>& localIndices = indexDictHandle_->localIndices;
  for (size_t i = 0; i < localIndices.size(); ++i) {
    uint32_t id = localIndices[i];
    if (id % numThreads == tid) {
      dest.checkIndex(id);
      simd::addTo(dest.getRow(id), getLocalRow(i), this->width_);
    }
  }
}

void SparseRowCpuMatrix::zeroMemThread(size_t tid, size_t numThreads) {
  std::vector<unsigned int>& localIndices = indexDictHandle_->localIndices;
  for (size_t i = 0; i < localIndices.size(); ++i) {
    uint32_t id = localIndices[i];
    if (id % numThreads == tid) {
      memset(this->getLocalRow(i), 0, this->width_ * sizeof(real));
    }
  }
}

void SparseAutoGrowRowCpuMatrix::mul(CpuSparseMatrix* a,
                                     CpuMatrix* b,
                                     real scaleAB,
                                     real scaleT) {
  CpuMatrix::mul<CpuMatrix, SparseAutoGrowRowCpuMatrix>(
      a, b, this, scaleAB, scaleT);
}

void CacheRowCpuMatrix::mul(CpuSparseMatrix* a,
                            CpuMatrix* b,
                            real scaleAB,
                            real scaleT) {
  CpuMatrix::mul<CpuMatrix, CacheRowCpuMatrix>(a, b, this, scaleAB, scaleT);
}

void SparsePrefetchRowCpuMatrix::addRows(const unsigned int* ids, size_t len) {
  std::vector<unsigned int>& localIndices = indexDictHandle_->localIndices;
  for (size_t i = 0; i < len; i++) {
    CHECK_LT(*(ids + i), this->getHeight())
        << "id:" << *(ids + i) << "Height:" << this->getHeight()
        << "sparse id value exceeds the max input dimension, "
        << "it could be caused invalid input data samples";
  }
  localIndices.insert(localIndices.end(), ids, ids + len);
}

void SparsePrefetchRowCpuMatrix::addRows(MatrixPtr input) {
  CpuSparseMatrix* mat = dynamic_cast<CpuSparseMatrix*>(input.get());
  CHECK(mat) << "only support sparse matrix";
  addRows(reinterpret_cast<const unsigned int*>(mat->getCols()),
          mat->getElementCnt());
}

void SparsePrefetchRowCpuMatrix::addRows(IVectorPtr ids) {
  std::vector<unsigned int>& localIndices = indexDictHandle_->localIndices;
  size_t numSamples = ids->getSize();
  int* index = ids->getData();
  for (size_t i = 0; i < numSamples; ++i) {
    if (index[i] == -1) continue;

    unsigned int id = (unsigned int)index[i];
    CHECK_LT(id, this->getHeight())
        << "id:" << id << "Height:" << this->getHeight()
        << "sparse id value exceeds the max input dimension, "
        << "it could be caused invalid input data samples";
    localIndices.push_back(id);
  }
}

void SparsePrefetchRowCpuMatrix::setupIndices() {
  auto& localIndices = indexDictHandle_->localIndices;
  uniqueIds(localIndices);
  // for each sparse row
  for (size_t id = 0; id < localIndices.size(); ++id) {
    globalIndices_[localIndices[id]] = id;  // sparse row -> local id
  }
  checkStoreSize();
}

void SparseRowCpuMatrix::checkIndices() {
  std::vector<unsigned int>& localIndices = indexDictHandle_->localIndices;
  for (size_t i = 0; i < localIndices.size(); ++i) {
    CHECK_EQ(globalIndices_[localIndices[i]], i);
  }
  checkStoreSize();
}

}  // namespace paddle
