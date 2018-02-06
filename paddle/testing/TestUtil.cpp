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

#include "TestUtil.h"
#include <gflags/gflags.h>
#include "paddle/math/SparseMatrix.h"

DEFINE_int32(fixed_seq_length, 0, "Produce some sequence of fixed length");

namespace paddle {

std::string randStr(const int len) {
  std::string str =
      "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
  std::string s = "";
  for (int i = 0; i < len; ++i) s += str[(rand() % 62)];  // NOLINT
  return s;
}

MatrixPtr makeRandomSparseMatrix(size_t height,
                                 size_t width,
                                 bool withValue,
                                 bool useGpu,
                                 bool equalNnzPerSample) {
#ifndef PADDLE_MOBILE_INFERENCE
  std::vector<int64_t> ids(height);
  std::vector<int64_t> indices(height + 1);
  indices[0] = 0;

  std::function<size_t()> randomer = [] { return uniformRandom(10); };
  if (equalNnzPerSample) {
    size_t n = 0;
    do {
      n = uniformRandom(10);
    } while (!n);
    randomer = [=] { return n; };
  }
  for (size_t i = 0; i < height; ++i) {
    indices[i + 1] = indices[i] + std::min(randomer(), width);
    ids[i] = i;
  }

  if (!withValue) {
    std::vector<sparse_non_value_t> data;
    data.resize(indices[height] - indices[0]);
    for (size_t i = 0; i < data.size(); ++i) {
      data[i].col = uniformRandom(width);
    }
    auto mat = Matrix::createSparseMatrix(
        height, width, data.size(), NO_VALUE, SPARSE_CSR, false, useGpu);
    if (useGpu) {
      std::dynamic_pointer_cast<GpuSparseMatrix>(mat)->copyFrom(
          ids.data(), indices.data(), data.data(), HPPL_STREAM_DEFAULT);
    } else {
      std::dynamic_pointer_cast<CpuSparseMatrix>(mat)->copyFrom(
          ids.data(), indices.data(), data.data());
    }
    return mat;
  } else {
    std::vector<sparse_float_value_t> data;
    data.resize(indices[height] - indices[0]);
    for (size_t i = 0; i < data.size(); ++i) {
      data[i].col = uniformRandom(width);
      data[i].value = rand() / static_cast<float>(RAND_MAX);  // NOLINT
    }
    auto mat = Matrix::createSparseMatrix(
        height, width, data.size(), FLOAT_VALUE, SPARSE_CSR, false, useGpu);
    if (useGpu) {
      std::dynamic_pointer_cast<GpuSparseMatrix>(mat)->copyFrom(
          ids.data(), indices.data(), data.data(), HPPL_STREAM_DEFAULT);
    } else {
      std::dynamic_pointer_cast<CpuSparseMatrix>(mat)->copyFrom(
          ids.data(), indices.data(), data.data());
    }
    return mat;
  }
#endif
  return nullptr;
}

void generateSequenceStartPositions(size_t batchSize,
                                    IVectorPtr& sequenceStartPositions) {
  ICpuGpuVectorPtr gpuCpuVec;
  generateSequenceStartPositions(batchSize, gpuCpuVec);
  sequenceStartPositions = gpuCpuVec->getMutableVector(false);
}

void generateSequenceStartPositions(size_t batchSize,
                                    ICpuGpuVectorPtr& sequenceStartPositions) {
  int numSeqs;
  if (FLAGS_fixed_seq_length != 0) {
    numSeqs = std::ceil((float)batchSize / (float)FLAGS_fixed_seq_length);
  } else {
    numSeqs = batchSize / 10 + 1;
  }
  sequenceStartPositions =
      ICpuGpuVector::create(numSeqs + 1, /* useGpu= */ false);
  int* buf = sequenceStartPositions->getMutableData(false);
  int64_t pos = 0;
  int len = FLAGS_fixed_seq_length;
  int maxLen = 2 * batchSize / numSeqs;
  for (int i = 0; i < numSeqs; ++i) {
    if (FLAGS_fixed_seq_length == 0) {
      len = uniformRandom(
                std::min<int64_t>(maxLen, batchSize - pos - numSeqs + i)) +
            1;
    }
    buf[i] = pos;
    pos += len;
    VLOG(1) << " len=" << len;
  }
  buf[numSeqs] = batchSize;
}

void generateSubSequenceStartPositions(
    const ICpuGpuVectorPtr& sequenceStartPositions,
    ICpuGpuVectorPtr& subSequenceStartPositions) {
  int numSeqs = sequenceStartPositions->getSize() - 1;
  const int* buf = sequenceStartPositions->getData(false);
  int numOnes = 0;
  for (int i = 0; i < numSeqs; ++i) {
    if (buf[i + 1] - buf[i] == 1) {
      ++numOnes;
    }
  }
  // each seq has two sub-seq except length 1
  int numSubSeqs = numSeqs * 2 - numOnes;
  subSequenceStartPositions =
      ICpuGpuVector::create(numSubSeqs + 1, /* useGpu= */ false);
  int* subBuf = subSequenceStartPositions->getMutableData(false);
  int j = 0;
  for (int i = 0; i < numSeqs; ++i) {
    if (buf[i + 1] - buf[i] == 1) {
      subBuf[j++] = buf[i];
    } else {
      int len = uniformRandom(buf[i + 1] - buf[i] - 1) + 1;
      subBuf[j++] = buf[i];
      subBuf[j++] = buf[i] + len;
    }
  }
  subBuf[j] = buf[numSeqs];
}

void generateMDimSequenceData(const IVectorPtr& sequenceStartPositions,
                              IVectorPtr& cpuSequenceDims) {
  /* generate sequences with 2 dims */
  int numSeqs = sequenceStartPositions->getSize() - 1;
  int numDims = 2;

  cpuSequenceDims = IVector::create(numSeqs * numDims, /* useGpu= */ false);
  int* bufStarts = sequenceStartPositions->getData();
  int* bufDims = cpuSequenceDims->getData();

  for (int i = 0; i < numSeqs; i++) {
    int len = bufStarts[i + 1] - bufStarts[i];
    /* get width and height randomly */
    std::vector<int> dimVec;
    for (int j = 0; j < len; j++) {
      if (len % (j + 1) == 0) {
        dimVec.push_back(1);
      }
    }
    int idx = rand() % dimVec.size();  // NOLINT use rand_r
    bufDims[i * numDims] = dimVec[idx];
    bufDims[i * numDims + 1] = len / dimVec[idx];
  }
}

void generateMDimSequenceData(const ICpuGpuVectorPtr& sequenceStartPositions,
                              IVectorPtr& cpuSequenceDims) {
  /* generate sequences with 2 dims */
  int numSeqs = sequenceStartPositions->getSize() - 1;
  int numDims = 2;

  cpuSequenceDims = IVector::create(numSeqs * numDims, /* useGpu= */ false);
  const int* bufStarts = sequenceStartPositions->getData(false);
  int* bufDims = cpuSequenceDims->getData();

  for (int i = 0; i < numSeqs; i++) {
    int len = bufStarts[i + 1] - bufStarts[i];
    /* get width and height randomly */
    std::vector<int> dimVec;
    for (int j = 0; j < len; j++) {
      if (len % (j + 1) == 0) {
        dimVec.push_back(1);
      }
    }
    int idx = rand() % dimVec.size();  // NOLINT use rand_r
    bufDims[i * numDims] = dimVec[idx];
    bufDims[i * numDims + 1] = len / dimVec[idx];
  }
}

void checkMatrixEqual(const MatrixPtr& a, const MatrixPtr& b) {
  EXPECT_EQ(a->getWidth(), b->getWidth());
  EXPECT_EQ(a->getHeight(), b->getHeight());
  EXPECT_EQ(a->isTransposed(), b->isTransposed());
  for (size_t r = 0; r < a->getHeight(); ++r) {
    for (size_t c = 0; c < a->getWidth(); ++c) {
      EXPECT_FLOAT_EQ(a->getElement(r, c), b->getElement(r, c));
    }
  }
}

void checkVectorEqual(const IVectorPtr& a, const IVectorPtr& b) {
  EXPECT_EQ(a->getSize(), b->getSize());
  for (size_t r = 0; r < a->getSize(); ++r) {
    EXPECT_FLOAT_EQ(a->get(r), b->get(r));
  }
}
}  // namespace paddle
