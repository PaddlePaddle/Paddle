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

#include <algorithm>
#include "hl_base.h"
#include "paddle/utils/Logging.h"

namespace paddle {

/**
 * \brief The tensor cpu evaluate api.
 */
template <class T, typename LeftType, typename RightType>
inline void TensorCpuApply(LeftType& lhs, const RightType& rhs) {
  TensorApply<LeftType, T> lhs_(lhs);
  TensorApply<const RightType, T> rhs_(rhs);
  CHECK_EQ(lhs_.getWidth(), rhs_.getWidth());
  CHECK_EQ(lhs_.getHeight(), rhs_.getHeight());
  CHECK_EQ(lhs_.useGpu(), rhs_.useGpu());

  int height = lhs_.getHeight();
  int width = lhs_.getWidth();
  if (lhs_.isContiguous() && rhs_.isContiguous()) {
    int size = height * width;
    for (int index = 0; index < size; index++) {
      lhs_.applyRef(index) = rhs_.apply(index);
    }
  } else {
    for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
        lhs_.applyRef(i, j) = rhs_.apply(i, j);
      }
    }
  }
}

#ifdef __NVCC__
template <typename LeftType, typename RightType>
__global__ void TensorElementWiseOp(LeftType lhs,
                                    RightType rhs,
                                    const int border) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < border) {
    lhs.applyRef(idx) = rhs.apply(idx);
  }
}

template <typename LeftType, typename RightType>
__global__ void TensorElementWiseOp(LeftType lhs, RightType rhs) {
  const int colIdx = blockIdx.x * blockDim.x + threadIdx.x;
  const int rowIdx = blockIdx.y * blockDim.y + threadIdx.y;
  for (int i = rowIdx; i < lhs.getHeight(); i += gridDim.y * blockDim.y) {
    for (int j = colIdx; j < lhs.getWidth(); j += gridDim.x * blockDim.x) {
      lhs.applyRef(i, j) = rhs.apply(i, j);
    }
  }
}

/**
 * \brief The tensor gpu evaluate api.
 */
template <class T, typename LeftType, typename RightType>
inline void TensorGpuApply(LeftType& lhs, const RightType& rhs) {
  TensorApply<LeftType, T> lhs_(lhs);
  TensorApply<const RightType, T> rhs_(rhs);
  CHECK_EQ(lhs_.getWidth(), rhs_.getWidth());
  CHECK_EQ(lhs_.getHeight(), rhs_.getHeight());
  CHECK_EQ(lhs_.useGpu(), rhs_.useGpu());

  int dimM = lhs_.getHeight();
  int dimN = lhs_.getWidth();

  if (lhs_.isContiguous() && rhs_.isContiguous()) {
    int size = dimM * dimN;
    int blockSize = size <= 1024 ? size : 1024;
    int gridSize = (size + 1024 - 1) / 1024;
    TensorElementWiseOp<<<gridSize, blockSize, 0, STREAM_DEFAULT>>>(
        lhs_, rhs_, size);
  } else {
    int blockSizeY = std::min(32, dimM);
    int blockSizeX = (32 / blockSizeY) * 32;
    int gridSizeX = std::min(32, (dimN + blockSizeX - 1) / blockSizeX);
    int gridSizeY = std::min(32, (dimM + blockSizeY - 1) / blockSizeY);
    dim3 threads(blockSizeX, blockSizeY);
    dim3 grid(gridSizeX, gridSizeY);
    TensorElementWiseOp<<<grid, threads, 0, STREAM_DEFAULT>>>(lhs_, rhs_);
  }

  CHECK_SYNC("TensorGpuApply failed");
}
#else
template <class T, typename LeftType, typename RightType>
inline void TensorGpuApply(LeftType& lhs, RightType& rhs) {
  LOG(FATAL) << "Since it is gcc compiled, "
                "this calculation does not support GPU implementation.";
}
#endif

}  // namespace paddle
