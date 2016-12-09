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
#include "paddle/utils/Logging.h"

namespace paddle {

/**
 * \brief Tensor Assign Expression(return by lazyAssign,
 * and evaluated by AssignEvaluate)
 */
template <typename LhsType, typename RhsType, class T>
class TensorAssignOp {
public:
  explicit TensorAssignOp(const LhsType& lhs, const RhsType& rhs)
      : lhs_(lhs), rhs_(rhs) {
#ifndef __CUDA_ARCH__
    CHECK_EQ(lhs_.getWidth(), rhs_.getWidth());
    CHECK_EQ(lhs_.getHeight(), rhs_.getHeight());
    CHECK_EQ(lhs_.useGpu(), rhs_.useGpu());
#endif
  }

  INLINE void apply(const int i, const int j) {
    lhs_.applyRef(i, j) = rhs_.apply(i, j);
  }
  INLINE void apply(const int index) {
    lhs_.applyRef(index) = rhs_.apply(index);
  }

  INLINE size_t getWidth() const { return lhs_.getWidth(); }
  INLINE size_t getHeight() const { return rhs_.getHeight(); }
  INLINE bool isContiguous() const {
    return lhs_.isContiguous() && rhs_.isContiguous();
  }
  INLINE bool useGpu() const { return lhs_.useGpu(); }

private:
  TensorApply<LhsType, T> lhs_;
  TensorApply<const RhsType, T> rhs_;
};

template <typename Assign, typename... AssignOp>
void AssignCpuEvaluate(int height,
                       int width,
                       bool isContiguous,
                       Assign&& assign,
                       AssignOp&&... args) {
  if (isContiguous) {
    int size = height * width;
    for (int index = 0; index < size; index++) {
      assign.apply(index);
      __attribute__((unused)) int dummy[] = {(((args)).apply(index), 0)...};
    }
  } else {
    for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
        assign.apply(i, j);
        __attribute__((unused)) int dummy[] = {(((args)).apply(i, j), 0)...};
      }
    }
  }
}

#ifdef __NVCC__
template <typename Assign, typename... AssignOp>
__global__ void AssignGpuEvaluate1(const int border,
                                   Assign assign,
                                   AssignOp... args) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < border) {
    assign.apply(idx);
    __attribute__((unused)) int dummy[] = {(((args)).apply(idx), 0)...};
  }
}

template <typename Assign, typename... AssignOp>
__global__ void AssignGpuEvaluate2(const int height,
                                   const int width,
                                   Assign assign,
                                   AssignOp... args) {
  const int colIdx = blockIdx.x * blockDim.x + threadIdx.x;
  const int rowIdx = blockIdx.y * blockDim.y + threadIdx.y;
  for (int i = rowIdx; i < height; i += gridDim.y * blockDim.y) {
    for (int j = colIdx; j < width; j += gridDim.x * blockDim.x) {
      assign.apply(i, j);
      __attribute__((unused)) int dummy[] = {(((args)).apply(i, j), 0)...};
    }
  }
}
#endif

/**
 * \brief Evaluate one or more TensorAssignOp objects.
 *
 * \note At least one assignment expression is required
 */
template <typename Assign, typename... AssignOp>
void AssignEvaluate(Assign&& assign, AssignOp&&... args) {
  const bool useGpu_ = assign.useGpu();
  bool isContiguous_ = assign.isContiguous();
  const size_t height = assign.getHeight();
  const size_t width = assign.getWidth();

  const int packSize = sizeof...(args);
  const bool packUseGpu[] = {((args)).useGpu()...};
  const bool packIsContiguous[] = {((args)).isContiguous()...};
  const size_t packHeight[] = {((args)).getHeight()...};
  const size_t packWidth[] = {((args)).getWidth()...};

  for (int i = 0; i < packSize; i++) {
    CHECK_EQ(useGpu_, packUseGpu[i]);
    CHECK_EQ(height, packHeight[i]);
    CHECK_EQ(width, packWidth[i]);
    isContiguous_ = isContiguous_ && packIsContiguous[i];
  }

  if (useGpu_) {
#ifdef __NVCC__
    if (isContiguous_) {
      int size = height * width;
      int blockSize = size <= 1024 ? size : 1024;
      int gridSize = (size + 1024 - 1) / 1024;
      AssignGpuEvaluate1<<<gridSize, blockSize, 0, STREAM_DEFAULT>>>(
          size, assign, args...);
    } else {
      int blockSizeY = std::min(32, (int)height);
      int blockSizeX = (32 / blockSizeY) * 32;
      int gridSizeX = std::min(32, (int)(width + blockSizeX - 1) / blockSizeX);
      int gridSizeY = std::min(32, (int)(height + blockSizeY - 1) / blockSizeY);
      dim3 threads(blockSizeX, blockSizeY);
      dim3 grid(gridSizeX, gridSizeY);
      AssignGpuEvaluate2<<<grid, threads, 0, STREAM_DEFAULT>>>(
          height, width, assign, args...);
    }

    CHECK_SYNC("AssignEvaluate failed");
#endif
  } else {
    AssignCpuEvaluate(height, width, isContiguous_, assign, args...);
  }
}

}  // namespace paddle
