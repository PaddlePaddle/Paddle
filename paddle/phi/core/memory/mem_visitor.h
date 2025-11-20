// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include <cstdint>
#include <vector>
#include "paddle/phi/core/enforce.h"

namespace paddle {
namespace memory {

namespace allocation {
class Allocator;
class RetryAllocator;
class StatAllocator;
class StreamSafeCUDAAllocator;
class VirtualMemoryAutoGrowthBestFitAllocator;
class VirtualMemoryAutoGrowthBestFitMultiScalePoolAllocator;
}  // namespace allocation

using allocation::Allocator;
using allocation::RetryAllocator;
using allocation::StatAllocator;
using allocation::StreamSafeCUDAAllocator;
using allocation::VirtualMemoryAutoGrowthBestFitAllocator;
using allocation::VirtualMemoryAutoGrowthBestFitMultiScalePoolAllocator;

/**
 * @brief AllocatorVisitorReqImpl serves as the Abstract Visitor interface in
 * the Visitor design pattern.
 *
 * It defines the pure virtual function signatures for all required Visit
 * methods necessary to interact with different concrete allocator types.
 * Derived classes must implement these Visit methods to perform specific
 * operations on each allocator type.
 */
class AllocatorVisitorReqImpl {
 public:
  virtual ~AllocatorVisitorReqImpl() = default;
  virtual void Visit(RetryAllocator* allocator) = 0;
  virtual void Visit(StatAllocator* allocator) = 0;
  virtual void Visit(Allocator* allocator) {}
#ifdef PADDLE_WITH_CUDA
  virtual void Visit(StreamSafeCUDAAllocator* allocator) = 0;
  virtual void Visit(VirtualMemoryAutoGrowthBestFitAllocator* allocator) = 0;
  virtual void Visit(
      VirtualMemoryAutoGrowthBestFitMultiScalePoolAllocator* allocator) = 0;
#endif
};

/**
 * @brief AllocatorVisitor is an abstract base class that implements the
 * AllocatorVisitorReqImpl interface.
 *
 * It inherits all the Visit interfaces and can provide default (often recursive
 * call) implementations for them. It serves as a convenient base class for
 * concrete visitors (like FreeMemoryMetricsVisitor), simplifying the
 * implementation by handling cases that do not require specialized logic.
 */
class AllocatorVisitor : public AllocatorVisitorReqImpl {
 public:
  virtual ~AllocatorVisitor() = default;
  virtual void Visit(RetryAllocator* allocator);
  virtual void Visit(StatAllocator* allocator);
  virtual void Visit(Allocator* allocator) {}
#ifdef PADDLE_WITH_CUDA
  virtual void Visit(StreamSafeCUDAAllocator* allocator);
  virtual void Visit(VirtualMemoryAutoGrowthBestFitAllocator* allocator);
  virtual void Visit(
      VirtualMemoryAutoGrowthBestFitMultiScalePoolAllocator* allocator);
#endif
};

#ifdef PADDLE_WITH_CUDA
/**
 * @brief FreeMemoryMetricsVisitor is a Concrete Visitor class designed to
 * inspect allocators for free memory information.
 *
 * Its primary goal is to gather statistics, specifically focusing on the
 * largest contiguous free block size within the visited allocators. Currently,
 * it provides specialized logic for the
 * VirtualMemoryAutoGrowthBestFitAllocator.
 */
class FreeMemoryMetricsVisitor : public AllocatorVisitor {
 public:
  /**
   * @brief Constructor for FreeMemoryMetricsVisitor.
   * @param nums_blocks The number of largest free blocks to potentially track
   * (defaults to 1).
   */
  explicit FreeMemoryMetricsVisitor(int32_t nums_blocks = 1)
      : nums_blocks_(nums_blocks) {}

  /**
   * @brief Implements the visit operation for
   * VirtualMemoryAutoGrowthBestFitAllocator. This is where the logic to query
   * and record the largest and total free sizes resides.
   * @param allocator The VirtualMemoryAutoGrowthBestFitAllocator instance to
   * visit.
   */
  void Visit(VirtualMemoryAutoGrowthBestFitAllocator* allocator) override;

  /**
   * @brief Retrieves the size of the largest free block found during the
   * visitation process.
   * @return The size of the largest free block in bytes.
   */
  size_t GetLargeSize() const { return large_size_; }

  /**
   * @brief Retrieves the total size of all free memory blocks found during the
   * visitation process.
   * @return The sum of `nums_blocks` free block sizes in bytes.
   */
  size_t GetSumSize() const { return sum_size_; }

 private:
  int32_t nums_blocks_ = 1;
  size_t large_size_ = 0;
  size_t sum_size_ = 0;
};

/**
 * @brief Visitor class to attempt memory allocation.
 *
 * To execute a series of memory allocation attempts (based on the
 * sizes_ list provided in the constructor) on a specific memory allocator
 * (typically VirtualMemoryAutoGrowthBestFitAllocator) and record if all
 * attempts were successful.
 */
class TryAllocVisitor : public AllocatorVisitor {
 public:
  /**
   * @brief Constructor.
   *
   * @param sizes A constant reference to a vector containing the sizes
   * of the memory blocks to be attempted for allocation. Defaults to an empty
   * list.
   */
  explicit TryAllocVisitor(const std::vector<size_t>& sizes = {})
      : sizes_(sizes) {}
  /**
   * @brief Visits the VirtualMemoryAutoGrowthBestFitAllocator.
   *
   * This is the core implementation of the Visitor Pattern for this specific
   * allocator. It iterates through the sizes_ list and attempts to call the
   * allocator's TryAllocate() method for each size. The flag
   * is_try_alloc_success_ is only set to true if ALL TryAllocate calls succeed.
   *
   * @param allocator Pointer to the memory allocator object to be visited and
   * tested.
   */
  void Visit(VirtualMemoryAutoGrowthBestFitAllocator* allocator) override;
  /**
   * @brief Queries the result of the allocation attempt.
   *
   * @return Returns true if all TryAllocate attempts were successful;
   * otherwise, returns false.
   */
  bool IsTryAllocSuccess() const { return is_try_alloc_success_; }

 private:
  const std::vector<size_t>& sizes_;
  bool is_try_alloc_success_ = false;
};
#endif

}  // namespace memory
}  // namespace paddle
