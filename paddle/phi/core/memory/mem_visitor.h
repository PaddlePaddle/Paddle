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
#ifdef PADDLE_WITH_CUDA
#include "paddle/phi/core/memory/allocation/vmm_ipc_allocation.h"
#endif

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
 * @brief AllocatorComputeStreamVisitor is a Concrete Visitor class designed to
 * only visit compute stream allocators.
 */
class AllocatorComputeStreamVisitor : public AllocatorVisitor {
 public:
  using AllocatorVisitor::Visit;
  void Visit(StreamSafeCUDAAllocator* allocator) override;
};

/**
 * @brief FreeMemoryMetricsVisitor is a Concrete Visitor class designed to
 * inspect allocators for free memory information.
 *
 * Its primary goal is to gather statistics, specifically focusing on the
 * largest contiguous free block size within the visited allocators. Currently,
 * it provides specialized logic for the
 * VirtualMemoryAutoGrowthBestFitAllocator.
 */
class FreeMemoryMetricsVisitor : public AllocatorComputeStreamVisitor {
 public:
  using AllocatorComputeStreamVisitor::Visit;
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
class TryAllocVisitor : public AllocatorComputeStreamVisitor {
  using AllocatorComputeStreamVisitor::Visit;

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

/**
 * @brief Visitor class to retrieve free block information from a VMM allocator.
 *
 * Inherits from AllocatorVisitor, implementing the Visitor Pattern.
 * The purpose of this class is to access a specific memory allocator's
 * internal state (the list of free memory blocks) and extract key information
 * (size and address) for external analysis or debugging.
 */
class VMMFreeBlocksInfoVisitor : public AllocatorComputeStreamVisitor {
  using AllocatorComputeStreamVisitor::Visit;

 public:
  /**
   * @brief Retrieves the collected information about the free memory blocks.
   *
   * The structure is a nested vector:
   * Outer Vector: Represents different categories or lists within the
   * allocator. Inner Vector: Contains pairs of (size, address) for the free
   * blocks in that category. uintptr_t is used to safely store the memory
   * address (void*) as an integer.
   *
   * @return A nested vector structure containing the size and integer address
   * of all free blocks.
   */
  std::vector<std::vector<std::pair<size_t, uintptr_t>>> GetFreeBlocksInfo()
      const {
    return free_blocks_info_;
  }

  /**
   * @brief Visits the VirtualMemoryAutoGrowthBestFitAllocator.
   *
   * This is the core implementation of the Visitor Pattern. When called,
   * it accesses the `allocator` object's internal structure that holds the
   * free block list(s) and populates the `free_blocks_info_` member variable
   * with the necessary data.
   *
   * @param allocator Pointer to the memory allocator object whose free blocks
   * information is to be extracted.
   */
  void Visit(VirtualMemoryAutoGrowthBestFitAllocator* allocator) override;

 private:
  /**
   * @brief Stores the extracted free block information.
   *
   * This member is populated during the Visit() call. It is structured to
   * hold lists of (size, address) pairs, where the outer vector typically
   * distinguishes between different allocators (e.g., small, large allocator).
   */
  std::vector<std::vector<std::pair<size_t, uintptr_t>>> free_blocks_info_;
};

/**
 * @brief Visitor class to retrieve All block information from a VMM allocator.
 *
 * Inherits from AllocatorVisitor, implementing the Visitor Pattern.
 * The purpose of this class is to access a specific memory allocator's
 * internal state (the list of all memory blocks) and extract key information
 * (size, address and free_info) for external analysis or debugging.
 */
class VMMAllBlocksInfoVisitor : public AllocatorComputeStreamVisitor {
  using AllocatorComputeStreamVisitor::Visit;

 public:
  /**
   * @brief Retrieves the collected information about the free memory blocks.
   *
   * The structure is a nested vector:
   * Outer Vector: Represents different categories or lists within the
   * allocator. Inner Vector: Contains tuple of (size, address, free_info) for
   * the all blocks in that category. uintptr_t is used to safely store the
   * memory address (void*) as an integer.
   *
   * @return A nested vector structure containing the size, integer address,
   * free info of all blocks.
   */
  std::vector<std::vector<std::tuple<size_t, uintptr_t, bool>>>
  GetAllBlocksInfo() const {
    return all_blocks_info_;
  }

  /**
   * @brief Visits the VirtualMemoryAutoGrowthBestFitAllocator.
   *
   * This is the core implementation of the Visitor Pattern. When called,
   * it accesses the `allocator` object's internal structure that holds the
   * free block list(s) and populates the `all_blocks_info_` member variable
   * with the necessary data.
   *
   * @param allocator Pointer to the memory allocator object whose free blocks
   * information is to be extracted.
   */
  void Visit(VirtualMemoryAutoGrowthBestFitAllocator* allocator) override;

 private:
  /**
   * @brief Stores the extracted all block information.
   *
   * This member is populated during the Visit() call. It is structured to
   * hold lists of (size, address, free_info) tuples, where the outer vector
   * typically distinguishes between different allocators (e.g., small, large
   * allocator).
   */
  std::vector<std::vector<std::tuple<size_t, uintptr_t, bool>>>
      all_blocks_info_;
};

class VMMAllocateRecordEventsVisitor : public AllocatorComputeStreamVisitor {
  using AllocatorComputeStreamVisitor::Visit;

 public:
  std::vector<std::tuple<uintptr_t, bool, uint64_t, size_t, int64_t, int64_t>>
  GetAllocateRecordEvents() const {
    return allocate_record_event_;
  }

  void Visit(VirtualMemoryAutoGrowthBestFitMultiScalePoolAllocator* allocator)
      override;

 private:
  std::vector<std::tuple<uintptr_t, bool, uint64_t, size_t, int64_t, int64_t>>
      allocate_record_event_;
};

class VMMAllocateCompactSizeVisitor : public AllocatorComputeStreamVisitor {
  using AllocatorComputeStreamVisitor::Visit;

 public:
  std::vector<size_t> GetCompactSize() const { return allocate_compact_size_; }

  void Visit(VirtualMemoryAutoGrowthBestFitMultiScalePoolAllocator* allocator)
      override;

 private:
  std::vector<size_t> allocate_compact_size_;
};

class VmmTensorPartsVisitor : public AllocatorVisitor {
 public:
  using BlockPart = allocation::BlockPart;
  explicit VmmTensorPartsVisitor(void* ptr) : target_ptr_(ptr) {}

  void Visit(VirtualMemoryAutoGrowthBestFitAllocator* allocator) override;

  bool Found() const { return found_; }
  const std::vector<BlockPart>& Parts() const { return parts_; }

 private:
  void* target_ptr_{nullptr};
  bool found_{false};
  std::vector<BlockPart> parts_;
};
#endif

}  // namespace memory
}  // namespace paddle
