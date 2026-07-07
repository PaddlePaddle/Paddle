// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

#if defined(PADDLE_WITH_CUDA)

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "paddle/phi/core/memory/allocation/spin_lock.h"
#include "paddle/phi/core/memory/allocation/vmm_allocator_v2_types.h"

namespace paddle {
namespace memory {
namespace allocation {

// Page-granular backing state for VMM V2. Allocation blocks keep only logical
// VA layout; backing ownership and release safety are decided from this map.
class VMMBackingMap {
 public:
  struct MappedPage {
    VMMDevicePtr va{0};
    VMMAllocHandle handle{0};
    std::shared_ptr<VMMHandleMeta> meta;
    uint64_t epoch{0};
  };
  struct UnmappedPage {
    VMMDevicePtr va{0};
    uint64_t epoch{0};
  };
  void Configure(VMMDevicePtr base, size_t size, size_t page_size, int device);

  bool IsConfigured() const { return configured_; }

  void MarkMapped(VMMDevicePtr va, VMMAllocHandle handle, size_t size);
  void MarkMapped(VMMDevicePtr va,
                  const std::shared_ptr<VMMHandleMeta>& meta,
                  size_t size);
  void MarkUnmapped(VMMDevicePtr va, size_t size);
  void MarkReleased(VMMDevicePtr va, VMMAllocHandle handle, size_t size);

  bool ValidateLayout(const HandleLayout& layout, const char* context) const;
  bool IsRangeMapped(VMMDevicePtr va, size_t size) const;
  bool IsRangeUnmapped(VMMDevicePtr va, size_t size) const;
  bool IsRangeReleasable(VMMDevicePtr va, size_t size) const;
  size_t total_mapped_bytes() const;

 private:
  struct Page {
    VMMAllocHandle handle{0};
    std::shared_ptr<VMMHandleMeta> meta;
    bool mapped{false};
    uint64_t epoch{0};
  };

  bool CheckRangeLocked(VMMDevicePtr va,
                        size_t size,
                        const char* context,
                        size_t* start,
                        size_t* count) const;
  void MarkPageMappedLocked(Page* page,
                            VMMDevicePtr page_va,
                            VMMAllocHandle handle,
                            const std::shared_ptr<VMMHandleMeta>& meta);
  void ResetPageToUnmappedLocked(Page* page);
  bool PageCanUseBackingLocked(Page* page, const char* context) const;

  VMMDevicePtr base_{0};
  size_t size_{0};
  size_t page_size_{0};
  int device_{-1};
  bool configured_{false};
  mutable std::vector<Page> pages_;
  size_t mapped_page_count_{0};
  mutable SpinLock spinlock_;
};

}  // namespace allocation
}  // namespace memory
}  // namespace paddle

#endif
