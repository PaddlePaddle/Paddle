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

#include "paddle/phi/core/memory/allocation/vmm_backing_map.h"

#if defined(PADDLE_WITH_CUDA)

#include <algorithm>
#include <mutex>

#include "glog/logging.h"
#include "paddle/phi/core/enforce.h"

namespace paddle {
namespace memory {
namespace allocation {

namespace {

bool AddOverflow(VMMDevicePtr base, size_t size) { return base + size < base; }

bool ComputeOverlappedPages(VMMDevicePtr base,
                            size_t backing_size,
                            size_t page_size,
                            VMMDevicePtr va,
                            size_t size,
                            const char* context,
                            size_t* start,
                            size_t* count) {
  if (size == 0 || page_size == 0 || AddOverflow(base, backing_size) ||
      va < base || va + size < va || va + size > base + backing_size) {
    VLOG(3) << "VMM V2 BackingMap invalid overlap range in " << context
            << ": va=" << reinterpret_cast<void*>(va) << " size=" << size
            << " base=" << reinterpret_cast<void*>(base)
            << " backing_size=" << backing_size << " page_size=" << page_size;
    return false;
  }
  const size_t begin_offset = va - base;
  const size_t end_offset = va + size - base;
  *start = begin_offset / page_size;
  const size_t end_page = (end_offset + page_size - 1) / page_size;
  *count = end_page - *start;
  return true;
}

}  // namespace

void VMMBackingMap::Configure(VMMDevicePtr base,
                              size_t size,
                              size_t page_size,
                              int device) {
  std::lock_guard<SpinLock> guard(spinlock_);
  PADDLE_ENFORCE_GT(page_size,
                    0UL,
                    common::errors::InvalidArgument(
                        "VMM V2 BackingMap page_size must be positive."));
  PADDLE_ENFORCE_EQ(
      size % page_size,
      0UL,
      common::errors::InvalidArgument(
          "VMM V2 BackingMap size %zu must be page-aligned by page_size %zu.",
          size,
          page_size));
  if (configured_) {
    if (base_ != base || size_ != size || page_size_ != page_size ||
        device_ != device) {
      VLOG(3) << "VMM V2 BackingMap reconfigure mismatch: old_base="
              << reinterpret_cast<void*>(base_)
              << " new_base=" << reinterpret_cast<void*>(base)
              << " old_size=" << size_ << " new_size=" << size
              << " old_page_size=" << page_size_
              << " new_page_size=" << page_size << " old_device=" << device_
              << " new_device=" << device;
    }
    return;
  }

  base_ = base;
  size_ = size;
  page_size_ = page_size;
  device_ = device;
  configured_ = true;
  pages_.resize(size_ / page_size_);
  mapped_page_count_ = 0;
}

bool VMMBackingMap::CheckRangeLocked(VMMDevicePtr va,
                                     size_t size,
                                     const char* context,
                                     size_t* start,
                                     size_t* count) const {
  if (!configured_) {
    VLOG(3) << "VMM V2 BackingMap " << context
            << " before Configure, va=" << reinterpret_cast<void*>(va)
            << " size=" << size;
    return false;
  }
  if (size == 0 || page_size_ == 0 || size % page_size_ != 0 ||
      AddOverflow(base_, size_) || va < base_ || va + size < va ||
      va + size > base_ + size_ || (va - base_) % page_size_ != 0) {
    VLOG(3) << "VMM V2 BackingMap invalid range in " << context
            << ": va=" << reinterpret_cast<void*>(va) << " size=" << size
            << " base=" << reinterpret_cast<void*>(base_)
            << " backing_size=" << size_ << " page_size=" << page_size_;
    return false;
  }
  *start = (va - base_) / page_size_;
  *count = size / page_size_;
  return true;
}

void VMMBackingMap::MarkPageMappedLocked(
    Page* page,
    VMMDevicePtr page_va,
    VMMAllocHandle handle,
    const std::shared_ptr<VMMHandleMeta>& meta,
    bool remap_destination_owned) {
  PADDLE_ENFORCE_EQ(
      page->mapped && handle != 0 && page->handle != handle,
      false,
      common::errors::PreconditionNotMet(
          "VMM V2 BackingMap cannot overwrite mapped page at %p from "
          "handle %p to %p.",
          reinterpret_cast<void*>(page_va),
          reinterpret_cast<void*>(page->handle),
          reinterpret_cast<void*>(handle)));
  if (!page->mapped) {
    mapped_page_count_++;
  }
  page->handle = handle;
  page->meta = meta;
  page->mapped = true;
  page->remap_destination_owned = remap_destination_owned;
  page->pending_events.clear();
  page->epoch++;
}

void VMMBackingMap::ResetPageToUnmappedLocked(Page* page,
                                              bool clear_ipc_exported) {
  if (page->mapped && mapped_page_count_ > 0) {
    mapped_page_count_--;
  }
  page->handle = 0;
  page->meta.reset();
  page->mapped = false;
  page->remap_destination_owned = false;
  if (clear_ipc_exported) {
    page->ipc_exported = false;
  }
  page->pending_events.clear();
  page->epoch++;
}

void VMMBackingMap::MarkMapped(VMMDevicePtr va,
                               VMMAllocHandle handle,
                               size_t size) {
  std::lock_guard<SpinLock> guard(spinlock_);
  size_t start = 0;
  size_t count = 0;
  if (!CheckRangeLocked(va, size, "MarkMapped(handle)", &start, &count)) {
    return;
  }
  for (size_t i = 0; i < count; ++i) {
    auto& page = pages_[start + i];
    MarkPageMappedLocked(&page,
                         va + i * page_size_,
                         handle,
                         std::shared_ptr<VMMHandleMeta>(),
                         false);
  }
}

void VMMBackingMap::MarkMapped(VMMDevicePtr va,
                               const std::shared_ptr<VMMHandleMeta>& meta,
                               size_t size) {
  std::lock_guard<SpinLock> guard(spinlock_);
  size_t start = 0;
  size_t count = 0;
  if (!CheckRangeLocked(va, size, "MarkMapped", &start, &count)) {
    return;
  }
  const VMMAllocHandle handle =
      meta == nullptr ? static_cast<VMMAllocHandle>(0) : meta->handle();
  for (size_t i = 0; i < count; ++i) {
    auto& page = pages_[start + i];
    MarkPageMappedLocked(&page, va + i * page_size_, handle, meta, false);
  }
}

void VMMBackingMap::MarkRemapDestinationMapped(
    VMMDevicePtr va, const std::shared_ptr<VMMHandleMeta>& meta, size_t size) {
  std::lock_guard<SpinLock> guard(spinlock_);
  size_t start = 0;
  size_t count = 0;
  if (!CheckRangeLocked(
          va, size, "MarkRemapDestinationMapped", &start, &count)) {
    return;
  }
  const VMMAllocHandle handle =
      meta == nullptr ? static_cast<VMMAllocHandle>(0) : meta->handle();
  for (size_t i = 0; i < count; ++i) {
    auto& page = pages_[start + i];
    MarkPageMappedLocked(&page, va + i * page_size_, handle, meta, true);
  }
}

bool VMMBackingMap::ClearRemapDestinationOwnership(VMMDevicePtr va,
                                                   size_t size) {
  std::lock_guard<SpinLock> guard(spinlock_);
  size_t start = 0;
  size_t count = 0;
  if (!CheckRangeLocked(
          va, size, "ClearRemapDestinationOwnership", &start, &count)) {
    return false;
  }
  for (size_t i = 0; i < count; ++i) {
    auto& page = pages_[start + i];
    if (!page.mapped || page.meta == nullptr) {
      VLOG(3) << "VMM V2 BackingMap cannot clear remap destination ownership "
              << "for non-mapped page at "
              << reinterpret_cast<void*>(va + i * page_size_);
      return false;
    }
    bool changed = false;
    if (page.remap_destination_owned) {
      page.remap_destination_owned = false;
      changed = true;
    }
    if (page.meta->IsOwnedByRemapDestination()) {
      page.meta->RestoreOriginalOwnership();
      changed = true;
    }
    if (changed) {
      page.epoch++;
    }
  }
  return true;
}

size_t VMMBackingMap::ClearRemapDestinationOwnershipInRange(VMMDevicePtr va,
                                                            size_t size) {
  std::lock_guard<SpinLock> guard(spinlock_);
  if (!configured_) {
    VLOG(3) << "VMM V2 BackingMap ClearRemapDestinationOwnershipInRange "
            << "before Configure, va=" << reinterpret_cast<void*>(va)
            << " size=" << size;
    return 0;
  }
  if (size == 0 || page_size_ == 0 || AddOverflow(base_, size_) || va < base_ ||
      va + size < va || va + size > base_ + size_) {
    VLOG(3) << "VMM V2 BackingMap invalid range in "
            << "ClearRemapDestinationOwnershipInRange"
            << ": va=" << reinterpret_cast<void*>(va) << " size=" << size
            << " base=" << reinterpret_cast<void*>(base_)
            << " backing_size=" << size_ << " page_size=" << page_size_;
    return 0;
  }

  const VMMDevicePtr range_end = va + size;
  const size_t start_offset = va - base_;
  const size_t end_offset = range_end - base_;
  // Only pages fully covered by the requested range can safely lose remap
  // destination ownership. Partially overlapped backing pages keep their state.
  const size_t first_page = (start_offset + page_size_ - 1) / page_size_;
  const size_t end_page = end_offset / page_size_;
  if (first_page >= end_page) {
    return 0;
  }

  size_t cleared_pages = 0;
  for (size_t page_idx = first_page; page_idx < end_page; ++page_idx) {
    auto& page = pages_[page_idx];
    if (!page.mapped || page.meta == nullptr) {
      continue;
    }
    bool changed = false;
    if (page.remap_destination_owned) {
      page.remap_destination_owned = false;
      changed = true;
    }
    if (page.meta->IsOwnedByRemapDestination()) {
      page.meta->RestoreOriginalOwnership();
      changed = true;
    }
    if (changed) {
      page.epoch++;
      cleared_pages++;
    }
  }
  return cleared_pages * page_size_;
}

void VMMBackingMap::MarkUnmapped(VMMDevicePtr va, size_t size) {
  std::lock_guard<SpinLock> guard(spinlock_);
  size_t start = 0;
  size_t count = 0;
  if (!CheckRangeLocked(va, size, "MarkUnmapped", &start, &count)) {
    return;
  }
  for (size_t i = 0; i < count; ++i) {
    auto& page = pages_[start + i];
    if (!page.mapped) {
      VLOG(5) << "VMM V2 BackingMap unmapping already-unmapped page at "
              << reinterpret_cast<void*>(va + i * page_size_);
    }
    ResetPageToUnmappedLocked(&page, false);
  }
}

void VMMBackingMap::MarkReleased(VMMDevicePtr va,
                                 VMMAllocHandle handle,
                                 size_t size) {
  std::lock_guard<SpinLock> guard(spinlock_);
  size_t start = 0;
  size_t count = 0;
  if (!CheckRangeLocked(va, size, "MarkReleased", &start, &count)) {
    return;
  }
  for (size_t i = 0; i < count; ++i) {
    auto& page = pages_[start + i];
    if (handle != 0 && page.handle != 0 && page.handle != handle) {
      VLOG(3) << "VMM V2 BackingMap release handle mismatch at "
              << reinterpret_cast<void*>(va + i * page_size_)
              << " tracked=" << reinterpret_cast<void*>(page.handle)
              << " released=" << reinterpret_cast<void*>(handle);
    }
    ResetPageToUnmappedLocked(&page, true);
  }
}

void VMMBackingMap::MarkIPCExported(VMMDevicePtr va, size_t size) {
  std::lock_guard<SpinLock> guard(spinlock_);
  size_t start = 0;
  size_t count = 0;
  if (!ComputeOverlappedPages(base_,
                              size_,
                              page_size_,
                              va,
                              size,
                              "MarkIPCExported",
                              &start,
                              &count)) {
    return;
  }
  for (size_t i = 0; i < count; ++i) {
    auto& page = pages_[start + i];
    if (!page.mapped) {
      VLOG(4) << "VMM V2 BackingMap marks unmapped page as IPC-exported at "
              << reinterpret_cast<void*>(va + i * page_size_);
    }
    page.ipc_exported = true;
    page.epoch++;
  }
}

void VMMBackingMap::MarkPendingEvent(VMMDevicePtr va,
                                     size_t size,
                                     gpuStream_t stream,
                                     std::shared_ptr<CUDAEventGuard> event) {
  std::lock_guard<SpinLock> guard(spinlock_);
  size_t start = 0;
  size_t count = 0;
  if (!CheckRangeLocked(va, size, "MarkPendingEvent", &start, &count)) {
    return;
  }
  for (size_t i = 0; i < count; ++i) {
    auto& page = pages_[start + i];
    if (!page.mapped) {
      VLOG(6) << "VMM V2 BackingMap marks unmapped page event-pending at "
              << reinterpret_cast<void*>(va + i * page_size_);
    }
    auto same_stream = std::find_if(page.pending_events.begin(),
                                    page.pending_events.end(),
                                    [stream](const PendingEvent& pending) {
                                      return pending.stream == stream;
                                    });
    if (same_stream != page.pending_events.end()) {
      same_stream->event = event;
    } else {
      page.pending_events.push_back(PendingEvent{stream, event});
    }
    page.epoch++;
  }
}

bool VMMBackingMap::MarkPendingEventForRange(
    VMMDevicePtr va,
    size_t size,
    gpuStream_t stream,
    std::shared_ptr<CUDAEventGuard> event) {
  std::lock_guard<SpinLock> guard(spinlock_);
  size_t start = 0;
  size_t count = 0;
  if (!ComputeOverlappedPages(base_,
                              size_,
                              page_size_,
                              va,
                              size,
                              "MarkPendingEventForRange",
                              &start,
                              &count)) {
    return false;
  }
  for (size_t i = 0; i < count; ++i) {
    auto& page = pages_[start + i];
    if (!page.mapped) {
      VLOG(6) << "VMM V2 BackingMap marks unmapped page event-pending at "
              << reinterpret_cast<void*>(base_ + (start + i) * page_size_);
    }
    auto same_stream = std::find_if(page.pending_events.begin(),
                                    page.pending_events.end(),
                                    [stream](const PendingEvent& pending) {
                                      return pending.stream == stream;
                                    });
    if (same_stream != page.pending_events.end()) {
      same_stream->event = event;
    } else {
      page.pending_events.push_back(PendingEvent{stream, event});
    }
    page.epoch++;
  }
  return true;
}

bool VMMBackingMap::ValidateLayout(const HandleLayout& layout,
                                   const char* context) const {
  std::lock_guard<SpinLock> guard(spinlock_);
  bool ok = true;
  for (const auto& meta : layout) {
    size_t start = 0;
    size_t count = 0;
    if (!CheckRangeLocked(
            meta->base(), meta->size(), context, &start, &count)) {
      ok = false;
      continue;
    }
    for (size_t i = 0; i < count; ++i) {
      const auto& page = pages_[start + i];
      const bool expected_mapped = !meta->IsOwnedByRemapDestination();
      if (page.mapped != expected_mapped) {
        VLOG(3) << "VMM V2 BackingMap mapped-state mismatch in " << context
                << " va="
                << reinterpret_cast<void*>(meta->base() + i * page_size_)
                << " tracked_mapped=" << page.mapped
                << " meta_owned_by_remap_destination="
                << meta->IsOwnedByRemapDestination();
        ok = false;
      }
      if (expected_mapped && page.handle != meta->handle()) {
        VLOG(3) << "VMM V2 BackingMap handle mismatch in " << context << " va="
                << reinterpret_cast<void*>(meta->base() + i * page_size_)
                << " tracked=" << reinterpret_cast<void*>(page.handle)
                << " meta=" << reinterpret_cast<void*>(meta->handle());
        ok = false;
      }
    }
  }
  return ok;
}

bool VMMBackingMap::CollectIPCPartDescriptors(
    VMMDevicePtr va,
    size_t size,
    std::vector<IPCBlockPartDescriptor>* descriptors) const {
  std::lock_guard<SpinLock> guard(spinlock_);
  return CollectIPCPartDescriptorsLocked(va, size, descriptors);
}

bool VMMBackingMap::IsRangeMapped(VMMDevicePtr va, size_t size) const {
  std::lock_guard<SpinLock> guard(spinlock_);
  size_t start = 0;
  size_t count = 0;
  if (!CheckRangeLocked(va, size, "IsRangeMapped", &start, &count)) {
    return false;
  }
  for (size_t i = 0; i < count; ++i) {
    if (!pages_[start + i].mapped) {
      return false;
    }
  }
  return true;
}

bool VMMBackingMap::IsRangeUnmapped(VMMDevicePtr va, size_t size) const {
  std::lock_guard<SpinLock> guard(spinlock_);
  size_t start = 0;
  size_t count = 0;
  if (!CheckRangeLocked(va, size, "IsRangeUnmapped", &start, &count)) {
    return false;
  }
  for (size_t i = 0; i < count; ++i) {
    if (pages_[start + i].mapped) {
      return false;
    }
  }
  return true;
}

bool VMMBackingMap::IsRangeReleasable(VMMDevicePtr va, size_t size) const {
  std::lock_guard<SpinLock> guard(spinlock_);
  size_t start = 0;
  size_t count = 0;
  if (!ComputeOverlappedPages(base_,
                              size_,
                              page_size_,
                              va,
                              size,
                              "IsRangeReleasable",
                              &start,
                              &count)) {
    return false;
  }
  for (size_t i = 0; i < count; ++i) {
    if (pages_[start + i].ipc_exported ||
        !PageCanUseBackingLocked(&pages_[start + i], "IsRangeReleasable")) {
      return false;
    }
  }
  return true;
}

bool VMMBackingMap::CanReleaseHandle(VMMDevicePtr va,
                                     VMMAllocHandle handle,
                                     const std::shared_ptr<VMMHandleMeta>& meta,
                                     size_t size) const {
  std::lock_guard<SpinLock> guard(spinlock_);
  size_t start = 0;
  size_t count = 0;
  if (!CheckRangeLocked(va, size, "CanReleaseHandle", &start, &count)) {
    return false;
  }
  for (size_t i = 0; i < count; ++i) {
    auto& page = pages_[start + i];
    if (!page.mapped || page.handle != handle || page.meta != meta ||
        page.ipc_exported ||
        !PageEventsReadyLocked(&page, "CanReleaseHandle")) {
      return false;
    }
  }
  return true;
}

bool VMMBackingMap::HasIPCExportedPages(VMMDevicePtr va, size_t size) const {
  std::lock_guard<SpinLock> guard(spinlock_);
  size_t start = 0;
  size_t count = 0;
  if (!ComputeOverlappedPages(base_,
                              size_,
                              page_size_,
                              va,
                              size,
                              "HasIPCExportedPages",
                              &start,
                              &count)) {
    return true;
  }
  for (size_t i = 0; i < count; ++i) {
    if (pages_[start + i].ipc_exported) {
      return true;
    }
  }
  return false;
}

size_t VMMBackingMap::CountIPCExportedBytes(
    const std::vector<std::pair<VMMDevicePtr, size_t>>& ranges) const {
  std::lock_guard<SpinLock> guard(spinlock_);
  size_t bytes = 0;
  for (const auto& range : ranges) {
    const auto va = range.first;
    const auto size = range.second;
    size_t start = 0;
    size_t count = 0;
    if (!ComputeOverlappedPages(base_,
                                size_,
                                page_size_,
                                va,
                                size,
                                "CountIPCExportedBytes",
                                &start,
                                &count)) {
      continue;
    }
    for (size_t i = 0; i < count; ++i) {
      const auto& page = pages_[start + i];
      if (!page.ipc_exported) {
        continue;
      }
      const VMMDevicePtr page_va = base_ + (start + i) * page_size_;
      const VMMDevicePtr slice_begin = std::max(va, page_va);
      const VMMDevicePtr slice_end = std::min(va + size, page_va + page_size_);
      bytes += slice_end - slice_begin;
    }
  }
  return bytes;
}

std::vector<VMMBackingMap::MappedPage> VMMBackingMap::CollectMappedPages(
    const std::vector<std::pair<VMMDevicePtr, size_t>>& ranges,
    size_t target_bytes) const {
  std::lock_guard<SpinLock> guard(spinlock_);
  std::vector<MappedPage> mapped_pages;
  if (page_size_ == 0) {
    return mapped_pages;
  }

  const size_t target_pages =
      target_bytes == 0 ? 0 : (target_bytes + page_size_ - 1) / page_size_;
  for (const auto& range : ranges) {
    AppendMappedPagesLocked(range.first,
                            range.second,
                            "CollectMappedPages",
                            target_pages,
                            &mapped_pages);
    if (target_pages != 0 && mapped_pages.size() >= target_pages) {
      break;
    }
  }
  return mapped_pages;
}

std::vector<VMMBackingMap::MappedPage>
VMMBackingMap::CollectMappedPagesFullyInRange(
    const std::vector<std::pair<VMMDevicePtr, size_t>>& ranges,
    size_t target_bytes) const {
  std::lock_guard<SpinLock> guard(spinlock_);
  std::vector<MappedPage> mapped_pages;
  if (page_size_ == 0) {
    return mapped_pages;
  }

  const size_t target_pages =
      target_bytes == 0 ? 0 : (target_bytes + page_size_ - 1) / page_size_;
  for (const auto& range : ranges) {
    AppendMappedPagesFullyInRangeLocked(range.first,
                                        range.second,
                                        "CollectMappedPagesFullyInRange",
                                        target_pages,
                                        false,
                                        false,
                                        &mapped_pages);
    if (target_pages != 0 && mapped_pages.size() >= target_pages) {
      break;
    }
  }
  return mapped_pages;
}

std::vector<VMMBackingMap::MappedPage>
VMMBackingMap::CollectRemapSourcePagesFullyInRange(
    const std::vector<std::pair<VMMDevicePtr, size_t>>& ranges,
    size_t target_bytes) const {
  std::lock_guard<SpinLock> guard(spinlock_);
  std::vector<MappedPage> mapped_pages;
  const size_t target_pages =
      (target_bytes == 0 || page_size_ == 0)
          ? 0
          : (target_bytes + page_size_ - 1) / page_size_;
  for (const auto& range : ranges) {
    AppendMappedPagesFullyInRangeLocked(range.first,
                                        range.second,
                                        "CollectRemapSourcePagesFullyInRange",
                                        0,
                                        false,
                                        true,
                                        &mapped_pages);
    if (target_pages != 0) {
      size_t ready_pages = 0;
      for (const auto& page : mapped_pages) {
        if (page.remap_source_state == RemapSourceState::kReady) {
          ++ready_pages;
        }
      }
      if (ready_pages >= target_pages) {
        break;
      }
    }
  }
  if (target_pages != 0) {
    size_t ready_pages = 0;
    size_t keep_pages = mapped_pages.size();
    for (size_t i = 0; i < mapped_pages.size(); ++i) {
      if (mapped_pages[i].remap_source_state == RemapSourceState::kReady) {
        ++ready_pages;
        if (ready_pages >= target_pages) {
          keep_pages = i + 1;
          break;
        }
      }
    }
    if (keep_pages < mapped_pages.size()) {
      mapped_pages.resize(keep_pages);
    }
  }
  return mapped_pages;
}

std::vector<VMMBackingMap::UnmappedPage>
VMMBackingMap::CollectUnmappedPagesFullyInRange(
    const std::vector<std::pair<VMMDevicePtr, size_t>>& ranges,
    size_t target_bytes) const {
  std::lock_guard<SpinLock> guard(spinlock_);
  std::vector<UnmappedPage> unmapped_pages;
  if (page_size_ == 0) {
    return unmapped_pages;
  }

  const size_t target_pages =
      target_bytes == 0 ? 0 : (target_bytes + page_size_ - 1) / page_size_;
  for (const auto& range : ranges) {
    AppendUnmappedPagesFullyInRangeLocked(range.first,
                                          range.second,
                                          "CollectUnmappedPagesFullyInRange",
                                          target_pages,
                                          &unmapped_pages);
    if (target_pages != 0 && unmapped_pages.size() >= target_pages) {
      break;
    }
  }
  return unmapped_pages;
}

VMMBackingMap::CompactCandidates VMMBackingMap::CollectCompactCandidates(
    const std::vector<std::pair<VMMDevicePtr, size_t>>& source_ranges,
    const std::vector<std::pair<VMMDevicePtr, size_t>>& target_ranges,
    size_t target_bytes) const {
  std::lock_guard<SpinLock> guard(spinlock_);
  CompactCandidates candidates;
  const size_t max_pages = (target_bytes == 0 || page_size_ == 0)
                               ? 0
                               : (target_bytes + page_size_ - 1) / page_size_;

  for (const auto& range : source_ranges) {
    AppendMappedPagesFullyInRangeLocked(range.first,
                                        range.second,
                                        "CollectCompactCandidates.source",
                                        max_pages,
                                        true,
                                        false,
                                        &candidates.source_pages);
    if (max_pages != 0 && candidates.source_pages.size() >= max_pages) {
      break;
    }
  }
  for (const auto& range : target_ranges) {
    AppendUnmappedPagesFullyInRangeLocked(range.first,
                                          range.second,
                                          "CollectCompactCandidates.target",
                                          max_pages,
                                          &candidates.target_pages);
    if (max_pages != 0 && candidates.target_pages.size() >= max_pages) {
      break;
    }
  }
  return candidates;
}

bool VMMBackingMap::ValidateMappedPages(
    const std::vector<MappedPage>& mapped_pages, const char* context) const {
  std::lock_guard<SpinLock> guard(spinlock_);
  bool ok = true;
  for (const auto& mapped_page : mapped_pages) {
    size_t start = 0;
    size_t count = 0;
    if (!CheckRangeLocked(
            mapped_page.va, page_size_, context, &start, &count)) {
      ok = false;
      continue;
    }
    const auto& page = pages_[start];
    if (!page.mapped) {
      VLOG(3) << "VMM V2 BackingMap mapped page became unmapped in " << context
              << ": va=" << reinterpret_cast<void*>(mapped_page.va)
              << " snapshot_epoch=" << mapped_page.epoch
              << " current_epoch=" << page.epoch;
      ok = false;
      continue;
    }
    if (page.handle != mapped_page.handle) {
      VLOG(3) << "VMM V2 BackingMap mapped page handle changed in " << context
              << ": va=" << reinterpret_cast<void*>(mapped_page.va)
              << " snapshot_handle="
              << reinterpret_cast<void*>(mapped_page.handle)
              << " current_handle=" << reinterpret_cast<void*>(page.handle);
      ok = false;
    }
    if (page.epoch != mapped_page.epoch) {
      VLOG(3) << "VMM V2 BackingMap mapped page epoch changed in " << context
              << ": va=" << reinterpret_cast<void*>(mapped_page.va)
              << " snapshot_epoch=" << mapped_page.epoch
              << " current_epoch=" << page.epoch;
      ok = false;
    }
  }
  return ok;
}

bool VMMBackingMap::ValidateUnmappedPages(
    const std::vector<UnmappedPage>& unmapped_pages,
    const char* context) const {
  std::lock_guard<SpinLock> guard(spinlock_);
  bool ok = true;
  for (const auto& unmapped_page : unmapped_pages) {
    size_t start = 0;
    size_t count = 0;
    if (!CheckRangeLocked(
            unmapped_page.va, page_size_, context, &start, &count)) {
      ok = false;
      continue;
    }
    const auto& page = pages_[start];
    if (page.mapped) {
      VLOG(3) << "VMM V2 BackingMap unmapped page became mapped in " << context
              << ": va=" << reinterpret_cast<void*>(unmapped_page.va)
              << " snapshot_epoch=" << unmapped_page.epoch
              << " current_epoch=" << page.epoch;
      ok = false;
      continue;
    }
    if (page.handle != 0) {
      VLOG(3) << "VMM V2 BackingMap unmapped page retains handle in " << context
              << ": va=" << reinterpret_cast<void*>(unmapped_page.va)
              << " handle=" << reinterpret_cast<void*>(page.handle);
      ok = false;
    }
    if (page.epoch != unmapped_page.epoch) {
      VLOG(3) << "VMM V2 BackingMap unmapped page epoch changed in " << context
              << ": va=" << reinterpret_cast<void*>(unmapped_page.va)
              << " snapshot_epoch=" << unmapped_page.epoch
              << " current_epoch=" << page.epoch;
      ok = false;
    }
  }
  return ok;
}

void VMMBackingMap::AppendMappedPagesLocked(
    VMMDevicePtr va,
    size_t size,
    const char* context,
    size_t max_pages,
    std::vector<MappedPage>* mapped_pages) const {
  size_t start = 0;
  size_t count = 0;
  if (!CheckRangeLocked(va, size, context, &start, &count)) {
    return;
  }

  for (size_t i = 0; i < count; ++i) {
    if (max_pages != 0 && mapped_pages->size() >= max_pages) {
      break;
    }
    const auto& page = pages_[start + i];
    if (!page.mapped || page.ipc_exported ||
        !PageCanUseBackingLocked(&pages_[start + i], context)) {
      continue;
    }
    mapped_pages->push_back(
        MappedPage{va + i * page_size_, page.handle, page.meta, page.epoch});
  }
}

void VMMBackingMap::AppendMappedPagesFullyInRangeLocked(
    VMMDevicePtr va,
    size_t size,
    const char* context,
    size_t max_pages,
    bool require_events_ready,
    bool annotate_remap_source_state,
    std::vector<MappedPage>* mapped_pages) const {
  if (!configured_) {
    VLOG(3) << "VMM V2 BackingMap " << context
            << " before Configure, va=" << reinterpret_cast<void*>(va)
            << " size=" << size;
    return;
  }
  if (size == 0 || page_size_ == 0 || AddOverflow(base_, size_) || va < base_ ||
      va + size < va || va + size > base_ + size_) {
    VLOG(3) << "VMM V2 BackingMap invalid range in " << context
            << ": va=" << reinterpret_cast<void*>(va) << " size=" << size
            << " base=" << reinterpret_cast<void*>(base_)
            << " backing_size=" << size_ << " page_size=" << page_size_;
    return;
  }

  const VMMDevicePtr range_end = va + size;
  const size_t start_offset = va - base_;
  const size_t end_offset = range_end - base_;
  const size_t first_page = (start_offset + page_size_ - 1) / page_size_;
  const size_t end_page = end_offset / page_size_;
  if (first_page >= end_page) {
    return;
  }

  for (size_t page_idx = first_page; page_idx < end_page; ++page_idx) {
    if (max_pages != 0 && mapped_pages->size() >= max_pages) {
      break;
    }
    const auto& page = pages_[page_idx];
    if (!page.mapped || page.ipc_exported) {
      continue;
    }
    auto remap_source_state = RemapSourceState::kReady;
    if (annotate_remap_source_state) {
      remap_source_state =
          GetRemapSourceStateLocked(&pages_[page_idx], context);
    } else if (require_events_ready &&
               !PageCanUseBackingLocked(&pages_[page_idx], context)) {
      continue;
    }
    mapped_pages->push_back(MappedPage{base_ + page_idx * page_size_,
                                       page.handle,
                                       page.meta,
                                       page.epoch,
                                       remap_source_state});
  }
}

bool VMMBackingMap::CollectIPCPartDescriptorsLocked(
    VMMDevicePtr va,
    size_t size,
    std::vector<IPCBlockPartDescriptor>* descriptors) const {
  size_t start = 0;
  size_t count = 0;
  if (!ComputeOverlappedPages(base_,
                              size_,
                              page_size_,
                              va,
                              size,
                              "CollectIPCPartDescriptors",
                              &start,
                              &count)) {
    return false;
  }
  if (descriptors != nullptr) {
    descriptors->clear();
    descriptors->reserve(count);
  }
  for (size_t i = 0; i < count; ++i) {
    const auto& page = pages_[start + i];
    if (!page.mapped || page.meta == nullptr ||
        page.meta->IsOwnedByRemapDestination()) {
      return false;
    }
    if (descriptors != nullptr) {
      const VMMDevicePtr page_va = base_ + (start + i) * page_size_;
      const VMMDevicePtr slice_begin = std::max(va, page_va);
      const VMMDevicePtr slice_end = std::min(va + size, page_va + page_size_);
      descriptors->push_back(IPCBlockPartDescriptor{
          page.meta->base(),
          page.meta->size(),
          page.meta->handle(),
          page.meta->device(),
          static_cast<size_t>(slice_begin - page_va),
          static_cast<size_t>(slice_end - slice_begin),
      });
    }
  }
  return true;
}

void VMMBackingMap::AppendUnmappedPagesFullyInRangeLocked(
    VMMDevicePtr va,
    size_t size,
    const char* context,
    size_t max_pages,
    std::vector<UnmappedPage>* unmapped_pages) const {
  if (!configured_) {
    VLOG(3) << "VMM V2 BackingMap " << context
            << " before Configure, va=" << reinterpret_cast<void*>(va)
            << " size=" << size;
    return;
  }
  if (size == 0 || page_size_ == 0 || AddOverflow(base_, size_) || va < base_ ||
      va + size < va || va + size > base_ + size_) {
    VLOG(3) << "VMM V2 BackingMap invalid range in " << context
            << ": va=" << reinterpret_cast<void*>(va) << " size=" << size
            << " base=" << reinterpret_cast<void*>(base_)
            << " backing_size=" << size_ << " page_size=" << page_size_;
    return;
  }

  const VMMDevicePtr range_end = va + size;
  const size_t start_offset = va - base_;
  const size_t end_offset = range_end - base_;
  const size_t first_page = (start_offset + page_size_ - 1) / page_size_;
  const size_t end_page = end_offset / page_size_;
  if (first_page >= end_page) {
    return;
  }

  for (size_t page_idx = first_page; page_idx < end_page; ++page_idx) {
    if (max_pages != 0 && unmapped_pages->size() >= max_pages) {
      break;
    }
    const auto& page = pages_[page_idx];
    if (page.mapped) {
      continue;
    }
    unmapped_pages->push_back(
        UnmappedPage{base_ + page_idx * page_size_, page.epoch});
  }
}

bool VMMBackingMap::PageEventsReadyLocked(Page* page,
                                          const char* context) const {
  for (auto it = page->pending_events.begin();
       it != page->pending_events.end();) {
    if (it->event == nullptr || it->event->event == nullptr) {
      gpuEvent_t event;
      PADDLE_ENFORCE_GPU_SUCCESS(
          cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
      PADDLE_ENFORCE_GPU_SUCCESS(cudaEventRecord(event, it->stream));
      it->event = std::make_shared<CUDAEventGuard>(event);
      VLOG(6) << "VMM V2 BackingMap lazily recorded pending event in "
              << context;
      return false;
    }
    gpuError_t err = cudaEventQuery(it->event->event);
    if (err != cudaSuccess && err != cudaErrorNotReady) {
      PADDLE_ENFORCE_GPU_SUCCESS(err);
    }
    if (err == cudaSuccess) {
      it = page->pending_events.erase(it);
      continue;
    }
    VLOG(6) << "VMM V2 BackingMap page blocked by pending event in " << context;
    return false;
  }
  return true;
}

bool VMMBackingMap::PageCanUseBackingLocked(Page* page,
                                            const char* context) const {
  return PageEventsReadyLocked(page, context);
}

VMMBackingMap::RemapSourceState VMMBackingMap::GetRemapSourceStateLocked(
    Page* page, const char* context) const {
  if (page == nullptr || page->meta == nullptr) {
    return RemapSourceState::kPartialOrInvalid;
  }
  if (page->remap_destination_owned ||
      page->meta->IsOwnedByRemapDestination()) {
    return RemapSourceState::kRemapDestinationOwned;
  }
  return PageCanUseBackingLocked(page, context)
             ? RemapSourceState::kReady
             : RemapSourceState::kPendingEvent;
}

size_t VMMBackingMap::total_mapped_bytes() const {
  std::lock_guard<SpinLock> guard(spinlock_);
  return mapped_page_count_ * page_size_;
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle

#endif
