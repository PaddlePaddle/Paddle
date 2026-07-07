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
    VLOG(0) << "VMM V2 BackingMap invalid overlap range in " << context
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
      VLOG(0) << "VMM V2 BackingMap reconfigure mismatch: old_base="
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
    VLOG(0) << "VMM V2 BackingMap " << context
            << " before Configure, va=" << reinterpret_cast<void*>(va)
            << " size=" << size;
    return false;
  }
  if (size == 0 || page_size_ == 0 || size % page_size_ != 0 ||
      AddOverflow(base_, size_) || va < base_ || va + size < va ||
      va + size > base_ + size_ || (va - base_) % page_size_ != 0) {
    VLOG(0) << "VMM V2 BackingMap invalid range in " << context
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
    const std::shared_ptr<VMMHandleMeta>& meta) {
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
  page->epoch++;
}

void VMMBackingMap::ResetPageToUnmappedLocked(Page* page) {
  if (page->mapped && mapped_page_count_ > 0) {
    mapped_page_count_--;
  }
  page->handle = 0;
  page->meta.reset();
  page->mapped = false;
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
    MarkPageMappedLocked(
        &page, va + i * page_size_, handle, std::shared_ptr<VMMHandleMeta>());
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
    MarkPageMappedLocked(&page, va + i * page_size_, handle, meta);
  }
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
    ResetPageToUnmappedLocked(&page);
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
      VLOG(0) << "VMM V2 BackingMap release handle mismatch at "
              << reinterpret_cast<void*>(va + i * page_size_)
              << " tracked=" << reinterpret_cast<void*>(page.handle)
              << " released=" << reinterpret_cast<void*>(handle);
    }
    ResetPageToUnmappedLocked(&page);
  }
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
      if (!page.mapped) {
        VLOG(0) << "VMM V2 BackingMap mapped-state mismatch in " << context
                << " va="
                << reinterpret_cast<void*>(meta->base() + i * page_size_)
                << " tracked_mapped=" << page.mapped;
        ok = false;
      }
      if (page.mapped && page.handle != meta->handle()) {
        VLOG(0) << "VMM V2 BackingMap handle mismatch in " << context << " va="
                << reinterpret_cast<void*>(meta->base() + i * page_size_)
                << " tracked=" << reinterpret_cast<void*>(page.handle)
                << " meta=" << reinterpret_cast<void*>(meta->handle());
        ok = false;
      }
    }
  }
  return ok;
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
    if (!PageCanUseBackingLocked(&pages_[start + i], "IsRangeReleasable")) {
      return false;
    }
  }
  return true;
}

bool VMMBackingMap::PageCanUseBackingLocked(Page* page,
                                            const char* context) const {
  if (page == nullptr || !page->mapped || page->meta == nullptr) {
    VLOG(6) << "VMM V2 BackingMap page cannot use backing in " << context;
    return false;
  }
  return true;
}

size_t VMMBackingMap::total_mapped_bytes() const {
  std::lock_guard<SpinLock> guard(spinlock_);
  return mapped_page_count_ * page_size_;
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle

#endif
