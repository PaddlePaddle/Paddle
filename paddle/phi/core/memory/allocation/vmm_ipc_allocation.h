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

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#if defined(PADDLE_WITH_CUDA)
#include "paddle/phi/backends/dynload/cuda_driver.h"
using VmmDevicePtr = CUdeviceptr;
using VmmAllocHandle = CUmemGenericAllocationHandle;
#else
using VmmDevicePtr = uintptr_t;
using VmmAllocHandle = uint64_t;
#endif

#include "paddle/phi/common/place.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/memory/allocation/allocator.h"

namespace paddle {
namespace memory {
namespace allocation {

struct ImportedVmmMulti {
  VmmDevicePtr base{0};
  size_t reserved_size{0};
  std::vector<VmmAllocHandle> hs;
#if defined(PADDLE_WITH_CUDA)
  ~ImportedVmmMulti() {
    if (base && reserved_size) {
      phi::dynload::cuMemUnmap(base, reserved_size);
    }
    for (auto h : hs) {
      if (h) phi::dynload::cuMemRelease(h);
    }
    if (base && reserved_size) {
      phi::dynload::cuMemAddressFree(base, reserved_size);
    }
  }
#else
  ~ImportedVmmMulti() = default;
#endif
};

class VmmImportedAllocation : public phi::Allocation {
 public:
  VmmImportedAllocation(void* ptr,
                        size_t bytes,
                        Place place,
                        std::shared_ptr<ImportedVmmMulti> keep)
      : Allocation(ptr, bytes, place), keep_(std::move(keep)) {}

 private:
  std::shared_ptr<ImportedVmmMulti> keep_;
};

struct VmmChunkMeta {
  VmmDevicePtr base;
  size_t size;
  VmmAllocHandle handle;
  int device;
};

struct BlockPart {
  std::shared_ptr<VmmChunkMeta> chunk;
  size_t chunk_rel_off;
  size_t len;
};

inline bool TryConcatAdjacentBlockPart(BlockPart* a, const BlockPart& b) {
  if (!a) return false;
  if (a->chunk.get() != b.chunk.get()) return false;
  if (a->chunk_rel_off + a->len != b.chunk_rel_off) return false;
  a->len += b.len;
  return true;
}

inline std::vector<BlockPart> SliceBlockPartsForRange(
    const std::vector<BlockPart>& parts,
    size_t range_offset,
    size_t range_len) {
  // parts describes one logical block as an ordered list of VMM chunk slices.
  // The target range is expressed in that logical block address space.
  std::vector<BlockPart> sliced_parts;
  if (range_len == 0 || parts.empty()) {
    return sliced_parts;
  }

  PADDLE_ENFORCE_LE(
      range_offset,
      std::numeric_limits<size_t>::max() - range_len,
      common::errors::InvalidArgument(
          "Invalid VMM block-part slice range: offset %zu plus length %zu "
          "overflows.",
          range_offset,
          range_len));

  if (parts.size() == 1) {
    const auto& part = parts.front();
    PADDLE_ENFORCE_LE(
        range_offset,
        part.len,
        common::errors::InvalidArgument(
            "Invalid VMM block-part slice offset %zu for part length %zu.",
            range_offset,
            part.len));
    PADDLE_ENFORCE_LE(
        range_len,
        part.len - range_offset,
        common::errors::InvalidArgument(
            "Invalid VMM block-part slice length %zu at offset %zu for part "
            "length %zu.",
            range_len,
            range_offset,
            part.len));
    return {
        BlockPart{part.chunk, part.chunk_rel_off + range_offset, range_len}};
  }

  sliced_parts.reserve(parts.size());
  const size_t range_end = range_offset + range_len;
  size_t cursor = 0;
  size_t sliced_len = 0;

  for (const auto& part : parts) {
    const size_t part_block_begin = cursor;
    const size_t part_block_end = cursor + part.len;
    cursor = part_block_end;

    if (part_block_end <= range_offset) {
      continue;
    }
    if (part_block_begin >= range_end) {
      break;
    }

    const size_t slice_begin = std::max(part_block_begin, range_offset);
    const size_t slice_end = std::min(part_block_end, range_end);
    BlockPart slice{part.chunk,
                    part.chunk_rel_off + (slice_begin - part_block_begin),
                    slice_end - slice_begin};

    if (!sliced_parts.empty() &&
        TryConcatAdjacentBlockPart(&sliced_parts.back(), slice)) {
      sliced_len += slice.len;
      continue;
    }
    sliced_parts.push_back(std::move(slice));
    sliced_len += sliced_parts.back().len;
  }

  PADDLE_ENFORCE_EQ(
      sliced_len,
      range_len,
      common::errors::InvalidArgument(
          "Invalid VMM block-part slice range: requested %zu bytes at offset "
          "%zu, but only sliced %zu bytes from %zu parts.",
          range_len,
          range_offset,
          sliced_len,
          parts.size()));
  return sliced_parts;
}

inline void AppendBlockPartsTail(std::vector<BlockPart>* dst,
                                 std::vector<BlockPart>* src) {
  if (src->empty()) return;
  dst->reserve(dst->size() + src->size());
  auto begin = src->begin();
  if (!dst->empty() && TryConcatAdjacentBlockPart(&dst->back(), src->front())) {
    ++begin;
  }
  dst->insert(dst->end(),
              std::make_move_iterator(begin),
              std::make_move_iterator(src->end()));
}

#pragma pack(push, 1)
struct VmmIpcHeader {
  uint8_t version;
  uint16_t flags;
  uint32_t pid;
  uint32_t num_entries;
  uint64_t alloc_size;
  uint64_t offset;
  uint64_t reserved_size;
};

struct VmmIpcEntry {
  uint8_t handle_type;
  uint8_t reserved[7];
  uint64_t rel_offset;
  uint64_t chunk_size;
  uint64_t chunk_rel_off;
};
#pragma pack(pop)

static_assert(sizeof(VmmIpcHeader) == 35, "VmmIpcHeader size changed");
static_assert(sizeof(VmmIpcEntry) == 32, "VmmIpcEntry size changed");

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
