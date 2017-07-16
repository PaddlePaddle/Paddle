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

#include "paddle/memory/detail/memory_block.h"

#include <stddef.h>

namespace paddle {
namespace memory {
namespace detail {

class Metadata {
 public:
  Metadata(MemoryBlock::Type t, size_t i, size_t s, size_t ts, MemoryBlock* l,
           MemoryBlock* r);
  Metadata();

 public:
  /*! \brief Update the guards when metadata is changed */
  void update_guards();

  /*! \brief Check consistency to previous modification */
  bool check_guards() const;

 public:
  // TODO(gangliao): compress this
  // clang-format off
  size_t            guard_begin = 0;
  MemoryBlock::Type type        = MemoryBlock::INVALID_CHUNK;
  size_t            index       = 0;
  size_t            size        = 0;
  size_t            total_size  = 0;
  MemoryBlock*      left_buddy  = nullptr;
  MemoryBlock*      right_buddy = nullptr;
  size_t            guard_end   = 0;
  // clang-format on
};

}  // namespace detail
}  // namespace memory
}  // namespace paddle
