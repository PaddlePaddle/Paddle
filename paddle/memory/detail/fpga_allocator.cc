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

#include "paddle/memory/detail/fpga_allocator.h"
#include "paddle/platform/assert.h"
#include "paddle/platform/enforce.h"

#include <stdlib.h>    // for malloc and free
#include <sys/mman.h>  // for mlock and munlock

#include "gflags/gflags.h"

#ifdef PADDLE_WITH_FPGA
#include "polaris.h"

// If use_pinned_memory is true, CPUAllocator calls mlock, which
// returns pinned and locked memory as staging areas for data exchange
// between host and device.  Allocates too much would reduce the amount
// of memory available to the system for paging.  So, by default, we
// should set false to use_pinned_memory.

namespace paddle {
namespace memory {
namespace detail {

FPGAAllocator::FPGAAllocator(int fpga_id) {
  _fpga_id = fpga_id;
  _ctxt = polaris_create_context(fpga_id);
  PADDLE_ENFORCE_NOT_NULL(_ctxt);
}

FPGAAllocator::~FPGAAllocator() {
  if (_ctxt != NULL) {
    polaris_destroy_context(_ctxt);
  }
}

void* FPGAAllocator::Alloc(size_t size) {
  void* p = NULL;
  polaris_malloc(_ctxt, size, &p);
  return p;
}

void FPGAAllocator::Free(void* p) {
  polaris_free(_ctxt, p);
}

size_t FPGAAllocator::Used() {
  // currently not support
  return 0;
}
}  // namespace detail
}  // namespace memory
}  // namespace paddle
#endif
