// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include <stdlib.h>

#include "paddle/cinn/runtime/cinn_runtime.h"

int cinn_x86_malloc(void* context, cinn_buffer_t* buf) {
  // ASSERT_NOT_NULL(context)
  ASSERT_NOT_NULL(buf)
  uint64_t memory_size;
  bool need_malloc = false;
  if (buf->memory_size > 0 && !buf->memory) {
    memory_size = buf->memory_size;
    need_malloc = true;
  } else {
    memory_size = buf->num_elements() * buf->type.bytes();
  }
  CINN_CHECK(memory_size > 0);
  if (buf->memory_size < memory_size || need_malloc) {
    if (buf->memory) {
      free(buf->memory);
    }
    if (buf->align == 0) {
      buf->memory = (unsigned char*)malloc(memory_size);
    } else {
      buf->memory = (unsigned char*)aligned_alloc(buf->align, memory_size);
    }
    buf->memory_size = memory_size;
  }
  ASSERT_NOT_NULL(buf->memory);
  return 0;
}

int cinn_x86_free(void* context, cinn_buffer_t* buf) {
  // ASSERT_NOT_NULL(context);
  ASSERT_NOT_NULL(buf);
  if (buf->memory) {
    free(buf->memory);
    buf->memory = NULL;
  }
  return 0;
}

// All the following operations are not support by X86 device, just leave them
// empty.
// @{
int cinn_x86_sync(void* context, cinn_buffer_t* buf) { return 0; }
int cinn_x86_release(void* context) { return 0; }
int cinn_x86_copy_to_host(void* context, cinn_buffer_t* buf) { return 0; }
int cinn_x86_copy_to_device(void* context, cinn_buffer_t* buf) { return 0; }
int cinn_x86_buffer_copy(void* context,
                         cinn_buffer_t* src,
                         cinn_buffer_t* dst) {
  return 0;
}
// @}

cinn_device_interface_impl_t cinn_x86_device_impl{&cinn_x86_malloc,
                                                  &cinn_x86_free,
                                                  &cinn_x86_sync,
                                                  &cinn_x86_release,
                                                  &cinn_x86_copy_to_host,
                                                  &cinn_x86_copy_to_device,
                                                  &cinn_x86_buffer_copy};

cinn_device_interface_t cinn_x86_device_interface_interface{
    &cinn_buffer_malloc,
    &cinn_buffer_free,
    &cinn_device_sync,
    &cinn_device_release,
    &cinn_buffer_copy_to_host,
    &cinn_buffer_copy_to_device,
    &cinn_buffer_copy,
    &cinn_x86_device_impl};

struct cinn_device_interface_t* cinn_x86_device_interface() {
  return &cinn_x86_device_interface_interface;
}
