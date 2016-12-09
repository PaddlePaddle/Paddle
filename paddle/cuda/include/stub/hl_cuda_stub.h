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

#ifndef HL_CUDA_STUB_H_
#define HL_CUDA_STUB_H_

#include "hl_cuda.h"

inline void hl_start() {}

inline void hl_specify_devices_start(int *device, int number) {}

inline void hl_init(int device) {}

inline int hl_get_cuda_lib_version(int device) { return 0; }

inline void hl_fini() {}

inline void hl_set_sync_flag(bool flag) {}

inline bool hl_get_sync_flag() { return false; }

inline int hl_get_device_count() { return 0; }

inline void hl_set_device(int device) {}

inline int hl_get_device() { return 0; }

inline void *hl_malloc_device(size_t size) { return NULL; }

inline void hl_free_mem_device(void *dest_d) {}

inline void *hl_malloc_host(size_t size) { return NULL; }

inline void hl_free_mem_host(void *dest_h) {}

inline void hl_memcpy(void *dst, void *src, size_t size) {}

inline void hl_memset_device(void *dest_d, int value, size_t size) {}

inline void hl_memcpy_host2device(void *dest_d, void *src_h, size_t size) {}

inline void hl_memcpy_device2host(void *dest_h, void *src_d, size_t size) {}

inline void hl_memcpy_device2device(void *dest_d, void *src_d, size_t size) {}

inline void hl_rand(real *dest_d, size_t num) {}

inline void hl_srand(unsigned int seed) {}

inline void hl_memcpy_async(void *dst,
                            void *src,
                            size_t size,
                            hl_stream_t stream) {}

inline void hl_stream_synchronize(hl_stream_t stream) {}

inline void hl_create_event(hl_event_t *event) {}

inline void hl_destroy_event(hl_event_t event) {}

inline float hl_event_elapsed_time(hl_event_t start, hl_event_t end) {
  return 0;
}

inline void hl_stream_record_event(hl_stream_t stream, hl_event_t event) {}

inline void hl_stream_wait_event(hl_stream_t stream, hl_event_t event) {}

inline void hl_event_synchronize(hl_event_t event) {}

inline int hl_get_device_last_error() { return 0; }

inline const char *hl_get_device_error_string() { return NULL; }

inline const char *hl_get_device_error_string(size_t err) { return NULL; }

inline bool hl_cuda_event_is_ready(hl_event_t event) { return true; }

inline void hl_device_synchronize() {}

inline void hl_profiler_start() {}

inline void hl_profiler_end() {}

#endif  // HL_CUDA_STUB_H_
