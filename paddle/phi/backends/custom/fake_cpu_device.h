// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/phi/backends/device_ext.h"

constexpr size_t global_total_memory = 1024 * 1024UL;
static size_t global_free_memory = global_total_memory;

C_Status Init() { return C_SUCCESS; }

C_Status InitDevice(const C_Device device) { return C_SUCCESS; }

C_Status SetDevice(const C_Device device) { return C_SUCCESS; }

C_Status GetDevice(const C_Device device) {
  device->id = 0;
  return C_SUCCESS;
}

C_Status DestroyDevice(const C_Device device) { return C_SUCCESS; }

C_Status Finalize() { return C_SUCCESS; }

C_Status GetDevicesCount(size_t *count) {
  *count = 1;
  return C_SUCCESS;
}

C_Status GetDevicesList(size_t *device) {
  *device = 0;
  return C_SUCCESS;
}

C_Status MemCpy(const C_Device device,
                void *dst,
                const void *src,
                size_t size) {
  memcpy(dst, src, size);
  return C_SUCCESS;
}

C_Status AsyncMemCpy(const C_Device device,
                     C_Stream stream,
                     void *dst,
                     const void *src,
                     size_t size) {
  memcpy(dst, src, size);
  return C_SUCCESS;
}

C_Status Allocate(const C_Device device, void **ptr, size_t size) {
  if (global_free_memory >= size) {
    *ptr = malloc(size);
    global_free_memory -= size;
    return C_SUCCESS;
  } else {
    *ptr = nullptr;
    return C_FAILED;
  }
}

C_Status Deallocate(const C_Device device, void *ptr, size_t size) {
  free(ptr);
  global_free_memory += size;
  return C_SUCCESS;
}

C_Status CreateStream(const C_Device device, C_Stream *stream) {
  return C_SUCCESS;
}

C_Status DestroyStream(const C_Device device, C_Stream stream) {
  return C_SUCCESS;
}

C_Status CreateEvent(const C_Device device, C_Event *event) {
  return C_SUCCESS;
}

C_Status RecordEvent(const C_Device device, C_Stream stream, C_Event event) {
  return C_SUCCESS;
}

C_Status DestroyEvent(const C_Device device, C_Event event) {
  return C_SUCCESS;
}

C_Status SyncDevice(const C_Device device) { return C_SUCCESS; }

C_Status SyncStream(const C_Device device, C_Stream stream) {
  return C_SUCCESS;
}

C_Status SyncEvent(const C_Device device, C_Event event) { return C_SUCCESS; }

C_Status StreamWaitEvent(const C_Device device,
                         C_Stream stream,
                         C_Event event) {
  return C_SUCCESS;
}

C_Status VisibleDevices(size_t *devices) { return C_SUCCESS; }

C_Status DeviceMemStats(const C_Device device,
                        size_t *total_memory,
                        size_t *free_memory) {
  *total_memory = global_total_memory;
  *free_memory = global_free_memory;
  return C_SUCCESS;
}

C_Status DeviceMinChunkSize(const C_Device device, size_t *size) {
  *size = 4 * 1024;
  return C_SUCCESS;
}

C_Status DeviceMaxChunkSize(const C_Device device, size_t *size) {
  *size = 64 * 1024;
  return C_SUCCESS;
}

C_Status DeviceMaxAllocSize(const C_Device device, size_t *size) {
  *size = global_total_memory * 0.95;
  return C_SUCCESS;
}

#define DEVICE_TYPE "FakeCPU"
#define SUB_DEVICE_TYPE "V100"

void InitFakeCPUDevice(CustomRuntimeParams *params) {
  params->device_type = const_cast<char *>(DEVICE_TYPE);
  params->sub_device_type = const_cast<char *>(SUB_DEVICE_TYPE);
  params->version.major = PADDLE_CUSTOM_RUNTIME_MAJOR_VERSION;
  params->version.minor = PADDLE_CUSTOM_RUNTIME_MINOR_VERSION;
  params->version.patch = PADDLE_CUSTOM_RUNTIME_PATCH_VERSION;

  memset(reinterpret_cast<void *>(params->interface),
         0,
         sizeof(C_DeviceInterface));

  params->interface->initialize = Init;
  params->interface->finalize = Finalize;

  params->interface->init_device = InitDevice;
  params->interface->set_device = SetDevice;
  params->interface->get_device = GetDevice;
  params->interface->deinit_device = DestroyDevice;

  params->interface->create_stream = CreateStream;
  params->interface->destroy_stream = DestroyStream;

  params->interface->create_event = CreateEvent;
  params->interface->destroy_event = DestroyEvent;
  params->interface->record_event = RecordEvent;

  params->interface->synchronize_device = SyncDevice;
  params->interface->synchronize_stream = SyncStream;
  params->interface->synchronize_event = SyncEvent;
  params->interface->stream_wait_event = StreamWaitEvent;

  params->interface->memory_copy_h2d = MemCpy;
  params->interface->memory_copy_d2d = MemCpy;
  params->interface->memory_copy_d2h = MemCpy;
  params->interface->async_memory_copy_h2d = AsyncMemCpy;
  params->interface->async_memory_copy_d2d = AsyncMemCpy;
  params->interface->async_memory_copy_d2h = AsyncMemCpy;
  params->interface->device_memory_allocate = Allocate;
  params->interface->host_memory_allocate = Allocate;
  params->interface->unified_memory_allocate = Allocate;
  params->interface->device_memory_deallocate = Deallocate;
  params->interface->host_memory_deallocate = Deallocate;
  params->interface->unified_memory_deallocate = Deallocate;

  params->interface->get_device_count = GetDevicesCount;
  params->interface->get_device_list = GetDevicesList;
  params->interface->device_memory_stats = DeviceMemStats;

  params->interface->device_max_chunk_size = DeviceMaxChunkSize;
  params->interface->device_min_chunk_size = DeviceMinChunkSize;
  params->interface->device_max_alloc_size = DeviceMaxAllocSize;
}
