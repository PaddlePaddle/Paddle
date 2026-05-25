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

#include <cstdio>
#include <cstring>

#include "paddle/phi/backends/device_ext.h"

// =============================================================================
// FakeCinnStub Device: A CPU-based mock device that implements the full
// C_CinnInterface with stub functions. This enables CINN CustomDevice code
// coverage testing WITHOUT requiring any GPU hardware.
//
// All CINN interface functions (compile, module_load, launch_kernel, etc.)
// return success with mock data, so the Paddle-side framework code
// (CinnCustomDevicePlugin, DefaultCompilerToolchain, DefaultRuntimeStrategy,
// CustomBackendAPI, cinn_call_custom_device_kernel) is fully exercised.
// =============================================================================

// ============================================================
// Device Interface Implementations (CPU-backed, same as fake_cpu_device.h)
// ============================================================

constexpr size_t kFakeCinnTotalMemory = 4 * 1024 * 1024UL;  // 4MB
static size_t g_fake_cinn_free_memory = kFakeCinnTotalMemory;

C_Status FakeCinnInit() { return C_SUCCESS; }
C_Status FakeCinnFinalize() { return C_SUCCESS; }
C_Status FakeCinnInitDevice(const C_Device device) { return C_SUCCESS; }
C_Status FakeCinnDestroyDevice(const C_Device device) { return C_SUCCESS; }

C_Status FakeCinnSetDevice(const C_Device device) { return C_SUCCESS; }
C_Status FakeCinnGetDevice(const C_Device device) {
  device->id = 0;
  return C_SUCCESS;
}

C_Status FakeCinnGetDevicesCount(size_t *count) {
  *count = 1;
  return C_SUCCESS;
}

C_Status FakeCinnGetDevicesList(size_t *device) {
  *device = 0;
  return C_SUCCESS;
}

C_Status FakeCinnAllocate(const C_Device device, void **ptr, size_t size) {
  if (g_fake_cinn_free_memory >= size) {
    *ptr = malloc(size);
    if (*ptr) {
      g_fake_cinn_free_memory -= size;
      return C_SUCCESS;
    }
  }
  *ptr = nullptr;
  return C_FAILED;
}

C_Status FakeCinnDeallocate(const C_Device device, void *ptr, size_t size) {
  free(ptr);
  g_fake_cinn_free_memory += size;
  return C_SUCCESS;
}

C_Status FakeCinnMemCpy(const C_Device device,
                        void *dst,
                        const void *src,
                        size_t size) {
  memcpy(dst, src, size);
  return C_SUCCESS;
}

C_Status FakeCinnAsyncMemCpy(const C_Device device,
                             C_Stream stream,
                             void *dst,
                             const void *src,
                             size_t size) {
  memcpy(dst, src, size);
  return C_SUCCESS;
}

C_Status FakeCinnMemSet(const C_Device device,
                        void *ptr,
                        unsigned char value,
                        size_t size) {
  memset(ptr, value, size);
  return C_SUCCESS;
}

C_Status FakeCinnCreateStream(const C_Device device, C_Stream *stream) {
  // Return a non-null fake stream handle
  *stream = reinterpret_cast<C_Stream>(0xDEAD0001);
  return C_SUCCESS;
}

C_Status FakeCinnDestroyStream(const C_Device device, C_Stream stream) {
  return C_SUCCESS;
}

C_Status FakeCinnSyncStream(const C_Device device, C_Stream stream) {
  return C_SUCCESS;
}

C_Status FakeCinnCreateEvent(const C_Device device, C_Event *event) {
  *event = reinterpret_cast<C_Event>(0xDEAD0002);
  return C_SUCCESS;
}

C_Status FakeCinnDestroyEvent(const C_Device device, C_Event event) {
  return C_SUCCESS;
}

C_Status FakeCinnRecordEvent(const C_Device device,
                             C_Stream stream,
                             C_Event event) {
  return C_SUCCESS;
}

C_Status FakeCinnSyncEvent(const C_Device device, C_Event event) {
  return C_SUCCESS;
}

C_Status FakeCinnStreamWaitEvent(const C_Device device,
                                 C_Stream stream,
                                 C_Event event) {
  return C_SUCCESS;
}

C_Status FakeCinnSyncDevice(const C_Device device) { return C_SUCCESS; }

C_Status FakeCinnDeviceMemStats(const C_Device device,
                                size_t *total_memory,
                                size_t *free_memory) {
  *total_memory = kFakeCinnTotalMemory;
  *free_memory = g_fake_cinn_free_memory;
  return C_SUCCESS;
}

C_Status FakeCinnDeviceMinChunkSize(const C_Device device, size_t *size) {
  *size = 512;
  return C_SUCCESS;
}

C_Status FakeCinnDeviceMaxChunkSize(const C_Device device, size_t *size) {
  *size = 64 * 1024;
  return C_SUCCESS;
}

C_Status FakeCinnDeviceMaxAllocSize(const C_Device device, size_t *size) {
  *size = static_cast<size_t>(kFakeCinnTotalMemory * 0.95);
  return C_SUCCESS;
}

// Device property stubs for CustomBackendAPI coverage
C_Status FakeCinnGetComputeCapability(const C_Device device,
                                      size_t *compute_capability) {
  *compute_capability = 80;  // Fake SM80
  return C_SUCCESS;
}

C_Status FakeCinnGetMultiProcess(const C_Device device, size_t *multi_process) {
  *multi_process = 108;  // Fake 108 SMs
  return C_SUCCESS;
}

C_Status FakeCinnGetMaxThreadsPerMP(const C_Device device,
                                    size_t *threads_per_mp) {
  *threads_per_mp = 2048;
  return C_SUCCESS;
}

C_Status FakeCinnGetWarpSize(const C_Device device, size_t *warp_size) {
  *warp_size = 32;
  return C_SUCCESS;
}

C_Status FakeCinnGetMaxRegistersPerMP(const C_Device device,
                                      size_t *max_registers) {
  *max_registers = 65536;
  return C_SUCCESS;
}

C_Status FakeCinnGetMaxThreadsPerBlock(const C_Device device,
                                       size_t *threads_per_block) {
  *threads_per_block = 1024;
  return C_SUCCESS;
}

C_Status FakeCinnGetMaxSharedMemPerBlock(const C_Device device,
                                         size_t *shared_mem_per_block) {
  *shared_mem_per_block = 49152;  // 48KB
  return C_SUCCESS;
}

C_Status FakeCinnGetMaxBlocksPerMP(const C_Device device,
                                   size_t *blocks_per_mp) {
  *blocks_per_mp = 32;
  return C_SUCCESS;
}

C_Status FakeCinnGetMaxGridDimSize(const C_Device device,
                                   std::array<unsigned int, 3> *grid_dim_size) {
  *grid_dim_size = {2147483647u, 65535u, 65535u};
  return C_SUCCESS;
}

C_Status FakeCinnGetMaxBlockDimSize(
    const C_Device device, std::array<unsigned int, 3> *block_dim_size) {
  *block_dim_size = {1024u, 1024u, 64u};
  return C_SUCCESS;
}

// XCCL stubs (minimal)
C_Status FakeCinnXcclGetUniqueIdSize(size_t *size) {
  *size = sizeof(size_t);
  return C_SUCCESS;
}
C_Status FakeCinnXcclGetUniqueId(C_CCLRootId *unique_id) { return C_SUCCESS; }
C_Status FakeCinnXcclCommInitRank(size_t ranks,
                                  C_CCLRootId *unique_id,
                                  size_t rank,
                                  C_CCLComm *comm) {
  return C_SUCCESS;
}
C_Status FakeCinnXcclDestroyComm(C_CCLComm comm) { return C_SUCCESS; }
C_Status FakeCinnXcclAllReduce(void *send_buf,
                               void *recv_buf,
                               size_t count,
                               C_DataType data_type,
                               C_CCLReduceOp op,
                               C_CCLComm comm,
                               C_Stream stream) {
  return C_SUCCESS;
}
C_Status FakeCinnXcclBroadcast(void *buf,
                               size_t count,
                               C_DataType data_type,
                               size_t root,
                               C_CCLComm comm,
                               C_Stream stream) {
  return C_SUCCESS;
}

// ============================================================
// C_CinnInterface Stub Implementations
// ============================================================

// Fake compile: writes a fake "compiled binary path" into out_path.
// No real compilation happens.
C_Status FakeCinnCompile(void *dev_ptr,
                         const char *code,
                         char *out_path,
                         size_t len) {
  const char *fake_path = "/tmp/fake_cinn_stub_kernel.bin";
  size_t path_len = strlen(fake_path);
  if (path_len < len) {
    strncpy(out_path, fake_path, len);
    // Create the fake file so module_load can find it
    FILE *f = fopen(fake_path, "w");
    if (f) {
      fprintf(f, "FAKE_MODULE_BINARY");
      fclose(f);
    }
    return C_SUCCESS;
  }
  return C_FAILED;
}

// Fake get_runtime_source: returns a minimal runtime header string.
static const char *kFakeRuntimeSource =
    "// FakeCinnStub runtime source\n"
    "#define FAKE_WARP_SIZE 32\n";

const char *FakeCinnGetRuntimeSource(void *dev_ptr) {
  return kFakeRuntimeSource;
}

// Fake module_load: returns a fake module handle.
// Uses a static counter to generate unique "handles".
static int g_fake_module_counter = 0;

C_Status FakeCinnModuleLoad(void *dev_ptr, const char *path, void **mod_out) {
  // Return a unique non-null fake handle
  g_fake_module_counter++;
  *mod_out = reinterpret_cast<void *>(
      static_cast<uintptr_t>(0xA0000000 + g_fake_module_counter));
  return C_SUCCESS;
}

// Fake module_unload: no-op (nothing to actually unload).
C_Status FakeCinnModuleUnload(void *dev_ptr, void *module_handle) {
  return C_SUCCESS;
}

// Fake get_kernel_address: returns a fake function pointer.
static int g_fake_kernel_counter = 0;

C_Status FakeCinnGetKernelAddress(void *dev_ptr,
                                  void *module_handle,
                                  const char *func_name,
                                  void **func_out) {
  g_fake_kernel_counter++;
  *func_out = reinterpret_cast<void *>(
      static_cast<uintptr_t>(0xB0000000 + g_fake_kernel_counter));
  return C_SUCCESS;
}

// Fake launch_kernel: no-op (no actual kernel execution).
// This still exercises all the Paddle-side dispatch code.
C_Status FakeCinnLaunchKernel(void *dev_ptr,
                              void *func_ptr,
                              void **args,
                              int num_args,
                              int gx,
                              int gy,
                              int gz,
                              int bx,
                              int by,
                              int bz,
                              int shm,
                              void *stream) {
  // No-op: the framework code path is fully exercised by reaching here.
  return C_SUCCESS;
}

// Fake apply_custom_pass: no-op.
C_Status FakeCinnApplyCustomPass(void *dev_ptr, void *ir_module) {
  return C_SUCCESS;
}

// ============================================================
// Static C_CinnInterface instance
// ============================================================
static C_CinnInterface g_fake_cinn_stub_interface;

// ============================================================
// Device Registration
// ============================================================

#define FAKE_CINN_DEVICE_TYPE "FakeCinnStub"
#define FAKE_CINN_SUB_DEVICE_TYPE "CPU"

void InitFakeCinnStubDevice(CustomRuntimeParams *params) {
  params->device_type = const_cast<char *>(FAKE_CINN_DEVICE_TYPE);
  params->sub_device_type = const_cast<char *>(FAKE_CINN_SUB_DEVICE_TYPE);
  params->version.major = PADDLE_CUSTOM_RUNTIME_MAJOR_VERSION;
  params->version.minor = PADDLE_CUSTOM_RUNTIME_MINOR_VERSION;
  params->version.patch = PADDLE_CUSTOM_RUNTIME_PATCH_VERSION;

  memset(reinterpret_cast<void *>(params->interface),
         0,
         sizeof(C_DeviceInterface));
  params->interface->size = sizeof(C_DeviceInterface);

  // --- Core device operations ---
  params->interface->initialize = FakeCinnInit;
  params->interface->finalize = FakeCinnFinalize;
  params->interface->init_device = FakeCinnInitDevice;
  params->interface->set_device = FakeCinnSetDevice;
  params->interface->get_device = FakeCinnGetDevice;
  params->interface->deinit_device = FakeCinnDestroyDevice;

  // --- Stream ---
  params->interface->create_stream = FakeCinnCreateStream;
  params->interface->destroy_stream = FakeCinnDestroyStream;
  params->interface->synchronize_stream = FakeCinnSyncStream;

  // --- Event ---
  params->interface->create_event = FakeCinnCreateEvent;
  params->interface->destroy_event = FakeCinnDestroyEvent;
  params->interface->record_event = FakeCinnRecordEvent;
  params->interface->synchronize_event = FakeCinnSyncEvent;
  params->interface->stream_wait_event = FakeCinnStreamWaitEvent;

  // --- Synchronization ---
  params->interface->synchronize_device = FakeCinnSyncDevice;

  // --- Memory ---
  params->interface->memory_copy_h2d = FakeCinnMemCpy;
  params->interface->memory_copy_d2d = FakeCinnMemCpy;
  params->interface->memory_copy_d2h = FakeCinnMemCpy;
  params->interface->async_memory_copy_h2d = FakeCinnAsyncMemCpy;
  params->interface->async_memory_copy_d2d = FakeCinnAsyncMemCpy;
  params->interface->async_memory_copy_d2h = FakeCinnAsyncMemCpy;
  params->interface->device_memory_allocate = FakeCinnAllocate;
  params->interface->host_memory_allocate = FakeCinnAllocate;
  params->interface->unified_memory_allocate = FakeCinnAllocate;
  params->interface->device_memory_deallocate = FakeCinnDeallocate;
  params->interface->host_memory_deallocate = FakeCinnDeallocate;
  params->interface->unified_memory_deallocate = FakeCinnDeallocate;
  params->interface->device_memory_set = FakeCinnMemSet;

  // --- Device info ---
  params->interface->get_device_count = FakeCinnGetDevicesCount;
  params->interface->get_device_list = FakeCinnGetDevicesList;
  params->interface->device_memory_stats = FakeCinnDeviceMemStats;
  params->interface->device_min_chunk_size = FakeCinnDeviceMinChunkSize;
  params->interface->device_max_chunk_size = FakeCinnDeviceMaxChunkSize;
  params->interface->device_max_alloc_size = FakeCinnDeviceMaxAllocSize;

  // --- Device properties (needed by CustomBackendAPI) ---
  params->interface->get_compute_capability = FakeCinnGetComputeCapability;
  params->interface->get_multi_process = FakeCinnGetMultiProcess;
  params->interface->get_max_threads_per_mp = FakeCinnGetMaxThreadsPerMP;
  params->interface->get_warp_size = FakeCinnGetWarpSize;
  params->interface->get_max_registers_per_mp = FakeCinnGetMaxRegistersPerMP;
  params->interface->get_max_threads_per_block = FakeCinnGetMaxThreadsPerBlock;
  params->interface->get_max_shared_mem_per_block =
      FakeCinnGetMaxSharedMemPerBlock;
  params->interface->get_max_blocks_per_mp = FakeCinnGetMaxBlocksPerMP;
  params->interface->get_max_grid_dim_size = FakeCinnGetMaxGridDimSize;
  params->interface->get_max_block_dim_size = FakeCinnGetMaxBlockDimSize;

  // --- XCCL (minimal stubs) ---
  params->interface->xccl_get_unique_id_size = FakeCinnXcclGetUniqueIdSize;
  params->interface->xccl_get_unique_id = FakeCinnXcclGetUniqueId;
  params->interface->xccl_comm_init_rank = FakeCinnXcclCommInitRank;
  params->interface->xccl_destroy_comm = FakeCinnXcclDestroyComm;
  params->interface->xccl_all_reduce = FakeCinnXcclAllReduce;
  params->interface->xccl_broadcast = FakeCinnXcclBroadcast;

  // --- CINN Interface (the key part for coverage) ---
  memset(&g_fake_cinn_stub_interface, 0, sizeof(C_CinnInterface));
  g_fake_cinn_stub_interface.size = sizeof(C_CinnInterface);
  g_fake_cinn_stub_interface.dev_ptr = nullptr;
  g_fake_cinn_stub_interface.compile = FakeCinnCompile;
  g_fake_cinn_stub_interface.get_runtime_source = FakeCinnGetRuntimeSource;
  g_fake_cinn_stub_interface.module_load = FakeCinnModuleLoad;
  g_fake_cinn_stub_interface.module_unload = FakeCinnModuleUnload;
  g_fake_cinn_stub_interface.get_kernel_address = FakeCinnGetKernelAddress;
  g_fake_cinn_stub_interface.launch_kernel = FakeCinnLaunchKernel;
  g_fake_cinn_stub_interface.apply_custom_pass = FakeCinnApplyCustomPass;

  params->interface->cinn_interface = &g_fake_cinn_stub_interface;
}
